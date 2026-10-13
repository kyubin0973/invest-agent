"""Technology & Product Agent — 10쪽 최종 설계서 기준.

현재 기업 기술자료의 하위 질문별 검색 → 충분성 점검 → 부족 질문만 1회 보완
→ 구조화 분석 → Python 근거 검증. 함수/State 계약은 기존 그래프와 동일하다.
채점·기업 간 비교·시장 평가는 하지 않는다. API/검색 장애는 호출자에게 전파하며
자료가 없는 경우와 구분한다. 원문 의미의 정확성은 별도 실제 API 검토가 필요하다.

Supervisor 재작업(rework_request.agent == "technology")을 받으면 처음부터 다시 분석하지 않는다.
- 근거 부족 평가 문항(B02/B04/B08/B09/H11/H12)을 내부 질문으로 바꿔 그 질문만 다시 검색한다.
  최초 검색과 다른 관점의 질의(rework_query)와 넓힌 Top-K(REWORK_K)를 쓴다.
- 품질 평가가 지적한 문장은 기존 결과에서 빼고, 그 문장이 속한 질문을 다시 원문과 대조한다.
- 재작업 대상이 아닌 질문의 검증된 결과는 그대로 두고, 대상 질문만 새 결과로 바꾼다.
  새 결과가 검증을 통과하지 못하면 기존 검증 결과를 유지한다(지적된 문장은 복원하지 않음).
- 재작업 실행은 LangSmith에 technology_rework로 묶이고, 결과 요약은 trace_id와 함께 로그로 남긴다.
"""

from datetime import datetime, timezone
from typing import Literal
import logging
import re
import unicodedata

from langchain_core.documents import Document
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field

from core.llm import get_structured_llm
from core.state import AnalysisResult, InvestmentState, ReworkRequest
from documents.company import company_key
from prompts.technology import (
    CRITERION_QUESTIONS,
    QUESTIONS,
    QUESTIONS_BY_ID,
    TechnologyQuestion,
    build_analysis_messages,
    build_review_messages,
    build_rework_note,
)
from rag.config import TOP_K
from rag.retriever import search_technology, to_evidence

logger = logging.getLogger(__name__)

AGENT = "technology"
REWORK_K = TOP_K * 2  # 재작업 검색은 최초보다 넓게 본다
MAX_REWORK_NOTE_ITEMS = 10  # 재작업 지시에 넣는 이전 부족 정보 상한 (프롬프트 비대 방지)


# Technology 내부 하위 질문 ID. B02/H11 등의 투자평가 Criterion ID와 별개다.
TechnologyQuestionId = Literal[
    "product", "architecture", "autonomy", "reliability",
    "generalization", "safety", "manufacturing", "supply_chain",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceReference(_Model):
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class QuestionReview(_Model):
    question_id: TechnologyQuestionId = Field(
        description="Technology 내부 분석 ID만 사용. 투자평가 문항 ID(B02/B04/B08/B09/H11/H12)는 금지."
    )
    relevant_refs: list[EvidenceReference]
    sufficient: bool
    missing_information: list[str]
    retry_query: str


class RetrievalReview(_Model):
    reviews: list[QuestionReview]


class EvidenceDraft(EvidenceReference):
    fact: str = Field(min_length=1)
    supporting_quote: str = Field(
        min_length=1,
        description="Exception to Korean output: copy a continuous verbatim quote from the specified "
        "source_id + chunk_id in its original language (English source -> English quote). "
        "Do not translate, paraphrase, correct grammar, replace words, summarize, combine sentences, "
        "or add text. E2라면 실제 Prototype/Demo/제한 테스트 근거가 필요."
    )
    evidence_level: Literal["E1", "E2", "E3", "E4", "E5"] = Field(
        description="E1: 기업 주장/계획. E2: 실제 Prototype/Demo/제한 테스트 근거가 있을 때만 사용하며 "
        "일반 기술 설명이나 불확실성의 중간 등급이 아님. E3: 고객/파트너/독립 출처가 확인한 현재 사실의 범위만. "
        "예: TI가 확인한 Apollo 모터 제어 기술 적용은 E3 가능하나 장기 신뢰성이나 미래 성능 개선으로 확대 금지. "
        "company는 E1/E2만 허용. E4/E5는 외부 확인된 실제 운영·계약/반복·규모화에만 부여."
    )


class FindingDraft(_Model):
    question_id: TechnologyQuestionId = Field(
        description="Technology 내부 분석 ID만 사용. 투자평가 문항 ID(B02/B04/B08/B09/H11/H12)는 금지."
    )
    statement: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference]


class TechnologyAnalysis(_Model):
    evidence: list[EvidenceDraft]
    findings: list[FindingDraft]
    risks: list[FindingDraft]
    missing_information: list[str]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _normalize_quote(text: str) -> str:
    """인용 비교용 정규화만 수행한다. 소실된 글자나 의미는 추정하지 않는다."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        " " if char.isspace() else char
        for char in text
        if char.isspace() or unicodedata.category(char) not in ("Cc", "Cf")
    )
    return " ".join(text.split())


def _matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _confirmed_sentences(text: str) -> str:
    """계획·가능성·부정 문장을 실적 판정에서 제외한다. 의미를 추정하지 않는다."""
    return " ".join(
        sentence for sentence in re.split(r"(?<=[.!?])\s+", _normalize_quote(text))
        if not _matches(
            r"\b(will|would|could|can|may|might|aim\w*|plan\w*|goal\w*|upcoming|future|"
            r"potential|designed to|not|never|no|said|claims?|according to)\b|"
            r"계획|목표|예정|향후|가능성|미확인|않|없", sentence
        )
    )


_DEMO = r"\b(demonstrated|tested|prototype|performed a .{0,30}test|controlled test)\b|시연했|시험했|테스트했|시제품"
_CURRENT = r"\b(uses?|utilizes?|incorporated|integrated|addressed|provides?|demonstrated|tested|operated)\b|적용되어|사용하고|확인했다"
_OPERATION = r"\b(operated|deployed|in operation|signed .{0,40}contract)\b|실제 운영했|가동 중|계약을 체결"
_REPEATED = r"\b(repeatedly|daily|every shift|at scale|fleet of)\b|매일|반복 운영|규모화"
_COMPONENT = r"\b(components?|parts?|motor.control|technology|technologies|microcontrollers?|gate drivers?)\b|부품|기술|모터|구동"
_SAFETY = r"\bsafe(?:ty)?\b|안전"


def _system_safety_confirmed(text: str) -> bool:
    # 부품 인증이나 'upcoming system-level certification'은 이 조건을 충족하지 않는다.
    return _matches(
        r"\b(?:robot|apollo|humanoid|robot system) (?:has been|was|is) (?:safety.certified|certified safe)\b|"
        r"\bsystem.level (?:functional )?safety (?:certification|testing) (?:was|is|has been) (?:completed|passed)\b|"
        r"로봇 전체 시스템.{0,20}(?:인증을 받았|안전 시험을 통과)",
        _confirmed_sentences(text),
    )


def _scope_error(statement: str, quote: str) -> str | None:
    confirmed = _confirmed_sentences(quote)
    if _matches(_SAFETY, quote) and _matches(_SAFETY, statement) and not _system_safety_confirmed(quote):
        # 부품 안전 인증의 적용 사실은 허용하지만 전체 안전성의 보장은 제외한다.
        positive = _confirmed_sentences(statement)
        if _matches(_SAFETY, positive) and (
            not _matches(_COMPONENT, positive)
            or _matches(r"안전성.{0,12}(?:확보|보장|검증|인증)|안전하게.{0,15}(?:작업|운영|작동)|"
                        r"전체.{0,20}(?:안전|인증)|\b(?:robot|apollo) is (?:safe|certified)\b", positive)
        ):
            return "부품·적용 기술의 안전 근거를 로봇 전체 안전성으로 확대함"
    if _matches(r"장기.{0,10}(?:안정|신뢰)|성공률|가동률|반복.{0,10}(?:검증|성과)|상용.{0,10}(?:검증|운영)", statement):
        if not _matches(_OPERATION, confirmed):
            return "적용 기술 설명을 장기 신뢰성·반복 성과·상용 운영으로 확대함"
    return None


def _level_error(item: EvidenceDraft, source_type: str) -> str | None:
    """허용 근거가 명시되지 않은 등급은 제외한다. 자동 승급/강등하지 않는다."""
    level = item.evidence_level
    confirmed = _confirmed_sentences(item.supporting_quote)
    if source_type == "company" and level not in ("E1", "E2"):
        return "기업 자체 자료에 외부 검증 수준(E3 이상)을 부여함"
    if level == "E2" and not _matches(_DEMO, confirmed):
        return "E2에 필요한 실제 Prototype/Demo/제한 테스트 근거 없음"
    if level in ("E3", "E4", "E5"):
        if source_type not in ("partner", "customer", "independent"):
            return "외부 확인 출처가 아님"
        required = _CURRENT if level == "E3" else _OPERATION
        if not _matches(required, confirmed) or (level == "E5" and not _matches(_REPEATED, confirmed)):
            return f"{level}에 필요한 현재 사실·운영·반복 확인 근거 없음"
    return _scope_error(item.fact, item.supporting_quote)


def _search(company: str, queries: list[str], scope: dict | None, k: int = TOP_K) -> list[Document]:
    """검색기의 기업/분야 필터를 사용하고 프로필의 허용 범위도 확인한다."""
    docs = {}
    key = company_key(company)
    for query in _unique(queries):
        for doc in search_technology(f"{company} {query}", company, k=k):
            m = doc.metadata
            if m.get("company") != key or m.get("domain") != "technology":
                continue
            if scope is not None and (
                m.get("source_id") not in scope["source_ids"]
                or m.get("document_type") not in scope["document_types"]
            ):
                continue
            if doc.page_content.strip():
                # 원본 출처가 다른 동일 내용은 합치지 않는다.
                docs.setdefault(m["chunk_id"], doc)
    return list(docs.values())


def _valid_ref(ref: EvidenceReference, available: dict[str, dict]) -> bool:
    item = available.get(ref.chunk_id)
    return item is not None and item["source_id"] == ref.source_id


def _review(
    company: str, docs: list[Document], *, can_retry: bool,
    questions: tuple[TechnologyQuestion, ...] = QUESTIONS, rework_note: str = "",
    run_name: str = "technology_review",
) -> list[QuestionReview]:
    targets = {q.id for q in questions}
    if docs:
        raw = get_structured_llm(RetrievalReview).invoke(
            build_review_messages(company, docs, can_retry=can_retry, questions=questions, rework_note=rework_note),
            config={"run_name": run_name},
        )
        review = RetrievalReview.model_validate(raw)
        by_id = {}
        for item in review.reviews:
            if item.question_id not in QUESTIONS_BY_ID or item.question_id in by_id:
                raise ValueError(f"Technology 근거 점검의 잘못된/중복 question_id: {item.question_id}")
            if item.question_id in targets:  # 재작업 범위 밖 질문의 점검은 쓰지 않는다
                by_id[item.question_id] = item
    else:
        by_id = {}

    available = {d.metadata["chunk_id"]: d.metadata for d in docs}
    checked = []
    for question in questions:
        item = by_id.get(question.id)
        if item is None:
            item = QuestionReview(
                question_id=question.id, relevant_refs=[], sufficient=False,
                missing_information=["질문에 대한 근거 점검 결과 없음" if docs else "검색된 근거 없음"],
                retry_query="",
            )
        refs = [ref for ref in item.relevant_refs if _valid_ref(ref, available)]
        invalid = len(refs) != len(item.relevant_refs)
        missing = _unique(item.missing_information)
        if invalid:
            missing.append("근거 점검의 인용 ID가 검색 원본과 일치하지 않음")
        if question.id == "safety":
            safety_docs = [d.page_content for d in docs if any(r.chunk_id == d.metadata["chunk_id"] for r in refs)]
            if any(_matches(_SAFETY, t) and _matches(_COMPONENT, t) for t in safety_docs) and not any(
                _system_safety_confirmed(t) for t in safety_docs
            ):
                missing.append("부품·적용 기술 수준의 안전 근거만 확인됨; 로봇 전체 시스템·실환경 안전 검증 미확인")
        sufficient = item.sufficient and bool(refs) and not invalid and not missing
        if not sufficient and not missing:
            missing = ["질문에 답할 근거가 충분하지 않음"]
        checked.append(QuestionReview(
            question_id=question.id, relevant_refs=refs, sufficient=sufficient,
            missing_information=missing,
            retry_query=(item.retry_query or f"{question.query} evidence measured results limitations")
            if can_retry and not sufficient else "",
        ))
    return checked


def _render_statement(statement: str, refs: list[dict], evidence: dict[str, dict]) -> str:
    if any(evidence[ref["chunk_id"]]["source_type"] == "company" for ref in refs):
        statement = f"기업 자체 발표 포함(외부 확인과 구분): {statement}"
    citations = " ".join(f"[{ref['chunk_id']} | {ref['source_id']}]" for ref in refs)
    return f"{statement} {citations}"


def _assemble(
    company: str, docs: list[Document], draft: TechnologyAnalysis, missing: list[str],
    questions: tuple[TechnologyQuestion, ...] = QUESTIONS,
) -> AnalysisResult:
    """LLM의 Metadata를 사용하지 않고 실제 원문에서 근거를 재구성한다."""
    originals = {d.metadata["chunk_id"]: d for d in docs}
    available = {key: to_evidence(doc) for key, doc in originals.items()}
    missing = [*missing, *draft.missing_information]
    evidence = {}
    # 같은 청크에 상충하는 분석이 있으면 임의로 한쪽을 선택하지 않는다.
    counts = {}
    for item in draft.evidence:
        counts[item.chunk_id] = counts.get(item.chunk_id, 0) + 1
    for item in draft.evidence:
        reason = None
        if not _valid_ref(item, available):
            reason = "인용 ID가 검색 원본과 일치하지 않음"
        elif counts[item.chunk_id] > 1:
            reason = "동일 청크의 근거 분석이 중복됨"
        elif not _normalize_quote(item.supporting_quote) or (
            _normalize_quote(item.supporting_quote) not in _normalize_quote(originals[item.chunk_id].page_content)
        ):
            reason = "뒷받침 인용문이 원문에 없음"
        else:
            reason = _level_error(item, available[item.chunk_id]["source_type"])
        if reason:
            missing.append(f"근거 검증 실패({item.chunk_id}): {reason}")
            continue
        fact = item.fact
        if available[item.chunk_id]["source_type"] == "company":
            fact = f"{company} 자체 발표에 따르면: {fact}"
        evidence[item.chunk_id] = {
            **available[item.chunk_id], "fact": fact, "evidence_level": item.evidence_level,
        }

    # 참조의 존재 여부와 LLM의 별도 근거 분석 누락은 구분한다.
    # 누락 항목은 원본 Metadata만 보존하고 fact/level은 None으로 둔다.
    # 명시적으로 제출됐으나 위 검증에서 탈락한 근거는 복원하지 않는다.
    for finding in [*draft.findings, *draft.risks]:
        for ref in finding.evidence_refs:
            if ref.chunk_id not in counts and _valid_ref(ref, available):
                evidence.setdefault(ref.chunk_id, available[ref.chunk_id])
                missing.append(f"{ref.chunk_id}: LLM 근거 분석 누락으로 fact/evidence_level 미분류")

    used_ids = set()
    covered_questions = set()
    targets = {q.id for q in questions}

    def validate_findings(items: list[FindingDraft]) -> list[dict]:
        findings = []
        for item in items:
            if item.question_id not in QUESTIONS_BY_ID:
                missing.append(f"분석 범위 밖 question_id 제외: {item.question_id}")
                continue
            if item.question_id not in targets:  # 재작업 대상이 아닌 질문은 기존 결과를 유지한다
                continue
            # 일부 인용을 제거한 뒤 원래 복합 주장을 그대로 남기지 않는다.
            if not item.evidence_refs or not all(_valid_ref(ref, evidence) for ref in item.evidence_refs):
                missing.append(f"{QUESTIONS_BY_ID[item.question_id].dimension}: 유효 근거가 없는 분석 문장 제외")
                continue
            quotes = " ".join(
                next((e.supporting_quote for e in draft.evidence if e.chunk_id == ref.chunk_id),
                     originals[ref.chunk_id].page_content)
                for ref in item.evidence_refs
            )
            scope_error = _scope_error(item.statement, quotes)
            if scope_error:
                missing.append(f"{QUESTIONS_BY_ID[item.question_id].dimension}: {scope_error}")
                continue
            refs = list({(r.chunk_id, r.source_id): r.model_dump() for r in item.evidence_refs}.values())
            used_ids.update(ref["chunk_id"] for ref in refs)
            covered_questions.add(item.question_id)
            findings.append({
                "dimension": QUESTIONS_BY_ID[item.question_id].dimension,
                "statement": _render_statement(item.statement, refs, evidence),
                "evidence_refs": refs,
            })
        return findings

    findings = validate_findings(draft.findings)
    risks = validate_findings(draft.risks)
    for question in questions:
        if question.id not in covered_questions:
            missing.append(f"{question.dimension}: 검증된 분석 결과 없음")
    # 자유 요약에 검증 실패한 주장이 남지 않도록 검증된 문장만 재사용한다.
    summary_items = findings or risks
    summary = "\n".join(item["statement"] for item in summary_items[:3])
    return {
        "summary": summary or f"{company}: 기술·제품을 판단할 검증된 근거가 부족합니다.",
        "findings": findings,
        "evidence": [item for key, item in evidence.items() if key in used_ids],
        "risks": [item["statement"] for item in risks],
        "missing_information": _unique(missing),
    }


def _analyze(
    company: str, scope: dict | None, questions: tuple[TechnologyQuestion, ...] = QUESTIONS, *,
    queries: list[str] | None = None, k: int = TOP_K, rework_note: str = "", run_prefix: str = "technology",
) -> AnalysisResult:
    """하위 질문별 검색 → 충분성 점검 → 부족 질문만 1회 보완 → 구조화 분석 → Python 근거 검증."""
    docs = _search(company, queries or [q.query for q in questions], scope, k=k)
    reviews = _review(company, docs, can_retry=True, questions=questions,
                      rework_note=rework_note, run_name=f"{run_prefix}_review")
    retries = [r.retry_query for r in reviews if not r.sufficient]
    if retries:
        extra = _search(company, retries, scope, k=k)
        docs = list({d.metadata["chunk_id"]: d for d in [*docs, *extra]}.values())
        # 1회 보완 라운드 후에는 더 이상 검색하지 않는다.
        reviews = _review(company, docs, can_retry=False, questions=questions,
                          rework_note=rework_note, run_name=f"{run_prefix}_review_final")

    missing = [
        f"{QUESTIONS_BY_ID[r.question_id].dimension}: {message}"
        for r in reviews if not r.sufficient for message in r.missing_information
    ]
    relevant_ids = {ref.chunk_id for r in reviews for ref in r.relevant_refs}
    relevant_docs = [d for d in docs if d.metadata["chunk_id"] in relevant_ids]
    if relevant_docs:
        raw = get_structured_llm(TechnologyAnalysis).invoke(
            build_analysis_messages(company, relevant_docs, missing, questions=questions, rework_note=rework_note),
            config={"run_name": f"{run_prefix}_analysis"},
        )
        draft = TechnologyAnalysis.model_validate(raw)
    else:
        draft = TechnologyAnalysis(evidence=[], findings=[], risks=[], missing_information=[])
    return _assemble(company, docs, draft, missing, questions=questions)


# ---------------------------------------------------------------------------
# Supervisor 재작업
# ---------------------------------------------------------------------------
_CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\s*\|\s*([A-Za-z0-9_.:-]+)\]")  # QA와 같은 인용 형식
_COMPANY_CLAIM = "기업 자체 발표 포함(외부 확인과 구분):"
_DIMENSION_ORDER = {q.dimension: i for i, q in enumerate(QUESTIONS)}
_QUESTION_BY_DIMENSION = {q.dimension: q for q in QUESTIONS}


def _rework_request(state: InvestmentState, company: str) -> ReworkRequest | None:
    """현재 기업의 Technology 재작업 요청만 받는다. 다른 Agent·기업의 요청은 무시한다."""
    request = state.get("rework_request")
    if not request or request.get("agent") != AGENT or request.get("company") != company:
        return None
    return request


def _cited(text: str) -> set[str]:
    return {chunk_id for chunk_id, _ in _CITATION.findall(text)}


def _claim_text(text: str) -> str:
    """인용 표시·기업 발표 표기·말줄임을 지운 비교용 문장."""
    text = _CITATION.sub(" ", text).replace(_COMPANY_CLAIM, " ").replace("…", " ")
    return " ".join(text.split())


def _is_flagged(statement: str, note: str) -> bool:
    """품질 평가 지적이 이 분석 문장을 가리키는지: 같은 청크 인용 또는 문장 일치(보고서 말줄임 포함)."""
    if _cited(statement) & _cited(note):
        return True
    claim, text = _claim_text(statement), _claim_text(note)
    return len(claim) >= 10 and (claim in text or claim[:40] in text)


def _rework_scope(
    request: ReworkRequest, previous: AnalysisResult,
) -> tuple[tuple[TechnologyQuestion, ...], set[str], list[str]]:
    """재작업할 질문, 빼야 할 기존 문장, 대응 질문이 없는 평가 문항을 정한다."""
    target_ids: set[str] = set()
    unmapped = []
    for criterion_id in request.get("criterion_ids") or []:
        if criterion_id in CRITERION_QUESTIONS:
            target_ids.update(CRITERION_QUESTIONS[criterion_id])
        else:
            unmapped.append(criterion_id)

    flagged: set[str] = set()
    for note in request.get("feedback") or []:
        findings = [f for f in previous.get("findings", []) if _is_flagged(f["statement"], note)]
        risks = [r for r in previous.get("risks", []) if _is_flagged(r, note)]
        flagged.update(f["statement"] for f in findings)
        flagged.update(risks)
        target_ids.update(_QUESTION_BY_DIMENSION[f["dimension"]].id for f in findings
                          if f["dimension"] in _QUESTION_BY_DIMENSION)
        risk_chunks = set().union(*(_cited(r) for r in risks)) if risks else set()
        target_ids.update(_QUESTION_BY_DIMENSION[f["dimension"]].id for f in previous.get("findings", [])
                          if f["dimension"] in _QUESTION_BY_DIMENSION
                          and risk_chunks & {ref["chunk_id"] for ref in f["evidence_refs"]})
        if not findings and not risks:
            # 지적 문장을 기존 분석과 연결하지 못하면 모든 질문을 다시 원문과 대조한다.
            target_ids.update(QUESTIONS_BY_ID)

    if not target_ids:  # 대응 질문이 없는 요청이면 전체 질문을 다시 점검한다
        target_ids.update(QUESTIONS_BY_ID)
    return tuple(q for q in QUESTIONS if q.id in target_ids), flagged, unmapped


def _merge(
    previous: AnalysisResult, fresh: AnalysisResult,
    questions: tuple[TechnologyQuestion, ...], flagged: set[str], company: str,
) -> AnalysisResult:
    """대상 질문 중 새로 검증된 결과가 나온 질문만 바꾸고, 나머지 검증 결과와 부족 정보는 유지한다."""
    target_dims = {q.dimension for q in questions}
    replaced = target_dims & {f["dimension"] for f in fresh["findings"]}
    kept = [
        f for f in previous.get("findings", [])
        if f["statement"] not in flagged and f["dimension"] not in replaced
    ]
    findings = sorted([*kept, *fresh["findings"]], key=lambda f: _DIMENSION_ORDER.get(f["dimension"], len(QUESTIONS)))
    risks = _unique([*(r for r in previous.get("risks", []) if r not in flagged), *fresh["risks"]])

    used = {ref["chunk_id"] for f in findings for ref in f["evidence_refs"]}
    used |= set().union(*(_cited(r) for r in risks)) if risks else set()
    evidence = {e["chunk_id"]: e for e in previous.get("evidence", [])}
    evidence.update({e["chunk_id"]: e for e in fresh["evidence"]})  # 새로 검증한 근거가 우선

    covered = {f["dimension"] for f in findings}
    resolved_notes = {f"{dim}: 검증된 분석 결과 없음" for dim in covered}
    missing = [
        m for m in previous.get("missing_information", [])
        if m not in resolved_notes and not any(m.startswith(f"{dim}:") for dim in replaced)
    ]
    missing += [m for m in fresh["missing_information"] if m not in resolved_notes]
    missing += [f"{q.dimension}: 검증된 분석 결과 없음" for q in questions if q.dimension not in covered]

    summary = "\n".join(f["statement"] for f in (findings or [{"statement": r} for r in risks])[:3])
    return {
        "summary": summary or f"{company}: 기술·제품을 판단할 검증된 근거가 부족합니다.",
        "findings": findings,
        "evidence": [item for key, item in evidence.items() if key in used],
        "risks": risks,
        "missing_information": _unique(missing),
    }


@traceable(name="technology_rework")
def _rework(
    company: str, scope: dict | None, previous: AnalysisResult, request: ReworkRequest, trace_id: str | None = None,
) -> AnalysisResult:
    """Supervisor가 지정한 부족 질문만 다시 검색·검증하고 기존 결과와 병합한다."""
    questions, flagged, unmapped = _rework_scope(request, previous)
    target_dims = {q.dimension for q in questions}
    previous_missing = [
        m for m in previous.get("missing_information", []) if any(m.startswith(f"{dim}:") for dim in target_dims)
    ][:MAX_REWORK_NOTE_ITEMS]
    note = build_rework_note(
        request.get("reason", ""), list(request.get("criterion_ids") or []), list(request.get("feedback") or []),
        questions, previous_missing,
    )
    # 최초 질의(넓힌 Top-K)로 기존 근거를 다시 대조하고, 다른 관점 질의로 새 근거를 찾는다.
    queries = [query for q in questions for query in (q.rework_query, q.query) if query]
    fresh = _analyze(company, scope, questions, queries=queries, k=REWORK_K,
                     rework_note=note, run_prefix="technology_rework")
    result = _merge(previous, fresh, questions, flagged, company)

    covered = {f["dimension"] for f in result["findings"]}
    outcome = {
        "trace_id": trace_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "company": company,
        "criterion_ids": list(request.get("criterion_ids") or []),
        "unmapped_criterion_ids": unmapped,
        "feedback_count": len(request.get("feedback") or []),
        "questions": [q.id for q in questions],
        "resolved": [q.id for q in questions if q.dimension in covered],
        "unresolved": [q.id for q in questions if q.dimension not in covered],
        "removed_statements": len(flagged),
        "evidence_before": len(previous.get("evidence", [])),
        "evidence_after": len(result["evidence"]),
    }
    logger.info(
        "%s [%s] technology rework company=%s criteria=%s questions=%s resolved=%s unresolved=%s "
        "removed=%d evidence=%d→%d",
        outcome["ts"], trace_id, company, outcome["criterion_ids"] or "-", outcome["questions"],
        outcome["resolved"] or "-", outcome["unresolved"] or "-", outcome["removed_statements"],
        outcome["evidence_before"], outcome["evidence_after"],
    )
    return result


def technology_node(state: InvestmentState) -> dict:
    company = state.get("current_company")
    if not isinstance(company, str) or not company.strip():
        raise ValueError("Technology Agent에는 비어 있지 않은 current_company가 필요합니다.")
    scope = state.get("company_profiles", {}).get(company, {}).get("document_scope")
    request = _rework_request(state, company)
    previous = (state.get("technology_results") or {}).get(company)
    if request and previous:
        result = _rework(company, scope, previous, request, trace_id=state.get("trace_id"))
    else:
        result = _analyze(company, scope)
    return {"technology_results": {company: result}, "technology_done": True}
