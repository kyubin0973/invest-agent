"""Market & Traction Agent (설계 산출물 2.1)

역할: 시장·고객·계약·사업모델·팀·상용화 Risk 분석 (현재 기업)
입력: current_company, 현재 기업 자료 + 공통 시장자료, company_profiles (Target Market 등 사전 입력값)
출력: market_traction_results[current_company], market_traction_done

처리 흐름 (LLM 호출 3회)
  1. 검색 계획 (코드)      담당 문항별 영어 하위 질문을 시장자료용 / 기업자료용으로 나눠 둔다.
  2. 검색 (코드)            시장자료: search_industry / 기업자료: 현재 기업 문서만. chunk_id로 중복 제거
  3. 충분성 판단 (LLM)      하위 질문별 sufficient, 답이 되는 원문 문장(answer_quote), 보완 검색어를 한 번에 받는다.
  4. 충분성 확인 (코드)      answer_quote가 실제 청크에 없으면 '충분'이라 해도 부족으로 처리한다.
     재검색 (코드)          부족한 하위 질문만 보완 검색어로 최대 1회 재검색 (설계 2.2.2)
  5. 분석 (LLM)             findings(분석축별), 사용 근거, risks, missing_information
  6. 검증 (코드)            없는 chunk_id 제거, Evidence Level 상·하한, 기업 고유 분석축의 근거 범위 확인
  7. 근거 대조 (LLM)        분석 문장을 인용 원문과 대조: 과장은 고치고, 근거 없는 문장은 제외

- 담당 평가 문항: evaluation.criteria.criteria_for("market_traction") → B01~B03, B05~B10, H12
- Market Evidence와 Company Evidence를 섞지 않는다 (설계 2.2.4).
- 제품 성능은 재평가하지 않는다 (Technology 담당).
- 공개 Team·Founder·가격 정보가 부족하면 추정하지 않는다 (해당 문항은 Judge가 N/A 처리).
"""

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from core.llm import get_structured_llm
from core.state import AnalysisResult, EvidenceItem, Finding, InvestmentState
from documents.company import company_key
from prompts.common import CITATION_RULES, EVIDENCE_RULES, build_system_prompt, build_user_prompt, format_documents
from prompts.market_traction import (
    COMPANY_SPECIFIC_DIMENSIONS,
    DIMENSIONS,
    MARKET_RULES,
    ROLE,
    analysis_request,
    grounding_request,
    sufficiency_request,
)
from rag.retriever import search, search_industry, to_evidence

MARKET_K = 3  # 공통 시장자료: 여러 하위 질문에서 겹치므로 적게
COMPANY_K = 4  # 기업 자료: 일반 검색
COMPANY_TYPE_K = 5  # 기업 자료: 하위 질문에 맞는 document_type으로 좁힌 검색 (설계 2.2.2)
DEFAULT_MARKET = "general-purpose humanoid robots"

# 기업 단독 발표는 외부 확인 전까지 E1~E2, 파트너 발표는 외부 확인(E3) 이상 (설계 3.3 Evidence Level 규칙)
MAX_LEVEL_BY_SOURCE = {"company": "E2"}
MIN_LEVEL_BY_SOURCE = {"partner": "E3"}
DEFAULT_LEVEL_BY_SOURCE = {"company": "E1", "partner": "E3", "industry_report": "E3"}
LEVEL_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5")

# 충분성 판단의 answer_quote가 실제 청크에 있는지 확인하는 기준
# PDF 추출 텍스트의 줄바꿈·합자(ﬁ) 차이를 감안해, 인용 문장의 단어 80% 이상이 청크에 있으면 인정한다.
MIN_QUOTE_WORDS = 4
QUOTE_MATCH_RATIO = 0.8
WORD = re.compile(r"[0-9a-z가-힣]{3,}")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _quote_found(quote: str | None, chunk_texts: list[str]) -> bool:
    """LLM이 옮겨 적은 답 문장이 검색된 청크(제목 포함) 중 하나에 실제로 있는지 확인한다."""
    words = WORD.findall(_normalize(quote or ""))
    if len(words) < MIN_QUOTE_WORDS:
        return False
    for text in map(_normalize, chunk_texts):
        if sum(w in text for w in words) / len(words) >= QUOTE_MATCH_RATIO:
            return True
    return False


# 문장 안에 LLM이 적은 인용 표기 (근거는 evidence_refs에 따로 있으므로 제거)
INLINE_CITATION = re.compile(r"\s*[\(\[](?:인용:\s*)?[A-Z]\d_P\d{2}_C\d{2}[^\)\]]*[\)\]]")


def _clean(statement: str) -> str:
    return INLINE_CITATION.sub("", statement).strip()


# ---------------------------------------------------------------------------
# 1. 검색 계획
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubQuestion:
    id: str
    scope: Literal["market", "company"]
    query: str  # {company}, {market} 자리표시자
    criteria: tuple[str, ...]  # 관련 문항 (Q번호)
    document_types: tuple[str, ...] = ()  # 기업 자료에서 우선 찾을 문서 유형


SUB_QUESTIONS = (
    SubQuestion("SQ01", "market", "{market} market size and growth forecast", ("Q1", "Q8")),
    SubQuestion("SQ02", "market", "labor shortage and automation demand for {market}", ("Q1", "Q2")),
    SubQuestion("SQ03", "market", "total cost of ownership, payback and ROI of humanoid robots versus human labor", ("Q3",)),
    SubQuestion("SQ04", "market", "commercialization barriers, safety, regulation and customer adoption of humanoid robots", ("Q9",)),
    SubQuestion("SQ05", "market", "humanoid robot supply chain constraints, manufacturing cost and scaling", ("Q12",)),
    SubQuestion("SQ06", "company", "{company} customers, pilots, deployments and commercial contracts", ("Q6", "Q2"),
                ("deployment",)),
    SubQuestion("SQ07", "company", "{company} tasks and use cases solved for customers", ("Q2", "Q8"),
                ("deployment", "product")),
    SubQuestion("SQ08", "company", "{company} pricing, business model, subscription or robots-as-a-service", ("Q7", "Q3"),
                ("product",)),
    SubQuestion("SQ09", "company", "{company} founders, CEO, leadership team and experience", ("Q5", "Q10")),
    SubQuestion("SQ10", "company", "{company} manufacturing partners, production capacity and fleet operations", ("Q12",),
                ("manufacturing",)),
)


def _retrieve(scope: str, query: str, company: str, document_types: tuple[str, ...] = ()) -> list[Document]:
    if scope == "market":
        return search_industry(query, k=MARKET_K)
    # 현재 기업 자료만 (다른 후보 자료와 섞지 않음): 일반 검색 + 문서 유형별 검색을 합친다
    found = search(query, company=company, k=COMPANY_K)
    for dt in document_types:
        found += search(query, company=company, document_type=dt, k=COMPANY_TYPE_K)
    return list({d.metadata["chunk_id"]: d for d in found}.values())


# ---------------------------------------------------------------------------
# LLM 출력 형식
# ---------------------------------------------------------------------------
class QuestionCheck(BaseModel):
    question_id: str
    sufficient: bool
    answer_quote: str | None = Field(default=None, description="sufficient=true일 때, 질문에 답하는 원문 문장 (영어 원문 그대로)")
    missing_aspects: list[str] = Field(default_factory=list)
    refined_query: str | None = Field(default=None, description="부족할 때 쓸 영어 검색어 (항상 작성)")


class SufficiencyResult(BaseModel):
    checks: list[QuestionCheck]


class FindingOut(BaseModel):
    question_id: str = Field(description="이 판단이 답하는 하위 질문 (SQ01~SQ10)")
    dimension: Literal[DIMENSIONS]
    statement: str
    evidence_chunk_ids: list[str] = Field(description="이 판단의 근거 chunk_id (source_id는 코드가 채운다)")


class UsedEvidenceOut(BaseModel):
    chunk_id: str
    evidence_level: Literal["E1", "E2", "E3", "E4", "E5"]
    fact: str


class MarketAnalysisOut(BaseModel):
    summary: str
    findings: list[FindingOut]
    evidence: list[UsedEvidenceOut]
    risks: list[str]
    missing_information: list[str]


class FindingCheck(BaseModel):
    finding_id: str = Field(description="F1, F2, ...")
    verdict: Literal["supported", "overstated", "unsupported"]
    revised_statement: str | None = Field(default=None, description="overstated일 때만, 근거에 맞게 고친 문장")
    reason: str


class GroundingResult(BaseModel):
    checks: list[FindingCheck]


# ---------------------------------------------------------------------------
# 6. 검증
# ---------------------------------------------------------------------------
def _cap_level(level: str, source_type: str) -> str:
    """출처 성격에 따라 Evidence Level의 상한·하한을 적용한다."""
    cap, floor = MAX_LEVEL_BY_SOURCE.get(source_type), MIN_LEVEL_BY_SOURCE.get(source_type)
    if cap and LEVEL_ORDER.index(level) > LEVEL_ORDER.index(cap):
        return cap
    if floor and LEVEL_ORDER.index(level) < LEVEL_ORDER.index(floor):
        return floor
    return level


def _validate(out: MarketAnalysisOut, docs: dict[str, Document], company_key: str) -> AnalysisResult:
    """LLM 출력을 검색 결과와 대조해 AnalysisResult로 만든다."""
    notes: list[str] = []

    evidence: dict[str, EvidenceItem] = {}

    def add_evidence(chunk_id: str, level: str | None = None, fact: str | None = None) -> None:
        doc = docs[chunk_id]
        source_type = doc.metadata["source_type"]
        level = _cap_level(level or DEFAULT_LEVEL_BY_SOURCE.get(source_type, "E1"), source_type)
        if chunk_id not in evidence:
            evidence[chunk_id] = {**to_evidence(doc), "evidence_level": level, "fact": fact}

    for e in out.evidence:
        if e.chunk_id in docs:
            add_evidence(e.chunk_id, e.evidence_level, e.fact)
        else:
            notes.append(f"검색 결과에 없는 chunk_id 제외: {e.chunk_id}")

    findings: list[Finding] = []
    for f in out.findings:
        ids = [c for c in dict.fromkeys(f.evidence_chunk_ids) if c in docs]
        if f.dimension in COMPANY_SPECIFIC_DIMENSIONS:
            # 기업 고유 분석축은 현재 기업 자료로만 뒷받침한다 (Market Evidence ≠ Company Evidence)
            ids = [c for c in ids if docs[c].metadata["company"] == company_key]
        if not ids:
            notes.append(f"유효한 근거가 없어 제외한 판단 ({f.dimension}): {f.statement}")
            continue
        for c in ids:
            add_evidence(c)
        findings.append({
            "dimension": f.dimension,
            "statement": _clean(f.statement),
            "evidence_refs": [{"chunk_id": c, "source_id": docs[c].metadata["source_id"]} for c in ids],
        })

    return {
        "summary": out.summary,
        "findings": findings,
        "evidence": list(evidence.values()),
        "risks": out.risks,
        "missing_information": list(dict.fromkeys(out.missing_information + notes)),
    }


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------
def analyze_market_traction(company: str, profile: dict) -> AnalysisResult:
    key = company_key(company)
    market = profile.get("target_market") or DEFAULT_MARKET

    # 2. 검색
    docs: dict[str, Document] = {}
    questions = []
    for sq in SUB_QUESTIONS:
        query = sq.query.format(company=company, market=market)
        found = _retrieve(sq.scope, query, company, sq.document_types)
        docs.update({d.metadata["chunk_id"]: d for d in found})
        questions.append({"id": sq.id, "scope": sq.scope, "query": query, "chunk_ids": [d.metadata["chunk_id"] for d in found]})

    system = build_system_prompt(ROLE)

    # 3. 충분성 판단 (LLM 1회)
    check = get_structured_llm(SufficiencyResult).invoke(
        [("system", system),
         ("user", build_user_prompt(format_documents(list(docs.values())), sufficiency_request(company, questions)))],
        config={"run_name": "market_sufficiency_check"},
    )

    # 4. 충분성 확인 (코드): LLM이 '충분'이라 해도 답 문장이 실제 청크에 없으면 부족으로 본다.
    #    LLM이 판단을 빠뜨린 하위 질문도 부족으로 본다.
    #    확인 범위: 이번에 검색한 모든 청크의 본문과 문서 제목 (다른 하위 질문에서 찾은 근거로도 답할 수 있다)
    checks = {c.question_id: c for c in check.checks}
    collected_texts = [f"{d.metadata['title']}\n{d.page_content}" for d in docs.values()]
    gaps = []
    for q in questions:
        c = checks.get(q["id"])
        if c and c.sufficient and _quote_found(c.answer_quote, collected_texts):
            q["sufficient"] = True
            continue
        q["sufficient"] = False
        if c and c.sufficient:
            gaps.append(f"{q['id']}: 질문에 직접 답하는 원문 문장을 확인하지 못함")
        # 부족한 하위 질문만 1회 재검색 (횟수 제한은 코드가 한다)
        refined = (c.refined_query if c else None) or q["query"]
        if refined != q["query"]:
            found = _retrieve(q["scope"], refined, company)
            docs.update({d.metadata["chunk_id"]: d for d in found})
            q["chunk_ids"] = list(dict.fromkeys(q["chunk_ids"] + [d.metadata["chunk_id"] for d in found]))
            q["retry_query"] = refined
        if c:
            gaps += [f"{q['id']}: {a}" for a in c.missing_aspects]

    # 5. 분석 (LLM 1회)
    out = get_structured_llm(MarketAnalysisOut).invoke(
        [("system", system),
         ("user", build_user_prompt(
             format_documents(list(docs.values())),
             analysis_request(company, profile.get("target_market"), questions, gaps),
             EVIDENCE_RULES, CITATION_RULES, MARKET_RULES,
         ))],
        config={"run_name": "market_analysis"},
    )

    # 6. 검증 (코드)
    result = _validate(out, docs, key)
    answered = {f.question_id for f in out.findings}
    skipped = [q["id"] for q in questions if q["chunk_ids"] and q["id"] not in answered]
    if skipped:
        result["missing_information"].append(f"근거는 검색됐으나 분석되지 않은 하위 질문: {', '.join(skipped)}")

    # 7. 근거 대조 (LLM 1회): 과장은 원문에 맞게 고치고, 근거 없는 문장은 제외
    return _ground(result, docs, company, system)


def _ground(result: AnalysisResult, docs: dict[str, Document], company: str, system: str) -> AnalysisResult:
    if not result["findings"]:
        return result
    cited = list(dict.fromkeys(r["chunk_id"] for f in result["findings"] for r in f["evidence_refs"]))
    check = get_structured_llm(GroundingResult).invoke(
        [("system", system),
         ("user", build_user_prompt(
             format_documents([docs[c] for c in cited]),
             grounding_request(company, result["findings"]),
             EVIDENCE_RULES, CITATION_RULES, MARKET_RULES,
         ))],
        config={"run_name": "market_grounding_check"},
    )
    verdicts = {c.finding_id: c for c in check.checks}
    kept, notes = [], []
    for i, f in enumerate(result["findings"], start=1):
        c = verdicts.get(f"F{i}")
        if c is None or c.verdict == "supported":
            kept.append(f)
        elif c.verdict == "overstated" and c.revised_statement:
            kept.append({**f, "statement": _clean(c.revised_statement)})
        else:
            notes.append(f"근거와 맞지 않아 제외한 판단 ({f['dimension']}): {f['statement']}")

    # 남은 판단이 인용한 근거만 유지한다 (REFERENCE에 실제 사용 출처만 남도록)
    used = {r["chunk_id"] for f in kept for r in f["evidence_refs"]}
    return {
        **result,
        "findings": kept,
        "evidence": [e for e in result["evidence"] if e["chunk_id"] in used],
        "missing_information": list(dict.fromkeys(result["missing_information"] + notes)),
    }


def market_traction_node(state: InvestmentState) -> dict:
    company = state["current_company"]
    result = analyze_market_traction(company, state["company_profiles"][company])
    return {"market_traction_results": {company: result}, "market_traction_done": True}
