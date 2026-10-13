"""Independent QA adapter, deterministic rules and validated Judge composition.

Original documents must come from a trusted caller. LLM facts are not originals.
No routing, counters, retrieval or shared State mutation is performed here.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
import re
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ITEMS = ("groundedness", "neutrality", "bias_control")
SCHEMA_VERSION = "report-quality-v1"
CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\s*\|\s*([A-Za-z0-9_.:-]+)\]")
HEADER = re.compile(r"^\s*#{1,6}\s+(.+)$")
SECRET_PATTERN = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|lsv2_[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|(?:AKIA|ASIA)[A-Z0-9]{16})")


def redact_secrets(value: object) -> object:
    """Defense for QA-owned traces, never logs environment values."""
    if isinstance(value, str):
        value = SECRET_PATTERN.sub("[REDACTED]", value)
        for key, secret in os.environ.items():
            if re.search(r"API_KEY|TOKEN|SECRET|PASSWORD", key, re.I) and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


class QualityJudgeError(RuntimeError):
    """Sanitized exception for the existing worker retry boundary."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceID(StrictModel):
    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class JudgeIssue(StrictModel):
    claim_id: str | None = Field(description="Existing claim ID, or JSON null for a whole-report issue; never the string 'null'.")
    evidence_ids: list[EvidenceID]
    suggested_owner: Literal["technology", "market_traction", "report_generator"]
    company: str | None = Field(description="Candidate company name, or JSON null; analysis owners require a company name.")
    reason: str = Field(min_length=1)
    requested_fix: str = Field(min_length=1)

    @field_validator("claim_id", "company", mode="before")
    @classmethod
    def normalize_null_literal(cls, value):
        # Some structured LLM responses encode nullable IDs as a string.
        # Normalize only this sentinel; unknown IDs and missing analysis owners
        # must still fail the existing reference checks.
        return None if value == "null" else value


class JudgeCheck(StrictModel):
    item: Literal["groundedness", "neutrality", "bias_control"]
    verdict: Literal["pass", "fail", "uncertain"]
    reason: str = Field(min_length=1)
    issues: list[JudgeIssue]
    missing_information: list[str]

    @model_validator(mode="after")
    def consistent(self):
        if self.verdict == "pass" and (self.issues or self.missing_information):
            raise ValueError("pass contains unresolved issues")
        if self.verdict != "pass" and not self.issues:
            raise ValueError("non-pass requires issues")
        if self.verdict == "uncertain" and not self.missing_information:
            raise ValueError("uncertain requires missing_information")
        return self


class JudgeOutput(StrictModel):
    checked_claim_ids: list[str]
    checks: list[JudgeCheck]


def fingerprint(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode()).hexdigest()


def report_identity(state: dict) -> tuple[int | str | None, str]:
    """독립 QA의 작성 차수와 실제 본문 hash. 공통 State 필드를 추가하지 않는다."""
    version = state.get("report_version", state.get("report_attempts"))
    report = state.get("final_report")
    return version, sha256((report if isinstance(report, str) else "").encode()).hexdigest()


def context_fingerprint(state: dict) -> str:
    return fingerprint({key: state.get(key) for key in (
        "candidate_companies", "company_profiles", "references", "technology_results", "market_traction_results",
        "competition_result", "investment_results")})


def result_is_current(state: dict, result: dict | None = None, *, original_documents: dict | None = None) -> bool:
    """독립 QA 상세 결과의 버전·본문·근거가 현재 입력과 일치하는지 검사한다.

    evaluate_report_details()의 결과를 result에 전달한다. 유효성은 합격과 다르다.
    공통 evaluate_report()/quality_result에는 hash metadata가 없으므로 그 결과만
    전달하면 False다. Supervisor의 QA 무효화·재검사 제어를 대신하지 않는다.
    original_documents를 전달하면 실제 원문 입력 fingerprint도 비교한다.
    """
    result = result if result is not None else state.get("quality_result")
    version, digest = report_identity(state)
    current = bool(result and state.get("final_report") and version is not None
                and state.get("node_status", {}).get("quality_evaluator", "DONE") == "DONE"
                and result.get("report_version") == version and result.get("report_hash") == digest
                and result.get("context_hash") == context_fingerprint(state))
    if current and original_documents is not None:
        payload, _, _ = prepare_input(state, original_documents)
        current = result.get("input_fingerprint") == fingerprint(payload)
    return current


def _split_report(report: str) -> tuple[list[dict], str, set[str]]:
    claims, references, sections = [], [], set()
    section = ""
    for number, line in enumerate(report.splitlines(), 1):
        heading = HEADER.match(line)
        if heading:
            section = heading.group(1)
            for name in ("SUMMARY", "REFERENCE"):
                if re.search(rf"\b{name}\b", section, re.I):
                    sections.add(name)
            continue
        if not line.strip():
            continue
        if re.search(r"\bREFERENCE\b", section, re.I):
            references.append(line)
        elif not re.fullmatch(r"[\s|:\-]+", line):
            claims.append({"claim_id": f"line:{number}", "location": f"line:{number}",
                           "statement": line.strip(), "section": section,
                           "evidence_ids": [{"source_id": s, "chunk_id": c}
                                            for c, s in CITATION.findall(line)]})
    return claims, "\n".join(references), sections


def _issue(criterion: str, reason: str, fix: str, claim: dict | None = None,
           refs: list[dict] | None = None, owner: str = "report_generator",
           company: str | None = None) -> dict:
    return {"criterion": criterion, "claim_id": claim["claim_id"] if claim else None,
            "location": claim["location"] if claim else "report",
            "statement": claim["statement"] if claim else "보고서 전체",
            "evidence_ids": refs or [], "owner": owner, "suggested_owner": owner,
            "company": company, "reason": reason, "requested_fix": fix}


def _is_computed_summary(statement: str) -> bool:
    """Recognize policy labels and verdict rows, while keeping them for Judge review."""
    text = re.sub(r"\*\*|__", "", statement).strip()
    if re.match(r"^(?:[-*+]\s+)?판정\s*기준\s*[:：]", text):
        return True
    if not (text.startswith("|") and text.endswith("|")):
        return False
    cells = [cell.strip() for cell in text.strip("|").split("|")]
    decisions = {"INVEST", "HOLD", "HOLD_INSUFFICIENT_EVIDENCE"}
    if len(cells) < 3:
        return False
    if not (cells[1] in decisions or re.fullmatch(r"Q\d+|-", cells[0], re.I)):
        return False
    # A factual sentence appended to a table cell must still require a citation.
    return all(cell in decisions or re.fullmatch(r"(?:[\d\s.,/%()+\-]|개|문항)+|N/A|None|—", cell)
               for cell in cells[2:])


def _add_evidence(registry: dict, item: dict, issues: list[dict]) -> None:
    pair = (item.get("source_id"), item.get("chunk_id"))
    if not all(isinstance(v, str) and v.strip() for v in pair):
        issues.append(_issue("groundedness", "근거 ID 형식 오류", "근거 ID를 복구하세요."))
        return
    old = registry.get(pair)
    fields = ("page", "title", "publisher", "published_date", "source_type", "reference_metadata")
    if old and any(old.get(k) != item.get(k) for k in fields):
        issues.append(_issue("groundedness", "동일 근거 ID의 원본 메타데이터 충돌", "원본을 대조하세요."))
    else:
        registry.setdefault(pair, deepcopy(item))


def prepare_input(state: dict, original_documents: dict | None = None) -> tuple[dict, list[dict], list[str]]:
    """QA-local contract: originals[chunk_id] = {source_id, chunk_id, page, text}."""
    report = state.get("final_report")
    report = report if isinstance(report, str) else ""
    claims, ref_text, sections = _split_report(report)
    version, digest = report_identity(state)
    issues, missing = [], []
    if not report.strip() or report.strip() == "# Markdown Report Generated" or not claims:
        issues.append(_issue("groundedness", "검사할 실제 보고서 본문이 없습니다.",
                             "PDF와 동일한 실제 본문을 final_report로 제공하세요."))
    if not ((type(version) is int and version > 0) or (isinstance(version, str) and version.strip())):
        issues.append(_issue("groundedness", "유효한 보고서 버전이 없습니다.",
                             "report_version 또는 양수 report_attempts를 제공하세요."))
    for section in ("SUMMARY", "REFERENCE"):
        if section not in sections:
            issues.append(_issue("groundedness", f"{section} 목차가 없습니다.",
                                 f"Markdown {section} 제목과 내용을 포함하세요."))
    if "REFERENCE" in sections and not ref_text.strip():
        issues.append(_issue("groundedness", "REFERENCE가 비어 있습니다.", "사용한 출처를 기재하세요."))
    evidence, owners = {}, {}
    known_statements = set()
    for agent, key in (("technology", "technology_results"), ("market_traction", "market_traction_results")):
        for company, analysis in (state.get(key) or {}).items():
            for finding in analysis.get("findings", []):
                statement = CITATION.sub("", finding.get("statement", "")).strip()
                if statement and finding.get("evidence_refs"):
                    known_statements.add(statement)
            for item in analysis.get("evidence", []):
                _add_evidence(evidence, item, issues)
                owners.setdefault((item.get("source_id"), item.get("chunk_id")), (agent, company))
    for investment in (state.get("investment_results") or {}).values():
        for criterion in investment.get("criteria", []):
            for item in criterion.get("evidence", []):
                _add_evidence(evidence, item, issues)
    references = {}
    for ref in state.get("references") or []:
        if ref.get("source_id") in references:
            issues.append(_issue("groundedness", "REFERENCE source_id 중복", "출처별로 병합하세요."))
        references[ref.get("source_id")] = ref
    originals, used = original_documents or {}, set()
    for claim in claims:
        pairs = [(r["source_id"], r["chunk_id"]) for r in claim["evidence_ids"]]
        # Only numeric assertions are flagged by this heuristic; Judge checks all other uncited claims.
        text = re.sub(r"^\s*\d+[.)]\s+", "", CITATION.sub("", claim["statement"]))
        numeric_assertion = re.search(
            r"\d+(?:[.,]\d+)?\s*(?:%|대|명|개|억|만|조|달러|원|년|개월|시간|일|회|USD|KRW|billion|million)"
            r"|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", text, re.I)
        if (not pairs and not _is_computed_summary(claim["statement"])
                and (numeric_assertion or any(s in text for s in known_statements))):
            issues.append(_issue("groundedness", "수치 또는 근거 기반 분석 문장에 인용이 없습니다.",
                                 "근거를 연결하거나 사실이 아닌 표기임을 명확히 하세요.", claim))
        for pair, ref_id in zip(pairs, claim["evidence_ids"]):
            item = evidence.get(pair)
            if item is None:
                issues.append(_issue("groundedness", "인용 ID가 근거에 존재하지 않습니다.",
                                     "존재하는 source_id·chunk_id 쌍으로 수정하세요.", claim, [ref_id]))
                continue
            used.add(pair)
            ref, page = references.get(pair[0]), item.get("page")
            valid_page = type(page) is int and page > 0
            owner, company = owners.get(pair, ("report_generator", None))
            if not valid_page or not item.get("title") or not item.get("publisher"):
                issues.append(_issue("groundedness", "출처·원본 페이지를 추적할 수 없습니다.",
                                     "원본 메타데이터를 복구하세요.", claim, [ref_id], owner, company))
            ref_pages = ref.get("pages", []) if ref else []
            valid_ref_pages = isinstance(ref_pages, list) and all(type(p) is int and p > 0 for p in ref_pages)
            if (not ref or not valid_page or not valid_ref_pages or page not in ref_pages
                    or not (f"[{pair[0]}]" in ref_text or (ref.get("title") and ref["title"] in ref_text))):
                issues.append(_issue("groundedness", "본문 인용과 REFERENCE 출처·페이지 연결 누락",
                                     "해당 출처 ID/제목과 페이지를 REFERENCE에 연결하세요.", claim, [ref_id]))
            elif any(ref.get(k) != item.get(k) for k in ("title", "publisher", "published_date", "reference_metadata")):
                issues.append(_issue("groundedness", "REFERENCE와 근거의 원본 메타데이터가 다릅니다.",
                                     "동일 원본의 메타데이터로 수정하세요.", claim, [ref_id]))
            original = originals.get(pair[1])
            if (not isinstance(original, dict) or original.get("source_id") != pair[0]
                    or original.get("chunk_id") != pair[1] or type(original.get("page")) is not int
                    or original.get("page") != page
                    or not isinstance(original.get("text"), str) or not original["text"].strip()):
                reason = f"{pair[1]}: 검증 가능한 원문 청크·위치가 없습니다. fact는 원문이 아닙니다."
                missing.append(reason)
                issues.append(_issue("groundedness", reason, "신뢰 가능한 원문 조회로 청크를 제공하세요.",
                                     claim, [ref_id], owner, company))
    for pair in sorted(used):
        meta = evidence[pair].get("reference_metadata") or {}
        if not meta.get("source_family_id") and not meta.get("origin_source_id"):
            missing.append(f"{pair[0]}: 동일 발표 재인용 여부를 확인할 출처 계보가 없습니다.")
    if claims and not used:
        issues.append(_issue("groundedness", "검증 가능한 본문 인용이 없습니다.",
                             "보고서 주장과 실제 근거의 연결을 제공하세요."))
    payload_evidence = []
    for pair in sorted(used):
        item = deepcopy(evidence[pair])
        original = originals.get(pair[1]) or {}
        item.update(original_text=original.get("text"), original_hash=fingerprint(original) if original else None)
        payload_evidence.append(item)
    payload = {"report_version": version, "report_hash": digest, "report": report, "claims": claims,
               "references": deepcopy(state.get("references") or []), "evidence": payload_evidence,
               "candidate_companies": state.get("candidate_companies") or [],
               "input_limitations": list(dict.fromkeys(missing))}
    for key in ("company_profiles", "technology_results", "market_traction_results", "competition_result", "investment_results"):
        payload[key] = deepcopy(state.get(key))
    # Investment QA uses the existing agreed rubric, without rewriting it.
    from evaluation.criteria import CRITERIA, INVEST_SCORE_THRESHOLD, MIN_SCORED_CRITERIA
    payload["investment_policy"] = {
        "min_scored_criteria": MIN_SCORED_CRITERIA,
        "invest_score_threshold": INVEST_SCORE_THRESHOLD,
        "criteria": [{"criterion_id": c.id, "name": c.name, "question": c.question} for c in CRITERIA],
    }
    if redact_secrets(payload) != payload:
        # Do not send credentials found in report/source data to an LLM.
        raise QualityJudgeError("QA 입력 데이터에 비밀값이 포함되어 검사를 중단했습니다.")
    return payload, issues, list(dict.fromkeys(missing))


def validate_judge(raw: object, payload: dict) -> JudgeOutput:
    try:
        raw = raw.model_dump() if isinstance(raw, BaseModel) else raw
        if redact_secrets(raw) != raw:
            raise ValueError("sensitive response")
        output = JudgeOutput.model_validate(raw)
        if sorted(c.item for c in output.checks) != sorted(ITEMS):
            raise ValueError("missing/duplicate criteria")
        claims = {c["claim_id"]: c for c in payload["claims"]}
        if sorted(output.checked_claim_ids) != sorted(claims):
            raise ValueError("incomplete claim coverage")
        available = {(e["source_id"], e["chunk_id"]) for e in payload["evidence"]}
        for check in output.checks:
            if not check.reason.strip() or any(not text.strip() for text in check.missing_information):
                raise ValueError("blank reason")
            for issue in check.issues:
                if not issue.reason.strip() or not issue.requested_fix.strip():
                    raise ValueError("blank issue")
                if issue.claim_id is not None and issue.claim_id not in claims:
                    raise ValueError("unknown claim")
                if any((r.source_id, r.chunk_id) not in available for r in issue.evidence_ids):
                    raise ValueError("unknown evidence")
                if issue.claim_id is not None:
                    cited = {(r["source_id"], r["chunk_id"]) for r in claims[issue.claim_id]["evidence_ids"]}
                    if any((r.source_id, r.chunk_id) not in cited for r in issue.evidence_ids):
                        raise ValueError("issue cites evidence unrelated to its claim")
                if issue.suggested_owner != "report_generator" and issue.company not in payload["candidate_companies"]:
                    raise ValueError("analysis issue needs company")
                if issue.company is not None and issue.company not in payload["candidate_companies"]:
                    raise ValueError("unknown company")
        return output
    except Exception:
        raise QualityJudgeError("QA Judge 응답 구조 또는 입력 참조 검증 실패") from None


def _judge_issues(item: str, check: JudgeCheck, payload: dict, state: dict) -> list[dict]:
    by_id = {c["claim_id"]: c for c in payload["claims"]}
    issues = []
    for issue in check.issues:
        claim = by_id.get(issue.claim_id)
        refs = [r.model_dump() for r in issue.evidence_ids]
        if not re.search(r"source_family_id|origin_source_id|출처\s*계보|소스\s*계보", issue.reason + " " + issue.requested_fix, re.I):
            issues.append(_issue(item, issue.reason, issue.requested_fix, claim, refs,
                                 issue.suggested_owner, issue.company))
            continue
        # Metadata repair belongs to the analyses that supplied the cited source.
        selected = refs or (claim["evidence_ids"] if claim else [])
        pairs = {(r["source_id"], r["chunk_id"]) for r in selected}
        if not pairs:
            pairs = {(e["source_id"], e["chunk_id"]) for e in payload["evidence"]
                     if not (e.get("reference_metadata") or {}).get("source_family_id")
                     and not (e.get("reference_metadata") or {}).get("origin_source_id")}
        targets = {}
        for owner, key in (("technology", "technology_results"), ("market_traction", "market_traction_results")):
            for company, analysis in (state.get(key) or {}).items():
                if company not in payload["candidate_companies"] or (issue.company and company != issue.company):
                    continue
                owned = [{"source_id": e.get("source_id"), "chunk_id": e.get("chunk_id")}
                         for e in analysis.get("evidence", [])
                         if (e.get("source_id"), e.get("chunk_id")) in pairs]
                if owned:
                    targets[owner, company] = owned
        if not targets:
            raise QualityJudgeError("QA 출처 계보 지적의 분석 담당·기업을 근거에서 확인할 수 없습니다.")
        for (owner, company), owned in targets.items():
            issues.append(_issue(item, issue.reason, issue.requested_fix, claim, owned, owner, company))
    return issues


def evaluate(state: dict, *, original_documents: dict | None = None,
             judge: Callable[[dict], object] | None = None) -> dict:
    from prompts.report_quality import QA_PROMPT_VERSION
    state = deepcopy(state)
    payload, rules, limitations = prepare_input(state, original_documents)
    checks = [{"item": item, "passed": False, "verdict": "not_evaluated",
               "reason": "결정적 검사 미달로 의미 검사를 실행하지 않았습니다.",
               "issues": [], "missing_information": []} for item in ITEMS]
    judge_used = False
    if not payload["claims"]:
        checks[0].update(verdict="fail", reason="근거·보고서 입력 규칙 미달", issues=rules,
                         missing_information=limitations)
    else:
        if judge is None:
            from core.llm import get_llm_settings, get_structured_llm
            from prompts.report_quality import build_quality_messages
            def judge(data):
                return get_structured_llm(JudgeOutput).invoke(build_quality_messages(data), config={
                    "run_name": "report_quality_judge", "metadata": {
                        "trace_id": state.get("trace_id"), "report_version": data["report_version"],
                        "report_hash": data["report_hash"], "prompt_version": QA_PROMPT_VERSION,
                        **get_llm_settings()}})
        try:
            raw = judge(payload)
        except Exception:
            raise QualityJudgeError("QA Judge 호출 실패; Supervisor 재시도 필요") from None
        output = validate_judge(raw, payload)
        judge_used = True
        checks = []
        for item in ITEMS:
            check = next(c for c in output.checks if c.item == item)
            issues = _judge_issues(item, check, payload, state)
            checks.append({"item": item, "passed": check.verdict == "pass", "verdict": check.verdict,
                           "reason": check.reason, "issues": issues, "missing_information": check.missing_information})
        if rules:
            checks[0].update(passed=False, verdict="fail",
                             reason="근거·보고서 입력 규칙 미달; " + checks[0]["reason"],
                             issues=rules + checks[0]["issues"],
                             missing_information=list(dict.fromkeys(limitations + checks[0]["missing_information"])))
        # 출처 계보 필드는 기존 계약의 필수값이 아니다. 기존 URL·발행기관·원문을
        # 함께 본 Judge의 내용 판정을 사용하며 불확실 판정은 여전히 불합격이다.
    return {"passed": judge_used and all(c["passed"] for c in checks), "checks": checks,
            "feedback": [f"[{c['item']}] {i['statement']} — {i['reason']} 수정: {i['requested_fix']}"
                         for c in checks for i in c["issues"] if i["owner"] == "report_generator"],
            "report_version": payload["report_version"], "report_hash": payload["report_hash"],
            "context_hash": context_fingerprint(state), "schema_version": SCHEMA_VERSION,
            "prompt_version": QA_PROMPT_VERSION, "judge_used": judge_used,
            "input_fingerprint": fingerprint(payload),
            "missing_information": list(dict.fromkeys(limitations + [m for c in checks for m in c["missing_information"]])),
            "trace_id": state.get("trace_id")}
