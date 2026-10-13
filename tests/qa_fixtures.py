"""Synthetic QA data only; never a submission report or real trace."""
from copy import deepcopy

from evaluation.report_quality import ITEMS

NORMAL = "A 발표에 따르면 실험실 데모를 공개했으며 상용 배치는 확인되지 않았다. [S1_P01_C01 | S1]"
EXAGGERATED = "A는 대규모 상용 배치 성과를 달성했다. [S1_P01_C01 | S1]"
SUPERIOR = "A의 기술은 모든 경쟁 기술보다 우수하므로 반드시 채택해야 한다. [S1_P01_C01 | S1]"


def fixture(statement: str = NORMAL, version: int = 1) -> tuple[dict, dict]:
    metadata = {"reference_type": "web", "url": "https://example.test/demo",
                "source_family_id": "announcement:demo"}
    evidence = {"source_id": "S1", "chunk_id": "S1_P01_C01", "page": 1,
                "title": "Demo announcement", "publisher": "A", "published_date": "2026-01-01",
                "source_type": "company", "reference_metadata": metadata,
                "fact": "A claims a lab demonstration, not commercial deployment.", "evidence_level": "E1"}
    reference = {k: deepcopy(evidence[k]) for k in (
        "source_id", "title", "publisher", "published_date", "reference_metadata")}
    reference["pages"] = [1]
    analysis = {"summary": "기업 발표에 한정된 데모", "findings": [
        {"dimension": "제품", "statement": NORMAL,
         "evidence_refs": [{"source_id": "S1", "chunk_id": "S1_P01_C01"}]}],
        "evidence": [evidence], "risks": ["외부 운영 검증 없음"],
        "missing_information": ["상용 운영 자료"]}
    state = {"candidate_companies": ["A"], "company_profiles": {"A": {"name": "A"}},
             "trace_id": "synthetic-qa-test", "report_attempts": version,
             "final_report": f"# SUMMARY\n{statement}\n# REFERENCE\n[S1] Demo announcement, A, p. 1\n",
             "references": [reference], "technology_results": {"A": analysis},
             "market_traction_results": {}, "competition_result": None, "investment_results": {}}
    originals = {"S1_P01_C01": {"source_id": "S1", "chunk_id": "S1_P01_C01", "page": 1,
                                "text": "A announced a laboratory demonstration. Commercial deployment has not been verified."}}
    return state, originals


def passing_judge(payload: dict) -> dict:
    return {"checked_claim_ids": [c["claim_id"] for c in payload["claims"]], "checks": [
        {"item": item, "verdict": "pass", "reason": "합성 테스트의 유효한 통과 응답",
         "issues": [], "missing_information": []} for item in ITEMS]}


def issue_judge(item: str, reason: str, owner: str = "report_generator", *, uncertain: bool = False):
    """Scripted semantic response, not an LLM-quality benchmark."""
    def judge(payload):
        output = passing_judge(payload)
        check = next(c for c in output["checks"] if c["item"] == item)
        claim = payload["claims"][0]
        check.update(verdict="uncertain" if uncertain else "fail", reason=reason,
                     missing_information=["추가 원문·반대 근거 확인"] if uncertain else [], issues=[
                         {"claim_id": claim["claim_id"], "evidence_ids": claim["evidence_ids"],
                          "suggested_owner": owner, "company": "A" if owner != "report_generator" else None,
                          "reason": reason, "requested_fix": "원문 범위와 한계를 반영해 수정하세요."}])
        return output
    return judge
