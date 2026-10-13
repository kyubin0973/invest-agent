"""QA 상세 결과를 기존 공통 계약으로 변환한다. 공통 State는 변경하지 않는다."""
from core.state import QualityCheck, QualityResult


def to_common_check(check: dict) -> QualityCheck:
    issues = []
    for issue in check["issues"]:
        refs = " ".join(f"[{r['chunk_id']} | {r['source_id']}]" for r in issue.get("evidence_ids", []))
        context = f"위치: {issue.get('location', 'report')}"
        if refs:
            context += f", 근거: {refs}"
        reason = f"{issue['reason']} ({context}) 수정 요청: {issue['requested_fix']}"
        issues.append({"owner": issue["owner"], "company": issue["company"],
                       "statement": issue["statement"], "reason": reason})
    return {"item": check["item"], "passed": check["passed"], "issues": issues}


def to_common_result(result: dict) -> QualityResult:
    checks = [to_common_check(check) for check in result["checks"]]
    return {"passed": result["passed"], "checks": checks,
            "feedback": [f"[{c['item']}] {issue['statement']} — {issue['reason']}"
                         for c in checks if not c["passed"] for issue in c["issues"]
                         if issue["owner"] == "report_generator"]}
