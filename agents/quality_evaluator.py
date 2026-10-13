"""Report Quality Evaluator (보고서 품질 평가)

보고서 작성 뒤 Supervisor가 호출한다. 평가 결과가 미달이면 Supervisor가 Report Generator에
재작성을 요청한다 (재작성 횟수 상한은 Supervisor가 관리).

입력: final_report, references, technology_results, market_traction_results, competition_result, investment_results
출력: quality_result (QualityResult)

평가 항목:
- groundedness : 보고서의 주장이 검색된 출처(REFERENCE)로 추적되는가
- neutrality   : 근거 없는 우열·추천 표현이 없는가
- bias_control : 단일 출처나 기업에 유리한 근거에만 치우치지 않았는가

미달 지적(QualityIssue)에는 수정할 담당(owner)을 적는다. Supervisor가 이 값으로 되돌려 보낼 곳을 정한다.
- owner가 technology / market_traction : 분석 근거 자체의 문제. company를 함께 적으면 그 Agent가 재작업한다.
- owner가 report_generator            : 서술·표현의 문제. 보고서만 다시 쓴다.

TODO: 아래 _check_* 함수는 자리만 잡아 둔 것이다. 항목별 검사를 구현하면 된다.
      `passed`는 LLM이 아니라 이 모듈의 Python 규칙으로 계산한다.
"""

from copy import deepcopy

from langsmith import traceable

from core.state import InvestmentState, QualityCheck, QualityResult
from evaluation.report_quality import evaluate, redact_secrets, report_identity, result_is_current

# 위 원본 주석을 보존한다. 원본 TODO의 세 검사는 이제 Hybrid QA 결과를 반환한다.
# 공통 결과는 한 번만 평가하고 _check_*에 전달하므로 Judge를 세 번 호출하지 않는다.
# 원문 주입·상세 필드는 QA 내부 확장이며 공통 State 계약은 담당자 합의 후 연결한다.


def _check_groundedness(state: InvestmentState, *, _evaluation: dict | None = None,
                        original_documents: dict | None = None, judge=None) -> QualityCheck:
    """원문·인용·REFERENCE에 대한 규칙 검사와 Judge의 근거성 결과를 반환한다."""
    result = _evaluation if _evaluation is not None else evaluate(
        state, original_documents=original_documents, judge=judge)
    return deepcopy(next(check for check in result["checks"] if check["item"] == "groundedness"))


def _check_neutrality(state: InvestmentState, *, _evaluation: dict | None = None,
                      original_documents: dict | None = None, judge=None) -> QualityCheck:
    """기술 우열·추천 및 사실·해석·전망 구분에 대한 중립성 결과를 반환한다."""
    result = _evaluation if _evaluation is not None else evaluate(
        state, original_documents=original_documents, judge=judge)
    return deepcopy(next(check for check in result["checks"] if check["item"] == "neutrality"))


def _check_bias_control(state: InvestmentState, *, _evaluation: dict | None = None,
                        original_documents: dict | None = None, judge=None) -> QualityCheck:
    """홍보 편중·출처 계보·위험 누락·기준 일관성에 대한 편향 검사 결과를 반환한다."""
    result = _evaluation if _evaluation is not None else evaluate(
        state, original_documents=original_documents, judge=judge)
    return deepcopy(next(check for check in result["checks"] if check["item"] == "bias_control"))


def _trace_input(inputs: dict) -> dict:
    state = inputs.get("state") or {}
    version, digest = report_identity(state)
    ids = set()
    for key in ("technology_results", "market_traction_results"):
        for analysis in (state.get(key) or {}).values():
            ids.update((e.get("source_id"), e.get("chunk_id")) for e in analysis.get("evidence", []))
    for result in (state.get("investment_results") or {}).values():
        for criterion in result.get("criteria", []):
            ids.update((e.get("source_id"), e.get("chunk_id")) for e in criterion.get("evidence", []))
    return redact_secrets({"trace_id": state.get("trace_id"), "report_version": version, "report_hash": digest,
                           "evidence_ids": [{"source_id": s, "chunk_id": c} for s, c in sorted(ids, key=str)]})


def _trace_output(output: dict) -> dict:
    # Do not duplicate the report or originals in QA traces.
    if not isinstance(output, dict):
        return {"error": "QA evaluation failed"}
    return redact_secrets({**output, "feedback": [], "checks": [
        {**check, "issues": [{k: v for k, v in issue.items() if k != "statement"}
                             for issue in check["issues"]]} for check in output["checks"]]})


@traceable(name="report_quality_evaluation", process_inputs=_trace_input, process_outputs=_trace_output)
def evaluate_report(state: InvestmentState, *, original_documents: dict | None = None, judge=None) -> QualityResult:
    # 동일 스냅샷으로 한 번 평가한 결과를 기존 세 함수에서 항목별로 받아 합산한다.
    snapshot = deepcopy(state)
    result = evaluate(snapshot, original_documents=original_documents, judge=judge)
    checks = [_check_groundedness(snapshot, _evaluation=result),
              _check_neutrality(snapshot, _evaluation=result),
              _check_bias_control(snapshot, _evaluation=result)]
    return {
        **result,
        "passed": result["judge_used"] and all(c["passed"] for c in checks),
        "checks": checks,
        "feedback": [
            f"[{c['item']}] {issue['statement']} — {issue['reason']} 수정: {issue['requested_fix']}"
            for c in checks if not c["passed"]
            for issue in c["issues"] if issue["owner"] == "report_generator"
        ],
    }


def quality_evaluator_node(state: InvestmentState) -> dict:
    # Judge 예외는 기존 worker에서 기록·재시도하며 합격으로 바꾸지 않는다.
    return {"quality_result": evaluate_report(state)}


__all__ = ["evaluate_report", "quality_evaluator_node", "result_is_current"]
