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

from core.state import InvestmentState, QualityCheck, QualityResult


def _check_groundedness(state: InvestmentState) -> QualityCheck:
    return {"item": "groundedness", "passed": True, "issues": []}


def _check_neutrality(state: InvestmentState) -> QualityCheck:
    return {"item": "neutrality", "passed": True, "issues": []}


def _check_bias_control(state: InvestmentState) -> QualityCheck:
    return {"item": "bias_control", "passed": True, "issues": []}


def evaluate_report(state: InvestmentState) -> QualityResult:
    checks = [_check_groundedness(state), _check_neutrality(state), _check_bias_control(state)]
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "feedback": [
            f"[{c['item']}] {issue['statement']} — {issue['reason']}"
            for c in checks if not c["passed"]
            for issue in c["issues"] if issue["owner"] == "report_generator"
        ],
    }


def quality_evaluator_node(state: InvestmentState) -> dict:
    return {"quality_result": evaluate_report(state)}
