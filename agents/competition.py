"""Competition Agent (설계 산출물 2.2절)

역할: 후보 기업 간 차별성과 경쟁 Risk 비교 (전체 분석 완료 후 1회 실행)
입력: technology_results, market_traction_results (앞선 Agent의 구조화된 결과)
출력: competition_result

원문을 다시 검색하지 않는다 (설계 2.3절).

TODO(담당자): stub을 실제 구현으로 교체
  - 담당 평가 항목: evaluation.criteria.criteria_for("competition") → B04, B09
"""

from core.state import CompetitionResult, InvestmentState


def competition_node(state: InvestmentState) -> dict:
    result: CompetitionResult = {
        "summary": "[STUB] 후보 기업 간 경쟁 비교",
        "comparisons": [
            {"company": c, "differentiation": [], "competitive_risks": []}
            for c in state["candidate_companies"]
        ],
        "missing_information": ["[STUB] 아직 구현되지 않음"],
    }
    return {"competition_result": result}
