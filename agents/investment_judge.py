"""Investment Judge (설계 산출물 2.2절, 5장)

역할: 12개 평가 항목(evaluation.criteria.CRITERIA)으로 현재 기업을 평가하고 INVEST / HOLD 결정
입력: technology_results, market_traction_results, competition_result
출력: investment_results[current_company], completed_companies

신규 검색을 하지 않고, 앞선 Agent가 제공한 Evidence만으로 평가한다 (설계 3.8절).
점수 합산과 INVEST / HOLD 판정은 LLM이 아니라 evaluation.criteria.decide()가 한다.

TODO(담당자): _score_criteria의 더미 점수를 LLM 평가로 교체
  - 항목별 1~5점 또는 N/A(None), reasoning, confidence, 사용한 evidence
"""

from core.state import CriterionResult, InvestmentResult, InvestmentState
from evaluation.criteria import CRITERIA, decide


def _score_criteria(state: InvestmentState, company: str) -> list[CriterionResult]:
    # [STUB] 흐름 확인용 더미 점수: 짝수 번째 후보 4점, 홀수 번째 후보 3점
    dummy = 4 if state["current_company_index"] % 2 == 0 else 3
    return [
        {
            "criterion_id": c.id,
            "criterion_name": c.name,
            "score": dummy,
            "status": "SCORED",
            "evidence": [],
            "reasoning": "[STUB] 더미 점수",
            "confidence": "Low",
            "missing_information": [],
        }
        for c in CRITERIA
    ]


def investment_judge_node(state: InvestmentState) -> dict:
    company = state["current_company"]
    criteria = _score_criteria(state, company)
    verdict = decide({c["criterion_id"]: c["score"] for c in criteria})

    result: InvestmentResult = {
        "criteria": criteria,
        **verdict,
        "key_strengths": [],
        "key_risks": [],
        "missing_information": [m for c in criteria for m in c["missing_information"]],
    }
    return {
        "investment_results": {company: result},
        "completed_companies": [*state["completed_companies"], company],
    }
