"""Investment Judge (설계 산출물 2.1, 3장)

역할: 12문항 Score·Evidence·Confidence·Decision (현재 기업)
입력: technology_results[기업], market_traction_results[기업], competition_result
출력: investment_results[current_company], completed_companies

- 신규 검색을 하지 않는다. 앞선 Agent가 전달한 Evidence만으로 12문항을 동일 Rubric으로 평가한다.
- LLM은 문항별 score·status·evidence 참조(chunk_id, source_id)·reasoning·confidence·missing_information을 낸다.
- Python이 검증(evaluation.criteria.validate_criteria)한 뒤 FinalScore·Coverage·Decision을 계산(decide)한다.
- 계획·주장·Demo를 실제 성과로 바꾸지 않고, 근거가 부족하면 N/A(score=None)를 반환한다.
- 점수: 1~5점 척도 중 1·3·5점만 사용 (evaluation.criteria.ALLOWED_SCORES).
  근거가 없으면 1점이 아니라 N/A. 2·4점은 검증 단계에서 '형식 오류'로 처리된다.

TODO(담당자): _llm_score_criteria의 더미 결과를 LLM 평가로 교체
  - Rubric: evaluation.criteria의 SCORE_RUBRIC, SCORE_RULE, EVIDENCE_LEVELS, EVIDENCE_LEVEL_RULE,
    CONFIDENCE_LEVELS, CRITERION_BOUNDARIES를 프롬프트에 넣는다.
  - 출력 형식에서 점수를 1·3·5 또는 None으로 제한하고, 형식 오류 문항은 1회 재평가한 뒤 검증한다.
  - 현재 stub의 더미 점수(4점)는 1·3·5 규칙에 맞지 않아 검증 단계에서 형식 오류로 처리된다.
"""

from core.state import InvestmentResult, InvestmentState
from evaluation.criteria import CRITERIA, decide, validate_criteria


def _available_evidence(state: InvestmentState, company: str) -> dict[str, dict]:
    """앞선 Agent가 실제 검색해 Judge에 전달한 Evidence (chunk_id → EvidenceItem)."""
    evidence = {}
    for results in (state["technology_results"], state["market_traction_results"]):
        for item in results.get(company, {}).get("evidence", []):
            evidence[item["chunk_id"]] = item
    return evidence


def _llm_score_criteria(state: InvestmentState, company: str, evidence: dict[str, dict]) -> list[dict]:
    # [STUB] 흐름 확인용 더미 결과: 짝수 번째 후보 4점, 홀수 번째 후보 3점.
    # 실제 구현에서는 LLM이 이 형식으로 12문항 결과를 낸다 (evidence는 chunk_id/source_id 참조).
    dummy = 4 if state["current_company_index"] % 2 == 0 else 3
    refs = [{"chunk_id": e["chunk_id"], "source_id": e["source_id"]} for e in list(evidence.values())[:1]]
    return [
        {
            "criterion_id": c.id,
            "score": dummy,
            "status": "SCORED",
            "evidence": refs,
            "reasoning": "[STUB] 더미 점수",
            "confidence": "Low",
            "missing_information": [],
        }
        for c in CRITERIA
    ]


def investment_judge_node(state: InvestmentState) -> dict:
    company = state["current_company"]
    evidence = _available_evidence(state, company)
    criteria = validate_criteria(_llm_score_criteria(state, company, evidence), evidence)
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
