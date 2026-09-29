"""Technology & Product Agent (설계 산출물 2.2절)

역할: 현재 기업의 기술·제품 역량 및 기술 Risk 분석
입력: current_company, 기업 기술자료
출력: technology_results[current_company], technology_done

TODO(담당자): stub을 실제 구현으로 교체
  - 검색: rag.retriever.search_technology(query, company) → 현재 기업의 기술자료만
  - 근거 변환: rag.retriever.to_evidence(doc)
  - 담당 평가 항목: evaluation.criteria.criteria_for("technology")
  - 근거 부족 시 Query 보완 후 재검색 1회, 그래도 부족하면 missing_information에 기록
"""

from core.state import AnalysisResult, InvestmentState


def technology_node(state: InvestmentState) -> dict:
    company = state["current_company"]
    result: AnalysisResult = {
        "summary": f"[STUB] {company} 기술·제품 분석",
        "findings": [],
        "evidence": [],
        "risks": [],
        "missing_information": ["[STUB] 아직 구현되지 않음"],
    }
    return {"technology_results": {company: result}, "technology_done": True}
