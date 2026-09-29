"""Market & Traction Agent (설계 산출물 2.2절)

역할: 시장·고객·사업·팀 관련 Evidence 분석
입력: current_company, 공통 시장자료, 현재 기업 자료, company_profiles (팀·투자 정보)
출력: market_traction_results[current_company], market_traction_done

TODO(담당자): stub을 실제 구현으로 교체
  - 검색: rag.retriever.search_market(query, company) → {"market": 공통 시장자료, "company": 기업 자료}
    Market Evidence와 Company Evidence를 섞지 않는다 (설계 3.8절)
  - 근거 변환: rag.retriever.to_evidence(doc)
  - 담당 평가 항목: evaluation.criteria.criteria_for("market_traction")
"""

from core.state import AnalysisResult, InvestmentState


def market_traction_node(state: InvestmentState) -> dict:
    company = state["current_company"]
    result: AnalysisResult = {
        "summary": f"[STUB] {company} 시장·Traction 분석",
        "findings": [],
        "evidence": [],
        "risks": [],
        "missing_information": ["[STUB] 아직 구현되지 않음"],
    }
    return {"market_traction_results": {company: result}, "market_traction_done": True}
