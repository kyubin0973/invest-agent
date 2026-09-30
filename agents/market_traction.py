"""Market & Traction Agent (설계 산출물 2.1)

역할: 시장·고객·계약·사업모델·팀·상용화 Risk 분석 (현재 기업)
입력: current_company, 현재 기업 자료 + 공통 시장자료, company_profiles (Target Market 등 사전 입력값)
출력: market_traction_results[current_company], market_traction_done

TODO(담당자): stub을 실제 구현으로 교체
  - 검색: rag.retriever.search_market(query, company) → {"market": 공통 시장자료, "company": 기업 자료}
    Market Evidence와 Company Evidence를 섞지 않는다 (설계 2.2.4)
  - 근거 변환: rag.retriever.to_evidence(doc)
  - 담당 평가 문항: evaluation.criteria.criteria_for("market_traction") → B01~B03, B05~B10, H12
  - 분석축 (설계 2.1.1)
      Robot AI/HW/VLA          : 제품 성능을 재평가하지 않음 (Technology 담당)
      Demo/Pilot/Deployment    : 고객검증·실제 배치·유료계약
      제조·배치 확장성         : 제조 파트너·가격·원가·Fleet 운영
      시장·팀·사업             : 시장 크기·WTP·사업모델·팀·파트너십
  - 공개 Team·Founder·가격 정보가 부족하면 추정하지 않는다 (해당 문항은 Judge가 N/A 처리).
  - findings는 {dimension, statement, evidence_refs[{chunk_id, source_id}]} 형식 (core.state.Finding)
  - Evidence Sufficiency Check: 부족한 하위 질문만 Query 보완 후 최대 1회 재검색
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
