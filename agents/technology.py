"""Technology & Product Agent (설계 산출물 2.1)

역할: AI/HW·제품 Capability·실환경 수행·기술 Risk 분석 (현재 기업)
입력: current_company, 기업 기술자료
출력: technology_results[current_company], technology_done

TODO(담당자): stub을 실제 구현으로 교체
  - 검색: rag.retriever.search_technology(query, company) → 현재 기업의 기술자료만
  - 근거 변환: rag.retriever.to_evidence(doc)
  - 담당 평가 문항: evaluation.criteria.criteria_for("technology") → B02, B04, B08, B09, H11, H12
  - 분석축 (설계 2.1.1)
      Robot AI/HW/VLA          : 기술 역량·제품 성능·기술 Risk
      Demo/Pilot/Deployment    : Autonomy·반복 수행·실패 복구
      제조·배치 확장성         : 생산기술·부품·공급망·생산능력
      시장·팀·사업             : 필요한 기술 맥락만 참고 (평가는 Market & Traction 담당)
  - findings는 {dimension, statement, evidence_refs[{chunk_id, source_id}]} 형식 (core.state.Finding)
  - Evidence Sufficiency Check: 부족한 하위 질문만 Query 보완 후 최대 1회 재검색, 그래도 부족하면 missing_information에 기록
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
