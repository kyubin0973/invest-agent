"""Competition Agent (설계 산출물 2.1)

역할: Target Market 맥락을 보존한 차별성·전략·상대 Risk 비교 (전체 후보 분석 후 1회 실행)
입력: technology_results, market_traction_results (전체 후보의 구조화 결과), company_profiles
도구: State Reader, Comparator (설계 2.1)
출력: competition_result

- 신규 검색을 하지 않고, 앞선 Agent의 구조화 결과와 EvidenceItem만 참조한다 (새 Evidence 생성 금지).
- Target Market 차이를 자동 우열로 바꾸지 않는다. 산업용·가정용 등 시장 차이는 target_market_context에 보존한다.
- 비교축: 기술, 제품, AI/Data, 기술 성숙도, 상용화, 제조·배치 확장성, 시장전략, 차별성, Risk
- 비교 근거는 comparisons[].evidence_refs(chunk_id + source_id)로 추적한다.
- 각 comparison은 criterion_ids에 해당 근거를 허용할 Judge 문항(B04 및/또는 B09)을 명시한다.
  차별성 비교는 B04, 상대 Risk 비교는 B09이며 두 문항에 모두 필요한 비교만 둘 다 지정한다.

TODO(담당자): stub을 실제 구현으로 교체
  - 담당 평가 문항: evaluation.criteria.criteria_for("competition") → B04(Q4), B09(Q9)
"""

from core.state import CompetitionResult, InvestmentState


def competition_node(state: InvestmentState) -> dict:
    companies = state["candidate_companies"]
    profiles = state["company_profiles"]
    result: CompetitionResult = {
        "target_market_context": {c: profiles[c]["target_market"] or "[STUB] 미입력" for c in companies},
        "comparisons": [],
        "differentiation": {c: "[STUB] 아직 구현되지 않음" for c in companies},
        "relative_risks": {c: "[STUB] 아직 구현되지 않음" for c in companies},
    }
    return {"competition_result": result}
