"""Competition Agent (설계 산출물 2.1, 6.3)

역할: Target Market 맥락을 보존한 차별성·전략·상대 Risk 비교 (전체 후보 분석 후 1회 실행)
입력: technology_results, market_traction_results (전체 후보의 구조화 결과), company_profiles
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
import json
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import get_llm
from core.state import CompetitionResult, InvestmentState, Comparison
from prompts.competition import COMPETITION_SYSTEM_PROMPT


def competition_node(state: InvestmentState) -> dict:
    companies = state["candidate_companies"]
    profiles = state["company_profiles"]
    tech_results = state.get("technology_results", {})
    market_results = state.get("market_traction_results", {})

    llm = get_llm()

    # 1. Target Market Context 보존 (설계 2.2, 6.3)
    target_market_context = {
        c: profiles.get(c, {}).get("target_market") or "세부 시장 미입력"
        for c in companies
    }

    # 2. State에 축적된 기업별 분석 데이터 및 evidence_refs 취합
    analysis_lines = []
    all_refs = []

    for comp in companies:
        t_res = tech_results.get(comp, {})
        m_res = market_results.get(comp, {})
        market_desc = target_market_context.get(comp, "")

        t_summary = t_res.get("summary", "기술 분석 결과 없음")
        m_summary = m_res.get("summary", "시장 분석 결과 없음")

        for f in t_res.get("findings", []):
            all_refs.extend(f.get("evidence_refs", []))
        for f in m_res.get("findings", []):
            all_refs.extend(f.get("evidence_refs", []))

        analysis_lines.append(
            f"=== 대상 기업: {comp} (세부 Target Market: {market_desc}) ===\n"
            f"[기술/제품 분석 요약]: {t_summary}\n"
            f"[기술 리스크]: {', '.join(t_res.get('risks', []))}\n"
            f"[시장/고객/BM 요약]: {m_summary}\n"
            f"[시장/상용화 리스크]: {', '.join(m_res.get('risks', []))}\n"
        )

    context_text = "\n".join(analysis_lines)

    # 3. LLM 비교 분석 요청
    messages = [
        SystemMessage(content=COMPETITION_SYSTEM_PROMPT),
        HumanMessage(content=f"비교 대상 기업: {', '.join(companies)}\n\n[각 기업 분석 데이터]\n{context_text}"),
    ]
    response = llm.invoke(messages)

    # 4. JSON 응답 정제 및 파싱
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {
            "comparisons": [],
            "differentiation": {c: f"{c} 차별성 분석 파싱 실패" for c in companies},
            "relative_risks": {c: f"{c} 상대 리스크 파싱 실패" for c in companies},
        }

    # 5. comparisons 항목 정제
    formatted_comparisons: list[Comparison] = []
    for item in parsed.get("comparisons", []):
        formatted_comparisons.append({
            "dimension": item.get("dimension", ""),
            "company_findings": item.get("company_findings", {}),
            "evidence_refs": item.get("evidence_refs", all_refs[:2]),
        })

    result: CompetitionResult = {
        "target_market_context": target_market_context,
        "comparisons": formatted_comparisons,
        "differentiation": parsed.get("differentiation", {c: "" for c in companies}),
        "relative_risks": parsed.get("relative_risks", {c: "" for c in companies}),
    }

    return {"competition_result": result}