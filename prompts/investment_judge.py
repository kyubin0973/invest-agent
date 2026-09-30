"""Investment Judge 전용 프롬프트 구성.

Judge는 검색하지 않고 State에 저장된 분석 결과와 Evidence만 사용한다.
"""

import json

from evaluation.criteria import (
    CONFIDENCE_LEVELS,
    CRITERIA,
    CRITERION_BOUNDARIES,
    EVIDENCE_LEVEL_RULE,
    EVIDENCE_LEVELS,
    SCORE_RUBRIC,
    SCORE_RULE,
)
from prompts.common import (
    CITATION_RULES,
    EVIDENCE_RULES,
    NO_NEW_EVIDENCE_RULE,
    TARGET_MARKET_RULE,
    build_system_prompt,
    build_user_prompt,
    format_evidence_items,
)


def _criterion_spec(criterion_ids: set[str]) -> list[dict]:
    return [
        {
            "criterion_id": criterion.id,
            "q": criterion.q,
            "name": criterion.name,
            "question": criterion.question,
            "evidence_focus": list(criterion.evidence_focus),
            "source_agents": list(criterion.agents),
        }
        for criterion in CRITERIA
        if criterion.id in criterion_ids
    ]


def _analysis_context(state: dict, company: str) -> dict:
    competition = state.get("competition_result") or {}
    return {
        "company": company,
        "company_profile": state.get("company_profiles", {}).get(company, {}),
        "technology_analysis": state.get("technology_results", {}).get(company, {}),
        "market_traction_analysis": state.get("market_traction_results", {}).get(company, {}),
        "competition": competition,
    }


def build_judge_messages(
    state: dict,
    company: str,
    evidence: list[dict],
    *,
    criterion_ids: set[str] | None = None,
    repair_issues: dict[str, list[str]] | None = None,
    previous_assessments: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """초기 일괄 평가 또는 문제 문항의 1회 보정 메시지를 만든다."""
    requested_ids = criterion_ids or {criterion.id for criterion in CRITERIA}
    context = json.dumps(_analysis_context(state, company), ensure_ascii=False, indent=2, default=str)
    evidence_text = format_evidence_items(evidence) or "제공된 Evidence 없음"
    specs = json.dumps(_criterion_spec(requested_ids), ensure_ascii=False, indent=2)
    rubric = json.dumps(
        {
            "evidence_levels": EVIDENCE_LEVELS,
            "evidence_level_rule": EVIDENCE_LEVEL_RULE,
            "score_rubric": {"N/A" if key is None else str(key): value for key, value in SCORE_RUBRIC.items()},
            "score_rule": SCORE_RULE,
            "confidence_levels": CONFIDENCE_LEVELS,
            "criterion_boundaries": list(CRITERION_BOUNDARIES),
        },
        ensure_ascii=False,
        indent=2,
    )

    if repair_issues:
        mode = (
            "아래 문항은 첫 평가에서 형식 또는 근거 검증에 실패했다. "
            "실패한 문항만 다시 평가하고 다른 문항은 출력하지 않는다.\n"
            f"검증 실패 사유:\n{json.dumps(repair_issues, ensure_ascii=False, indent=2)}\n"
            f"이전 출력:\n{json.dumps(previous_assessments or [], ensure_ascii=False, indent=2, default=str)}"
        )
    else:
        mode = "요청된 12개 문항을 모두 한 번씩 평가한다. 누락하거나 중복하지 않는다."

    request = f"""{mode}

[현재 기업 및 앞선 Agent 결과]
{context}

[평가할 문항]
{specs}

[평가 기준]
{rubric}

[출력 요구]
- score는 1, 3, 5 또는 null만 사용한다.
- SCORED이면 유효한 Evidence 참조가 하나 이상 있어야 한다.
- INSUFFICIENT_EVIDENCE이면 score는 null이어야 한다.
- 5점은 E3 이상 Evidence가 실제로 연결될 때만 가능하다. E1~E2 근거만 있으면 최대 3점이다.
- 근거가 없다는 이유로 1점을 주지 않는다. 1점은 실제 부정 Evidence가 있을 때만 사용한다.
- Q9는 점수가 높을수록 Risk가 관리 가능하다는 뜻이다.
- confidence는 High, Medium, Low 중 하나만 사용한다.
- Evidence의 원문 Metadata를 새로 만들지 말고 chunk_id와 source_id만 참조한다.
- reasoning에는 해당 문항의 판단과 한계를 함께 적고, 다른 문항과 같은 의미로 중복 가점·감점하지 않는다.
- key_strengths와 key_risks는 criterion_ids와 evidence 참조를 포함해야 한다. 확인되지 않은 요약은 만들지 않는다.
- FinalScore, EvidenceCoverage, Decision은 계산하지 않는다. Python이 계산한다."""

    user = build_user_prompt(
        evidence_text,
        request,
        EVIDENCE_RULES,
        CITATION_RULES,
        NO_NEW_EVIDENCE_RULE,
        TARGET_MARKET_RULE,
    )
    system = build_system_prompt("확보된 근거만으로 휴머노이드 스타트업을 평가하는 Investment Judge")
    return [("system", system), ("user", user)]
