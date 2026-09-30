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

JUDGE_PROMPT_VERSION = "investment-judge-v2"


def _analysis_view(result: object) -> dict:
    """EvidenceItem 원문은 별도 Evidence 블록에만 두고 분석 결론만 Context에 남긴다."""
    if not isinstance(result, dict):
        return {}
    return {
        key: result.get(key)
        for key in ("summary", "findings", "risks", "missing_information")
    }


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
        "technology_analysis": _analysis_view(
            state.get("technology_results", {}).get(company, {})
        ),
        "market_traction_analysis": _analysis_view(
            state.get("market_traction_results", {}).get(company, {})
        ),
        "competition": competition,
    }


def _json(value: object) -> str:
    """프롬프트 fingerprint가 실행마다 흔들리지 않도록 canonical JSON을 만든다."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def build_judge_messages(
    state: dict,
    company: str,
    evidence: list[dict],
    *,
    competition_evidence: list[tuple[str, dict, list[str]]] | None = None,
    criterion_ids: set[str] | None = None,
    repair_issues: dict[str, list[str]] | None = None,
    previous_assessments: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """초기 일괄 평가 또는 문제 문항의 1회 보정 메시지를 만든다."""
    requested_ids = criterion_ids or {criterion.id for criterion in CRITERIA}
    context = _json(_analysis_context(state, company))
    sorted_evidence = sorted(evidence, key=lambda item: (item["source_id"], item["chunk_id"]))
    own_evidence_text = format_evidence_items(sorted_evidence) or "제공된 현재 기업 Evidence 없음"
    competition_blocks = []
    sorted_competition_evidence = sorted(
        competition_evidence or [],
        key=lambda value: (value[1]["source_id"], value[1]["chunk_id"]),
    )
    for owner, item, allowed_criteria in sorted_competition_evidence:
        competition_blocks.append(
            f"<competition_evidence_owner>{owner}</competition_evidence_owner>\n"
            f"<allowed_criteria>{','.join(allowed_criteria)}</allowed_criteria>\n"
            f"{format_evidence_items([item])}"
        )
    competition_evidence_text = "\n".join(competition_blocks) or "제공된 Competition Evidence 없음"
    evidence_text = (
        f"[현재 평가 기업 Evidence: {company}]\n{own_evidence_text}\n\n"
        "[Competition Evidence: B04(Q4)·B09(Q9)에서만 사용 가능]\n"
        f"{competition_evidence_text}"
    )
    specs = _json(_criterion_spec(requested_ids))
    rubric = _json(
        {
            "evidence_levels": EVIDENCE_LEVELS,
            "evidence_level_rule": EVIDENCE_LEVEL_RULE,
            "score_rubric": {"N/A" if key is None else str(key): value for key, value in SCORE_RUBRIC.items()},
            "score_rule": SCORE_RULE,
            "confidence_levels": CONFIDENCE_LEVELS,
            "criterion_boundaries": list(CRITERION_BOUNDARIES),
        },
    )

    if repair_issues:
        mode = (
            "아래 문항은 첫 평가에서 형식 또는 근거 검증에 실패했다. "
            "실패한 문항만 다시 평가하고 다른 문항은 출력하지 않는다.\n"
            f"검증 실패 사유:\n{_json(repair_issues)}\n"
            f"이전 출력:\n{_json(previous_assessments or [])}"
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
- Competition Evidence는 각 항목의 allowed_criteria에 표시된 B04(Q4) 또는 B09(Q9)에만 인용한다. 다른 문항에는 현재 평가 기업 Evidence만 사용한다.
- Competition Evidence의 owner를 확인하고 타사 Evidence를 현재 기업의 직접 성과·고객·시장 근거로 바꾸지 않는다.
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
