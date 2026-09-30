"""Investment Judge (설계 산출물 2.1, 5장).

현재 기업의 Technology/Market 분석과 전체 Competition 결과만 사용해 12문항을
평가한다. 신규 검색, 최종 점수 계산, 투자판정은 LLM에 맡기지 않는다.
"""

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from core.judge_logging import log_criteria, log_decision, log_repair_issues, log_stage
from core.llm import get_structured_llm
from core.state import InvestmentResult, InvestmentState
from evaluation.criteria import (
    ALLOWED_SCORES,
    CONFIDENCE_LEVELS,
    CRITERIA_BY_ID,
    decide,
    validate_criteria,
)
from prompts.investment_judge import build_judge_messages

COMPETITION_EVIDENCE_CRITERIA = {"B04", "B09"}


class EvidenceRefOutput(BaseModel):
    chunk_id: str
    source_id: str


class CriterionAssessmentOutput(BaseModel):
    criterion_id: str
    # int로 받아 2·4점 같은 오류를 Python이 식별하고 한 번 보정할 수 있게 한다.
    score: int | None = None
    status: str
    evidence: list[EvidenceRefOutput] = Field(default_factory=list)
    reasoning: str = ""
    confidence: str = "Low"
    missing_information: list[str] = Field(default_factory=list)


class SummaryClaimOutput(BaseModel):
    text: str
    criterion_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRefOutput] = Field(default_factory=list)


class JudgeBatchOutput(BaseModel):
    criteria: list[CriterionAssessmentOutput] = Field(default_factory=list)
    key_strengths: list[SummaryClaimOutput] = Field(default_factory=list)
    key_risks: list[SummaryClaimOutput] = Field(default_factory=list)


def _available_evidence(state: InvestmentState, company: str) -> dict[str, dict]:
    """앞선 Agent가 현재 기업에 대해 실제로 전달한 Evidence (chunk_id → EvidenceItem)."""
    evidence: dict[str, dict] = {}
    for results in (state["technology_results"], state["market_traction_results"]):
        for item in results.get(company, {}).get("evidence", []):
            evidence[item["chunk_id"]] = item
    return evidence


def _all_analysis_evidence(state: InvestmentState) -> tuple[dict[str, dict], dict[str, str]]:
    """모든 후보의 분석 Evidence와 각 Evidence의 소유 기업을 수집한다."""
    evidence: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for company in state["candidate_companies"]:
        for results in (state["technology_results"], state["market_traction_results"]):
            for item in results.get(company, {}).get("evidence", []):
                chunk_id = item["chunk_id"]
                existing = evidence.get(chunk_id)
                if existing and existing["source_id"] != item["source_id"]:
                    raise ValueError(f"chunk_id 충돌: {chunk_id}")
                existing_owner = owners.get(chunk_id)
                if existing_owner and existing_owner != company:
                    raise ValueError(f"Evidence 소유 기업 충돌: {chunk_id} ({existing_owner}, {company})")
                evidence[chunk_id] = item
                owners[chunk_id] = company
    return evidence, owners


def _competition_evidence(
    state: InvestmentState,
) -> tuple[dict[str, dict], dict[str, str], list[str]]:
    """Competition이 실제 참조한 Evidence만 전체 후보 분석 결과에서 역참조한다."""
    catalog, owners = _all_analysis_evidence(state)
    resolved: dict[str, dict] = {}
    resolved_owners: dict[str, str] = {}
    unresolved: list[str] = []

    competition = state.get("competition_result") or {}
    for comparison in competition.get("comparisons", []):
        for ref in comparison.get("evidence_refs", []):
            if not isinstance(ref, dict):
                unresolved.append(repr(ref))
                continue
            chunk_id = ref.get("chunk_id")
            source_id = ref.get("source_id")
            item = catalog.get(chunk_id)
            if not item or item.get("source_id") != source_id:
                unresolved.append(f"{chunk_id} | {source_id}")
                continue
            resolved[chunk_id] = item
            resolved_owners[chunk_id] = owners[chunk_id]

    return resolved, resolved_owners, list(dict.fromkeys(unresolved))


def _allowed_evidence_by_criterion(
    own_evidence: dict[str, dict],
    competition_evidence: dict[str, dict],
) -> dict[str, set[tuple[str, str]]]:
    own_refs = {(item["chunk_id"], item["source_id"]) for item in own_evidence.values()}
    competition_refs = {
        (item["chunk_id"], item["source_id"]) for item in competition_evidence.values()
    }
    allowed = {criterion_id: set(own_refs) for criterion_id in CRITERIA_BY_ID}
    for criterion_id in COMPETITION_EVIDENCE_CRITERIA:
        allowed[criterion_id].update(competition_refs)
    return allowed


def _as_dict(value: Any) -> dict:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Judge 구조화 출력 형식이 아닙니다: {type(value).__name__}")


def _as_dict_list(values: Any) -> list[dict]:
    if not isinstance(values, list):
        return []
    return [_as_dict(value) for value in values if isinstance(value, (BaseModel, dict))]


def _criterion_issues(
    assessments: list[dict],
    available_evidence: dict[str, dict],
    expected_ids: set[str],
    allowed_evidence_by_criterion: dict[str, set[tuple[str, str]]] | None = None,
) -> dict[str, list[str]]:
    """보정 호출이 필요한 문항과 검증 실패 사유를 찾는다."""
    counts = Counter(item.get("criterion_id") for item in assessments)
    by_id = {item.get("criterion_id"): item for item in assessments}
    issues: dict[str, list[str]] = {}

    for criterion_id in expected_ids:
        reasons: list[str] = []
        raw = by_id.get(criterion_id)
        if raw is None:
            reasons.append("문항 누락")
        elif counts[criterion_id] > 1:
            reasons.append("동일 criterion_id 중복")

        if raw is not None:
            score = raw.get("score")
            status = raw.get("status")
            confidence = raw.get("confidence")
            refs = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
            valid_items = []
            invalid_refs = []
            for ref in refs:
                if not isinstance(ref, dict):
                    invalid_refs.append(ref)
                    continue
                item = available_evidence.get(ref.get("chunk_id"))
                evidence_key = (ref.get("chunk_id"), ref.get("source_id"))
                allowed_refs = (
                    allowed_evidence_by_criterion.get(criterion_id, set())
                    if allowed_evidence_by_criterion is not None
                    else None
                )
                if (
                    item
                    and item.get("source_id") == ref.get("source_id")
                    and (allowed_refs is None or evidence_key in allowed_refs)
                ):
                    valid_items.append(item)
                else:
                    invalid_refs.append(ref)

            valid_score = isinstance(score, int) and not isinstance(score, bool) and score in ALLOWED_SCORES
            if status == "SCORED":
                if not valid_score:
                    reasons.append(f"SCORED의 허용되지 않은 score: {score!r}")
                if not valid_items:
                    reasons.append("SCORED에 검증 가능한 Evidence가 없음")
            elif status == "INSUFFICIENT_EVIDENCE":
                if score is not None:
                    reasons.append("INSUFFICIENT_EVIDENCE의 score가 null이 아님")
            else:
                reasons.append(f"허용되지 않은 status: {status!r}")

            if invalid_refs:
                reasons.append("존재하지 않거나 source_id·문항 허용 범위가 불일치하는 Evidence 참조")
            if confidence not in CONFIDENCE_LEVELS:
                reasons.append(f"허용되지 않은 confidence: {confidence!r}")
            if score == 5 and not any(
                item.get("evidence_level") in ("E3", "E4", "E5") for item in valid_items
            ):
                reasons.append("5점에 필요한 E3 이상 외부검증 Evidence가 없음")

        if reasons:
            issues[criterion_id] = reasons
    return issues


def _merge_repair(
    initial: list[dict],
    repaired: list[dict],
    issue_ids: set[str],
) -> list[dict]:
    """문제가 없던 초기 결과는 보존하고, 문제 문항만 재평가 결과로 교체한다."""
    kept = [item for item in initial if item.get("criterion_id") not in issue_ids]
    replacements = [item for item in repaired if item.get("criterion_id") in issue_ids]
    return kept + replacements


def _invoke_judge(messages: list[tuple[str, str]]) -> dict:
    response = get_structured_llm(JudgeBatchOutput).invoke(messages)
    return _as_dict(response)


def _llm_score_criteria(
    state: InvestmentState,
    company: str,
    own_evidence: dict[str, dict],
    competition_evidence: dict[str, dict],
    competition_owners: dict[str, str],
    allowed_evidence_by_criterion: dict[str, set[tuple[str, str]]],
) -> dict:
    """12문항 일괄 평가 후 문제가 있는 문항만 최대 한 번 보정한다."""
    expected_ids = set(CRITERIA_BY_ID)
    available_evidence = {**competition_evidence, **own_evidence}
    own_evidence_items = list(own_evidence.values())
    competition_evidence_items = [
        (competition_owners[item["chunk_id"]], item) for item in competition_evidence.values()
    ]

    # 현재는 비용·지연을 줄이기 위해 12문항을 한 번에 평가한다. 실제 평가에서 긴 Context로
    # 문항 누락·상호간섭이 반복되면 이 호출 경계를 문항별 12회 호출로 바꿀 수 있다.
    log_stage(company, "LLM", "12개 평가 문항 일괄 호출")
    initial = _invoke_judge(
        build_judge_messages(
            state,
            company,
            own_evidence_items,
            competition_evidence=competition_evidence_items,
        )
    )
    initial_criteria = _as_dict_list(initial.get("criteria"))
    log_criteria(company, "LLM 최초 평가", initial_criteria)
    issues = _criterion_issues(
        initial_criteria,
        available_evidence,
        expected_ids,
        allowed_evidence_by_criterion,
    )
    if not issues:
        return {
            "criteria": initial_criteria,
            "key_strengths": _as_dict_list(initial.get("key_strengths")),
            "key_risks": _as_dict_list(initial.get("key_risks")),
        }

    issue_ids = set(issues)
    log_repair_issues(company, issues)
    log_stage(company, "REPAIR", f"문제 문항 {len(issue_ids)}개만 1회 재평가")
    previous = [item for item in initial_criteria if item.get("criterion_id") in issue_ids]
    repair = _invoke_judge(
        build_judge_messages(
            state,
            company,
            own_evidence_items,
            competition_evidence=competition_evidence_items,
            criterion_ids=issue_ids,
            repair_issues=issues,
            previous_assessments=previous,
        )
    )
    repaired_criteria = _as_dict_list(repair.get("criteria"))
    log_criteria(company, "LLM 재평가", repaired_criteria)
    return {
        "criteria": _merge_repair(initial_criteria, repaired_criteria, issue_ids),
        "key_strengths": [
            *_as_dict_list(initial.get("key_strengths")),
            *_as_dict_list(repair.get("key_strengths")),
        ],
        "key_risks": [
            *_as_dict_list(initial.get("key_risks")),
            *_as_dict_list(repair.get("key_risks")),
        ],
    }


def _validated_summary_texts(
    summaries: list[dict],
    criteria: list[dict],
    *,
    kind: str,
) -> list[str]:
    """검증된 문항과 Evidence에 연결된 Strength/Risk 요약만 문자열로 보존한다."""
    criteria_by_id = {item["criterion_id"]: item for item in criteria}
    accepted: list[str] = []

    for summary in summaries:
        text = str(summary.get("text") or "").strip()
        criterion_ids = summary.get("criterion_ids") or []
        refs = summary.get("evidence") or []
        linked = [criteria_by_id.get(criterion_id) for criterion_id in criterion_ids]
        if not text or not linked or any(item is None for item in linked):
            continue

        if kind == "strength" and not any(item["score"] == 5 for item in linked):
            continue
        if kind == "risk" and not any(item["score"] in (1, 3) for item in linked):
            continue

        allowed_refs = {
            (ev["chunk_id"], ev["source_id"])
            for item in linked
            for ev in item["evidence"]
        }
        summary_refs = {
            (ref.get("chunk_id"), ref.get("source_id"))
            for ref in refs
            if isinstance(ref, dict)
        }
        if not summary_refs or not summary_refs.issubset(allowed_refs):
            continue
        if text not in accepted:
            accepted.append(text)

    return accepted


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def investment_judge_node(state: InvestmentState) -> dict:
    company = state["current_company"]
    if not company:
        raise ValueError("Investment Judge 실행 전에 current_company가 설정되어야 합니다.")

    own_evidence = _available_evidence(state, company)
    competition_evidence, competition_owners, unresolved_competition_refs = _competition_evidence(state)
    available_evidence = {**competition_evidence, **own_evidence}
    allowed_evidence = _allowed_evidence_by_criterion(own_evidence, competition_evidence)
    technology_count = len(state["technology_results"].get(company, {}).get("evidence", []))
    market_count = len(state["market_traction_results"].get(company, {}).get("evidence", []))
    log_stage(
        company,
        "START",
        (
            f"technology_evidence={technology_count} | market_evidence={market_count} | "
            f"competition_evidence={len(competition_evidence)} | "
            f"unresolved_competition_refs={len(unresolved_competition_refs)}"
        ),
    )
    if unresolved_competition_refs:
        log_stage(
            company,
            "COMPETITION_EVIDENCE_WARNING",
            "해결하지 못한 참조: " + ", ".join(unresolved_competition_refs),
        )
    llm_result = _llm_score_criteria(
        state,
        company,
        own_evidence,
        competition_evidence,
        competition_owners,
        allowed_evidence,
    )
    criteria = validate_criteria(llm_result["criteria"], available_evidence, allowed_evidence)
    log_criteria(company, "Python 검증 후 최종 평가", criteria)
    verdict = decide({criterion["criterion_id"]: criterion["score"] for criterion in criteria})

    result: InvestmentResult = {
        "criteria": criteria,
        **verdict,
        "key_strengths": _validated_summary_texts(
            llm_result["key_strengths"], criteria, kind="strength"
        ),
        "key_risks": _validated_summary_texts(llm_result["key_risks"], criteria, kind="risk"),
        "missing_information": _deduplicate(
            [missing for criterion in criteria for missing in criterion["missing_information"]]
        ),
    }
    log_decision(company, result)
    completed = state["completed_companies"]
    return {
        "investment_results": {company: result},
        "completed_companies": completed if company in completed else [*completed, company],
    }
