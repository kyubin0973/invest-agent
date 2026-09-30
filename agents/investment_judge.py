"""Investment Judge (설계 산출물 2.1, 5장).

현재 기업의 Technology/Market 분석과 전체 Competition 결과만 사용해 12문항을
평가한다. 신규 검색, 최종 점수 계산, 투자판정은 LLM에 맡기지 않는다.
"""

import hashlib
import json
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from core.judge_logging import log_criteria, log_decision, log_repair_issues, log_stage
from core.llm import get_llm_settings, get_structured_llm
from core.state import InvestmentResult, InvestmentState, JudgeRunMetadata
from evaluation.criteria import (
    ALLOWED_SCORES,
    CONFIDENCE_LEVELS,
    CRITERIA_BY_ID,
    decide,
    validate_criteria,
)
from evaluation.judge_input import JudgeEvidenceContext, validate_and_build_evidence_context
from prompts.investment_judge import JUDGE_PROMPT_VERSION, build_judge_messages

JUDGE_RUN_SCHEMA_VERSION = "1"


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


def _canonical_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _invoke_judge(
    messages: list[tuple[str, str]],
    *,
    company: str,
    phase: str,
    prompt_fingerprint: str,
) -> dict:
    response = get_structured_llm(JudgeBatchOutput).invoke(
        messages,
        config={
            "run_name": f"investment_judge_{phase}",
            "metadata": {
                "company": company,
                "phase": phase,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "prompt_fingerprint": prompt_fingerprint,
                **get_llm_settings(),
            },
        },
    )
    return _as_dict(response)


def _judge_run_metadata(
    prompt_fingerprints: list[str],
    response_fingerprints: list[str],
    repaired_criterion_ids: list[str],
) -> JudgeRunMetadata:
    settings = get_llm_settings()
    return {
        "schema_version": JUDGE_RUN_SCHEMA_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "model_provider": str(settings["model_provider"]),
        "model": str(settings["model"]),
        "temperature": settings["temperature"],
        "seed": int(settings["seed"]),
        "input_fingerprint": prompt_fingerprints[0],
        "prompt_fingerprints": prompt_fingerprints,
        "response_fingerprints": response_fingerprints,
        "llm_call_count": len(prompt_fingerprints),
        "repaired_criterion_ids": repaired_criterion_ids,
    }


def _llm_score_criteria(
    state: InvestmentState,
    company: str,
    evidence_context: JudgeEvidenceContext,
) -> dict:
    """12문항 일괄 평가 후 문제가 있는 문항만 최대 한 번 보정한다."""
    expected_ids = set(CRITERIA_BY_ID)
    available_evidence = evidence_context.available_evidence
    allowed_evidence_by_criterion = evidence_context.allowed_evidence_by_criterion
    own_evidence_items = list(evidence_context.own_evidence.values())
    competition_evidence_items = evidence_context.competition_prompt_items()

    # 현재는 비용·지연을 줄이기 위해 12문항을 한 번에 평가한다. 실제 평가에서 긴 Context로
    # 문항 누락·상호간섭이 반복되면 이 호출 경계를 문항별 12회 호출로 바꿀 수 있다.
    log_stage(company, "LLM", "12개 평가 문항 일괄 호출")
    initial_messages = build_judge_messages(
        state,
        company,
        own_evidence_items,
        competition_evidence=competition_evidence_items,
    )
    initial_fingerprint = _canonical_fingerprint(initial_messages)
    prompt_fingerprints = [initial_fingerprint]
    settings = get_llm_settings()
    log_stage(
        company,
        "REPRO",
        (
            f"prompt={JUDGE_PROMPT_VERSION} | model={settings['model']} | "
            f"temperature={settings['temperature']} | seed={settings['seed']} | "
            f"input_sha256={initial_fingerprint}"
        ),
    )
    initial = _invoke_judge(
        initial_messages,
        company=company,
        phase="initial",
        prompt_fingerprint=initial_fingerprint,
    )
    response_fingerprints = [_canonical_fingerprint(initial)]
    log_stage(company, "REPRO", f"response_sha256={response_fingerprints[0]}")
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
            "judge_run": _judge_run_metadata(prompt_fingerprints, response_fingerprints, []),
        }

    issue_ids = set(issues)
    ordered_issue_ids = [criterion_id for criterion_id in CRITERIA_BY_ID if criterion_id in issue_ids]
    log_repair_issues(company, issues)
    log_stage(company, "REPAIR", f"문제 문항 {len(issue_ids)}개만 1회 재평가")
    previous_by_id = {
        item.get("criterion_id"): item
        for item in initial_criteria
        if item.get("criterion_id") in issue_ids
    }
    previous = [
        previous_by_id[criterion_id]
        for criterion_id in ordered_issue_ids
        if criterion_id in previous_by_id
    ]
    repair_messages = build_judge_messages(
        state,
        company,
        own_evidence_items,
        competition_evidence=competition_evidence_items,
        criterion_ids=issue_ids,
        repair_issues={criterion_id: issues[criterion_id] for criterion_id in ordered_issue_ids},
        previous_assessments=previous,
    )
    repair_fingerprint = _canonical_fingerprint(repair_messages)
    prompt_fingerprints.append(repair_fingerprint)
    log_stage(company, "REPRO", f"repair_sha256={repair_fingerprint}")
    repair = _invoke_judge(
        repair_messages,
        company=company,
        phase="repair",
        prompt_fingerprint=repair_fingerprint,
    )
    response_fingerprints.append(_canonical_fingerprint(repair))
    log_stage(company, "REPRO", f"repair_response_sha256={response_fingerprints[-1]}")
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
        "judge_run": _judge_run_metadata(
            prompt_fingerprints,
            response_fingerprints,
            ordered_issue_ids,
        ),
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

    evidence_context = validate_and_build_evidence_context(state, company)
    available_evidence = evidence_context.available_evidence
    allowed_evidence = evidence_context.allowed_evidence_by_criterion
    technology_count = len(state["technology_results"].get(company, {}).get("evidence", []))
    market_count = len(state["market_traction_results"].get(company, {}).get("evidence", []))
    log_stage(
        company,
        "START",
        (
            f"technology_evidence={technology_count} | market_evidence={market_count} | "
            f"competition_q4_evidence={len(evidence_context.competition_by_criterion['B04'])} | "
            f"competition_q9_evidence={len(evidence_context.competition_by_criterion['B09'])}"
        ),
    )
    llm_result = _llm_score_criteria(state, company, evidence_context)
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
        "judge_run": llm_result["judge_run"],
    }
    log_decision(company, result)
    completed = state["completed_companies"]
    return {
        "investment_results": {company: result},
        "completed_companies": completed if company in completed else [*completed, company],
    }
