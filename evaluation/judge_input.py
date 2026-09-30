"""Investment Judge 입력 계약 검증과 문항별 Evidence 범위 구성."""

from dataclasses import dataclass

from core.state import InvestmentState
from evaluation.criteria import CRITERIA_BY_ID

COMPETITION_EVIDENCE_CRITERIA = ("B04", "B09")
COMPETITION_EVIDENCE_CRITERIA_SET = frozenset(COMPETITION_EVIDENCE_CRITERIA)
EVIDENCE_LEVELS = frozenset({"E1", "E2", "E3", "E4", "E5"})
SOURCE_TYPES = frozenset({"company", "partner", "industry_report"})
ANALYSIS_KEYS = frozenset(
    {"summary", "findings", "evidence", "risks", "missing_information"}
)


@dataclass(frozen=True)
class JudgeEvidenceContext:
    own_evidence: dict[str, dict]
    competition_by_criterion: dict[str, dict[str, dict]]
    competition_owners: dict[str, str]
    excluded_evidence_ids: tuple[str, ...] = ()
    excluded_competition_refs: tuple[str, ...] = ()

    @property
    def competition_evidence(self) -> dict[str, dict]:
        combined: dict[str, dict] = {}
        for evidence in self.competition_by_criterion.values():
            combined.update(evidence)
        return combined

    @property
    def available_evidence(self) -> dict[str, dict]:
        return {**self.competition_evidence, **self.own_evidence}

    @property
    def allowed_evidence_by_criterion(self) -> dict[str, set[tuple[str, str]]]:
        own_refs = {
            (item["chunk_id"], item["source_id"]) for item in self.own_evidence.values()
        }
        allowed = {criterion_id: set(own_refs) for criterion_id in CRITERIA_BY_ID}
        for criterion_id, evidence in self.competition_by_criterion.items():
            allowed[criterion_id].update(
                (item["chunk_id"], item["source_id"]) for item in evidence.values()
            )
        return allowed

    def competition_prompt_items(self) -> list[tuple[str, dict, list[str]]]:
        scopes: dict[str, list[str]] = {}
        for criterion_id, evidence in self.competition_by_criterion.items():
            for chunk_id in evidence:
                scopes.setdefault(chunk_id, []).append(criterion_id)
        return sorted(
            [
                (self.competition_owners[chunk_id], item, sorted(scopes[chunk_id]))
                for chunk_id, item in self.competition_evidence.items()
            ],
            key=lambda value: (value[1]["source_id"], value[1]["chunk_id"]),
        )


def _require_non_empty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: 비어 있지 않은 문자열이어야 합니다.")
    return value


def _validate_string_list(value: object, location: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{location}: 문자열 list여야 합니다.")
    return value


def _validate_evidence_item(item: object, location: str) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"{location}: EvidenceItem object가 아닙니다.")
    required = {
        "source_id",
        "chunk_id",
        "page",
        "title",
        "publisher",
        "published_date",
        "source_type",
        "reference_metadata",
        "evidence_level",
        "fact",
    }
    missing = required - set(item)
    if missing:
        raise ValueError(f"{location}: 필수 필드 누락 {sorted(missing)}")

    _require_non_empty_string(item["source_id"], f"{location}.source_id")
    _require_non_empty_string(item["chunk_id"], f"{location}.chunk_id")
    if not isinstance(item["page"], int) or isinstance(item["page"], bool) or item["page"] < 1:
        raise ValueError(f"{location}.page: 1 이상의 정수여야 합니다.")
    for field in ("title", "publisher", "published_date"):
        if not isinstance(item[field], str):
            raise ValueError(f"{location}.{field}: 문자열이어야 합니다.")
    if item["source_type"] not in SOURCE_TYPES:
        raise ValueError(f"{location}.source_type: 허용되지 않은 값 {item['source_type']!r}")
    if not isinstance(item["reference_metadata"], dict):
        raise ValueError(f"{location}.reference_metadata: object여야 합니다.")
    if item["evidence_level"] is not None and item["evidence_level"] not in EVIDENCE_LEVELS:
        raise ValueError(f"{location}.evidence_level: null 또는 E1~E5 중 하나여야 합니다.")
    if item["fact"] is not None:
        _require_non_empty_string(item["fact"], f"{location}.fact")
    return item


def _is_scoreable_evidence(item: dict) -> bool:
    """앞단 분석이 fact와 level을 모두 확정한 Evidence만 Judge 채점에 사용한다."""
    return (
        item.get("evidence_level") in EVIDENCE_LEVELS
        and isinstance(item.get("fact"), str)
        and bool(item["fact"].strip())
    )


def _competition_criterion_ids(comparison: dict, location: str) -> list[str]:
    """신·구 Competition 출력 모두 수용하되 문항 범위는 Judge에서 제한한다."""
    raw = comparison.get("criterion_ids")
    if raw is not None:
        if (
            not isinstance(raw, list)
            or not raw
            or len(set(raw)) != len(raw)
            or not set(raw).issubset(COMPETITION_EVIDENCE_CRITERIA_SET)
        ):
            raise ValueError(f"{location}.criterion_ids: B04/B09만 중복 없이 허용합니다.")
        return raw

    # 현재 팀 Competition 계약에는 criterion_ids가 없다. Risk 축만 B09로,
    # 나머지 비교축은 차별성 B04로 보수적으로 연결한다.
    dimension = str(comparison.get("dimension") or "").casefold()
    return ["B09"] if "risk" in dimension or "리스크" in dimension else ["B04"]


def _validate_analysis_result(result: object, company: str, agent: str) -> dict[str, dict]:
    location = f"{agent}[{company}]"
    if not isinstance(result, dict):
        raise ValueError(f"{location}: AnalysisResult object가 없습니다.")
    missing_keys = ANALYSIS_KEYS - set(result)
    if missing_keys:
        raise ValueError(f"{location}: 필수 필드 누락 {sorted(missing_keys)}")
    if not isinstance(result["summary"], str):
        raise ValueError(f"{location}.summary: 문자열이어야 합니다.")
    _validate_string_list(result["risks"], f"{location}.risks")
    _validate_string_list(result["missing_information"], f"{location}.missing_information")

    evidence_items = result["evidence"]
    if not isinstance(evidence_items, list):
        raise ValueError(f"{location}.evidence: list여야 합니다.")
    local_evidence: dict[str, dict] = {}
    for index, raw_item in enumerate(evidence_items):
        item = _validate_evidence_item(raw_item, f"{location}.evidence[{index}]")
        chunk_id = item["chunk_id"]
        if chunk_id in local_evidence:
            raise ValueError(f"{location}: 중복 chunk_id {chunk_id}")
        local_evidence[chunk_id] = item

    findings = result["findings"]
    if not isinstance(findings, list):
        raise ValueError(f"{location}.findings: list여야 합니다.")
    for index, finding in enumerate(findings):
        finding_location = f"{location}.findings[{index}]"
        if not isinstance(finding, dict):
            raise ValueError(f"{finding_location}: Finding object가 아닙니다.")
        _require_non_empty_string(finding.get("dimension"), f"{finding_location}.dimension")
        _require_non_empty_string(finding.get("statement"), f"{finding_location}.statement")
        refs = finding.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{finding_location}.evidence_refs: 비어 있지 않은 list여야 합니다.")
        for ref in refs:
            if not isinstance(ref, dict):
                raise ValueError(f"{finding_location}: 잘못된 EvidenceRef")
            item = local_evidence.get(ref.get("chunk_id"))
            if not item or item["source_id"] != ref.get("source_id"):
                raise ValueError(
                    f"{finding_location}: AnalysisResult.evidence에 없는 참조 "
                    f"{ref.get('chunk_id')} | {ref.get('source_id')}"
                )
    return local_evidence


def validate_and_build_evidence_context(
    state: InvestmentState,
    current_company: str,
) -> JudgeEvidenceContext:
    """Judge 호출 전에 전체 분석·Competition Evidence 계약을 검증한다."""
    candidates = state.get("candidate_companies")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(company, str) or not company.strip() for company in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("candidate_companies: 중복 없는 비어 있지 않은 기업명 list여야 합니다.")
    if current_company not in candidates:
        raise ValueError(f"current_company이 후보 기업에 없습니다: {current_company!r}")
    for field in ("technology_results", "market_traction_results"):
        if not isinstance(state.get(field), dict):
            raise ValueError(f"{field}: 기업별 AnalysisResult object여야 합니다.")

    catalog: dict[str, dict] = {}
    scoreable_catalog: dict[str, dict] = {}
    owners: dict[str, set[str]] = {}
    own_evidence: dict[str, dict] = {}

    for company in candidates:
        for agent, results in (
            ("technology_results", state["technology_results"]),
            ("market_traction_results", state["market_traction_results"]),
        ):
            local = _validate_analysis_result(results.get(company), company, agent)
            for chunk_id, item in local.items():
                existing = catalog.get(chunk_id)
                existing_owners = owners.get(chunk_id, set())
                shared_industry_evidence = bool(
                    existing
                    and existing["source_id"] == item["source_id"]
                    and existing.get("source_type") == "industry_report"
                    and item.get("source_type") == "industry_report"
                )
                if existing and company not in existing_owners and not shared_industry_evidence:
                    raise ValueError(
                        f"전체 분석 결과에서 chunk_id 충돌: {chunk_id} "
                        f"({sorted(existing_owners)}, {company})"
                    )
                # 동일 회사의 Technology/Market이 같은 원문 청크를 서로 다른 관점으로
                # 분석하는 것은 허용한다. 공통 산업자료가 여러 기업에서 재사용되면 최초의
                # 검증 완료 표현을 Competition용 대표 item으로 유지한다.
                if not existing or company in existing_owners:
                    catalog[chunk_id] = item
                    if _is_scoreable_evidence(item):
                        scoreable_catalog[chunk_id] = item
                elif _is_scoreable_evidence(item) and chunk_id not in scoreable_catalog:
                    scoreable_catalog[chunk_id] = item
                owners.setdefault(chunk_id, set()).add(company)
                if company == current_company and _is_scoreable_evidence(item):
                    own_evidence[chunk_id] = item

    competition = state.get("competition_result")
    if not isinstance(competition, dict):
        raise ValueError("competition_result가 없습니다.")
    candidate_set = set(candidates)
    for field in ("target_market_context", "differentiation", "relative_risks"):
        values = competition.get(field)
        if not isinstance(values, dict) or not candidate_set.issubset(values):
            raise ValueError(
                f"competition_result.{field}: 모든 후보 기업을 포함한 object여야 합니다."
            )
        if any(not isinstance(values[company], str) for company in candidates):
            raise ValueError(f"competition_result.{field}: 기업별 문자열이어야 합니다.")
    comparisons = competition.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("competition_result.comparisons: list여야 합니다.")

    competition_by_criterion = {
        criterion_id: {} for criterion_id in COMPETITION_EVIDENCE_CRITERIA
    }
    excluded_competition_refs: list[str] = []
    for index, comparison in enumerate(comparisons):
        location = f"competition_result.comparisons[{index}]"
        if not isinstance(comparison, dict):
            raise ValueError(f"{location}: Comparison object가 아닙니다.")
        _require_non_empty_string(comparison.get("dimension"), f"{location}.dimension")
        criterion_ids = _competition_criterion_ids(comparison, location)
        company_findings = comparison.get("company_findings")
        if not isinstance(company_findings, dict) or not candidate_set.issubset(company_findings):
            raise ValueError(f"{location}.company_findings: 모든 후보 기업을 포함해야 합니다.")
        if any(not isinstance(company_findings[company], str) for company in candidates):
            raise ValueError(f"{location}.company_findings: 기업별 문자열이어야 합니다.")
        refs = comparison.get("evidence_refs")
        if not isinstance(refs, list):
            excluded_competition_refs.append(f"{location}: evidence_refs가 list가 아님")
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                excluded_competition_refs.append(f"{location}: {ref!r}")
                continue
            chunk_id = ref.get("chunk_id")
            source_id = ref.get("source_id")
            item = catalog.get(chunk_id)
            if not item or item["source_id"] != source_id:
                excluded_competition_refs.append(f"{location}: {chunk_id} | {source_id}")
                continue
            scoreable_item = scoreable_catalog.get(chunk_id)
            if not scoreable_item:
                excluded_competition_refs.append(f"{location}: {chunk_id} | 미분류")
                continue
            for criterion_id in criterion_ids:
                competition_by_criterion[criterion_id][chunk_id] = scoreable_item

    competition_evidence = {
        chunk_id: item
        for evidence in competition_by_criterion.values()
        for chunk_id, item in evidence.items()
    }
    competition_owners = {
        chunk_id: (
            next(iter(owners[chunk_id]))
            if len(owners[chunk_id]) == 1
            else "공통 산업자료"
        )
        for chunk_id in competition_evidence
    }
    return JudgeEvidenceContext(
        own_evidence=own_evidence,
        competition_by_criterion=competition_by_criterion,
        competition_owners=competition_owners,
        excluded_evidence_ids=tuple(sorted(set(catalog) - set(scoreable_catalog))),
        excluded_competition_refs=tuple(dict.fromkeys(excluded_competition_refs)),
    )
