"""LangGraph State Schema (설계 산출물 6장)

State는 Agent 간 분석 결과와 Graph 제어정보를 함께 관리한다.
Score, Evidence, Confidence, Insufficient Evidence는 별도 Top-level State로 분리하지 않고
investment_results 안에 함께 저장해 "평가항목 ↔ Evidence ↔ Score" 관계를 유지한다.
"""

from typing import Annotated, Literal, TypedDict

# ---------------------------------------------------------------------------
# Nested Schema
# ---------------------------------------------------------------------------
EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4", "E5"]
Confidence = Literal["High", "Medium", "Low"]


class EvidenceItem(TypedDict):
    source_id: str
    chunk_id: str
    page: int
    title: str
    publisher: str
    published_date: str
    source_type: str  # company / partner / industry_report
    reference_metadata: dict  # reference_type, url, site_name, ...
    evidence_level: EvidenceLevel | None
    fact: str | None


class AnalysisResult(TypedDict):
    summary: str
    findings: list[str]
    evidence: list[EvidenceItem]
    risks: list[str]
    missing_information: list[str]


class CompanyComparison(TypedDict):
    company: str
    differentiation: list[str]
    competitive_risks: list[str]


class CompetitionResult(TypedDict):
    summary: str
    comparisons: list[CompanyComparison]
    missing_information: list[str]


class CriterionResult(TypedDict):
    criterion_id: str  # B01 ~ B12
    criterion_name: str
    score: int | None  # 1~5, 판단 불가 시 None (N/A)
    status: Literal["SCORED", "INSUFFICIENT_EVIDENCE"]
    evidence: list[EvidenceItem]
    reasoning: str
    confidence: Confidence
    missing_information: list[str]


class InvestmentResult(TypedDict):
    criteria: list[CriterionResult]
    final_score: float | None
    evidence_coverage: float
    decision: Literal["INVEST", "HOLD"]
    decision_reason: str
    key_strengths: list[str]
    key_risks: list[str]
    missing_information: list[str]


class CompanyProfile(TypedDict):
    name: str
    company_key: str  # 문서 메타데이터의 company 값 (figure, apptronik, 1x)
    document_ids: list[str]  # 이 기업의 문서 범위 (manifest source_id)
    team: list[dict]  # 팀·창업자 사실 정보 (출처 포함)
    funding: list[dict]  # 투자 이력 (출처 포함)


class ReferenceItem(TypedDict):
    source_id: str
    title: str
    publisher: str
    published_date: str
    reference_metadata: dict
    pages: list[int]  # 실제 사용한 페이지


AnalysisRoute = Literal["NEXT_CANDIDATE", "COMPETITION"]
EvaluationRoute = Literal["NEXT_CANDIDATE", "FINAL_DECISION"]
FinalRoute = Literal["INVEST_FOUND", "NO_INVEST"]


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------
def merge_dict(left: dict, right: dict) -> dict:
    """기업별 결과 dict를 병합한다. 병렬 노드가 서로 다른 키를 써도 덮어쓰지 않는다."""
    return {**(left or {}), **(right or {})}


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class InvestmentState(TypedDict):
    # 후보 입력
    candidate_companies: list[str]
    company_profiles: dict[str, CompanyProfile]

    # 분석 루프 제어
    current_company: str | None
    current_company_index: int

    # 분석 결과 (기업명 → 결과)
    technology_results: Annotated[dict[str, AnalysisResult], merge_dict]
    market_traction_results: Annotated[dict[str, AnalysisResult], merge_dict]

    # 병렬 분석 제어
    technology_done: bool
    market_traction_done: bool
    analysis_join_ready: bool
    analysis_route: AnalysisRoute | None

    # 경쟁 분석 (전체 후보 1회)
    competition_result: CompetitionResult | None

    # 투자 평가 루프
    investment_results: Annotated[dict[str, InvestmentResult], merge_dict]
    completed_companies: list[str]
    evaluation_route: EvaluationRoute | None
    final_route: FinalRoute | None

    # 최종 산출물
    references: list[ReferenceItem]
    final_report: str | None
