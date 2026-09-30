"""LangGraph State Schema (설계 산출물 4.1 State 설계, 4.1.1 중첩 Schema)

State는 Agent 분석 결과와 Graph 제어정보를 함께 관리한다.
병렬 Agent는 서로 다른 결과 Key에 기록하며, 평가 문항의 Score·Evidence·Confidence·N/A는
investment_results 내부에서 일관되게 관리한다.

데이터 흐름: Document/Chunk → EvidenceItem → AnalysisResult → Competition/Judge → InvestmentResult
→ Report Statement/Reference. Agent 간에는 원문 전체가 아니라 구조화 결과와 검증된 Evidence 참조를 전달한다.
"""

from typing import Annotated, Literal, TypedDict

# ---------------------------------------------------------------------------
# 공통 값
# ---------------------------------------------------------------------------
EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4", "E5"]
Confidence = Literal["High", "Medium", "Low"]
CompetitionCriterion = Literal["B04", "B09"]
CriterionStatus = Literal["SCORED", "INSUFFICIENT_EVIDENCE"]
Decision = Literal["INVEST", "HOLD", "HOLD_INSUFFICIENT_EVIDENCE"]

AnalysisRoute = Literal["NEXT_CANDIDATE", "COMPETITION"]
EvaluationRoute = Literal["NEXT_CANDIDATE", "FINAL_DECISION"]
FinalRoute = Literal["INVEST_FOUND", "NO_INVEST"]


# ---------------------------------------------------------------------------
# 중첩 Schema
# ---------------------------------------------------------------------------
class EvidenceRef(TypedDict):
    """앞선 Agent가 검색한 EvidenceItem을 가리키는 참조 (원본 Metadata를 새로 쓰지 않는다)."""

    chunk_id: str
    source_id: str


class EvidenceItem(TypedDict):
    source_id: str
    chunk_id: str
    page: int  # 원본 page
    title: str
    publisher: str
    published_date: str
    source_type: str  # company / partner / industry_report
    reference_metadata: dict  # reference_type, url, site_name, ...
    evidence_level: EvidenceLevel | None
    fact: str | None


class Finding(TypedDict):
    dimension: str  # 분석축 (설계 2.1.1)
    statement: str  # Evidence 기반 분석 문장
    evidence_refs: list[EvidenceRef]  # AnalysisResult.evidence 안의 항목 참조


class AnalysisResult(TypedDict):
    summary: str
    findings: list[Finding]
    evidence: list[EvidenceItem]
    risks: list[str]
    missing_information: list[str]


class Comparison(TypedDict):
    dimension: str  # 기술, 제품, AI/Data, 기술 성숙도, 상용화, 제조·배치 확장성, 시장전략, 차별성, Risk
    criterion_ids: list[CompetitionCriterion]  # 이 비교 Evidence를 사용할 수 있는 Judge 문항
    company_findings: dict[str, str]  # 기업명 → 비교축별 Finding
    evidence_refs: list[EvidenceRef]


class CompetitionResult(TypedDict):
    target_market_context: dict[str, str]  # 기업명 → Target Market 설명 (시장 차이를 자동 우열로 바꾸지 않음)
    comparisons: list[Comparison]
    differentiation: dict[str, str]  # 기업명 → 핵심 차별점
    relative_risks: dict[str, str]  # 기업명 → 상대적 Risk


class CriterionResult(TypedDict):
    criterion_id: str  # B01~B10, H11~H12 (Q1~Q12와 일대일)
    criterion_name: str
    score: int | None  # 1~5, 판단 불가 시 None (N/A)
    status: CriterionStatus
    evidence: list[EvidenceItem]  # Python이 검증한 EvidenceItem만 연결
    reasoning: str
    confidence: Confidence
    missing_information: list[str]


class JudgeRunMetadata(TypedDict):
    schema_version: str
    prompt_version: str
    model_provider: str
    model: str
    temperature: int | float
    seed: int
    input_fingerprint: str  # 최초 호출 전체 메시지의 SHA-256
    prompt_fingerprints: list[str]  # 최초 호출과 선택적 보정 호출 순서
    response_fingerprints: list[str]  # LLM 구조화 응답의 canonical SHA-256
    llm_call_count: int
    repaired_criterion_ids: list[str]


class InvestmentResult(TypedDict):
    criteria: list[CriterionResult]  # N/A 문항을 포함해 항상 12개
    final_score: float | None
    evidence_coverage: float  # N_scored / 12 (0~1)
    decision: Decision
    decision_reason: str
    key_strengths: list[str]
    key_risks: list[str]
    missing_information: list[str]
    judge_run: JudgeRunMetadata


class Eligibility(TypedDict):
    is_private: bool | None
    exit_completed: bool | None
    evaluation_date: str | None  # YYYY-MM-DD


class DocumentScope(TypedDict):
    source_ids: list[str]  # 기업별 검색에 허용된 문서
    document_types: list[str]


class CompanyProfile(TypedDict):
    """실행 전 사전 입력한 값. LLM이 자격조건·투자 단계·기업 사실을 추정하거나 보완하지 않는다."""

    name: str
    target_market: str | None  # 해당 기업이 실제 진입하려는 세부 시장
    funding_stage: str | None  # 사전 확인한 투자 단계, 미확인 시 None
    eligibility: Eligibility
    document_scope: DocumentScope
    source_refs: dict[str, str]  # 확인 가능한 Profile 사실별 source_id 또는 근거 URL


class ReferenceItem(TypedDict):
    source_id: str
    title: str
    publisher: str
    published_date: str
    reference_metadata: dict
    pages: list[int]  # 실제 사용한 원본 페이지


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

    # 분석·평가 Loop 위치
    current_company: str | None
    current_company_index: int

    # 기업별 분석 결과 (기업명 → 결과)
    technology_results: Annotated[dict[str, AnalysisResult], merge_dict]
    market_traction_results: Annotated[dict[str, AnalysisResult], merge_dict]

    # 병렬 분석 완료 (동일 기업의 Fan-in 보장)
    technology_done: bool
    market_traction_done: bool
    analysis_join_ready: bool
    analysis_route: AnalysisRoute | None

    # 전체 후보 비교 (1회)
    competition_result: CompetitionResult | None

    # 투자 평가 Loop
    investment_results: Annotated[dict[str, InvestmentResult], merge_dict]
    completed_companies: list[str]
    evaluation_route: EvaluationRoute | None
    final_route: FinalRoute | None

    # 최종 산출물
    references: list[ReferenceItem]
    final_report: str | None
