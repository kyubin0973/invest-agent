"""LangGraph State Schema (Supervisor 패턴)

State는 "작업 페이로드"와 "제어 메타데이터"를 구분해 관리한다.
- 작업 페이로드: Agent 분석 결과. 기업명을 Key로 하는 dict에 덮어쓰므로 재작업해도 늘어나지 않는다.
- 제어 메타데이터: Supervisor가 라우팅·종료·재개를 판단하는 데 필요한 최소 값만 둔다.
- 결정 로그(라우팅 사유 등)는 State에 쌓지 않고 logger·LangSmith로 보내며, trace_id로 State와 잇는다.
평가 문항의 Score·Evidence·Confidence·N/A는 investment_results 내부에서 일관되게 관리한다.

데이터 흐름: Document/Chunk → EvidenceItem → AnalysisResult → Competition/Judge → InvestmentResult
→ Report Statement/Reference. Agent 간에는 원문 전체가 아니라 구조화 결과와 검증된 Evidence 참조를 전달한다.
"""

import operator
from typing import Annotated, Literal, TypedDict

# ---------------------------------------------------------------------------
# 공통 값
# ---------------------------------------------------------------------------
EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4", "E5"]
Confidence = Literal["High", "Medium", "Low"]
CompetitionCriterion = Literal["B04", "B09"]
CriterionStatus = Literal["SCORED", "INSUFFICIENT_EVIDENCE"]
Decision = Literal["INVEST", "HOLD", "HOLD_INSUFFICIENT_EVIDENCE"]

FinalRoute = Literal["INVEST_FOUND", "NO_INVEST"]

# Supervisor가 부를 수 있는 하위 Agent와 종료 신호
Worker = Literal[
    "technology", "market_traction", "competition", "investment_judge", "report_generator", "quality_evaluator"
]
NextStep = Literal[
    "technology", "market_traction", "competition", "investment_judge", "report_generator", "quality_evaluator",
    "FINISH",
]
AnalysisAgent = Literal["technology", "market_traction"]
RunStatus = Literal["RUNNING", "SUCCESS", "FAILED"]
NodeStatus = Literal["DONE", "FAILED"]
QualityItem = Literal["groundedness", "neutrality", "bias_control"]
QualityOwner = Literal["technology", "market_traction", "report_generator"]


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


class ReworkRequest(TypedDict):
    """Supervisor가 근거 부족으로 판단해 분석 Agent에 보내는 재작업 요청."""

    company: str
    agent: AnalysisAgent
    criterion_ids: list[str]  # 근거가 부족한 평가 문항 (B01~B10, H11~H12). 품질 평가에서 온 요청이면 빈 목록
    reason: str
    feedback: list[str]  # 품질 평가가 지적한 문장과 이유. 근거 부족에서 온 요청이면 빈 목록


class QualityIssue(TypedDict):
    owner: QualityOwner  # 수정할 담당
    company: str | None  # 담당이 분석 Agent면 대상 기업
    statement: str  # 문제가 된 보고서 문장
    reason: str  # 미달 이유


class QualityCheck(TypedDict):
    item: QualityItem
    passed: bool
    issues: list[QualityIssue]  # 통과하면 빈 목록


class QualityResult(TypedDict):
    passed: bool  # 모든 항목 통과 여부 (Python 규칙으로 계산)
    checks: list[QualityCheck]
    feedback: list[str]  # Report Generator가 재작성할 때 반영할 지시 (담당이 report_generator인 지적)


class ErrorItem(TypedDict):
    node: str
    company: str | None
    message: str
    ts: str  # ISO 8601


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
    # ── 작업 페이로드 ─────────────────────────────────────────────
    # 후보 입력
    candidate_companies: list[str]
    company_profiles: dict[str, CompanyProfile]

    # 기업별 분석 결과 (기업명 → 결과). 재작업 결과는 같은 Key를 덮어쓴다.
    technology_results: Annotated[dict[str, AnalysisResult], merge_dict]
    market_traction_results: Annotated[dict[str, AnalysisResult], merge_dict]

    # 전체 후보 비교
    competition_result: CompetitionResult | None

    # 투자 평가 (기업명 → 결과)
    investment_results: Annotated[dict[str, InvestmentResult], merge_dict]

    # 최종 산출물. PDF 본문은 파일로 저장하고 State에는 경로만 둔다.
    references: list[ReferenceItem]
    final_report: str | None
    report_path: str | None
    quality_result: QualityResult | None  # 보고서 품질 평가 결과 (미달 시 재작성 지시 포함)

    # ── 제어 메타데이터 (라우팅·종료·재개에 필요한 최소 값) ───────────
    trace_id: str  # 외부 로그·LangSmith Trace와 잇는 상관 키
    status: RunStatus

    # 라우팅: Supervisor만 쓰고, Gate 함수와 하위 Agent가 읽는다.
    next: NextStep | None  # 다음에 실행할 하위 Agent 또는 FINISH
    current_company: str | None  # 기업 단위 Agent(Technology, Market, Judge)의 작업 대상
    rework_request: ReworkRequest | None  # 재작업일 때만 값이 있다 (첫 분석은 None)

    # Supervisor의 작업 추적
    pending_judge: list[str]  # 평가(재평가)가 필요한 기업
    competition_stale: bool  # 재작업으로 분석이 바뀌어 비교를 다시 해야 하는지
    rework_counts: dict[str, int]  # "기업:agent" → 재작업 횟수 (상한 관리)
    report_attempts: int  # 보고서 작성 횟수 (상한 관리)
    report_evaluated: bool  # 마지막으로 작성한 보고서의 품질 평가를 요청했는지

    # 종료 보장
    step_count: int  # Supervisor가 하위 Agent를 부른 횟수
    max_steps: int

    # 재개·복구: 노드별 마지막 상태와 오류. 오류는 실패할 때만 한 건씩 추가된다.
    node_status: Annotated[dict[str, NodeStatus], merge_dict]  # "node" 또는 "node:기업" → 상태
    errors: Annotated[list[ErrorItem], operator.add]
    last_error: str | None

    # ── 기존 Agent가 반환하는 완료 표시 (호환용, 라우팅에는 쓰지 않는다) ──
    technology_done: bool
    market_traction_done: bool
    completed_companies: list[str]
    final_route: FinalRoute | None
