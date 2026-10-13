"""Supervisor (조정 계층)

하위 Agent는 Supervisor하고만 통신한다. 모든 하위 Agent는 작업 후 Supervisor로 돌아오고,
Supervisor가 현재 State를 보고 다음에 부를 Agent 하나를 정한다 (순서 하드코딩 없음).

- 판단(decide): State만 읽는 순수 함수. LLM을 쓰지 않아 같은 State면 항상 같은 결정을 낸다.
- 기록(supervisor_node): 결정을 State의 제어 필드에 쓰고, 사유는 logger·LangSmith로 보낸다.
- 분기(route): Supervisor가 정한 `next`를 그대로 돌려주는 Gate. add_conditional_edges에 연결한다.

근거 충분성: 평가 가능 문항이 MIN_SCORED_CRITERIA(9개) 미만인 기업은 근거 부족으로 보고,
N/A 문항의 근거를 맡은 분석 Agent에 재작업을 요청한다.
품질 평가: 미달 지적의 담당(owner)이 분석 Agent면 그 Agent에 재작업을 요청한 뒤 보고서를 다시 쓰고,
담당이 Report Generator면 보고서만 다시 쓴다.
보고서 작성 시점은 단계 수가 아니라 위 판단으로 정한다. 재작업·보고서 재작성·전체 단계 수의 상한은
끝나지 않는 Loop에서 빠져나오기 위한 장치다.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langsmith import traceable

from core.state import AnalysisResult, InvestmentState, ReworkRequest
from evaluation.criteria import CRITERIA_BY_ID, MARKET, MIN_SCORED_CRITERIA, TECH

logger = logging.getLogger(__name__)

FINISH = "FINISH"
COMPETITION = "competition"
JUDGE = "investment_judge"
REPORT = "report_generator"
QUALITY = "quality_evaluator"

ANALYSIS_AGENTS = (TECH, MARKET)
COMPANY_WORKERS = (TECH, MARKET, JUDGE)  # current_company 단위로 실행되는 Agent
WORKERS = (TECH, MARKET, COMPETITION, JUDGE, REPORT, QUALITY)
RESULT_KEY = {TECH: "technology_results", MARKET: "market_traction_results"}

MAX_REWORK_PER_AGENT = 1  # 기업·Agent별 재작업 횟수
MAX_REPORT_ATTEMPTS = 2  # 보고서 작성 횟수 (최초 1회 + 품질 미달 재작성 1회)
MAX_NODE_RETRIES = 1  # 하위 Agent가 예외로 실패했을 때 재시도 횟수


def default_max_steps(num_candidates: int) -> int:
    """하위 Agent 호출 횟수 상한: 기본 경로 + 재작업 경로 + 보고서 재작성 + 실패 재시도 여유."""
    base = 3 * num_candidates + 1  # 분석 2개·평가 1개 / 기업 + 비교 1회
    rework = MAX_REWORK_PER_AGENT * 2 * num_candidates + num_candidates + 1  # 재작업 + 재평가 + 비교 1회
    return base + rework + 2 * MAX_REPORT_ATTEMPTS + 4


def status_key(node: str, company: str | None) -> str:
    return f"{node}:{company}" if node in COMPANY_WORKERS else node


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Decision:
    next: str  # 하위 Agent 이름 또는 FINISH
    reason: str
    company: str | None = None
    rework: ReworkRequest | None = None
    updates: dict = field(default_factory=dict)  # 결정과 함께 바꿀 제어 필드


# ---------------------------------------------------------------------------
# 근거 충분성 평가
# ---------------------------------------------------------------------------
def assess_sufficiency(state: InvestmentState) -> list[ReworkRequest]:
    """근거가 부족한 기업의 N/A 문항을, 그 근거를 맡은 분석 Agent별 재작업 요청으로 바꾼다."""
    requests: list[ReworkRequest] = []
    for company in state["candidate_companies"]:
        result = state["investment_results"].get(company)
        if not result:
            continue
        missing = [c["criterion_id"] for c in result["criteria"] if c["status"] != "SCORED"]
        scored = len(result["criteria"]) - len(missing)
        if scored >= MIN_SCORED_CRITERIA:
            continue
        for agent in ANALYSIS_AGENTS:
            ids = [cid for cid in missing if agent in CRITERIA_BY_ID[cid].agents]
            if ids:
                requests.append({
                    "company": company,
                    "agent": agent,
                    "criterion_ids": ids,
                    "reason": f"평가 가능 문항 {scored}개 (최소 {MIN_SCORED_CRITERIA}개), 근거 부족 문항 {', '.join(ids)}",
                    "feedback": [],
                })
    return requests


def quality_rework_requests(state: InvestmentState) -> list[ReworkRequest]:
    """품질 평가의 미달 지적 중 담당이 분석 Agent인 것을 기업·Agent별 재작업 요청으로 묶는다."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for check in (state.get("quality_result") or {}).get("checks", []):
        if check["passed"]:
            continue
        for issue in check["issues"]:
            if issue["owner"] in ANALYSIS_AGENTS and issue["company"] in state["candidate_companies"]:
                note = f"[{check['item']}] {issue['statement']} — {issue['reason']}"
                grouped.setdefault((issue["company"], issue["owner"]), []).append(note)
    return [
        {"company": company, "agent": agent, "criterion_ids": [], "reason": f"품질 평가 지적 {len(notes)}건", "feedback": notes}
        for (company, agent), notes in grouped.items()
    ]


def _empty_analysis(message: str) -> AnalysisResult:
    return {"summary": "", "findings": [], "evidence": [], "risks": [], "missing_information": [message]}


# ---------------------------------------------------------------------------
# 판단: 다음에 부를 하위 Agent
# ---------------------------------------------------------------------------
def _failure(state: InvestmentState) -> Decision | None:
    """직전에 부른 Agent가 실패했으면 재시도·제외·중단 중 하나를 정한다."""
    node, company = state.get("next"), state.get("current_company")
    if not node or node == FINISH:
        return None
    key = status_key(node, company)
    if state["node_status"].get(key) != "FAILED":
        return None
    company = company if node in COMPANY_WORKERS else None
    failures = sum(1 for e in state["errors"] if status_key(e["node"], e["company"]) == key)
    if failures <= MAX_NODE_RETRIES and state["step_count"] < state["max_steps"]:
        return Decision(node, f"{key} 실패 → 재시도 ({failures}/{MAX_NODE_RETRIES})", company, state.get("rework_request"))
    if node in ANALYSIS_AGENTS:
        if company in state[RESULT_KEY[node]]:
            return None  # 재작업 실패: 기존 분석 결과를 그대로 쓴다
        # 최초 분석 실패: 빈 결과로 제외하고 계속 진행한다 (해당 문항은 N/A)
        empty = {RESULT_KEY[node]: {company: _empty_analysis(f"{node} 실행 실패로 분석하지 못함")}}
        return Decision(node, f"{key} 재시도 후에도 실패 → 제외", company, updates={**empty, "_skip": True})
    return Decision(FINISH, f"{key} 재시도 후에도 실패 → 중단", updates={"status": "FAILED"})


def decide(state: InvestmentState) -> Decision:
    companies = state["candidate_companies"]
    steps_left = state["max_steps"] - state["step_count"]

    failure = _failure(state)
    if failure:
        return failure
    if steps_left <= 0:
        done = bool(state.get("final_report")) and state["report_evaluated"]
        return Decision(FINISH, "단계 상한 도달", updates={"status": "SUCCESS" if done else "FAILED"})

    # 1) 분석이 없는 기업·관점부터 채운다
    for company in companies:
        for agent in ANALYSIS_AGENTS:
            if company not in state[RESULT_KEY[agent]]:
                return Decision(agent, "분석 결과 없음", company)

    # 2) 재작업 요청: 모든 후보를 한 번씩 평가한 뒤에 판단한다
    #    - 첫 보고서 전: 근거가 부족한 기업의 N/A 문항 담당 Agent
    #    - 품질 평가 미달 후: 지적된 문장의 담당 Agent (보고서를 다시 쓸 수 있을 때만)
    pending = state["pending_judge"]
    attempts = state["report_attempts"]
    quality = state.get("quality_result") or {}
    quality_failed = state["report_evaluated"] and not quality.get("passed", True)
    can_rewrite = attempts < MAX_REPORT_ATTEMPTS
    if all(c in state["investment_results"] for c in companies):
        if attempts == 0:
            requests, label = assess_sufficiency(state), "근거 부족"
        elif quality_failed and can_rewrite:
            requests, label = quality_rework_requests(state), "품질 평가 미달"
        else:
            requests, label = [], ""
        for request in requests:
            key = f"{request['company']}:{request['agent']}"
            count = state["rework_counts"].get(key, 0)
            rejudge = len(set(pending) | {request["company"]})
            needed = 1 + 1 + rejudge + 2  # 재작업 + 비교 + 재평가 + 보고서·품질 평가
            if count < MAX_REWORK_PER_AGENT and needed <= steps_left:
                return Decision(
                    request["agent"], f"{label} → 재작업 요청: {request['reason']}", request["company"], request,
                    updates={
                        "rework_counts": {**state["rework_counts"], key: count + 1},
                        "pending_judge": pending if request["company"] in pending else [*pending, request["company"]],
                        "competition_stale": True,
                    },
                )

    # 3) 비교: 없거나, 재작업으로 분석이 바뀌었으면 다시
    if state.get("competition_result") is None or state["competition_stale"]:
        reason = "분석 변경 반영" if state.get("competition_result") else "전체 후보 분석 완료"
        return Decision(COMPETITION, reason, updates={"competition_stale": False})

    # 4) 평가(재평가)가 필요한 기업
    if pending:
        reason = "재작업 결과 재평가" if pending[0] in state["investment_results"] else "분석·비교 완료"
        return Decision(JUDGE, reason, pending[0], updates={"pending_judge": pending[1:]})

    # 5) 보고서 작성 → 품질 평가 → 미달이면 재작성
    write = {"report_attempts": attempts + 1, "report_evaluated": False}
    if attempts == 0:
        invest = any(r["decision"] == "INVEST" for r in state["investment_results"].values())
        short = sorted({r["company"] for r in assess_sufficiency(state)})
        reason = (
            f"근거 부족 기업({', '.join(short)})의 재작업 상한 도달 → 남은 문항 N/A 확정" if short
            else f"근거 충분: 모든 후보의 평가 가능 문항 {MIN_SCORED_CRITERIA}개 이상"
        )
        return Decision(REPORT, reason, updates={**write, "final_route": "INVEST_FOUND" if invest else "NO_INVEST"})
    if not state["report_evaluated"]:
        return Decision(QUALITY, f"보고서 {attempts}차 작성 완료", updates={"report_evaluated": True})
    if quality.get("passed"):
        return Decision(FINISH, "보고서 품질 평가 통과", updates={"status": "SUCCESS"})
    if can_rewrite and steps_left >= 2:
        failed = [c["item"] for c in quality.get("checks", []) if not c["passed"]]
        return Decision(REPORT, f"품질 평가 미달({', '.join(failed)}) → 재작성", updates=write)
    return Decision(FINISH, "품질 평가 미달이지만 재작성 상한 도달", updates={"status": "SUCCESS"})


# ---------------------------------------------------------------------------
# Node / Gate
# ---------------------------------------------------------------------------
@traceable(name="supervisor_decision")
def record_decision(trace_id: str, step: int, next: str, company: str | None, reason: str) -> dict:
    """결정 로그는 State가 아니라 외부(logger, LangSmith)에 남긴다. trace_id로 State와 잇는다."""
    logger.info("[%s] step=%d next=%s company=%s reason=%s", trace_id, step, next, company or "-", reason)
    return {"trace_id": trace_id, "step": step, "next": next, "company": company, "reason": reason}


def supervisor_node(state: InvestmentState) -> dict:
    decision = decide(state)
    updates = dict(decision.updates)
    if updates.pop("_skip", False):
        # 실패한 분석을 빈 결과로 대체했으므로, 대체한 State로 다시 판단한다
        key = status_key(decision.next, decision.company)
        record_decision(state["trace_id"], state["step_count"], "SKIP", decision.company, decision.reason)
        merged = {**state, **{k: {**state[k], **v} for k, v in updates.items()}}
        merged["node_status"] = {**state["node_status"], key: "DONE"}
        following = supervisor_node(merged)
        return {**updates, "node_status": {key: "DONE"}, **following}

    step = state["step_count"] + (0 if decision.next == FINISH else 1)
    record_decision(state["trace_id"], step, decision.next, decision.company, decision.reason)
    return {
        **updates,
        "next": decision.next,
        "current_company": decision.company,
        "rework_request": decision.rework,
        "step_count": step,
    }


def route(state: InvestmentState) -> str:
    """Gate: Supervisor가 정한 다음 단계를 그대로 따른다 (판단 없음)."""
    return state["next"]
