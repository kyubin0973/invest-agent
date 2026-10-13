"""LangGraph Workflow (Supervisor 패턴)

- 모든 하위 Agent는 Supervisor하고만 연결된다 (하위 Agent 간 직접 Edge 없음).
- Supervisor가 State를 보고 다음 Agent를 정하고, add_conditional_edges의 Gate가 그대로 분기한다.
- 근거가 부족하면 분석 Agent로, 보고서 품질이 미달이면 Report Generator로 되돌아간다.
- 하위 Agent의 예외는 노드 밖으로 올리지 않고 State(node_status, errors)에 기록해 Supervisor가
  재시도·제외·중단을 정한다.
"""

import logging
import uuid

from langgraph.graph import END, START, StateGraph

from agents.competition import competition_node
from agents.investment_judge import investment_judge_node
from agents.market_traction import market_traction_node
from agents.quality_evaluator import quality_evaluator_node
from agents.report_generator import report_generator_node
from agents.supervisor import (
    COMPETITION, FINISH, JUDGE, QUALITY, REPORT, WORKERS,
    default_max_steps, now, route, status_key, supervisor_node,
)
from agents.technology import technology_node
from core.state import InvestmentState
from evaluation.criteria import MARKET, TECH

logger = logging.getLogger(__name__)

WORKER_NODES = {
    TECH: technology_node,
    MARKET: market_traction_node,
    COMPETITION: competition_node,
    JUDGE: investment_judge_node,
    REPORT: report_generator_node,
    QUALITY: quality_evaluator_node,
}


def initialize_state(state: InvestmentState) -> dict:
    candidates = state.get("candidate_companies") or []
    if not candidates:
        raise ValueError("candidate_companies가 비어 있습니다.")
    missing = [c for c in candidates if c not in state.get("company_profiles", {})]
    if missing:
        raise ValueError(f"company_profiles에 없는 후보: {missing}")
    return {
        # 작업 페이로드
        "technology_results": {},
        "market_traction_results": {},
        "competition_result": None,
        "investment_results": {},
        "references": [],
        "final_report": None,
        "report_path": None,
        "quality_result": None,
        # 제어 메타데이터
        "trace_id": state.get("trace_id") or uuid.uuid4().hex,
        "status": "RUNNING",
        "next": None,
        "current_company": None,
        "rework_request": None,
        "pending_judge": list(candidates),
        "competition_stale": False,
        "rework_counts": {},
        "report_attempts": 0,
        "report_evaluated": False,
        "step_count": 0,
        "max_steps": state.get("max_steps") or default_max_steps(len(candidates)),
        "node_status": {},
        "errors": [],
        "last_error": None,
        # 호환용
        "technology_done": False,
        "market_traction_done": False,
        "completed_companies": [],
        "final_route": None,
    }


def worker(name: str, node):
    """하위 Agent 실행 결과에 상태를 붙인다. 예외는 State에 기록하고 Supervisor로 돌아간다."""

    def run(state: InvestmentState) -> dict:
        company = state.get("current_company")
        key = status_key(name, company)
        try:
            update = node(state)
        except Exception as exc:  # 실패 처리(재시도·제외·중단)는 Supervisor가 정한다
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] %s 실패", state["trace_id"], key)
            error = {"node": name, "company": company, "message": message, "ts": now()}
            return {"node_status": {key: "FAILED"}, "errors": [error], "last_error": message}
        return {**update, "node_status": {key: "DONE"}}

    run.__name__ = name
    return run


def build_graph(checkpointer=None):
    g = StateGraph(InvestmentState)

    g.add_node("initialize_state", initialize_state)
    g.add_node("supervisor", supervisor_node)
    g.add_edge(START, "initialize_state")
    g.add_edge("initialize_state", "supervisor")

    for name in WORKERS:
        g.add_node(name, worker(name, WORKER_NODES[name]))
        g.add_edge(name, "supervisor")  # 작업 후 항상 Supervisor로 복귀

    g.add_conditional_edges("supervisor", route, {**{name: name for name in WORKERS}, FINISH: END})

    return g.compile(checkpointer=checkpointer)


def recursion_limit(num_candidates: int) -> int:
    """LangGraph 실행 단계 상한: Supervisor ↔ 하위 Agent 왕복 2단계 × max_steps + 여유."""
    return 2 * default_max_steps(num_candidates) + 10
