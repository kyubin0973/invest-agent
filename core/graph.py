"""LangGraph Workflow (설계 산출물 7장)

- 후보 기업 간 분석은 순차 Loop
- 한 기업 안에서는 Technology ∥ Market & Traction 병렬 실행(Fan-out) → Analysis Join(Fan-in)
- 모든 기업 분석 후 Competition Agent 1회 → Investment Judge가 후보를 순차 평가
- 첫 INVEST가 나와도 멈추지 않고 모든 후보를 끝까지 평가
- INVEST 기업 존재 여부로 Final Branch → Report Generator → END
"""

from langgraph.graph import END, START, StateGraph

from agents.competition import competition_node
from agents.investment_judge import investment_judge_node
from agents.market_traction import market_traction_node
from agents.report_generator import report_generator_node
from agents.technology import technology_node
from core.state import InvestmentState


# ---------------------------------------------------------------------------
# Control nodes
# ---------------------------------------------------------------------------
def initialize_state(state: InvestmentState) -> dict:
    candidates = state.get("candidate_companies") or []
    if not candidates:
        raise ValueError("candidate_companies가 비어 있습니다.")
    missing = [c for c in candidates if c not in state.get("company_profiles", {})]
    if missing:
        raise ValueError(f"company_profiles에 없는 후보: {missing}")
    return {
        "current_company": None,
        "current_company_index": 0,
        "technology_results": {},
        "market_traction_results": {},
        "technology_done": False,
        "market_traction_done": False,
        "analysis_join_ready": False,
        "analysis_route": None,
        "competition_result": None,
        "investment_results": {},
        "completed_companies": [],
        "evaluation_route": None,
        "final_route": None,
        "references": [],
        "final_report": None,
    }


def candidate_selection(state: InvestmentState) -> dict:
    return {
        "current_company": state["candidate_companies"][state["current_company_index"]],
        "technology_done": False,
        "market_traction_done": False,
        "analysis_join_ready": False,
    }


def analysis_join(state: InvestmentState) -> dict:
    return {"analysis_join_ready": state["technology_done"] and state["market_traction_done"]}


def analysis_router(state: InvestmentState) -> dict:
    if not state["analysis_join_ready"]:
        raise RuntimeError(f"{state['current_company']} 병렬 분석이 완료되지 않았습니다.")
    has_next = state["current_company_index"] + 1 < len(state["candidate_companies"])
    return {"analysis_route": "NEXT_CANDIDATE" if has_next else "COMPETITION"}


def next_analysis_candidate(state: InvestmentState) -> dict:
    return {"current_company_index": state["current_company_index"] + 1}


def evaluation_init(state: InvestmentState) -> dict:
    return {
        "current_company_index": 0,
        "current_company": state["candidate_companies"][0],
        "completed_companies": [],
    }


def evaluation_router(state: InvestmentState) -> dict:
    has_next = state["current_company_index"] + 1 < len(state["candidate_companies"])
    return {"evaluation_route": "NEXT_CANDIDATE" if has_next else "FINAL_DECISION"}


def next_evaluation_candidate(state: InvestmentState) -> dict:
    index = state["current_company_index"] + 1
    return {"current_company_index": index, "current_company": state["candidate_companies"][index]}


def final_decision_router(state: InvestmentState) -> dict:
    invest = any(r["decision"] == "INVEST" for r in state["investment_results"].values())
    return {"final_route": "INVEST_FOUND" if invest else "NO_INVEST"}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(InvestmentState)

    g.add_node("initialize_state", initialize_state)
    g.add_node("candidate_selection", candidate_selection)
    g.add_node("technology", technology_node)
    g.add_node("market_traction", market_traction_node)
    g.add_node("analysis_join", analysis_join)
    g.add_node("analysis_router", analysis_router)
    g.add_node("next_analysis_candidate", next_analysis_candidate)
    g.add_node("competition", competition_node)
    g.add_node("evaluation_init", evaluation_init)
    g.add_node("investment_judge", investment_judge_node)
    g.add_node("evaluation_router", evaluation_router)
    g.add_node("next_evaluation_candidate", next_evaluation_candidate)
    g.add_node("final_decision_router", final_decision_router)
    g.add_node("report_generator", report_generator_node)

    # Candidate Analysis Loop: 기업별 Technology ∥ Market 병렬 → Join
    g.add_edge(START, "initialize_state")
    g.add_edge("initialize_state", "candidate_selection")
    g.add_edge("candidate_selection", "technology")
    g.add_edge("candidate_selection", "market_traction")
    g.add_edge(["technology", "market_traction"], "analysis_join")
    g.add_edge("analysis_join", "analysis_router")
    g.add_conditional_edges(
        "analysis_router",
        lambda s: s["analysis_route"],
        {"NEXT_CANDIDATE": "next_analysis_candidate", "COMPETITION": "competition"},
    )
    g.add_edge("next_analysis_candidate", "candidate_selection")

    # Investment Evaluation Loop
    g.add_edge("competition", "evaluation_init")
    g.add_edge("evaluation_init", "investment_judge")
    g.add_edge("investment_judge", "evaluation_router")
    g.add_conditional_edges(
        "evaluation_router",
        lambda s: s["evaluation_route"],
        {"NEXT_CANDIDATE": "next_evaluation_candidate", "FINAL_DECISION": "final_decision_router"},
    )
    g.add_edge("next_evaluation_candidate", "investment_judge")

    # Final Branch: INVEST 존재 여부와 관계없이 보고서 생성 (NO_INVEST면 "Investment Recommendation: None")
    g.add_conditional_edges(
        "final_decision_router",
        lambda s: s["final_route"],
        {"INVEST_FOUND": "report_generator", "NO_INVEST": "report_generator"},
    )
    g.add_edge("report_generator", END)

    return g.compile()


def recursion_limit(num_candidates: int) -> int:
    """후보 수에 비례하는 실행 단계 상한 (분석 Loop 6단계 + 평가 Loop 3단계 / 기업 + 여유)."""
    return 10 * num_candidates + 20
