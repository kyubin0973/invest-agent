"""AI Startup Investment Evaluation Agent 실행 스크립트

    uv run python app.py
    uv run python app.py --companies "Figure AI" "1X Technologies"
"""

import argparse
import logging
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from core import tracing
from core.graph import build_graph, recursion_limit
from core.judge_logging import configure_judge_logging
from documents.profiles import load_company_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    profiles = load_company_profiles()

    parser = argparse.ArgumentParser(description="휴머노이드 스타트업 투자 평가")
    parser.add_argument("--companies", nargs="+", default=list(profiles), help=f"평가할 후보 (기본: 전체 {list(profiles)})")
    args = parser.parse_args()

    tracing.setup()  # 그래프 실행 전에 설정해야 전체 실행이 U_2_4 프로젝트에 한 번에 기록된다
    configure_judge_logging()
    logging.getLogger("agents.supervisor").setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s")

    # trace_id 하나로 State, 로그, LangSmith Trace, Checkpoint(thread_id)를 잇는다
    trace_id = uuid.uuid4().hex
    graph = build_graph(checkpointer=InMemorySaver())
    final_state = graph.invoke(
        {"candidate_companies": args.companies, "company_profiles": profiles, "trace_id": trace_id},
        config={
            "recursion_limit": recursion_limit(len(args.companies)),
            "configurable": {"thread_id": trace_id},
            "run_name": "invest-agent-supervisor",
            "tags": ["supervisor"],
            "metadata": {"trace_id": trace_id},
        },
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(final_state.get("final_report") or "", encoding="utf-8")

    for company in args.companies:
        r = final_state["investment_results"].get(company)
        if r:
            print(f"{company}: {r['decision']} (score {r['final_score']}, coverage {r['evidence_coverage']:.0%}) - {r['decision_reason']}")
    quality = final_state.get("quality_result") or {}
    print(f"Status: {final_state['status']} | trace_id: {trace_id}")
    print(f"Steps: {final_state['step_count']}/{final_state['max_steps']} | 재작업: {final_state['rework_counts'] or '없음'}")
    print(f"보고서 작성 {final_state['report_attempts']}회 | 품질 평가: {'통과' if quality.get('passed') else '미달'}")
    if final_state["errors"]:
        print(f"오류 {len(final_state['errors'])}건, 마지막: {final_state['last_error']}")
    print(f"보고서: {final_state.get('report_path') or out.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
