"""AI Startup Investment Evaluation Agent 실행 스크립트

    uv run python app.py
    uv run python app.py --companies "Figure AI" "1X Technologies"
"""

import argparse
import json
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


def save_run(graph, config: dict, final_state: dict, stamp: str) -> tuple[Path, Path]:
    """실행 결과를 다시 볼 수 있게 남긴다 (outputs/는 Git에 올리지 않는다).

    - state_*.json   : 최종 State 전체. 품질 평가 독립 실행 도구(--state)의 입력으로 쓸 수 있다.
    - quality_*.json : 품질 평가 회차별 결과와 그때 평가한 보고서 본문. State에는 마지막 회차만 남으므로
                       Checkpoint 이력에서 회차별로 꺼낸다.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    state_path = OUTPUT_DIR / f"state_{stamp}.json"
    state_path.write_text(json.dumps(final_state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rounds: dict[int, dict] = {}
    for snapshot in reversed(list(graph.get_state_history(config))):  # 오래된 것부터
        values = snapshot.values
        attempt, quality = values.get("report_attempts"), values.get("quality_result")
        # 품질 평가 노드가 막 끝난 시점(다음 차례가 Supervisor)의 결과만 모은다
        if quality and values.get("next") == "quality_evaluator" and snapshot.next == ("supervisor",):
            rounds[attempt] = ({
                "report_attempt": attempt,
                "passed": quality.get("passed"),
                "issue_count": sum(len(c.get("issues", [])) for c in quality.get("checks", [])),
                "quality_result": quality,
                "evaluated_report": values.get("final_report"),
            })
    quality_path = OUTPUT_DIR / f"quality_{stamp}.json"
    quality_path.write_text(
        json.dumps({"trace_id": final_state.get("trace_id"), "rounds": list(rounds.values())},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return state_path, quality_path


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
    config = {
        "recursion_limit": recursion_limit(len(args.companies)),
        "configurable": {"thread_id": trace_id},
        "run_name": "invest-agent-supervisor",
        "tags": ["supervisor"],
        "metadata": {"trace_id": trace_id},
    }
    final_state = graph.invoke(
        {"candidate_companies": args.companies, "company_profiles": profiles, "trace_id": trace_id},
        config=config,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"report_{stamp}.md"
    out.write_text(final_state.get("final_report") or "", encoding="utf-8")
    state_path, quality_path = save_run(graph, config, final_state, stamp)

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
    print(f"실행 기록: {state_path.relative_to(OUTPUT_DIR.parent)} | 품질 평가 회차별 지적: {quality_path.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
