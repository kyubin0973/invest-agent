"""AI Startup Investment Evaluation Agent 실행 스크립트

    uv run python app.py
    uv run python app.py --companies "Figure AI" "1X Technologies"
"""

import argparse
import time
from pathlib import Path

from core import tracing
from core.graph import build_graph, recursion_limit
from documents.profiles import load_company_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    profiles = load_company_profiles()

    parser = argparse.ArgumentParser(description="휴머노이드 스타트업 투자 평가")
    parser.add_argument("--companies", nargs="+", default=list(profiles), help=f"평가할 후보 (기본: 전체 {list(profiles)})")
    args = parser.parse_args()

    tracing.setup()  # 그래프 실행 전에 설정해야 전체 실행이 U_2_4 프로젝트에 한 번에 기록된다

    graph = build_graph()
    final_state = graph.invoke(
        {"candidate_companies": args.companies, "company_profiles": profiles},
        config={"recursion_limit": recursion_limit(len(args.companies))},
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(final_state["final_report"], encoding="utf-8")

    for company in args.companies:
        r = final_state["investment_results"][company]
        print(f"{company}: {r['decision']} (score {r['final_score']}, coverage {r['evidence_coverage']:.0%}) - {r['decision_reason']}")
    print(f"Final route: {final_state['final_route']}")
    print(f"보고서: {out.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
