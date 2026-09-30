"""테스트용 fixture 저장 (개발 중에만 사용, 제출 전 fixtures/와 함께 삭제)

앞 단계 Agent의 결과를 JSON으로 저장해, 다음 단계 Agent 담당자가 앞 단계를 다시 실행하지 않고 개발할 수 있게 한다.
실제 실행(app.py)은 fixture를 읽지 않고 State로 결과를 전달한다.

    uv run python -m scripts.save_fixture market_traction                  # 전체 기업
    uv run python -m scripts.save_fixture market_traction --companies Apptronik
"""

import argparse
import json
import time

from documents.company import company_key
from documents.config import ROOT
from documents.profiles import load_company_profiles

FIXTURE_DIR = ROOT / "fixtures"


def _market_traction(company: str, profile: dict) -> dict:
    from agents.market_traction import analyze_market_traction

    return analyze_market_traction(company, profile)


# 다른 Agent 담당자는 여기에 자기 Agent를 추가한다
AGENTS = {
    "market_traction": _market_traction,
}


def main() -> None:
    profiles = load_company_profiles()
    parser = argparse.ArgumentParser(description="Agent 결과를 fixture(JSON)로 저장")
    parser.add_argument("agent", choices=list(AGENTS))
    parser.add_argument("--companies", nargs="+", default=list(profiles))
    args = parser.parse_args()

    out_dir = FIXTURE_DIR / args.agent
    out_dir.mkdir(parents=True, exist_ok=True)
    for company in args.companies:
        t0 = time.perf_counter()
        result = AGENTS[args.agent](company, profiles[company])
        path = out_dir / f"{company_key(company)}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{company}: {len(result['findings'])} findings, {len(result['evidence'])} evidence "
              f"({time.perf_counter() - t0:.0f}초) → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
