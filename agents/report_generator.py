"""Report Generator (설계 산출물 2.1, 5장)

역할: 새 판단 없이 최대 5쪽 보고서와 실제 사용 출처 생성
입력: 확정된 State
출력: references, final_report

보고서 구성 (설계 5.1) - 최대 5쪽, 맨 앞 SUMMARY(핵심 결론, 1/2페이지 이하), 맨 끝 REFERENCE
  1. SUMMARY + Company Snapshot   ← company_profiles, investment_results, competition_result
  2. Technology & Product         ← technology_results
  3. Market & Traction            ← market_traction_results
  4. Competition / Risk / Evaluation ← competition_result, investment_results
     (Target Market Context, 12문항 Scorecard, 차별성·Risk·Decision Reason)
  5. REFERENCE                    ← references

SUMMARY 전달 규칙 (설계 5.2) - 항목별 유일한 Source State
  Investment Thesis / Decision Reason ← investment_results[기업].key_strengths + decision_reason
  Key Risks                           ← investment_results[기업].key_risks
  Missing Information / Due Diligence ← investment_results[기업].missing_information
  Company Snapshot                    ← company_profiles[기업] (사전 입력·검증 값)

Template + 선택적 Narrative (설계 2.1): Narrative를 쓸 때도 확정된 State의 사실·점수·Risk·Decision만 전달한다.

출력 제약 (설계 5.4)
  - 실제 State만 사용: 새 사실·Score·Risk·Decision·추가 실사 항목을 생성하지 않는다.
  - 점수 재계산 금지: Python Rule의 FinalScore·Coverage·Decision을 그대로 표시한다.
  - 출처 정확성: 동일 source_id 중복 제거, 사용 원본 page 병합, 누락 서지정보를 추정하지 않는다.
  - 추천 없음도 보고: 모두 HOLD 또는 근거 부족이어도 '추천 기업 없음'과 이유를 구분해 정상 생성한다.

TODO(담당자): 2·3쪽 본문, SUMMARY의 Investment Thesis·Key Risks·Missing Information 배치를 완성하고 PDF로 출력
"""

from core.state import InvestmentState, ReferenceItem
from evaluation.criteria import CRITERIA


def collect_references(state: InvestmentState) -> list[ReferenceItem]:
    """Judge가 사용한 Evidence에서 출처를 모은다. 같은 source_id는 하나로 합치고 페이지를 병합한다."""
    refs: dict[str, ReferenceItem] = {}
    for result in state["investment_results"].values():
        for criterion in result["criteria"]:
            for ev in criterion["evidence"]:
                ref = refs.setdefault(
                    ev["source_id"],
                    {
                        "source_id": ev["source_id"],
                        "title": ev["title"],
                        "publisher": ev["publisher"],
                        "published_date": ev["published_date"],
                        "reference_metadata": ev["reference_metadata"],
                        "pages": [],
                    },
                )
                if ev["page"] not in ref["pages"]:
                    ref["pages"].append(ev["page"])
    for ref in refs.values():
        ref["pages"].sort()
    return sorted(refs.values(), key=lambda r: r["source_id"])


def format_reference(ref: ReferenceItem) -> str:
    """출처 유형별 표기 형식 (설계 5.3). 누락된 값은 추정하지 않고 비워 둔다."""
    meta = ref["reference_metadata"]
    pages = f" (사용 페이지: p. {', '.join(map(str, ref['pages']))})"
    url = meta.get("url") or ""
    if meta.get("reference_type") == "web":
        # 웹페이지: 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL
        site = meta.get("site_name") or ref["publisher"]
        return f"{ref['publisher']}({ref['published_date']}). *{ref['title']}*. {site}, {url}{pages}"
    # 기관 보고서: 발행기관(YYYY). 보고서명. URL
    return f"{ref['publisher']}({ref['published_date'][:4]}). *{ref['title']}*. {url}{pages}"


def _coverage(result: dict) -> str:
    n_scored = sum(c["status"] == "SCORED" for c in result["criteria"])
    return f"{n_scored}/12 ({result['evidence_coverage']:.0%})"


def _render(state: InvestmentState, references: list[ReferenceItem]) -> str:
    results = state["investment_results"]
    companies = state["candidate_companies"]
    profiles = state["company_profiles"]
    fmt = lambda v: "N/A" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)

    # 1. SUMMARY + Company Snapshot
    lines = ["# SUMMARY", ""]
    if state["final_route"] == "NO_INVEST":
        lines += ["**Investment Recommendation: None**", ""]
    lines += ["| COMPANY | DECISION | FINAL SCORE | COVERAGE |", "|---|---|---|---|"]
    for c in companies:
        r = results[c]
        lines.append(f"| {c} | {r['decision']} | {fmt(r['final_score'])} / 5 | {_coverage(r)} |")
    lines += [""] + [f"- **{c}**: {results[c]['decision_reason']}" for c in companies]

    lines += ["", "## Company Snapshot", "", "| COMPANY | TARGET MARKET | FUNDING STAGE |", "|---|---|---|"]
    for c in companies:
        p = profiles[c]
        lines.append(f"| {c} | {p['target_market'] or '미입력'} | {p['funding_stage'] or '미확인'} |")

    # 4. 12문항 Scorecard
    lines += ["", "# Competition / Risk / Investment Evaluation", "",
              "| Criterion | " + " | ".join(companies) + " |", "|---|" + "---|" * len(companies)]
    for crit in CRITERIA:
        scores = [next(x["score"] for x in results[c]["criteria"] if x["criterion_id"] == crit.id) for c in companies]
        lines.append(f"| {crit.q}. {crit.name} | " + " | ".join(fmt(s) for s in scores) + " |")
    lines.append("| **Final Score** | " + " | ".join(fmt(results[c]["final_score"]) for c in companies) + " |")
    lines.append("| **Evidence Coverage** | " + " | ".join(_coverage(results[c]) for c in companies) + " |")
    lines.append("| **Decision** | " + " | ".join(results[c]["decision"] for c in companies) + " |")

    # 5. REFERENCE
    lines += ["", "# REFERENCE", ""]
    lines += [f"- {format_reference(r)}" for r in references] or ["- 평가에 사용된 Evidence 없음"]
    return "\n".join(lines)


def report_generator_node(state: InvestmentState) -> dict:
    references = collect_references(state)
    return {"references": references, "final_report": _render(state, references)}
