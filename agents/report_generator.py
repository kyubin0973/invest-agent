"""Report Generator (설계 산출물 2.2절, 8장)

역할: 확정된 State만으로 투자보고서 구성 (Presentation Layer)
입력: 확정된 전체 State
출력: references, final_report

새로운 사실·점수·Risk·투자판단을 만들거나 바꾸지 않는다 (설계 2.3절).
REFERENCE에는 실제 평가에 사용된 Evidence의 출처만 넣는다 (설계 8.5절).

TODO(담당자): _render의 초안 형식을 설계 8장 목차(5페이지)대로 완성하고 PDF로 출력
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


def _render(state: InvestmentState, references: list[ReferenceItem]) -> str:
    results = state["investment_results"]
    companies = state["candidate_companies"]
    fmt = lambda v: "N/A" if v is None else f"{v:.1f}" if isinstance(v, float) else str(v)

    lines = ["# SUMMARY", ""]
    if state["final_route"] == "NO_INVEST":
        lines += ["**Investment Recommendation: None**", ""]
    lines += ["| COMPANY | DECISION | SCORE | COVERAGE |", "|---|---|---|---|"]
    for c in companies:
        r = results[c]
        lines.append(f"| {c} | {r['decision']} | {fmt(r['final_score'])} / 5 | {r['evidence_coverage']:.0%} |")

    lines += ["", "# Bessemer Scorecard", "", "| Criterion | " + " | ".join(companies) + " |",
              "|---|" + "---|" * len(companies)]
    for crit in CRITERIA:
        scores = [next(x["score"] for x in results[c]["criteria"] if x["criterion_id"] == crit.id) for c in companies]
        lines.append(f"| {crit.id} {crit.name} | " + " | ".join(fmt(s) for s in scores) + " |")

    lines += ["", "# REFERENCE", ""]
    lines += [f"- {r['publisher']}({r['published_date'][:4]}). *{r['title']}*. "
              f"{r['reference_metadata'].get('url') or ''} (p. {', '.join(map(str, r['pages']))})"
              for r in references] or ["- [STUB] 사용된 Evidence 없음"]
    return "\n".join(lines)


def report_generator_node(state: InvestmentState) -> dict:
    references = collect_references(state)
    return {"references": references, "final_report": _render(state, references)}
