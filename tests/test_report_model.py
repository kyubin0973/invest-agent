"""Report Generator: 문장 단위 모델, 인용, 품질 평가 지적 반영(재작성)"""

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from agents.report_generator import (
    MAX_REPORT_PAGES, apply_quality_feedback, build_report_model, collect_references,
    generate_pdf_report, render_markdown, report_generator_node,
)
from evaluation.criteria import CRITERIA

COMPANIES = ["Alpha", "Beta"]


def ev(source_id: str, page: int = 1) -> dict:
    return {"source_id": source_id, "chunk_id": f"{source_id}-c{page}", "page": page, "title": f"제목 {source_id}",
            "publisher": "기관", "published_date": "2026-01-01", "source_type": "industry_report",
            "reference_metadata": {"reference_type": "report", "url": "https://example.com"},
            "evidence_level": "E3", "fact": "fact"}


def analysis(tag: str, cited: bool = True) -> dict:
    refs = [{"chunk_id": f"{tag}-c1", "source_id": tag}] if cited else []
    return {
        "summary": f"{tag} 요약 문장입니다 요약",
        "findings": [
            {"dimension": "제품", "statement": f"{tag} 근거가 있는 제품 서술입니다", "evidence_refs": refs},
            {"dimension": "AI", "statement": f"{tag} 가장 우수하다고 단정하는 서술입니다", "evidence_refs": refs},
        ],
        "evidence": [ev(tag)],
        "risks": [f"{tag} 리스크"],
        "missing_information": [],
    }


def state(**override) -> dict:
    result = {
        "decision": "HOLD", "final_score": 3.0, "evidence_coverage": 1.0, "decision_reason": "점수 3.00 < 기준",
        "key_strengths": ["강점"], "key_risks": ["리스크"],
        "criteria": [{"criterion_id": c.id, "score": 3, "status": "SCORED", "evidence": [ev("J1")]} for c in CRITERIA],
    }
    base = {
        "candidate_companies": COMPANIES,
        "company_profiles": {c: {"target_market": "산업용", "funding_stage": "Series A", "eligibility": {"is_private": True}} for c in COMPANIES},
        "technology_results": {c: analysis(f"T-{c}") for c in COMPANIES},
        "market_traction_results": {c: analysis(f"M-{c}") for c in COMPANIES},
        "competition_result": {"comparisons": [{
            "dimension": "차별성", "company_findings": {c: f"{c} 차별점 서술입니다 차별" for c in COMPANIES},
            "evidence_refs": [{"chunk_id": "x", "source_id": "T-Alpha"}],
        }]},
        "investment_results": {c: result for c in COMPANIES},
        "trace_id": "t1",
        "report_attempts": 1,
        "quality_result": None,
    }
    base.update(override)
    return base


def quality(statement: str, owner: str = "report_generator", item: str = "neutrality") -> dict:
    return {"passed": False, "feedback": [], "checks": [
        {"item": item, "passed": False, "issues": [{"owner": owner, "company": "Alpha", "statement": statement, "reason": "r"}]},
    ]}


class ReportModelTest(unittest.TestCase):
    def test_statements_carry_sources_and_ids_are_unique(self):
        model = build_report_model(state())
        ids = [s.id for s in model.statements]
        self.assertEqual(len(ids), len(set(ids)))
        finding = next(s for s in model.statements if s.id == "technology:Alpha:f0")
        self.assertEqual(finding.source_ids, ("T-Alpha",))
        summary = next(s for s in model.statements if s.id == "technology:Alpha:summary")
        self.assertEqual(summary.source_ids, ("T-Alpha",))  # 요약은 finding 출처를 합친다

    def test_markdown_is_real_report_with_numbered_citations(self):
        s = state()
        refs = collect_references(s)
        md = render_markdown(build_report_model(s), s, refs)
        self.assertNotEqual(md.strip(), "# Markdown Report Generated")
        index = {r["source_id"]: i for i, r in enumerate(refs, 1)}
        line = next(l for l in md.splitlines() if "T-Alpha 근거가 있는 제품 서술입니다" in l)
        self.assertTrue(line.endswith("[T-Alpha-c1 | T-Alpha]"))  # 품질 평가가 읽는 인용 형식
        self.assertIn(f"[{index['T-Alpha']}] ", md.split("## 부록. 참고문헌 (REFERENCE)")[1])  # REFERENCE 번호

    def test_issue_statement_from_markdown_line_matches(self):
        """품질 평가가 final_report에서 문장을 복사해 지적하면(인용 표기 포함) 문장을 찾는다."""
        s = state()
        refs = collect_references(s)
        md = render_markdown(build_report_model(s), s, refs)
        line = next(l for l in md.splitlines() if "가장 우수하다고" in l and "T-Alpha" in l)
        copied = line.split(":** ", 1)[-1]  # "- **라벨:** 본문 [n]" → "본문 [n]"
        model, unmatched = apply_quality_feedback(build_report_model(s), quality(copied))
        self.assertEqual(unmatched, [])
        self.assertEqual(model.omitted, {"technology:Alpha:f1": "neutrality"})


try:
    # judge_used 같은 상세 결과는 evaluate_report_details가 반환한다 (evaluate_report는 공통 형식만)
    from agents.quality_evaluator import evaluate_report_details as evaluate_report
    from tests.qa_fixtures import passing_judge
except ImportError:  # 품질 평가 코드가 아직 없는 브랜치
    evaluate_report = None


@unittest.skipIf(evaluate_report is None, "품질 평가(feat/qa) 코드가 없는 브랜치")
class QualityContractTest(unittest.TestCase):
    def test_report_passes_deterministic_quality_rules(self):
        """실제 품질 평가의 결정적 검사(인용 형식·REFERENCE·수치 인용)가 보고서에서 막히지 않는다 (Judge는 통과 응답 주입)."""
        s = state()
        s["competition_result"]["comparisons"][0]["evidence_refs"] = [{"chunk_id": "T-Alpha-c1", "source_id": "T-Alpha"}]
        s["final_report"] = report_generator_node(s)["final_report"]
        s["references"] = collect_references(s)
        originals = {e["chunk_id"]: {"source_id": e["source_id"], "chunk_id": e["chunk_id"], "page": e["page"], "text": "원문"}
                     for key in ("technology_results", "market_traction_results")
                     for a in s[key].values() for e in a["evidence"]}
        result = evaluate_report(s, original_documents=originals, judge=passing_judge)
        self.assertTrue(result["judge_used"], [i["reason"] for c in result["checks"] for i in c["issues"]])
        groundedness = next(c for c in result["checks"] if c["item"] == "groundedness")
        self.assertTrue(groundedness["passed"], groundedness["issues"])


class RewriteTest(unittest.TestCase):
    def test_report_generator_issue_omits_statement_everywhere(self):
        s = state(report_attempts=2, quality_result=quality("T-Alpha 가장 우수하다고 단정하는 서술입니다"))
        out = report_generator_node(s)
        self.assertNotIn("T-Alpha 가장 우수하다고", out["final_report"])
        self.assertIn("T-Alpha 근거가 있는 제품 서술입니다", out["final_report"])
        self.assertIn("서술 1건을 제외", out["final_report"])

    def test_analysis_owned_issue_is_not_applied_here(self):
        s = state(report_attempts=2, quality_result=quality("T-Alpha 가장 우수하다고 단정하는 서술입니다", owner="technology"))
        out = report_generator_node(s)
        self.assertIn("T-Alpha 가장 우수하다고", out["final_report"])

    def test_first_write_ignores_stale_quality_result(self):
        s = state(report_attempts=1, quality_result=quality("T-Alpha 가장 우수하다고 단정하는 서술입니다"))
        self.assertIn("T-Alpha 가장 우수하다고", report_generator_node(s)["final_report"])

    def test_computed_decision_reason_is_never_omitted(self):
        """판정 사유(Python 규칙 계산 결과)를 지적해도 보고서에 남고, 점수표도 그대로다."""
        s = state(report_attempts=2, quality_result=quality("점수 3.00 < 기준"))
        model, notes = apply_quality_feedback(build_report_model(s), s["quality_result"])
        self.assertEqual(model.omitted, {})
        self.assertEqual(len(notes), 1)
        out = report_generator_node(s)
        self.assertIn("점수 3.00 < 기준", out["final_report"])
        self.assertIn("Final Score", out["final_report"])

    def test_uncitable_statement_kept_for_citation_only_issue(self):
        """강점처럼 인용을 붙일 수 없는 서술은 '인용 누락' 지적으로는 지우지 않고, 다른 사유면 지운다."""
        def issue(reason):
            q = quality("강점")
            q["checks"][0]["issues"][0]["reason"] = reason
            return q
        model, notes = apply_quality_feedback(build_report_model(state()), issue("수치 또는 근거 기반 분석 문장에 인용이 없습니다."))
        self.assertEqual(model.omitted, {})
        self.assertEqual(len(notes), 1)
        model, _ = apply_quality_feedback(build_report_model(state()), issue("근거 없는 단정 표현"))
        self.assertIn("summary:Alpha:strengths", model.omitted)

    def test_unmatched_issue_is_reported_not_fatal(self):
        model, unmatched = apply_quality_feedback(build_report_model(state()), quality("보고서에 없는 완전히 다른 문장입니다"))
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(model.omitted, {})

    def test_node_returns_only_known_state_keys_and_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                out = report_generator_node(state())
                self.assertEqual(set(out), {"references", "final_report", "report_path"})
                self.assertTrue(1 <= len(PdfReader(out["report_path"]).pages) <= MAX_REPORT_PAGES)
            finally:
                os.chdir(cwd)


class RobustnessTest(unittest.TestCase):
    def _pages(self, s: dict) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            return len(PdfReader(generate_pdf_report(s, collect_references(s), str(Path(tmp) / "r.pdf"))).pages)

    def test_none_profile_values_and_markup_characters(self):
        s = state()
        s["company_profiles"]["Alpha"] = {"target_market": None, "funding_stage": None, "eligibility": {"is_private": None}}
        s["technology_results"]["Alpha"]["findings"][0]["statement"] = "R&D <b>비용 5 < 10 & 성장"
        self.assertTrue(1 <= self._pages(s) <= MAX_REPORT_PAGES)

    def test_other_company_counts_stay_five_pages(self):
        for companies in (["Alpha"], ["Alpha", "Beta", "Gamma", "Delta"]):
            s = state(candidate_companies=companies)
            for key in ("technology_results", "market_traction_results"):
                s[key] = {c: analysis(f"{key[0]}-{c}") for c in companies}
            s["company_profiles"] = {c: {} for c in companies}
            s["investment_results"] = {c: s["investment_results"]["Alpha"] for c in companies}
            self.assertTrue(1 <= self._pages(s) <= MAX_REPORT_PAGES, companies)

    def test_competition_result_none(self):
        self.assertTrue(1 <= self._pages(state(competition_result=None)) <= MAX_REPORT_PAGES)

    def test_page_limit_is_ten(self):
        from agents import report_generator as rg
        self.assertEqual(rg.MAX_REPORT_PAGES, 10)
        self.assertLessEqual(self._pages(state()), rg.MAX_REPORT_PAGES)


if __name__ == "__main__":
    unittest.main()
