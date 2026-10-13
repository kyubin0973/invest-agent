import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

from agents import report_generator
from agents.report_generator import MAX_FINDINGS, MAX_REPORT_PAGES, generate_pdf_report
from evaluation.criteria import CRITERIA

COMPANIES = ["Figure AI", "Apptronik", "1X Technologies"]
LONG = "매우 긴 분석 문장입니다. " * 60
MARKER = "끝문장마커"


def _state(heavy: bool) -> dict:
    findings = [{"dimension": "제품", "statement": LONG, "evidence_refs": []}] * (MAX_FINDINGS if heavy else 1)
    if heavy:  # 마지막으로 보이는 문장: 글자를 줄여 맞추더라도 내용이 빠지면 안 된다
        findings = findings[:-1] + [{"dimension": "마지막", "statement": f"{MARKER} 마지막 분석 문장", "evidence_refs": []}]
    analysis = {
        "summary": LONG if heavy else "요약",
        "findings": findings,
        "evidence": [],
        "risks": [LONG] * (10 if heavy else 1),
        "missing_information": [LONG] * (15 if heavy else 0),
    }
    result = {
        "decision": "HOLD", "final_score": 3.0, "evidence_coverage": 1.0, "decision_reason": LONG,
        "key_strengths": [LONG] * 5, "key_risks": [LONG] * 5,
        "criteria": [{"criterion_id": c.id, "score": 3, "status": "SCORED", "evidence": []} for c in CRITERIA],
    }
    return {
        "candidate_companies": COMPANIES,
        "company_profiles": {c: {"target_market": "산업용", "funding_stage": "Series A", "eligibility": {"is_private": True}} for c in COMPANIES},
        "technology_results": {c: analysis for c in COMPANIES},
        "market_traction_results": {c: analysis for c in COMPANIES},
        "competition_result": {"comparisons": [{"dimension": f"축{i}", "company_findings": {c: LONG for c in COMPANIES}} for i in range(9)]},
        "investment_results": {c: result for c in COMPANIES},
    }


class ReportPagesTest(unittest.TestCase):
    """쪽 수 조건: 최대 MAX_REPORT_PAGES(10)장 이내. 길어도 내용은 줄이지 않고 글자 크기만 줄인다."""

    def _build(self, state: dict, n_refs: int):
        refs = [{"source_id": f"S{i}", "title": "제목", "publisher": "기관", "published_date": "2026-01-01",
                 "reference_metadata": {"reference_type": "web", "url": "https://example.com"}, "pages": [1]} for i in range(n_refs)]
        with tempfile.TemporaryDirectory() as tmp:
            reader = PdfReader(generate_pdf_report(state, refs, str(Path(tmp) / "r.pdf")))
            return len(reader.pages), "\n".join(p.extract_text() for p in reader.pages)

    def test_heavy_report_stays_within_ten_pages(self):
        pages, _ = self._build(_state(heavy=True), 40)
        self.assertTrue(1 <= pages <= MAX_REPORT_PAGES, pages)

    def test_light_report_stays_within_limits(self):
        pages, _ = self._build(_state(heavy=False), 1)
        self.assertTrue(1 <= pages <= MAX_REPORT_PAGES, pages)

    def test_over_limit_shrinks_font_but_keeps_content(self):
        """자연 분량이 상한을 넘으면 5쪽으로 줄이지 않고, 글자만 줄여 상한 안에 맞춘다 (내용 유지)."""
        with patch.object(report_generator, "MAX_REPORT_PAGES", 99):
            natural, natural_text = self._build(_state(heavy=True), 40)
        self.assertGreater(natural, 5)
        limit = natural - 1
        with patch.object(report_generator, "MAX_REPORT_PAGES", limit):
            pages, text = self._build(_state(heavy=True), 40)
        self.assertLessEqual(pages, limit)
        self.assertGreater(pages, 5)  # 5쪽으로 압축하지 않는다
        self.assertEqual(text.count(MARKER), natural_text.count(MARKER))  # 문장이 빠지지 않는다
        self.assertGreater(text.count(MARKER), 0)

    def test_node_fits_ten_pages_and_markdown_matches_pdf(self):
        """극단적으로 긴 입력도 10장 안에 맞추고, 줄인 내용은 Markdown(품질 평가 입력)에도 똑같이 반영된다."""
        import os
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                out = report_generator.report_generator_node(_state(heavy=True))
                reader = PdfReader(out["report_path"])
                pdf_text = "\n".join(p.extract_text() for p in reader.pages)
                pages = len(reader.pages)
            finally:
                os.chdir(cwd)
        self.assertTrue(1 <= pages <= MAX_REPORT_PAGES, pages)
        self.assertEqual(MARKER in out["final_report"], MARKER in pdf_text)

    def test_impossible_limit_raises(self):
        with patch.object(report_generator, "MAX_REPORT_PAGES", 3), self.assertRaises(ValueError):
            self._build(_state(heavy=False), 1)


if __name__ == "__main__":
    unittest.main()
