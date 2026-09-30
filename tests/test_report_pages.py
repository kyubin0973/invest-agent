import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from agents.report_generator import REPORT_PAGES, generate_pdf_report
from evaluation.criteria import CRITERIA

COMPANIES = ["Figure AI", "Apptronik", "1X Technologies"]
LONG = "매우 긴 분석 문장입니다. " * 60


def _state(heavy: bool) -> dict:
    analysis = {
        "summary": LONG if heavy else "요약",
        "findings": [{"dimension": "제품", "statement": LONG, "evidence_refs": []}] * (20 if heavy else 1),
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
    def _pages(self, state: dict, n_refs: int) -> int:
        refs = [{"source_id": f"S{i}", "title": "제목", "publisher": "기관", "published_date": "2026-01-01",
                 "reference_metadata": {"reference_type": "web", "url": "https://example.com"}, "pages": [1]} for i in range(n_refs)]
        with tempfile.TemporaryDirectory() as tmp:
            return len(PdfReader(generate_pdf_report(state, refs, str(Path(tmp) / "r.pdf"))).pages)

    def test_heavy_report_stays_five_pages(self):
        self.assertEqual(self._pages(_state(heavy=True), 40), REPORT_PAGES)

    def test_light_report_stays_five_pages(self):
        self.assertEqual(self._pages(_state(heavy=False), 1), REPORT_PAGES)


if __name__ == "__main__":
    unittest.main()
