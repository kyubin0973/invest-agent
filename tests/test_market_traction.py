import unittest

from langchain_core.documents import Document

from agents.market_traction import (
    SUB_QUESTIONS,
    FindingOut,
    MarketAnalysisOut,
    _dimension_for,
    _on_topic,
    _validate,
)
from prompts.market_traction import DIMENSIONS


def _doc(chunk_id: str, text: str, company: str = "figure", source_type: str = "company") -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "source_id": chunk_id.split("_")[0],
            "company": company,
            "source_type": source_type,
            "page": 1,
            "title": "t",
            "publisher": "p",
            "published_date": "2025",
            "reference_metadata": "{}",
        },
    )


def _out(*findings: FindingOut) -> MarketAnalysisOut:
    return MarketAnalysisOut(summary="s", findings=list(findings), evidence=[], risks=[], missing_information=[])


class SubQuestionTest(unittest.TestCase):
    def test_sub_question_dimensions_are_known(self):
        for sq in SUB_QUESTIONS:
            self.assertTrue(sq.dimensions)
            self.assertTrue(set(sq.dimensions) <= set(DIMENSIONS), sq.id)

    def test_dimension_outside_question_falls_back_to_default(self):
        self.assertEqual(_dimension_for("SQ10", "사업모델·가격"), "제조 파트너·원가·Fleet 운영")
        self.assertEqual(_dimension_for("SQ08", "WTP·도입 경제성"), "WTP·도입 경제성")


class TopicCheckTest(unittest.TestCase):
    def test_business_model_needs_pricing_text(self):
        self.assertFalse(_on_topic("사업모델·가격", ["BotQ can produce up to 12,000 humanoid robots per year."]))
        self.assertTrue(_on_topic("사업모델·가격", ["NEO is available for $20,000 or $499 per month."]))

    def test_other_dimensions_are_not_checked(self):
        self.assertTrue(_on_topic("제조 파트너·원가·Fleet 운영", ["anything"]))


class ValidateTest(unittest.TestCase):
    def test_off_topic_business_model_finding_is_dropped(self):
        docs = {"F1_P02_C02": _doc("F1_P02_C02", "BotQ is Figure's dedicated manufacturing facility.")}
        result, answered = _validate(
            _out(FindingOut(question_id="SQ08", dimension="사업모델·가격", statement="공급망 구축", evidence_chunk_ids=["F1_P02_C02"])),
            docs,
            "figure",
        )
        self.assertEqual(result["findings"], [])
        self.assertEqual(answered, set())

    def test_manufacturing_finding_keeps_question_dimension(self):
        docs = {"F1_P02_C02": _doc("F1_P02_C02", "BotQ is Figure's dedicated manufacturing facility.")}
        result, answered = _validate(
            _out(FindingOut(question_id="SQ10", dimension="사업 확장성", statement="BotQ 생산", evidence_chunk_ids=["F1_P02_C02"])),
            docs,
            "figure",
        )
        self.assertEqual(result["findings"][0]["dimension"], "제조 파트너·원가·Fleet 운영")
        self.assertEqual(answered, {"SQ10"})

    def test_not_found_statement_moves_to_missing(self):
        docs = {"X5_P01_C01": _doc("X5_P01_C01", "NEO home robot.", company="1x")}
        result, answered = _validate(
            _out(FindingOut(question_id="SQ06", dimension="고객검증·배치·계약", statement="고객 계약 정보는 확인되지 않았다.", evidence_chunk_ids=["X5_P01_C01"])),
            docs,
            "1x",
        )
        self.assertEqual(result["findings"], [])
        self.assertIn("고객 계약 정보는 확인되지 않았다.", result["missing_information"])
        self.assertEqual(answered, set())

    def test_company_dimension_rejects_market_report(self):
        docs = {"M5_P02_C01": _doc("M5_P02_C01", "Supply chain constraints.", company="", source_type="industry_report")}
        result, _ = _validate(
            _out(FindingOut(question_id="SQ10", dimension="제조 파트너·원가·Fleet 운영", statement="공급망", evidence_chunk_ids=["M5_P02_C01"])),
            docs,
            "figure",
        )
        self.assertEqual(result["findings"], [])


if __name__ == "__main__":
    unittest.main()
