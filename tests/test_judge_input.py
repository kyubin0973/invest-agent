import unittest

from evaluation.judge_input import validate_and_build_evidence_context
from scripts.run_judge_demo import build_demo_state


class JudgeInputValidationTests(unittest.TestCase):
    def test_valid_input_builds_q4_q9_scopes(self):
        state = build_demo_state()

        context = validate_and_build_evidence_context(state, "Figure AI")

        self.assertEqual(len(context.competition_by_criterion["B04"]), 3)
        self.assertEqual(len(context.competition_by_criterion["B09"]), 3)
        competition_ref = ("DA4_C01", "DA4")
        self.assertIn(competition_ref, context.allowed_evidence_by_criterion["B04"])
        self.assertIn(competition_ref, context.allowed_evidence_by_criterion["B09"])
        self.assertNotIn(competition_ref, context.allowed_evidence_by_criterion["B01"])

    def test_invalid_evidence_item_is_rejected_before_llm(self):
        state = build_demo_state()
        state["technology_results"]["Figure AI"]["evidence"][0]["fact"] = None

        with self.assertRaisesRegex(ValueError, r"fact"):
            validate_and_build_evidence_context(state, "Figure AI")

    def test_unresolved_analysis_finding_reference_is_rejected(self):
        state = build_demo_state()
        state["technology_results"]["Figure AI"]["findings"][0]["evidence_refs"] = [
            {"chunk_id": "UNKNOWN", "source_id": "UNKNOWN"}
        ]

        with self.assertRaisesRegex(ValueError, r"AnalysisResult\.evidence에 없는 참조"):
            validate_and_build_evidence_context(state, "Figure AI")

    def test_invalid_competition_scope_is_rejected(self):
        state = build_demo_state()
        state["competition_result"]["comparisons"][0]["criterion_ids"] = ["B01"]

        with self.assertRaisesRegex(ValueError, r"B04/B09만"):
            validate_and_build_evidence_context(state, "Figure AI")

    def test_unresolved_competition_reference_is_rejected(self):
        state = build_demo_state()
        state["competition_result"]["comparisons"][0]["evidence_refs"][0] = {
            "chunk_id": "UNKNOWN",
            "source_id": "UNKNOWN",
        }

        with self.assertRaisesRegex(ValueError, r"전체 분석 Evidence에 없는 참조"):
            validate_and_build_evidence_context(state, "Figure AI")


if __name__ == "__main__":
    unittest.main()
