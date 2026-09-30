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

    def test_unclassified_evidence_is_accepted_but_excluded_from_scoring(self):
        state = build_demo_state()
        item = state["technology_results"]["Figure AI"]["evidence"][0]
        item["fact"] = None
        item["evidence_level"] = None

        context = validate_and_build_evidence_context(state, "Figure AI")

        self.assertNotIn(item["chunk_id"], context.own_evidence)
        self.assertIn(item["chunk_id"], context.excluded_evidence_ids)

    def test_same_chunk_from_two_agents_uses_scoreable_item_without_collision(self):
        state = build_demo_state()
        technology_item = state["technology_results"]["Figure AI"]["evidence"][0]
        market_variant = {**technology_item, "fact": None, "evidence_level": None}
        state["market_traction_results"]["Figure AI"]["evidence"].append(market_variant)

        context = validate_and_build_evidence_context(state, "Figure AI")

        self.assertEqual(
            context.own_evidence[technology_item["chunk_id"]]["fact"],
            technology_item["fact"],
        )

    def test_shared_industry_chunk_can_be_used_by_multiple_companies(self):
        state = build_demo_state()
        shared = next(
            item
            for item in state["market_traction_results"]["Figure AI"]["evidence"]
            if item["source_type"] == "industry_report"
        )
        state["market_traction_results"]["Apptronik"]["evidence"].append(
            {**shared, "fact": "같은 공통 보고서의 Apptronik 분석 관점"}
        )
        state["competition_result"]["comparisons"][0]["evidence_refs"].append(
            {"chunk_id": shared["chunk_id"], "source_id": shared["source_id"]}
        )

        context = validate_and_build_evidence_context(state, "Figure AI")

        owners = {
            item["chunk_id"]: owner
            for owner, item, _criterion_ids in context.competition_prompt_items()
        }
        self.assertEqual(owners[shared["chunk_id"]], "공통 산업자료")

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

    def test_unresolved_competition_reference_is_excluded(self):
        state = build_demo_state()
        state["competition_result"]["comparisons"][0]["evidence_refs"][0] = {
            "chunk_id": "UNKNOWN",
            "source_id": "UNKNOWN",
        }

        context = validate_and_build_evidence_context(state, "Figure AI")

        self.assertEqual(len(context.excluded_competition_refs), 1)
        self.assertNotIn("UNKNOWN", context.competition_by_criterion["B04"])

    def test_legacy_competition_without_criterion_ids_defaults_to_q4(self):
        state = build_demo_state()
        comparison = state["competition_result"]["comparisons"][0]
        comparison.pop("criterion_ids")

        context = validate_and_build_evidence_context(state, "Figure AI")

        self.assertEqual(len(context.competition_by_criterion["B04"]), 3)
        self.assertEqual(len(context.competition_by_criterion["B09"]), 0)


if __name__ == "__main__":
    unittest.main()
