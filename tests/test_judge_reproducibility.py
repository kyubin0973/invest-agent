import unittest
from unittest.mock import patch

from core import llm
from prompts.investment_judge import build_judge_messages
from tests.test_investment_judge import make_evidence, make_state


class JudgePromptTests(unittest.TestCase):
    def test_analysis_evidence_is_not_duplicated_in_context_and_evidence_block(self):
        evidence = make_evidence()
        messages = build_judge_messages(make_state(evidence), "Test Robotics", [evidence])
        user_prompt = messages[1][1]

        self.assertEqual(user_prompt.count("<chunk_id>TEST_C01</chunk_id>"), 1)
        self.assertNotIn('"evidence":[', user_prompt)
        self.assertIn('"technology_analysis"', user_prompt)

    def test_evidence_order_does_not_change_prompt(self):
        first = make_evidence("E3", "Z_C01", "Z_SOURCE")
        second = make_evidence("E3", "A_C01", "A_SOURCE")
        state = make_state()

        forward = build_judge_messages(state, "Test Robotics", [first, second])
        reverse = build_judge_messages(state, "Test Robotics", [second, first])

        self.assertEqual(forward, reverse)
        self.assertLess(
            forward[1][1].index("<chunk_id>A_C01</chunk_id>"),
            forward[1][1].index("<chunk_id>Z_C01</chunk_id>"),
        )

    def test_competition_evidence_order_does_not_change_prompt(self):
        first = make_evidence("E3", "Z_C01", "Z_SOURCE")
        second = make_evidence("E3", "A_C01", "A_SOURCE")
        state = make_state()
        forward_items = [("Z Corp", first, ["B09"]), ("A Corp", second, ["B04"])]

        forward = build_judge_messages(
            state,
            "Test Robotics",
            [],
            competition_evidence=forward_items,
        )
        reverse = build_judge_messages(
            state,
            "Test Robotics",
            [],
            competition_evidence=list(reversed(forward_items)),
        )

        self.assertEqual(forward, reverse)


class LlmReproducibilityTests(unittest.TestCase):
    @patch("core.llm.tracing.setup")
    @patch("core.llm.init_chat_model")
    def test_common_llm_uses_zero_temperature_and_fixed_seed(self, init_chat_model, _setup):
        llm.get_llm.cache_clear()

        llm.get_llm()

        init_chat_model.assert_called_once_with(
            "gpt-4o-mini",
            model_provider="openai",
            temperature=0,
            seed=42,
        )
        llm.get_llm.cache_clear()


if __name__ == "__main__":
    unittest.main()
