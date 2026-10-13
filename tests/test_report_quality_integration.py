"""Real graph + real QA with injected synthetic sources/Judge; not real LLM evidence.

Supervisor files are read-only. Expected failures expose owner-1 integration bugs
instead of weakening the required FAILED termination assertions.
"""
import logging
import json
import unittest
from unittest.mock import patch

from agents import supervisor
from agents.quality_evaluator import evaluate_report_details as evaluate_report
from agents.supervisor import FINISH, QUALITY, REPORT
from core import graph as graph_module
from evaluation.report_quality import QualityJudgeError, result_is_current
from evaluation.report_quality_contract import to_common_result
from tests.qa_fixtures import EXAGGERATED, NORMAL, fixture, issue_judge, passing_judge
from tests.test_supervisor import Workers, run


class QAWorkers(Workers):
    def __init__(self, *, always_fail=False, fail_owner="report_generator", judge_error=False, normal_first=False):
        super().__init__()
        self.always_fail = always_fail
        self.fail_owner = fail_owner
        self.judge_error = judge_error
        self.normal_first = normal_first
        self.results = []

    def report(self, state):
        self._run(REPORT, state)
        statement = EXAGGERATED if self.always_fail or (state["report_attempts"] == 1 and not self.normal_first) else NORMAL
        local, _ = fixture(statement, state["report_attempts"])
        # Only worker stubs provide missing team data. Production State is untouched.
        return {"final_report": local["final_report"], "references": local["references"], "report_path": "synthetic.pdf"}

    def quality_node(self, state):
        self._run(QUALITY, state)
        local, originals = fixture(version=state["report_attempts"])
        local["final_report"] = state["final_report"]
        local["candidate_companies"] = state["candidate_companies"]
        local["trace_id"] = state["trace_id"]
        if self.judge_error:
            def judge(_):
                raise QualityJudgeError("synthetic-error")
        elif EXAGGERATED in local["final_report"]:
            judge = issue_judge("groundedness", "데모 과장", self.fail_owner)
        else:
            judge = passing_judge
        result = evaluate_report(local, original_documents=originals, judge=judge)
        self.results.append(result)
        return {"quality_result": to_common_result(result)}


class QAIntegrationTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_failure_rewrite_recheck_uses_latest_body(self):
        workers = QAWorkers()
        state = run(workers)
        self.assertEqual([c[0] for c in workers.calls][-4:], [REPORT, QUALITY, REPORT, QUALITY])
        self.assertEqual(state["status"], "SUCCESS")
        self.assertEqual(len(workers.results), 2)
        self.assertFalse(workers.results[0]["passed"])
        self.assertTrue(workers.results[1]["passed"])
        self.assertEqual(workers.results[1]["report_version"], state["report_attempts"])
        self.assertNotEqual(workers.results[0]["report_hash"], workers.results[1]["report_hash"])

    def test_normal_path_finishes_after_one_report(self):
        workers = QAWorkers(normal_first=True)
        state = run(workers)
        self.assertEqual(state["status"], "SUCCESS")
        self.assertEqual(state["report_attempts"], 1)
        self.assertTrue(state["quality_result"]["passed"])
        self.assertEqual([c[0] for c in workers.calls][-2:], [REPORT, QUALITY])

    def test_analysis_issue_is_routed_only_by_supervisor(self):
        workers = QAWorkers(fail_owner="technology")
        state = run(workers)
        self.assertIn(("technology", "A", True), workers.calls)
        self.assertTrue(state["quality_result"]["passed"])

    def test_judge_errors_retry_then_terminate_failed(self):
        workers = QAWorkers(judge_error=True)
        state = run(workers)
        self.assertEqual(state["status"], "FAILED")
        self.assertEqual(state["next"], FINISH)
        self.assertEqual(sum(c[0] == QUALITY for c in workers.calls), supervisor.MAX_NODE_RETRIES + 1)
        self.assertEqual(state["node_status"][QUALITY], "FAILED")
        self.assertFalse(result_is_current(state))

    def test_production_placeholder_remains_failed_and_is_bounded(self):
        workers = Workers()
        from agents.quality_evaluator import quality_evaluator_node
        nodes = workers.nodes()
        nodes[QUALITY] = quality_evaluator_node
        with patch.dict(graph_module.WORKER_NODES, nodes), patch("core.llm.get_structured_llm") as factory:
            factory.return_value.invoke.side_effect = lambda messages, **kwargs: passing_judge(json.loads(messages[1][1]))
            state = graph_module.build_graph().invoke(
                {"candidate_companies": ["A", "B"], "company_profiles": {"A": {}, "B": {}}},
                config={"recursion_limit": graph_module.recursion_limit(2)})
        self.assertEqual(state["next"], FINISH)
        self.assertFalse(state["quality_result"]["passed"])
        self.assertEqual(set(state["quality_result"]), {"passed", "checks", "feedback"})
        self.assertEqual(state["report_attempts"], supervisor.MAX_REPORT_ATTEMPTS)

    @unittest.expectedFailure
    def test_report_cap_must_end_failed_owner1_fix_required(self):
        state = run(QAWorkers(always_fail=True))
        self.assertEqual(state["next"], FINISH)
        self.assertEqual(state["report_attempts"], supervisor.MAX_REPORT_ATTEMPTS)
        self.assertFalse(state["quality_result"]["passed"])
        self.assertTrue(state["quality_result"]["checks"][0]["issues"])
        self.assertEqual(state["status"], "FAILED")  # currently incorrectly SUCCESS

    @unittest.expectedFailure
    def test_step_cap_with_failed_qa_must_end_failed_owner1_fix_required(self):
        from core.graph import initialize_state
        state = initialize_state({"candidate_companies": ["A"], "company_profiles": {"A": {}}})
        state.update(final_report="report", report_evaluated=True,
                     quality_result={"passed": False}, step_count=state["max_steps"])
        self.assertEqual(supervisor.decide(state).updates["status"], "FAILED")

    @unittest.expectedFailure
    def test_stale_pass_must_trigger_recheck_owner1_fix_required(self):
        from core.graph import initialize_state
        local, originals = fixture()
        result = evaluate_report(local, original_documents=originals, judge=passing_judge)
        state = initialize_state({"candidate_companies": ["A"], "company_profiles": {"A": {}}})
        state.update(local)
        state.update(market_traction_results={"A": local["technology_results"]["A"]},
                     investment_results={"A": {"criteria": [], "decision": "HOLD"}},
                     competition_result={"comparisons": []}, pending_judge=[], report_evaluated=True,
                     report_attempts=2, quality_result=to_common_result(result))
        state["final_report"] += "수정된 본문\n"
        self.assertFalse(result_is_current(state, result))
        self.assertEqual(supervisor.decide(state).next, QUALITY)  # currently accepts the stale pass


if __name__ == "__main__":
    unittest.main()
