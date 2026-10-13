"""공통 State를 바꾸지 않는 QA 반환·원문 조회 경계 테스트."""
import unittest
from unittest.mock import Mock, patch

from core.state import QualityCheck, QualityIssue, QualityResult
from agents.quality_evaluator import evaluate_report, evaluate_report_details, quality_evaluator_node
from evaluation.report_quality import QualityJudgeError
from evaluation.report_quality_sources import load_original_documents
from tests.qa_fixtures import fixture, issue_judge, passing_judge


class QualityContractTests(unittest.TestCase):
    def setUp(self):
        self.state, self.originals = fixture()

    def assert_common(self, result):
        self.assertEqual(set(result), set(QualityResult.__annotations__))
        for check in result["checks"]:
            self.assertEqual(set(check), set(QualityCheck.__annotations__))
            for issue in check["issues"]:
                self.assertEqual(set(issue), set(QualityIssue.__annotations__))

    def test_pass_result_has_exact_common_keys(self):
        result = evaluate_report(self.state, original_documents=self.originals, judge=passing_judge)
        self.assert_common(result)
        self.assertTrue(result["passed"])

    def test_failure_keeps_requested_fix_in_existing_reason_and_feedback(self):
        result = evaluate_report(self.state, original_documents=self.originals,
                                 judge=issue_judge("groundedness", "원문과 다른 주장"))
        self.assert_common(result)
        self.assertFalse(result["passed"])
        self.assertIn("수정 요청:", result["checks"][0]["issues"][0]["reason"])
        self.assertIn("line:2", result["feedback"][0])
        self.assertIn("S1_P01_C01", result["feedback"][0])

    def test_analysis_fix_survives_existing_supervisor_feedback_conversion(self):
        from agents.supervisor import quality_rework_requests
        result = evaluate_report(self.state, original_documents=self.originals,
                                 judge=issue_judge("groundedness", "분석 근거 오류", "technology"))
        requests = quality_rework_requests({**self.state, "quality_result": result})
        self.assertEqual(requests[0]["agent"], "technology")
        self.assertIn("수정 요청:", requests[0]["feedback"][0])

    def test_details_are_available_outside_shared_state(self):
        details = evaluate_report_details(self.state, original_documents=self.originals, judge=passing_judge)
        self.assertEqual(details["report_version"], self.state["report_attempts"])
        self.assertIn("report_hash", details)
        self.assertIn("verdict", details["checks"][0])

    def test_node_loads_originals_and_writes_only_quality_result(self):
        with patch("agents.quality_evaluator.load_original_documents", return_value=self.originals) as loader:
            # Patch only the LLM boundary; source and contract wiring remain real.
            with patch("core.llm.get_structured_llm") as factory:
                from evaluation.report_quality import prepare_input
                factory.return_value.invoke.return_value = passing_judge(prepare_input(self.state, self.originals)[0])
                update = quality_evaluator_node(self.state)
        loader.assert_called_once_with(self.state)
        self.assertEqual(set(update), {"quality_result"})
        self.assert_common(update["quality_result"])
        self.assertTrue(update["quality_result"]["passed"])

    def test_default_source_loader_reuses_existing_vectorstore_get(self):
        with patch("rag.retriever.get_vectorstore") as vectorstore:
            vectorstore.return_value.get.return_value = self.source_response()
            originals = load_original_documents(self.state)
        self.assertEqual(originals, self.originals)
        vectorstore.return_value.get.assert_called_once_with(
            ids=["S1_P01_C01"], include=["documents", "metadatas"])
        vectorstore.return_value.similarity_search_with_relevance_scores.assert_not_called()

    def source_response(self):
        original = self.originals["S1_P01_C01"]
        return {"ids": [original["chunk_id"]], "documents": [original["text"]],
                "metadatas": [{k: original[k] for k in ("source_id", "chunk_id", "page")} ]}

    def test_originals_use_exact_id_batch_get_without_similarity_search(self):
        fetch = Mock(return_value=self.source_response())
        result = load_original_documents(self.state, fetch=fetch)
        self.assertEqual(result, self.originals)
        fetch.assert_called_once_with(ids=["S1_P01_C01"], include=["documents", "metadatas"])

    def test_duplicate_citations_fetch_once(self):
        self.state["final_report"] += "[S1_P01_C01 | S1]\n"
        fetch = Mock(return_value=self.source_response())
        load_original_documents(self.state, fetch=fetch)
        self.assertEqual(fetch.call_args.kwargs["ids"], ["S1_P01_C01"])

    def test_placeholder_never_opens_vectorstore(self):
        self.state["final_report"] = "# Markdown Report Generated"
        fetch = Mock()
        self.assertEqual(load_original_documents(self.state, fetch=fetch), {})
        fetch.assert_not_called()

    def test_unknown_citation_is_not_used_to_fetch_new_evidence(self):
        self.state["final_report"] = "# SUMMARY\n[UNKNOWN | S1]\n# REFERENCE\n[S1]"
        fetch = Mock()
        self.assertEqual(load_original_documents(self.state, fetch=fetch), {})
        fetch.assert_not_called()

    def test_missing_or_wrong_original_is_execution_error(self):
        responses = [
            {"ids": [], "documents": [], "metadatas": []},
            {"ids": ["S1_P01_C01"], "documents": ["text"], "metadatas": [{"source_id": "WRONG", "chunk_id": "S1_P01_C01", "page": 1}]},
            {"ids": ["S1_P01_C01"], "documents": [], "metadatas": []},
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(QualityJudgeError):
                load_original_documents(self.state, fetch=Mock(return_value=response))

    def test_source_exception_does_not_echo_credentials(self):
        with self.assertRaises(QualityJudgeError) as context:
            load_original_documents(self.state, fetch=Mock(side_effect=RuntimeError("secret-value")))
        self.assertNotIn("secret-value", str(context.exception))

    def test_source_failure_propagates_to_existing_worker_retry_boundary(self):
        with patch("agents.quality_evaluator.load_original_documents", side_effect=QualityJudgeError("원문 조회 실패")):
            with self.assertRaises(QualityJudgeError):
                quality_evaluator_node(self.state)


if __name__ == "__main__":
    unittest.main()
