"""독립 실행 도구의 무과금 기본 동작·오류·출력 경계 검증."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import report_quality_runner as runner
from evaluation.report_quality import QualityJudgeError
from tests.qa_fixtures import fixture


class QualityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.state, self.originals = fixture()

    def test_default_only_checks_input_and_never_claims_qa_pass(self):
        with patch.object(runner, "evaluate_report") as evaluation:
            result = runner.run_quality(self.state, self.originals)
        evaluation.assert_not_called()
        self.assertEqual(result["mode"], "input_check")
        self.assertTrue(result["input_ready"])
        self.assertIsNone(result["passed"])
        self.assertFalse(result["judge_used"])

    def test_placeholder_is_not_ready(self):
        self.state["final_report"] = "# Markdown Report Generated"
        result = runner.run_quality(self.state, self.originals)
        self.assertFalse(result["input_ready"])
        self.assertTrue(result["issues"])
        self.assertIsNone(result["passed"])

    def test_live_reuses_configuration_and_evaluator(self):
        with (patch("core.tracing.setup") as setup,
              patch.object(runner, "evaluate_report", return_value={"passed": True}) as evaluation):
            result = runner.run_quality(self.state, self.originals, live=True)
        setup.assert_called_once()
        evaluation.assert_called_once()
        self.assertEqual(evaluation.call_args.kwargs["original_documents"], self.originals)
        self.assertEqual(result["mode"], "live")
        self.assertTrue(result["passed"])

    def test_judge_failure_is_reported_as_error_not_pass(self):
        with (patch("core.tracing.setup"), patch.object(runner, "evaluate_report",
              side_effect=QualityJudgeError("QA Judge 호출 실패; Supervisor 재시도 필요"))):
            result = runner.run_quality(self.state, self.originals, live=True)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["passed"])

    def test_unexpected_input_errors_never_echo_data(self):
        self.state["technology_results"] = "private-value"
        result = runner.run_quality(self.state, self.originals)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("private-value", json.dumps(result))

    def test_output_rejects_path_traversal_and_existing_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "OUTPUT_DIR", Path(temp)):
            for name in ("../report.json", "agents/quality_evaluator.py", "/tmp/result.json"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.save_result({}, name)
            output = runner.save_result({"passed": None}, "qa.json")
            with self.assertRaises(FileExistsError):
                runner.save_result({"passed": True}, "qa.json")
            self.assertIsNone(json.loads(output.read_text())["passed"])

    def test_output_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "OUTPUT_DIR", Path(temp)):
            input_file = Path(temp) / "input.json"
            input_file.write_text("{}")
            with self.assertRaises(ValueError):
                runner.save_result({}, "input.json", input_paths=(input_file,))
            self.assertEqual(input_file.read_text(), "{}")

    def test_output_folder_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "protected").mkdir()
            (root / "qa").symlink_to(root / "protected", target_is_directory=True)
            with patch.object(runner, "OUTPUT_DIR", root / "qa"), self.assertRaises(ValueError):
                runner.save_result({}, "qa.json")
            self.assertEqual(list((root / "protected").iterdir()), [])

    def test_output_redacts_keys(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "OUTPUT_DIR", Path(temp)):
            output = runner.save_result({"reason": "sk-syntheticSecret0123456789"}, "qa.json")
            self.assertEqual(json.loads(output.read_text())["reason"], "[REDACTED]")

    def test_cli_input_check_works_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_file, originals_file = root / "state.json", root / "originals.json"
            state_file.write_text(json.dumps(self.state), encoding="utf-8")
            originals_file.write_text(json.dumps(self.originals), encoding="utf-8")
            with (patch.object(runner, "OUTPUT_DIR", root / "qa"),
                  patch.object(runner, "evaluate_report") as evaluation, redirect_stdout(io.StringIO())):
                status = runner.main(["--state", str(state_file), "--originals", str(originals_file),
                                      "--output-name", "input-check.json"])
            self.assertEqual(status, 0)
            evaluation.assert_not_called()
            self.assertEqual(json.loads((root / "qa" / "input-check.json").read_text())["mode"], "input_check")

    def test_cli_bad_json_returns_error_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = root / "bad.json"
            bad.write_text("not-json")
            with patch.object(runner, "OUTPUT_DIR", root / "qa"), redirect_stdout(io.StringIO()):
                status = runner.main(["--state", str(bad), "--originals", str(bad)])
            self.assertEqual(status, 2)
            self.assertFalse((root / "qa").exists())


if __name__ == "__main__":
    unittest.main()
