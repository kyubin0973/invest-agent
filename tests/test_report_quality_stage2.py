"""규칙·구조화 응답·대역 검증. 실제 모델 평가는 benchmark --live로 분리한다."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from evaluation.report_quality import QualityJudgeError, evaluate, prepare_input, result_is_current
from evaluation.report_quality_contract import to_common_result
from evaluation.report_quality_benchmark import run_cases
from tests.qa_fixtures import passing_judge, issue_judge
from tests.qa_semantic_cases import CASES, semantic_fixture


class StageTwoTests(unittest.TestCase):
    def test_all_semantic_cases_reach_judge_without_leaking_expected_answer(self):
        for name in CASES:
            with self.subTest(case=name):
                state, originals, _ = semantic_fixture(name)
                payload, issues, _ = prepare_input(state, originals)
                self.assertEqual(issues, [])
                self.assertNotIn('expected_failed_criterion', payload)
                self.assertTrue(payload['claims'])

    def test_scripted_semantic_failures_preserve_reason_fix_and_owner(self):
        for name in CASES:
            state, originals, criterion = semantic_fixture(name)
            if criterion is None:
                continue
            with self.subTest(case=name):
                result = evaluate(state, original_documents=originals,
                                  judge=issue_judge(criterion, '합성 사례의 위반 이유'))
                common = to_common_result(result)
                self.assertFalse(common['passed'])
                check = next(c for c in common['checks'] if c['item'] == criterion)
                self.assertIn('수정 요청:', check['issues'][0]['reason'])
                self.assertEqual(check['issues'][0]['owner'], 'report_generator')

    def test_policy_distinguishes_disclosed_limits_from_unsupported_success(self):
        from prompts.report_quality import SYSTEM_PROMPT, QA_PROMPT_VERSION
        self.assertEqual(QA_PROMPT_VERSION, 'report-quality-v5')
        self.assertIn('검사 범위는 보고서가 실제로 주장한 내용', SYSTEM_PROMPT)
        self.assertIn('기업 발표를 출처에 귀속해 제한적으로 소개하는 것 자체는 편향이 아니다', SYSTEM_PROMPT)
        self.assertIn('근거 유무와 관계없이 금지', SYSTEM_PROMPT)

    def test_investment_state_is_forwarded_without_rewriting_scores(self):
        state, originals, _ = semantic_fixture('investment_consistent')
        before = deepcopy(state)
        payload, issues, _ = prepare_input(state, originals)
        self.assertEqual(issues, [])
        self.assertEqual(payload['investment_results'], state['investment_results'])
        result = evaluate(state, original_documents=originals, judge=passing_judge)
        self.assertTrue(result['passed'])
        self.assertEqual(state, before)

    def test_observation_can_pass_with_valid_scripted_judge(self):
        state, originals, _ = semantic_fixture('observation')
        self.assertTrue(evaluate(state, original_documents=originals, judge=passing_judge)['passed'])

    def test_common_result_has_no_version_guard_metadata(self):
        state, originals, _ = semantic_fixture('normal')
        detail = evaluate(state, original_documents=originals, judge=passing_judge)
        self.assertTrue(result_is_current(state, detail, original_documents=originals))
        self.assertFalse(result_is_current(state, to_common_result(detail), original_documents=originals))

    def test_malformed_responses_cannot_pass(self):
        state, originals, _ = semantic_fixture('normal')
        mutations = {
            'duplicate_claim': lambda r: r['checked_claim_ids'].append(r['checked_claim_ids'][0]),
            'unknown_check': lambda r: r['checks'][0].update(item='profit'),
            'blank_reason': lambda r: r['checks'][0].update(reason=' '),
            'unknown_owner': lambda r: r['checks'][0]['issues'][0].update(suggested_owner='supervisor'),
            'blank_fix': lambda r: r['checks'][0]['issues'][0].update(requested_fix=' '),
            'unknown_company': lambda r: r['checks'][0]['issues'][0].update(suggested_owner='technology', company='B'),
            'unrelated_evidence': lambda r: r['checks'][0]['issues'][0].update(evidence_ids=[{'source_id':'S2','chunk_id':'S2_P01_C01'}]),
            'uncertain_no_missing': lambda r: r['checks'][0].update(verdict='uncertain', missing_information=[]),
        }
        for label, mutate in mutations.items():
            with self.subTest(case=label):
                response = issue_judge('groundedness', '재검증 필요')(prepare_input(state, originals)[0])
                mutate(response)
                with self.assertRaises(QualityJudgeError):
                    evaluate(state, original_documents=originals, judge=lambda _: deepcopy(response))

    def test_benchmark_default_has_no_model_call_or_pass_claim(self):
        with patch('evaluation.report_quality_runner.evaluate_report', side_effect=AssertionError('no API')):
            result = run_cases(['normal', 'supported_recommendation'])
        self.assertFalse(result['submission_evidence'])
        self.assertIsNone(result['all_expectations_matched'])
        self.assertTrue(all(r['result']['input_ready'] for r in result['cases']))
        self.assertTrue(all(r['result']['passed'] is None for r in result['cases']))

    def test_benchmark_caps_and_deduplicates_logical_evaluations(self):
        for cases in ([], ['normal'] * 2, list(CASES)[:5], ['unknown']):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                run_cases(cases, live=True)

    def test_benchmark_stops_after_judge_error(self):
        with patch('evaluation.report_quality_benchmark.run_quality', return_value={'status':'error','passed':False}) as run:
            result = run_cases(['normal', 'demo_exaggeration'], live=True)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(result['cases'][0]['expectation_matched'])
        self.assertIsNone(result['all_expectations_matched'])

    def test_benchmark_does_not_accept_rule_failure_as_semantic_success(self):
        with patch('evaluation.report_quality_benchmark.run_quality', return_value={
            'passed':False, 'judge_used':False, 'checks':[{'item':'neutrality','passed':False}]}):
            result = run_cases(['supported_recommendation'], live=True)
        self.assertFalse(result['all_expectations_matched'])


if __name__ == '__main__':
    unittest.main()
