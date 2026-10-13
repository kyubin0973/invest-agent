"""QA rules + semantic Judge composition and actionable lineage routing."""
from copy import deepcopy
import unittest
from unittest.mock import Mock

from agents.quality_evaluator import evaluate_report
from agents.supervisor import quality_rework_requests
from evaluation.report_quality import evaluate, QualityJudgeError
from tests.qa_fixtures import fixture, passing_judge


class HybridRoutingTests(unittest.TestCase):
    def setUp(self):
        self.state, self.originals = fixture()

    def test_rule_failure_still_runs_judge_and_keeps_semantic_issues(self):
        self.state['final_report'] = self.state['final_report'].replace('# REFERENCE',
            '로봇 100대 배치.\n# REFERENCE')
        def judge(payload):
            output = passing_judge(payload)
            output['checks'][1].update(verdict='fail', issues=[{
                'claim_id': payload['claims'][0]['claim_id'], 'evidence_ids': [],
                'suggested_owner': 'report_generator', 'company': None,
                'reason': '기술 우열 표현', 'requested_fix': '우열 표현 수정'}])
            return output
        callback = Mock(side_effect=judge)
        result = evaluate(self.state, original_documents=self.originals, judge=callback)
        callback.assert_called_once()
        self.assertTrue(result['judge_used'])
        self.assertFalse(result['passed'])
        self.assertEqual([c['verdict'] for c in result['checks']], ['fail', 'fail', 'pass'])
        self.assertIn('인용이 없습니다', result['checks'][0]['issues'][0]['reason'])
        self.assertEqual(result['checks'][1]['issues'][0]['reason'], '기술 우열 표현')

    def lineage_judge(self, payload):
        output = passing_judge(payload)
        output['checks'][2].update(verdict='fail', issues=[{
            'claim_id': None, 'evidence_ids': [], 'suggested_owner': 'report_generator',
            'company': None, 'reason': '출처 계보 source_family_id 누락',
            'requested_fix': '원본 발표의 계보를 확인하세요.'}])
        return output

    def test_lineage_only_issue_is_recorded_as_missing_information_not_failure(self):
        # 출처 계보는 문서 풀에 없는 정보라 재작업으로 채울 수 없다: 미달·재작업 사유가 아니라 미확인 정보로 남긴다
        self.state['technology_results']['A']['evidence'][0]['reference_metadata'].pop('source_family_id')
        self.state['references'][0]['reference_metadata'].pop('source_family_id')
        self.state['market_traction_results']['A'] = deepcopy(self.state['technology_results']['A'])
        result = evaluate_report(self.state, original_documents=self.originals, judge=self.lineage_judge)
        bias = result['checks'][2]
        self.assertTrue(bias['passed'])
        self.assertEqual(bias['issues'], [])
        self.assertTrue(result['passed'])
        self.assertEqual(quality_rework_requests({**self.state, 'quality_result': result}), [])
        detail = evaluate(self.state, original_documents=self.originals, judge=self.lineage_judge)
        self.assertIn('출처 계보', ' '.join(detail['missing_information']))

    def test_lineage_issue_does_not_hide_other_bias_issue(self):
        def judge(payload):
            output = self.lineage_judge(payload)
            output['checks'][2]['issues'].append({
                'claim_id': None, 'evidence_ids': [], 'suggested_owner': 'report_generator',
                'company': None, 'reason': '위험 근거 누락', 'requested_fix': '공개된 위험을 반영하세요.'})
            return output
        result = evaluate(self.state, original_documents=self.originals, judge=judge)
        bias = result['checks'][2]
        self.assertFalse(bias['passed'])
        self.assertEqual([i['reason'].split(' (')[0] for i in bias['issues']], ['위험 근거 누락'])

    def test_rule_failure_does_not_hide_judge_error(self):
        self.state['final_report'] += '\nロボット100台ではなく100대 배치'
        with self.assertRaises(QualityJudgeError):
            evaluate(self.state, original_documents=self.originals, judge=Mock(side_effect=RuntimeError('error')))
