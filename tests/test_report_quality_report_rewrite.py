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

    def test_whole_report_lineage_issue_routes_to_actual_analysis_owners(self):
        self.state['technology_results']['A']['evidence'][0]['reference_metadata'].pop('source_family_id')
        self.state['references'][0]['reference_metadata'].pop('source_family_id')
        self.state['market_traction_results']['A'] = deepcopy(self.state['technology_results']['A'])
        result = evaluate_report(self.state, original_documents=self.originals, judge=self.lineage_judge)
        issues = result['checks'][2]['issues']
        self.assertEqual({(i['owner'], i['company']) for i in issues},
                         {('technology', 'A'), ('market_traction', 'A')})
        self.assertEqual(result['feedback'], [])
        requests = quality_rework_requests({**self.state, 'quality_result': result})
        self.assertEqual({r['agent'] for r in requests}, {'technology', 'market_traction'})

    def test_specific_lineage_issue_does_not_route_unrelated_company(self):
        self.state['candidate_companies'].append('B')
        self.state['technology_results']['B'] = deepcopy(self.state['technology_results']['A'])
        def judge(payload):
            output = self.lineage_judge(payload)
            issue = output['checks'][2]['issues'][0]
            issue.update(company='A', evidence_ids=payload['claims'][0]['evidence_ids'])
            return output
        result = evaluate(self.state, original_documents=self.originals, judge=judge)
        self.assertEqual([(i['owner'], i['company']) for i in result['checks'][2]['issues']], [('technology', 'A')])

    def test_rule_failure_does_not_hide_judge_error(self):
        self.state['final_report'] += '\nロボット100台ではなく100대 배치'
        with self.assertRaises(QualityJudgeError):
            evaluate(self.state, original_documents=self.originals, judge=Mock(side_effect=RuntimeError('error')))
