"""Self-contained synthetic Judge input; no external demo script dependency."""
from copy import deepcopy
from tests.test_investment_judge import make_evidence, make_state


def build_demo_state():
    state = make_state(make_evidence('E3', 'DF_C01', 'DF'))
    for key in ('technology_results', 'market_traction_results', 'company_profiles'):
        state[key]['Figure AI'] = state[key].pop('Test Robotics')
    state['candidate_companies'] = ['Figure AI', 'Apptronik', '1X Technologies']
    state['company_profiles']['Figure AI']['name'] = 'Figure AI'
    technology = state['technology_results']['Figure AI']
    technology['findings'] = [{'dimension': '제품', 'statement': '기술 관측',
                              'evidence_refs': [{'chunk_id': 'DF_C01', 'source_id': 'DF'}]}]
    shared = make_evidence('E3', 'DM_C01', 'DM')
    shared['source_type'] = 'industry_report'
    state['market_traction_results']['Figure AI']['evidence'] = [shared]
    for company, source in (('Apptronik', 'DA4'), ('1X Technologies', 'DX')):
        state['company_profiles'][company] = deepcopy(state['company_profiles']['Figure AI'])
        state['company_profiles'][company]['name'] = company
        state['technology_results'][company] = {'summary': '기술 분석', 'findings': [],
            'evidence': [make_evidence('E3', f'{source}_C01', source)], 'risks': [], 'missing_information': []}
        state['market_traction_results'][company] = {'summary': '시장 분석', 'findings': [],
            'evidence': [], 'risks': [], 'missing_information': []}
    state['competition_result'] = {'target_market_context': {c: '산업용' for c in state['candidate_companies']},
        'differentiation': {c: '차별점' for c in state['candidate_companies']},
        'relative_risks': {c: '검증 한계' for c in state['candidate_companies']},
        'comparisons': [{'dimension': '차별성', 'criterion_ids': ['B04', 'B09'],
            'company_findings': {c: '관측된 차이' for c in state['candidate_companies']},
            'evidence_refs': [{'chunk_id': f'{source}_C01', 'source_id': source}
                              for source in ('DF', 'DA4', 'DX')]}]}
    return state
