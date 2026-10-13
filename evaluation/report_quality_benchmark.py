"""합성 의미 평가. 기본 입력 점검 / --live 명시 시 최대 4개 실제 LLM 평가."""
import argparse
import logging
from datetime import datetime, timezone
import uuid
from evaluation.report_quality_runner import run_quality, save_result


def run_cases(case_ids, *, live=False):
    from tests.qa_semantic_cases import CASES, semantic_fixture
    if not 1 <= len(case_ids) <= 4 or len(set(case_ids)) != len(case_ids) or any(c not in CASES for c in case_ids):
        raise ValueError("서로 다른 등록 사례 1~4개만 지정하세요.")
    rows = []
    for case_id in case_ids:
        state, originals, expected = semantic_fixture(case_id)
        result = run_quality(state, originals, live=live)
        match = None
        if live:
            checks = {c['item']: c for c in result.get('checks', [])}
            match = (result.get('passed') is True if expected is None else
                     result.get('passed') is False and checks.get(expected, {}).get('passed') is False)
            if result.get('status') == 'error' or not result.get('judge_used'):
                match = False
        rows.append({'case_id': case_id, 'expected_failed_criterion': expected,
                     'expectation_matched': match, 'result': result})
        if result.get('status') == 'error':
            break  # 호출 장애 시 남은 유료 호출을 중단한다.
    return {'synthetic': True, 'submission_evidence': False,
            'mode': 'live' if live else 'input_check', 'requested_cases': case_ids,
            'executed_cases': len(rows), 'cases': rows,
            'all_expectations_matched': all(r['expectation_matched'] for r in rows) if live and len(rows) == len(case_ids) else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['normal', 'supported_recommendation', 'demo_exaggeration', 'risk_omission'])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output-name')
    args = parser.parse_args(argv)
    # SDK 연결 오류 로그의 키 일부·요청 내용을 콘솔에 노출하지 않는다.
    # 실행 실패는 run_quality의 정제된 결과 JSON으로 확인한다.
    if args.live:
        logging.getLogger("langsmith.client").disabled = True
    try:
        result = run_cases(args.cases, live=args.live)
        name = args.output_name or f"synthetic_qa_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.json"
        path = save_result(result, name)
    except (ValueError, OSError):
        print('사례 목록 또는 출력 경로 오류. 등록 사례 1~4개와 새 파일명을 지정하세요.')
        return 2
    print(f"mode={result['mode']} executed={result['executed_cases']} matched={result['all_expectations_matched']} output={path}")
    if result['mode'] == 'live':
        return 0 if result['all_expectations_matched'] else 1
    return 0 if all(r['result'].get('input_ready') for r in result['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
