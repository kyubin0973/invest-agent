"""test_competition.py
Competition Agent 단독 동작 검증용 테스트 스크립트
"""
import json
from agents.competition import competition_node
from core.state import InvestmentState

# 가짜 이전 분석 데이터 (앞선 Tech, Market 에이전트가 만든 형태)
mock_state: InvestmentState = {
    "candidate_companies": ["Figure AI", "Apptronik", "1X Technologies"],
    "current_company": "Figure AI",
    "company_profiles": {
        "Figure AI": {"target_market": "제조 및 물류 산업 공정"},
        "Apptronik": {"target_market": "물류 및 제조 현장 물품 이송"},
        "1X Technologies": {"target_market": "가정용 일상 지원 및 가사 보조"}
    },
    "technology_results": {
        "Figure AI": {
            "summary": "Helix 기반 VLA 및 대규모 양산용 전동 액추에이터 탑재",
            "findings": [
                {
                    "dimension": "Robot AI/HW Capability",
                    "statement": "자체 설계 전동 액추에이터와 VLA 적용으로 산업 환경 정밀 조작 수행",
                    "evidence_refs": [{"chunk_id": "fig_tech_01", "source_id": "figure_whitepaper"}]
                }
            ],
            "evidence": [],
            "risks": ["높은 액추에이터 BOM 원가"],
            "missing_information": []
        },
        "Apptronik": {
            "summary": "NASA 발키리 기반 모듈형 설계 및 상체 중심 물류 작업 특화",
            "findings": [
                {
                    "dimension": "Robot AI/HW Capability",
                    "statement": "Apollo 모델을 통한 박스 이송 및 유연한 페이로드 대응",
                    "evidence_refs": [{"chunk_id": "app_tech_01", "source_id": "apptronik_doc"}]
                }
            ],
            "evidence": [],
            "risks": ["배터리 가동 시간의 제한"],
            "missing_information": []
        },
        "1X Technologies": {
            "summary": "EVE 바퀴형 및 NEO 이족보행 기반 저소음 부드러운 구동계",
            "findings": [
                {
                    "dimension": "Robot AI/HW Capability",
                    "statement": "인간과의 접촉 안전성을 극대화한 케이블 구동 및 World Model 적용",
                    "evidence_refs": [{"chunk_id": "1x_tech_01", "source_id": "1x_overview"}]
                }
            ],
            "evidence": [],
            "risks": ["산업용 고하중 작업 대비 페이로드 부족"],
            "missing_information": []
        }
    },
    "market_traction_results": {
        "Figure AI": {
            "summary": "BMW 상업 배치 및 대규모 후속 투자 유치",
            "findings": [],
            "evidence": [],
            "risks": ["양산 단가 회수 기간"],
            "missing_information": []
        },
        "Apptronik": {
            "summary": "메르세데스-벤츠 시범 배치 및 미 육군 공급 계약",
            "findings": [],
            "evidence": [],
            "risks": ["유료 본계약 전환 속도"],
            "missing_information": []
        },
        "1X Technologies": {
            "summary": "소비자용 사전 예약 확보 및 EVE 보안 순찰 상용화 실적",
            "findings": [],
            "evidence": [],
            "risks": ["가정 내 안전 규제 및 높은 가격 장벽"],
            "missing_information": []
        }
    },
    "technology_done": True,
    "market_traction_done": True,
    "competition_result": {},
    "criteria_evaluations": {},
    "final_decision": {},
    "report_path": ""
}

print(">>> Competition Agent 실행 중... (LLM 호출)")
result = competition_node(mock_state)

print("\n=== 실행 결과 요약 ===")
print("반환된 최상위 키:", list(result.keys()))

comp_res = result.get("competition_result", {})
print("1. Target Market Context 확인:")
print(json.dumps(comp_res.get("target_market_context"), indent=2, ensure_ascii=False))

print("\n2. 기업별 차별성 (B04 연계):")
print(json.dumps(comp_res.get("differentiation"), indent=2, ensure_ascii=False))

print("\n3. 기업별 상대 리스크 (B09 연계):")
print(json.dumps(comp_res.get("relative_risks"), indent=2, ensure_ascii=False))

print("\n4. 비교 항목 수:", len(comp_res.get("comparisons", [])))

# test_competition.py 맨 아래에 추가
print("\n5. 5대 비교 항목(comparisons) 세부 내용 검증:")
for idx, comp_item in enumerate(comp_res.get("comparisons", []), 1):
    print(f"\n[{idx}] 비교축: {comp_item.get('dimension')}")
    print("  - 기업별 내용:")
    for company, finding_text in comp_item.get("company_findings", {}).items():
        print(f"    * {company}: {finding_text}")
    print(f"  - evidence_refs: {comp_item.get('evidence_refs')}")