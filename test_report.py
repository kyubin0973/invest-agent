"""test_report.py: 실무 수준 데이터 기반 5쪽 PDF 보고서 생성 검증 스크립트"""
from agents.report_generator import report_generator_node
from core.state import InvestmentState

CRITERION_IDS = [f"B{str(i).zfill(2)}" for i in range(1, 11)] + ["H11", "H12"]

def build_criteria(base_score: int, comp_name: str = "Figure AI"):
    criteria = []
    for cid in CRITERION_IDS:
        ev_list = []
        if comp_name == "Figure AI" and cid == "B01":
            ev_list.append({
                "source_id": "rep_01",
                "title": "2026 글로벌 휴머노이드 로보틱스 시장 전망 보고서",
                "publisher": "한국로봇산업진흥원",
                "published_date": "2026-03-15",
                "page": 14,
                "reference_metadata": {"reference_type": "report", "url": "https://kiria.org/report/2026"}
            })
        elif comp_name == "Apptronik" and cid == "B04":
            ev_list.append({
                "source_id": "paper_02",
                "title": "Vision-Language-Action Models in Humanoid Manipulation",
                "publisher": "김로봇 외 3명",
                "published_date": "2026-01-20",
                "page": 45,
                "reference_metadata": {"reference_type": "paper", "journal": "IEEE Robotics and Automation Letters", "volume": "Vol.11, No.2"}
            })
        elif comp_name == "1X Technologies" and cid == "B06":
            ev_list.append({
                "source_id": "web_03",
                "title": "Figure AI and BMW Manufacturing Partnership Update",
                "publisher": "TechCrunch",
                "published_date": "2026-02-18",
                "page": 1,
                "reference_metadata": {"reference_type": "web", "site_name": "TechCrunch", "url": "https://techcrunch.com/2026/02/figure-bmw"}
            })
        criteria.append({
            "criterion_id": cid,
            "score": base_score,
            "status": "SCORED",
            "evidence": ev_list
        })
    return criteria

mock_state: InvestmentState = {
    "candidate_companies": ["Figure AI", "Apptronik", "1X Technologies"],
    "current_company": None,
    "current_company_index": 2,
    "company_profiles": {
        "Figure AI": {"target_market": "대규모 제조 및 물류 산업 공정 (산업용)", "funding_stage": "Series B", "eligibility": {"is_private": True}},
        "Apptronik": {"target_market": "물류 및 제조 현장 물품 이송/분류 (산업용)", "funding_stage": "Series A", "eligibility": {"is_private": True}},
        "1X Technologies": {"target_market": "가정용 일상 지원 및 가사 보조 서비스 (소비자용)", "funding_stage": "Series B", "eligibility": {"is_private": True}},
    },
    "technology_results": {
        "Figure AI": {
            "summary": "Helix 기반 비전-언어-행동(VLA) 통합 모델 및 자체 전동 액추에이터를 통한 완전 자율 휴머노이드 구현",
            "findings": [
                {"dimension": "Robot AI/HW Capability", "statement": "자체 개발 전동 액추에이터로 정밀 토크 제어 구현 및 자율 핸드-아이 코디네이션 확보.", "evidence_refs": []},
                {"dimension": "Technology Maturity", "statement": "공장 라인 10시간 이상 연속 가동 테스트를 통해 높은 반복 작업 정밀도 검증 완료.", "evidence_refs": []},
            ],
            "risks": ["대규모 액추에이터 양산 시 높은 초기 BOM 원가", "다양한 제조 환경 도입 시 비정형 물체 핸들링 한계"],
            "missing_information": ["장기 부품 내구성 테스트 데이터", "극저온/고온 환경 작동 신뢰성"],
            "evidence": []
        },
        "Apptronik": {
            "summary": "NASA 발키리 프로젝트 기반의 모듈형 하드웨어 설계 및 물류 상체 작업 최적화 이족보행 플랫폼",
            "findings": [
                {"dimension": "Robot AI/HW Capability", "statement": "모듈형 전원부와 팔 구조를 채택하여 공정 변화에 따른 유연한 하드웨어 재구성이 가능함.", "evidence_refs": []},
                {"dimension": "Technology Maturity", "statement": "창고 환경에서 박스 이송 및 리프팅 작업 중심의 파일럿 실증 수행 단계.", "evidence_refs": []},
            ],
            "risks": ["배터리 연속 가동 시간(약 2시간) 한계로 인한 교체 주기 문제", "보행 중 급격한 외란 발생 시 복구 알고리즘 미비"],
            "missing_information": ["차세대 배터리 팩 탑재 일정", "본계약 양산 전환용 부품 공급망 확보 내역"],
            "evidence": []
        },
        "1X Technologies": {
            "summary": "안전한 인간-로봇 상호작용(HRI)을 위한 저소음 고효율 케이블 구동계 및 바퀴/보행 하이브리드 로보틱스",
            "findings": [
                {"dimension": "Robot AI/HW Capability", "statement": "소프트 모터 및 케이블 텐션 구동으로 인간 충돌 시 부상 위험을 최소화한 설계 적용.", "evidence_refs": []},
                {"dimension": "Technology Maturity", "statement": "EVE 바퀴형 모델의 상용 보안 순찰 레퍼런스 보유, NEO 휴머노이드는 시제품 단계 검증.", "evidence_refs": []},
            ],
            "risks": ["가정 내 엄격한 안전/개인정보 보호 규제 승인 지연", "케이블 마모에 따른 주기적 유지보수 소요"],
            "missing_information": ["가정 내 복합 장애물 자율 회피 성공률", "소비자용 최종 판매 목표가(Target Price)"],
            "evidence": []
        },
    },
    "market_traction_results": {
        "Figure AI": {
            "summary": "BMW 사우스캐롤라이나 스파르탄버그 공장 실배치 및 6억 7,500만 달러 규모의 시리즈 B 투자 유치 완료",
            "findings": [
                {"dimension": "Commercialization", "statement": "글로벌 완성차 메이커 공장 현장에 실제 투입되어 차체 패널 조작 작업 파일럿 완료.", "evidence_refs": []},
                {"dimension": "Team Competence", "statement": "보스턴 다이내믹스, 테슬라, 구글 딥마인드 출신의 핵심 하드웨어/AI 엔지니어링 팀 구성.", "evidence_refs": []},
            ],
            "risks": ["고객사 단일 의존도(BMW) 완화를 위한 물류 고객 다변화 필요성"],
            "missing_information": ["대당 RaaS(Robot-as-a-Service) 과금 모델의 구체적 마진율"],
            "evidence": []
        },
        "Apptronik": {
            "summary": "메르세데스-벤츠 조립 공장 시범 배치 협약 및 미 육군 연구소(ARL) 물류 이송 프로젝트 수주",
            "findings": [
                {"dimension": "Commercialization", "statement": "벤츠 파일럿 라인에 투입되어 부품 키팅(Kitting) 작업 검증 중이나 정식 구매 계약 대기.", "evidence_refs": []},
                {"dimension": "Team Competence", "statement": "오스틴 텍사스대 로봇공학 연구진 기반으로 창업되어 국방 및 산업용 로봇 연구 경험 풍부.", "evidence_refs": []},
            ],
            "risks": ["시범 도입 후 대량 유료 전환 속도가 지연될 경우 런웨이 압박"],
            "missing_information": ["벤츠 파일럿 최종 완료 시점 및 본계약 발주 예상 수량"],
            "evidence": []
        },
        "1X Technologies": {
            "summary": "글로벌 사전 예약 물량 확보 및 OpenAI, EQT 등 탑티어 투자 유치 완료",
            "findings": [
                {"dimension": "Commercialization", "statement": "산업 시설 보안 로봇으로 상용 매출을 창출 중이며, 차세대 NEO 예약 고객군 구축.", "evidence_refs": []},
                {"dimension": "Team Competence", "statement": "기계공학 및 임베디드 제어 분야의 탄탄한 유럽 기반 엔지니어링 역량 보유.", "evidence_refs": []},
            ],
            "risks": ["일반 소비자 시장의 높은 가격 저항선 및 초기 A/S 망 구축 부담"],
            "missing_information": ["NEO 사전 예약 건수의 실제 유상 전환 보증금 비율"],
            "evidence": []
        },
    },
    "competition_result": {
        "target_market_context": {
            "Figure AI": "제조 및 물류 산업 공정",
            "Apptronik": "물류 및 제조 현장 물품 이송",
            "1X Technologies": "가정용 일상 지원 및 가사 보조"
        },
        "comparisons": [
            {
                "dimension": "Robot AI/HW Capability",
                "company_findings": {
                    "Figure AI": "Helix VLA와 대규모 양산용 전동 액추에이터로 정밀 제조 공정에 최적화",
                    "Apptronik": "NASA 발키리 헤리티지 기반의 모듈형 상체 물류 작업에 특화",
                    "1X Technologies": "소프트 케이블 구동계로 인간 접촉 안전성을 극대화한 가정용 설계"
                },
                "evidence_refs": []
            },
            {
                "dimension": "Technology Maturity",
                "company_findings": {
                    "Figure AI": "공장 라인 10시간 이상 연속 가동 테스트를 통해 높은 반복 작업 정밀도 검증",
                    "Apptronik": "창고 환경에서 박스 이송 및 리프팅 작업 중심의 파일럿 실증 수행 단계",
                    "1X Technologies": "EVE 바퀴형 모델의 상용 보안 순찰 레퍼런스 보유, NEO 휴머노이드는 시제품 단계"
                },
                "evidence_refs": []
            },
            {
                "dimension": "Commercialization & Early Response",
                "company_findings": {
                    "Figure AI": "BMW 상업 배치로 초기 시장 반응 우수하며 글로벌 양산 인프라 선점",
                    "Apptronik": "메르세데스-벤츠 시범 배치 및 미 육군 공급 계약으로 레퍼런스 확대 중",
                    "1X Technologies": "소비자용 사전 예약 확보로 높은 대중적 관심 유도 성공"
                },
                "evidence_refs": []
            },
            {
                "dimension": "Manufacturing & Scalability",
                "company_findings": {
                    "Figure AI": "초기 BOM 원가 부담이 있으나 규모의 경제 실현 시 높은 양산성 기대",
                    "Apptronik": "모듈형 구조로 생산 공정이 단순하나 배터리 지속 시간이 확장성 제약",
                    "1X Technologies": "부드러운 구동계로 제작비는 절감 가능하나 고하중 작업 제조 확장성 한계"
                },
                "evidence_refs": []
            },
            {
                "dimension": "Business Model & Strategy",
                "company_findings": {
                    "Figure AI": "완성차 및 대형 물류 엔터프라이즈 중심의 RaaS 모델 집중",
                    "Apptronik": "정부/국방 및 제조업체를 타깃으로 한 맞춤형 솔루션 판매 전략",
                    "1X Technologies": "B2B 보안 순찰 캐시카우를 기반으로 B2C 홈 로봇 구독 모델로 확장"
                },
                "evidence_refs": []
            }
        ],
        "differentiation": {
            "Figure AI": "BMW 상용 배치 및 독자 액추에이터 기반의 강력한 대량 양산 체계 확보",
            "Apptronik": "NASA 발키리 기반의 높은 모듈형 교체 유연성 및 국방/산업 맞춤 적용",
            "1X Technologies": "가정 내 인간 접촉 안전성을 극대화한 저소음 케이블 구동 메커니즘",
        },
        "relative_risks": {
            "Figure AI": "높은 초기 액추에이터 BOM 원가 회수 리스크",
            "Apptronik": "시범 도입 후 유료 본계약 전환 속도 지연 가능성",
            "1X Technologies": "가정 내 안전 규제 승인 지연 및 일반 소비자 가격 장벽",
        }
    },
    "investment_results": {
        "Figure AI": {
            "criteria": build_criteria(5, "Figure AI"),
            "final_score": 4.17,
            "evidence_coverage": 1.0,
            "decision": "INVEST",
            "decision_reason": "기술 완성도, 양산 체계, 완성차 고객 트랙션 모두 업계 최고 수준으로 즉시 투자 적격",
            "key_strengths": ["대규모 양산 액추에이터", "BMW 실제 배치 레퍼런스", "OpenAI 협업 VLA"],
            "key_risks": ["높은 초기 BOM 원가", "단일 고객 집중도"],
            "missing_information": ["장기 부품 내구성 검증 데이터"]
        },
        "Apptronik": {
            "criteria": build_criteria(3, "Apptronik"),
            "final_score": 3.00,
            "evidence_coverage": 1.0,
            "decision": "HOLD",
            "decision_reason": "기술적 유연성은 인정되나 시범 도입에서 유료 본계약으로의 전환 데이터 모니터링 필요",
            "key_strengths": ["NASA 헤리티지 모듈형 설계", "정부/국방 레퍼런스"],
            "key_risks": ["배터리 지속 시간 한계", "유료 전환 불확실성"],
            "missing_information": ["본계약 발주 확정 수량"]
        },
        "1X Technologies": {
            "criteria": build_criteria(3, "1X Technologies"),
            "final_score": 3.00,
            "evidence_coverage": 1.0,
            "decision": "HOLD",
            "decision_reason": "가정용 로봇 시장의 본격적인 개화 시점 및 안전 인증 불확실성으로 보류 판정",
            "key_strengths": ["안전 지향 케이블 구동", "B2C 대중 인지도"],
            "key_risks": ["가정 안전 규제 리스크", "소비자 가격 저항"],
            "missing_information": ["사전 예약의 실제 유상 전환율"]
        }
    },
    "final_route": "INVEST_FOUND",
    "completed_companies": ["Figure AI", "Apptronik", "1X Technologies"],
    "references": [],
    "final_report": None
}

output = report_generator_node(mock_state)
print(">>> 보고서 생성 성공!")
print("1. 생성된 PDF 경로:", output["report_path"])
print("2. 수집된 References 건수:", len(output["references"]))