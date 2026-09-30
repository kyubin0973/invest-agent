"""앞단 Agent의 [DEMO] 결과로 세 기업 Investment Judge를 실제 LLM 호출 테스트한다.

실행:
    uv run python -m scripts.run_judge_demo

이 데이터는 코드 경로 검증용이며 실제 기업 분석 결과가 아니다.
"""

import json
import time
from pathlib import Path

from agents.investment_judge import investment_judge_node
from core import tracing
from core.judge_logging import configure_judge_logging

COMPANIES = ["Figure AI", "Apptronik", "1X Technologies"]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _evidence(
    source_id: str,
    chunk_id: str,
    level: str,
    fact: str,
    *,
    source_type: str,
) -> dict:
    return {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "page": 1,
        "title": f"[DEMO] {source_id} synthetic source",
        "publisher": "[DEMO] Synthetic Publisher",
        "published_date": "2026-09-30",
        "source_type": source_type,
        "reference_metadata": {
            "reference_type": "web",
            "url": f"https://example.invalid/demo/{source_id.lower()}",
            "site_name": "[DEMO] Synthetic Source",
        },
        "evidence_level": level,
        "fact": f"[DEMO - 실제 기업 사실 아님] {fact}",
    }


def _analysis(summary: str, evidence: list[dict], risks: list[str], missing: list[str]) -> dict:
    return {
        "summary": f"[DEMO] {summary}",
        "findings": [
            {
                "dimension": "DEMO input",
                "statement": item["fact"],
                "evidence_refs": [{"chunk_id": item["chunk_id"], "source_id": item["source_id"]}],
            }
            for item in evidence
        ],
        "evidence": evidence,
        "risks": [f"[DEMO] {risk}" for risk in risks],
        "missing_information": [f"[DEMO] {item}" for item in missing],
    }


def _demo_evidence() -> dict[str, dict[str, list[dict]]]:
    return {
        "Figure AI": {
            "technology": [
                _evidence(
                    "DF1",
                    "DF1_C01",
                    "E2",
                    "기업 자료에서 제한된 공장 환경의 이동·조작 Demo를 제시했지만 장시간 반복성과 실패 복구 수치는 없다.",
                    source_type="company",
                ),
                _evidence(
                    "DF2",
                    "DF2_C01",
                    "E1",
                    "기업이 범용 AI와 Robot Data 확장 계획을 발표했으나 독립 성능 비교는 제공되지 않았다.",
                    source_type="company",
                ),
                _evidence(
                    "DF3",
                    "DF3_C01",
                    "E1",
                    "대규모 생산 목표를 발표했지만 실제 생산량·BOM·Fleet 유지보수 실적은 공개하지 않았다.",
                    source_type="company",
                ),
            ],
            "market": [
                _evidence(
                    "DM1",
                    "DM1_C01",
                    "E3",
                    "독립 산업자료가 제조·물류 휴머노이드 시장의 성장 가능성과 노동력 부족 수요를 확인했다.",
                    source_type="industry_report",
                ),
                _evidence(
                    "DF4",
                    "DF4_C01",
                    "E3",
                    "파트너 자료가 제한된 제조 Pilot 진행과 현장 작업 검증을 확인했지만 유료계약 규모는 공개하지 않았다.",
                    source_type="partner",
                ),
                _evidence(
                    "DF5",
                    "DF5_C01",
                    "E1",
                    "기업이 산업 고객 대상 서비스 모델을 언급했지만 가격·TCO·구체적인 반복 매출 구조는 확인되지 않았다.",
                    source_type="company",
                ),
            ],
        },
        "Apptronik": {
            "technology": [
                _evidence(
                    "DA1",
                    "DA1_C01",
                    "E3",
                    "외부 파트너가 실제 제조 작업 조건에서 반복 조작 시험과 안전 절차 검증을 확인했다.",
                    source_type="partner",
                ),
                _evidence(
                    "DA2",
                    "DA2_C01",
                    "E4",
                    "제조 파트너가 생산 라인 통합과 현장 운영을 확인했으며 정비·운영 담당 체계를 함께 설명했다.",
                    source_type="partner",
                ),
                _evidence(
                    "DA3",
                    "DA3_C01",
                    "E3",
                    "독립 시험 자료가 특정 작업에서 반복 수행 결과를 확인했으나 여러 산업으로의 일반화는 검증하지 않았다.",
                    source_type="partner",
                ),
            ],
            "market": [
                _evidence(
                    "DM2",
                    "DM2_C01",
                    "E3",
                    "독립 산업자료가 제조 자동화 수요와 접근 가능한 초기 시장을 확인했다.",
                    source_type="industry_report",
                ),
                _evidence(
                    "DA4",
                    "DA4_C01",
                    "E4",
                    "고객 자료가 유료 현장 운영과 후속 배치 검토를 확인했지만 계약 금액은 공개하지 않았다.",
                    source_type="partner",
                ),
                _evidence(
                    "DA5",
                    "DA5_C01",
                    "E1",
                    "기업 자료는 판매와 서비스 수익을 함께 제시하지만 가격·마진·장기 유지보수 원가는 공개하지 않았다.",
                    source_type="company",
                ),
            ],
        },
        "1X Technologies": {
            "technology": [
                _evidence(
                    "DX1",
                    "DX1_C01",
                    "E2",
                    "기업 Demo에서 가정 환경 이동과 단순 물체 조작을 보였으나 비정형 환경의 반복 성공률은 없다.",
                    source_type="company",
                ),
                _evidence(
                    "DX2",
                    "DX2_C01",
                    "E1",
                    "기업이 가정용 학습 데이터와 원격 지원 전략을 설명했지만 독립적인 안전성·복구 검증은 없다.",
                    source_type="company",
                ),
                _evidence(
                    "DX3",
                    "DX3_C01",
                    "E1",
                    "향후 생산 확대 계획은 있으나 현재 생산능력·원가·대규모 Fleet 운영 Evidence는 제공되지 않았다.",
                    source_type="company",
                ),
            ],
            "market": [
                _evidence(
                    "DM3",
                    "DM3_C01",
                    "E3",
                    "독립 시장자료는 가정 보조 로봇의 장기 잠재력과 함께 가격·안전·사용자 수용성 불확실성을 지적했다.",
                    source_type="industry_report",
                ),
                _evidence(
                    "DX4",
                    "DX4_C01",
                    "E1",
                    "기업이 초기 가정 테스트 참가자를 모집한다고 발표했지만 외부 확인된 유료 고객이나 반복 배치는 없다.",
                    source_type="company",
                ),
                _evidence(
                    "DX5",
                    "DX5_C01",
                    "E1",
                    "구독형 서비스 방향은 발표했으나 가격·TCO·유지보수 책임과 수익성 자료가 없다.",
                    source_type="company",
                ),
            ],
        },
    }


def build_demo_state() -> dict:
    evidence = _demo_evidence()
    target_markets = {
        "Figure AI": "[DEMO] 제조·물류 중심 산업용 시장",
        "Apptronik": "[DEMO] 제조 현장 중심 산업용 시장",
        "1X Technologies": "[DEMO] 가정용 보조 로봇 시장",
    }
    profiles = {
        company: {
            "name": company,
            "target_market": target_markets[company],
            "funding_stage": "[DEMO] 미확정",
            "eligibility": {
                "is_private": True,
                "exit_completed": False,
                "evaluation_date": "2026-09-30",
            },
            "document_scope": {
                "source_ids": [item["source_id"] for group in evidence[company].values() for item in group],
                "document_types": ["[DEMO] synthetic"],
            },
            "source_refs": {},
        }
        for company in COMPANIES
    }
    technology_results = {
        company: _analysis(
            f"{company} 기술·제품 더미 분석",
            evidence[company]["technology"],
            ["실환경 장기 안정성과 실패 복구 검증이 제한적임"],
            ["반복 성공률", "장시간 운영 데이터"],
        )
        for company in COMPANIES
    }
    market_results = {
        company: _analysis(
            f"{company} 시장·Traction 더미 분석",
            evidence[company]["market"],
            ["가격·원가·계약 규모 정보가 제한적임"],
            ["가격과 TCO", "계약 금액", "Founder 장기 Commitment 근거"],
        )
        for company in COMPANIES
    }

    return {
        "candidate_companies": COMPANIES,
        "company_profiles": profiles,
        "current_company": None,
        "current_company_index": 0,
        "technology_results": technology_results,
        "market_traction_results": market_results,
        "competition_result": {
            "target_market_context": target_markets,
            "comparisons": [
                {
                    "dimension": "[DEMO] 상용화와 기술 성숙도",
                    "criterion_ids": ["B04", "B09"],
                    "company_findings": {
                        "Figure AI": "[DEMO] 외부 확인된 Pilot이 있으나 장기 반복 운영은 미확인",
                        "Apptronik": "[DEMO] 고객·제조 파트너가 실제 현장 운영을 확인",
                        "1X Technologies": "[DEMO] 가정 환경 Demo 중심이며 외부 확인된 운영은 미확인",
                    },
                    "evidence_refs": [
                        {"chunk_id": "DF4_C01", "source_id": "DF4"},
                        {"chunk_id": "DA4_C01", "source_id": "DA4"},
                        {"chunk_id": "DX4_C01", "source_id": "DX4"},
                    ],
                }
            ],
            "differentiation": {
                "Figure AI": "[DEMO] 범용 AI·산업용 전략",
                "Apptronik": "[DEMO] 제조 파트너 연계 현장 배치 전략",
                "1X Technologies": "[DEMO] 가정용 Target Market과 원격 지원 전략",
            },
            "relative_risks": {
                "Figure AI": "[DEMO] 생산 목표와 실제 규모화 실적의 간극",
                "Apptronik": "[DEMO] 공개된 가격·마진 정보 부족",
                "1X Technologies": "[DEMO] 가정 시장의 안전·가격·수용성 불확실성",
            },
        },
        "investment_results": {},
        "completed_companies": [],
    }


def main() -> None:
    tracing.setup()
    configure_judge_logging()
    state = build_demo_state()

    for index, company in enumerate(COMPANIES):
        state["current_company_index"] = index
        state["current_company"] = company
        update = investment_judge_node(state)
        state["investment_results"].update(update["investment_results"])
        state["completed_companies"] = update["completed_companies"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"judge_demo_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(
        json.dumps(
            {
                "warning": "DEMO synthetic evidence; not an actual company evaluation",
                "investment_results": state["investment_results"],
                "completed_companies": state["completed_companies"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[DEMO] 결과 저장: {output_path.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
