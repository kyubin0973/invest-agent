"""투자 평가 기준: Bessemer's Checklist 10개 (휴머노이드용으로 재작성) + 추가 2개

- 12개 항목은 동일한 중요도를 가진다 (가중치 없음).
- 분석 Agent는 Evidence를 만들고, Investment Judge만 점수를 매긴다.
- 점수: 1~5 (Score Rubric), 판단 불가 시 N/A(None). N/A는 0점이나 1점으로 처리하지 않는다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    bessemer_question: str | None  # 추가 항목은 None
    question: str  # 휴머노이드 스타트업 평가용 질문
    evidence_focus: tuple[str, ...]
    agents: tuple[str, ...]  # Evidence를 제공하는 Agent


TECH = "technology"
MARKET = "market_traction"
COMPETITION = "competition"

CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "B01", "Market Size",
        "이 시장은 얼마나 큰가?",
        "휴머노이드 로봇이 진입하려는 시장의 규모와 성장 가능성이 충분한가?",
        ("시장 규모", "성장률", "적용 산업", "노동·자동화 수요"),
        (MARKET,),
    ),
    Criterion(
        "B02", "Problem / Product Fit",
        "실제 문제를 해결하는가?",
        "휴머노이드가 기존 인력·자동화 방식으로 해결하기 어려운 실제 문제를 해결하는가?",
        ("Use Case", "작업 수행 능력", "Demo/Pilot/Deployment"),
        (TECH, MARKET),
    ),
    Criterion(
        "B03", "Willingness to Pay",
        "고객이 돈을 지불할 이유가 있는가?",
        "고객이 기존 인력·산업로봇 대비 휴머노이드를 도입할 경제적·운영상 이유가 명확한가?",
        ("유료 계약", "가격", "TCO/ROI", "고객 도입 근거"),
        (MARKET,),
    ),
    Criterion(
        "B04", "Differentiation",
        "경쟁사보다 차별성이 있는가?",
        "경쟁 휴머노이드 대비 기술·성능·가격·데이터·적용분야에서 뚜렷한 차별성이 있는가?",
        ("AI", "하드웨어 성능", "가격", "데이터", "적용 분야"),
        (COMPETITION, TECH),
    ),
    Criterion(
        "B05", "Team",
        "창업자와 팀은 믿을 만한가?",
        "창업자와 핵심 인력이 AI·로보틱스·하드웨어·양산 분야에서 충분한 전문성과 실행력을 갖췄는가?",
        ("경력", "산업·기술 경험", "실행 기록"),
        (MARKET,),
    ),
    Criterion(
        "B06", "Early Customer Response",
        "초기 고객 반응은 어떠한가?",
        "PoC·파트너십·실제 배치·유료 고객 등 초기 시장 검증이 이루어지고 있는가?",
        ("PoC", "파트너십", "실제 배치", "유료 고객"),
        (MARKET,),
    ),
    Criterion(
        "B07", "Business Model",
        "수익모델은 명확한가?",
        "로봇 판매·RaaS·소프트웨어·유지보수 등 지속 가능한 수익모델이 명확한가?",
        ("판매", "RaaS", "소프트웨어", "유지보수", "고객군"),
        (MARKET,),
    ),
    Criterion(
        "B08", "Upside Potential",
        "성공하면 큰 기회인가?",
        "기술과 사업모델이 성공할 경우 여러 작업·산업으로 확장 가능한 큰 사업기회가 존재하는가?",
        ("작업 확장성", "산업 확장성"),
        (MARKET, TECH),
    ),
    Criterion(
        "B09", "Risk",
        "기술·운영·법률 리스크는?",
        "기술 안정성·안전·양산·원가·규제·고객 수용성 등 핵심 리스크가 관리 가능한 수준인가?",
        ("기술 안정성", "안전", "양산", "원가", "규제", "고객 수용성"),
        (TECH, MARKET, COMPETITION),
    ),
    Criterion(
        "B10", "Founder Commitment",
        "창업자가 10년을 투자할 각오인가?",
        "경영진이 장기간 필요한 기술개발·양산·시장확대 과정을 수행할 의지와 실행 근거를 보여주는가?",
        ("장기 활동", "관련 경력", "경영진 지속성"),
        (MARKET,),
    ),
    Criterion(
        "B11", "Technology Maturity",
        None,
        "휴머노이드가 데모 수준을 넘어 실제 환경에서 반복적으로 작업할 수 있는 기술 성숙도를 확보했는가?",
        ("실환경 반복 작업", "자율성 수준", "작업 성공률·가동 시간", "Demo vs Deployment"),
        (TECH,),
    ),
    Criterion(
        "B12", "Manufacturing & Scalability",
        None,
        "로봇을 경제적인 비용으로 생산하고 대규모 배치할 수 있는 양산·운영 확장성을 갖췄는가?",
        ("양산 파트너·설비", "생산 능력", "원가(Unit Economics)", "공급망", "배치·운영 확장"),
        (TECH, MARKET),
    ),
)

CRITERIA_BY_ID = {c.id: c for c in CRITERIA}

# Prototype 운영 기준 (업계 표준이 아닌 본 Prototype의 기준)
INVEST_SCORE_THRESHOLD = 3.5  # Partially Supported(3)와 Supported(4)의 경계
MIN_EVIDENCE_COVERAGE = 0.7  # 12개 중 최소 9개 항목에서 판단 가능한 Evidence 필요


def criteria_for(agent: str) -> list[Criterion]:
    """해당 Agent가 Evidence를 제공해야 하는 평가 항목."""
    return [c for c in CRITERIA if agent in c.agents]


def decide(scores: dict[str, int | None]) -> dict:
    """항목별 점수(1~5, 판단 불가는 None)로 INVEST / HOLD를 결정한다.

    FinalScore = 판단 가능한 항목의 단순 평균
    EvidenceCoverage = 판단 가능한 항목 수 / 전체 항목 수
    """
    missing = set(CRITERIA_BY_ID) - set(scores)
    if missing:
        raise ValueError(f"점수가 없는 항목: {sorted(missing)}")

    scored = [s for s in scores.values() if s is not None]
    coverage = len(scored) / len(CRITERIA)
    final_score = sum(scored) / len(scored) if scored else None

    if coverage < MIN_EVIDENCE_COVERAGE:
        decision, reason = "HOLD", "Insufficient Evidence"
    elif final_score >= INVEST_SCORE_THRESHOLD:
        decision, reason = "INVEST", f"Final Score {final_score:.2f} ≥ {INVEST_SCORE_THRESHOLD}"
    else:
        decision, reason = "HOLD", f"Final Score {final_score:.2f} < {INVEST_SCORE_THRESHOLD}"

    return {
        "final_score": None if final_score is None else round(final_score, 2),
        "evidence_coverage": round(coverage, 3),
        "decision": decision,
        "decision_reason": reason,
    }
