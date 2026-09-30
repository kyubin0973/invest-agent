"""투자 판단 기준 (설계 산출물 3장)

Bessemer 핵심 질문 10개를 범용 휴머노이드에 맞게 구체화하고, 휴머노이드 특화 2개 문항을 추가한 12문항.
- 모든 문항은 동일 중요도 (가중치 없음). Investment Judge가 1~5점 또는 N/A로 판단한다.
- 식별자: Bessemer 문항 B01~B10, 휴머노이드 특화 문항 H11~H12 (Q1~Q12와 일대일)
- 점수 부여 책임은 모두 Investment Judge에 있다. Analysis Agent는 근거를 제공하며
  공개 Team·Founder·가격 정보가 부족하면 추정하지 않고 N/A로 처리한다.
- FinalScore·Coverage·Decision은 LLM이 아니라 이 모듈의 Python 규칙으로 계산한다.
"""

from dataclasses import dataclass

TECH = "technology"
MARKET = "market_traction"
COMPETITION = "competition"


@dataclass(frozen=True)
class Criterion:
    id: str  # B01~B10, H11~H12
    q: str  # Q1~Q12 (보고서 표기)
    name: str
    bessemer_question: str | None  # 휴머노이드 특화 문항은 None
    question: str  # 도메인 구체화 질문
    evidence_focus: tuple[str, ...]  # 핵심 Evidence
    agents: tuple[str, ...]  # 근거를 제공하는 Agent


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "B01", "Q1", "Market Size",
        "이 시장은 얼마나 큰가?",
        "해당 기업이 진입하려는 휴머노이드 시장의 규모와 성장 가능성이 충분한가?",
        ("시장 규모", "성장", "적용 산업", "노동력 부족", "자동화 수요"),
        (MARKET,),
    ),
    Criterion(
        "B02", "Q2", "Problem / Product Fit",
        "실제 문제를 해결하는가?",
        "기존 인력·자동화 방식으로 해결하기 어려운 실제 문제를 해결하는가?",
        ("Use Case", "작업 능력", "고객 문제", "Demo/Pilot/Deployment"),
        (TECH, MARKET),
    ),
    Criterion(
        "B03", "Q3", "Willingness to Pay",
        "고객이 돈을 지불할 이유가 있는가?",
        "고객이 기존 인력·산업로봇 대비 도입할 경제적·운영상 이유가 명확한가?",
        ("유료 계약", "가격", "TCO", "ROI", "도입 근거"),
        (MARKET,),
    ),
    Criterion(
        "B04", "Q4", "Differentiation",
        "경쟁사보다 차별성이 있는가?",
        "경쟁사 대비 기술·성능·가격·데이터·적용분야의 뚜렷한 차별성이 있는가?",
        ("AI", "Data", "HW", "Capability", "제조전략", "비교"),
        (COMPETITION, TECH),
    ),
    Criterion(
        "B05", "Q5", "Team",
        "창업자와 팀은 믿을 만한가?",
        "창업자와 핵심 인력이 AI·로보틱스·HW·양산 전문성과 실행력을 갖췄는가?",
        ("관련 경력", "기술/제조 경험", "사업화", "실행 실적"),
        (MARKET,),
    ),
    Criterion(
        "B06", "Q6", "Early Customer Response",
        "초기 고객 반응은 어떠한가?",
        "PoC·파트너십·실제 배치·유료 고객 등 초기 시장 검증이 있는가?",
        ("Pilot", "Deployment", "고객", "계약", "반복 배치"),
        (MARKET,),
    ),
    Criterion(
        "B07", "Q7", "Business Model",
        "수익모델은 명확한가?",
        "판매·RaaS·Software·유지보수 등 지속 가능한 수익모델이 명확한가?",
        ("가격", "수익구조", "고객군", "유지보수"),
        (MARKET,),
    ),
    Criterion(
        "B08", "Q8", "Upside Potential",
        "성공하면 큰 기회인가?",
        "성공할 경우 여러 작업·고객·산업으로 확장 가능한 큰 기회인가?",
        ("작업·산업·Platform 확장성",),
        (MARKET, TECH),
    ),
    Criterion(
        "B09", "Q9", "Risk",
        "기술·운영·법률 리스크는?",
        "기술·안전·양산·원가·규제·고객수용 Risk가 관리 가능한가?",
        ("Risk의 중대성", "관리 가능성"),
        (TECH, MARKET, COMPETITION),
    ),
    Criterion(
        "B10", "Q10", "Founder Commitment",
        "창업자가 10년을 투자할 각오인가?",
        "장기간의 기술개발·양산·시장확대를 수행할 의지와 실행 근거가 있는가?",
        ("장기 활동", "Founder 경력", "경영진 지속성"),
        (MARKET,),
    ),
    Criterion(
        "H11", "Q11", "Technology Maturity",
        None,
        "Demo를 넘어 실제 환경에서 반복 작업 가능한 성숙도를 확보했는가?",
        ("Autonomy", "일반화", "신뢰성", "실패복구", "장기운영"),
        (TECH,),
    ),
    Criterion(
        "H12", "Q12", "Manufacturing & Deployment",
        None,
        "경제적으로 생산하고 대규모 배치·운영할 확장성을 갖췄는가?",
        ("생산능력", "파트너", "BOM", "공급망", "Fleet", "유지보수"),
        (TECH, MARKET),
    ),
)

CRITERIA_BY_ID = {c.id: c for c in CRITERIA}

# 문항 경계: 중복 평가 방지 규칙 (설계 3.1)
CRITERION_BOUNDARIES = (
    "Q1/Q4: Q1은 각 기업의 실제 Target Market 규모, Q4는 경쟁 차별성. 시장이 다르다는 사실만으로 열위 처리하지 않는다.",
    "Q5/Q10: Q5는 전문성·실행역량·과거 실적, Q10은 장기간 관여·경영진 연속성. 같은 경력 사실을 두 문항에 같은 의미로 반영하지 않는다.",
    "공통 시장자료는 시장기회 판단에만 사용한다. 기업별 역량·고객·성과는 해당 기업 Evidence가 필요하다.",
)

# ---------------------------------------------------------------------------
# 근거 수준·점수·신뢰도 (설계 3.3)
# ---------------------------------------------------------------------------
EVIDENCE_LEVELS = {
    "E0": "없음 - 판단 가능한 자료 없음",
    "E1": "주장·계획 - 기업 자체 주장, 목표, 향후 계획",
    "E2": "시연 - Prototype, Demo, 제한 환경 테스트",
    "E3": "외부 확인 - 고객·파트너·독립 자료의 외부 확인",
    "E4": "실제 운영·계약 - 고객·파트너·독립 출처로 확인된 실제 배치·운영·계약",
    "E5": "반복·규모화 - 고객·파트너·독립 출처로 확인된 복수 고객·환경의 반복 또는 규모화",
}
EVIDENCE_LEVEL_RULE = (
    "기업 단독 발표의 운영·계약 주장은 외부 확인 전까지 E1~E2로 관리한다. "
    "Evidence Level은 근거의 검증 수준이며 점수로 자동 변환하지 않는다."
)

SCORE_RUBRIC = {
    5: "강한 복수 근거·외부검증 (운영·배치 문항은 반복 실적까지 확인)",
    4: "충분한 긍정 근거와 외부검증 (운영·배치 문항은 실제 적용 근거 확인)",
    3: "초기단계·기업자료 중심·검증범위 제한, 또는 긍정·부정 근거 혼재",
    2: "자료 부족이 아니라 실제 근거에서 실질적 약점 확인",
    1: "높은 검증 수준의 근거에서 중대한 부정 요인 확인",
    None: "N/A - 판단 근거 부족 (0점이나 1점으로 계산하지 않음)",
}
SCORE_RULE = "Missing Evidence ≠ Negative Evidence. Q9 Risk도 점수가 높을수록 Risk가 관리 가능하다는 긍정 방향이다."

CONFIDENCE_LEVELS = {
    "High": "복수 독립 근거와 외부검증 또는 운영 결과 존재",
    "Medium": "판단 가능한 근거는 있으나 기업자료 비중이 크거나 검증범위 제한",
    "Low": "근거가 적거나 초기·소수 자료 중심",
}

# ---------------------------------------------------------------------------
# 계산과 최종 판단 규칙 (설계 3.4) - Prototype 운영 기준이며 업계 표준으로 주장하지 않는다
# ---------------------------------------------------------------------------
INVEST_SCORE_THRESHOLD = 3.5  # Rubric 3점(Partially Supported)과 4점(Supported)의 경계
MIN_SCORED_CRITERIA = 9  # Coverage 70% 이상 = 12문항 중 최소 9개 (9/12 = 75.0%)


def criteria_for(agent: str) -> list[Criterion]:
    """해당 Agent가 근거를 제공해야 하는 문항."""
    return [c for c in CRITERIA if agent in c.agents]


def validate_criteria(criteria: list[dict], available_evidence: dict[str, dict]) -> list[dict]:
    """LLM이 낸 문항별 결과를 Python으로 검증한다 (설계 3.4, 21쪽본 5.5).

    - 문항 완전성: 12문항이 각 1개씩. 누락 문항은 INSUFFICIENT_EVIDENCE / score=None으로 채운다.
    - Evidence ID: 앞선 Agent가 실제 검색해 전달한 Evidence(available_evidence, chunk_id → EvidenceItem)에
      존재하고 source_id가 일치하는 참조만 남긴다. LLM이 쓴 Metadata 대신 원본 EvidenceItem을 연결한다.
    - score/status: SCORED면 정수 1~5, INSUFFICIENT_EVIDENCE면 None.
      유효 Evidence가 남지 않거나 규칙에 맞지 않으면 None / INSUFFICIENT_EVIDENCE로 바꾸고 사유를 기록한다.

    criteria 항목의 evidence는 [{"chunk_id", "source_id"}] 참조 또는 EvidenceItem이어도 된다.
    """
    by_id = {c.get("criterion_id"): c for c in criteria}
    validated = []
    for crit in CRITERIA:
        raw = by_id.get(crit.id)
        missing = list(raw.get("missing_information", [])) if raw else []
        if raw is None:
            validated.append(_insufficient(crit, "LLM 출력에 문항 결과가 없음"))
            continue

        evidence = []
        for ref in raw.get("evidence", []):
            item = available_evidence.get(ref.get("chunk_id"))
            if item and item["source_id"] == ref.get("source_id"):
                evidence.append(item)

        score, status = raw.get("score"), raw.get("status")
        valid_score = isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 5
        reason = None
        if status == "SCORED" and not valid_score:
            reason = f"유효하지 않은 점수: {score!r}"
        elif status == "SCORED" and not evidence:
            reason = "검증된 Evidence가 없음"
        elif status not in ("SCORED", "INSUFFICIENT_EVIDENCE"):
            reason = f"유효하지 않은 status: {status!r}"

        if reason or status == "INSUFFICIENT_EVIDENCE":
            validated.append({
                **_insufficient(crit, reason),
                "evidence": evidence,
                "reasoning": raw.get("reasoning", ""),
                "missing_information": missing + ([reason] if reason else []),
            })
        else:
            validated.append({
                "criterion_id": crit.id,
                "criterion_name": crit.name,
                "score": score,
                "status": "SCORED",
                "evidence": evidence,
                "reasoning": raw.get("reasoning", ""),
                "confidence": raw.get("confidence", "Low"),
                "missing_information": missing,
            })
    return validated


def _insufficient(crit: Criterion, reason: str | None) -> dict:
    return {
        "criterion_id": crit.id,
        "criterion_name": crit.name,
        "score": None,
        "status": "INSUFFICIENT_EVIDENCE",
        "evidence": [],
        "reasoning": "",
        "confidence": "Low",
        "missing_information": [reason] if reason else [],
    }


def decide(scores: dict[str, int | None]) -> dict:
    """문항별 점수(1~5, N/A는 None)로 FinalScore·EvidenceCoverage·Decision을 계산한다.

    FinalScore       = 유효한 1~5점의 단순 평균 (N/A 제외, 평가 가능한 문항이 없으면 None)
    EvidenceCoverage = N_scored / 12
    N_scored < 9                       → HOLD_INSUFFICIENT_EVIDENCE
    N_scored ≥ 9 AND FinalScore ≥ 3.5  → INVEST
    N_scored ≥ 9 AND FinalScore < 3.5  → HOLD
    규칙은 표시용 반올림 이전 값으로 적용한다.
    """
    missing = set(CRITERIA_BY_ID) - set(scores)
    if missing:
        raise ValueError(f"점수가 없는 문항: {sorted(missing)}")

    scored = [s for s in scores.values() if s is not None]
    n_scored = len(scored)
    final_score = sum(scored) / n_scored if scored else None

    if n_scored < MIN_SCORED_CRITERIA:
        decision = "HOLD_INSUFFICIENT_EVIDENCE"
        reason = f"평가 가능 문항 {n_scored}/12 < {MIN_SCORED_CRITERIA} (Evidence Coverage 70% 미만)"
    elif final_score >= INVEST_SCORE_THRESHOLD:
        decision = "INVEST"
        reason = f"평가 가능 문항 {n_scored}/12, Final Score {final_score:.2f} ≥ {INVEST_SCORE_THRESHOLD}"
    else:
        decision = "HOLD"
        reason = f"평가 가능 문항 {n_scored}/12, Final Score {final_score:.2f} < {INVEST_SCORE_THRESHOLD}"

    return {
        "final_score": None if final_score is None else round(final_score, 2),
        "evidence_coverage": round(n_scored / len(CRITERIA), 3),
        "decision": decision,
        "decision_reason": reason,
    }
