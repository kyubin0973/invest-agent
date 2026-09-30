"""투자 판단 기준 (설계 산출물 3장)

Bessemer 핵심 질문 10개를 범용 휴머노이드에 맞게 구체화하고, 휴머노이드 특화 2개 문항을 추가한 12문항.
- 모든 문항은 동일 중요도 (가중치 없음). Investment Judge가 1~5점 척도 중 1·3·5점 또는 N/A로 판단한다.
  (2·4점은 경계가 모호해 사용하지 않는다)
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

ALLOWED_SCORES = (1, 3, 5)

SCORE_RUBRIC = {
    5: "해당 문항을 지지하는 외부 검증 근거(E3 이상)가 있음. 운영·배치 문항은 실제 적용 또는 반복 실적까지 확인됨",
    3: "긍정 신호는 있으나 기업자료(E1~E2) 중심이거나 검증범위가 제한적임. 또는 긍정·부정 근거가 혼재함",
    1: "자료 부족이 아니라 실제 근거에서 실질적인 약점 또는 부정 요인이 확인됨",
    None: "N/A - 판단 근거 부족 (0점이나 1점으로 계산하지 않음)",
}
SCORE_RULE = (
    "Missing Evidence ≠ Negative Evidence. 근거가 없으면 1점이 아니라 N/A이다. "
    "약점과 강점이 섞여 있으면 3점이며, 1점은 확인된 실질적 약점에만 쓴다. "
    "Q9 Risk도 점수가 높을수록 Risk가 관리 가능하다는 긍정 방향이다."
)

CONFIDENCE_LEVELS = {
    "High": "복수 독립 근거와 외부검증 또는 운영 결과 존재",
    "Medium": "판단 가능한 근거는 있으나 기업자료 비중이 크거나 검증범위 제한",
    "Low": "근거가 적거나 초기·소수 자료 중심",
}
CONFIDENCE_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def _confidence_cap(evidence: list[dict]) -> str:
    """근거 수·독립 출처·외부검증 수준으로 허용 가능한 confidence 상한을 정한다."""
    source_ids = {item["source_id"] for item in evidence}
    externally_verified_sources = {
        item["source_id"]
        for item in evidence
        if item.get("evidence_level") in ("E3", "E4", "E5")
    }
    if len(externally_verified_sources) >= 2:
        return "High"
    if externally_verified_sources or len(source_ids) >= 2:
        return "Medium"
    return "Low"

# ---------------------------------------------------------------------------
# 계산과 최종 판단 규칙 (설계 3.4) - Prototype 운영 기준이며 업계 표준으로 주장하지 않는다
# ---------------------------------------------------------------------------
# 1·3·5 척도에서 3.5는 '기업자료 중심(3점)'을 넘어 외부 검증된 강점(5점)이 충분해야 넘는 기준이다.
# 예) 12문항: 5점 3개 + 3점 9개 = 평균 3.5 → INVEST / 9문항: 5점 2개 + 3점 7개 = 평균 3.44 → HOLD
INVEST_SCORE_THRESHOLD = 3.2
MIN_SCORED_CRITERIA = 9  # Coverage 70% 이상 = 12문항 중 최소 9개 (9/12 = 75.0%)


def criteria_for(agent: str) -> list[Criterion]:
    """해당 Agent가 근거를 제공해야 하는 문항."""
    return [c for c in CRITERIA if agent in c.agents]


def validate_criteria(
    criteria: list[dict],
    available_evidence: dict[str, dict],
    allowed_evidence_by_criterion: dict[str, set[tuple[str, str]]] | None = None,
) -> list[dict]:
    """LLM이 낸 문항별 결과를 Python으로 검증한다 (설계 3.4, 21쪽본 5.5).

    - 문항 완전성: 12문항이 각 1개씩. 누락 문항은 INSUFFICIENT_EVIDENCE / score=None으로 채운다.
    - Evidence ID: 앞선 Agent가 실제 검색해 전달한 Evidence(available_evidence, chunk_id → EvidenceItem)에
      존재하고 source_id가 일치하는 참조만 남긴다. LLM이 쓴 Metadata 대신 원본 EvidenceItem을 연결한다.
      allowed_evidence_by_criterion이 주어지면 문항별 허용 범위도 확인한다. 이를 통해 Competition의 타사
      Evidence는 Q4(B04)·Q9(B09)에만 허용하고 다른 문항으로 섞이지 않게 할 수 있다.
    - score/status: SCORED면 1·3·5 중 하나, INSUFFICIENT_EVIDENCE면 None.
      허용되지 않은 점수(2·4점 등)는 '형식 오류'로 기록해 근거 부족과 구분한다.
      이 처리는 최후의 안전장치이며, Judge는 출력 형식을 1·3·5로 제한하고 형식 오류 문항은 재평가한 뒤 호출한다.
      유효 Evidence가 남지 않거나 규칙에 맞지 않으면 None / INSUFFICIENT_EVIDENCE로 바꾸고 사유를 기록한다.
    - 중복/confidence/5점 상한: 중복 문항은 N/A, 잘못된 confidence는 Low로 정규화한다.
      confidence는 근거 수준과 독립 source 수로 계산한 상한을 넘지 못한다. High는 서로 다른 source_id의
      E3 이상 근거가 2개 이상일 때만, Medium은 E3 이상 1개 또는 서로 다른 출처 2개 이상일 때까지 허용한다.
      E3 이상 Evidence가 없는 5점은 기업 주장·Demo 중심 Rubric의 상한인 3점으로 조정한다.

    criteria 항목의 evidence는 [{"chunk_id", "source_id"}] 참조 또는 EvidenceItem이어도 된다.
    """
    by_id: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for item in criteria:
        criterion_id = item.get("criterion_id")
        if criterion_id in by_id:
            duplicate_ids.add(criterion_id)
        else:
            by_id[criterion_id] = item

    validated = []
    for crit in CRITERIA:
        raw = by_id.get(crit.id)
        raw_missing = raw.get("missing_information") if raw else []
        missing = list(raw_missing) if isinstance(raw_missing, list) else []
        if raw is None:
            validated.append(_insufficient(crit, "LLM 출력에 문항 결과가 없음"))
            continue
        if crit.id in duplicate_ids:
            validated.append(_insufficient(crit, "형식 오류: 동일 criterion_id가 중복됨"))
            continue

        evidence = []
        seen_evidence: set[tuple[str, str]] = set()
        raw_evidence = raw.get("evidence")
        for ref in raw_evidence if isinstance(raw_evidence, list) else []:
            if not isinstance(ref, dict):
                continue
            item = available_evidence.get(ref.get("chunk_id"))
            evidence_key = (ref.get("chunk_id"), ref.get("source_id"))
            allowed_refs = (
                allowed_evidence_by_criterion.get(crit.id, set())
                if allowed_evidence_by_criterion is not None
                else None
            )
            if (
                item
                and item["source_id"] == ref.get("source_id")
                and (allowed_refs is None or evidence_key in allowed_refs)
                and evidence_key not in seen_evidence
            ):
                evidence.append(item)
                seen_evidence.add(evidence_key)

        score, status = raw.get("score"), raw.get("status")
        valid_score = isinstance(score, int) and not isinstance(score, bool) and score in ALLOWED_SCORES
        confidence = raw.get("confidence")
        reason = None
        if status == "SCORED" and not valid_score:
            reason = f"형식 오류: 허용되지 않은 점수 {score!r} (허용: 1, 3, 5)"
        elif status == "SCORED" and not evidence:
            reason = "검증된 Evidence가 없음"
        elif status == "INSUFFICIENT_EVIDENCE" and score is not None:
            reason = "형식 오류: INSUFFICIENT_EVIDENCE의 score는 null이어야 함"
        elif status not in ("SCORED", "INSUFFICIENT_EVIDENCE"):
            reason = f"형식 오류: 허용되지 않은 status {status!r}"

        if reason or status == "INSUFFICIENT_EVIDENCE":
            validated.append({
                **_insufficient(crit, reason),
                "evidence": evidence,
                "reasoning": raw.get("reasoning", ""),
                "missing_information": missing + ([reason] if reason else []),
            })
        else:
            # E3 이상은 5점의 필요조건일 뿐 충분조건은 아니다. E1~E2 근거를 5점으로
            # 과대평가한 경우에만 현행 Rubric의 상한인 3점으로 제한한다.
            validation_notes = []
            if score == 5 and not any(e.get("evidence_level") in ("E3", "E4", "E5") for e in evidence):
                score = 3
                validation_notes.append("E3 이상 외부검증 Evidence가 없어 5점을 3점으로 조정")

            if confidence not in CONFIDENCE_LEVELS:
                confidence = "Low"
                validation_notes.append("허용되지 않은 confidence를 Low로 정규화")
            else:
                confidence_cap = _confidence_cap(evidence)
                if CONFIDENCE_ORDER[confidence] > CONFIDENCE_ORDER[confidence_cap]:
                    confidence = confidence_cap
                    validation_notes.append(
                        f"Evidence 수준·독립 출처 수 기준 confidence 상한을 {confidence_cap}로 조정"
                    )

            reasoning = raw.get("reasoning", "")
            if validation_notes:
                reasoning = f"[Python 검증: {'; '.join(validation_notes)}] {reasoning}".strip()

            validated.append({
                "criterion_id": crit.id,
                "criterion_name": crit.name,
                "score": score,
                "status": "SCORED",
                "evidence": evidence,
                "reasoning": reasoning,
                "confidence": confidence,
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
    """문항별 점수(1·3·5, N/A는 None)로 FinalScore·EvidenceCoverage·Decision을 계산한다.

    FinalScore       = 유효한 점수의 단순 평균 (N/A 제외, 평가 가능한 문항이 없으면 None)
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
