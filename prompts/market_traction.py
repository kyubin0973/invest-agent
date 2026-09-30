"""Market & Traction Agent 프롬프트

공통 규칙(prompts.common)은 요청 바로 앞(user 메시지)에 넣고, 여기에는 Market 전용 지시만 둔다.
"""

ROLE = "휴머노이드 스타트업의 시장·고객·사업을 분석하는 투자 애널리스트"

# 분석축 (설계 2.1.1 Market & Traction 영역). findings.dimension은 이 중 하나만 쓴다.
DIMENSIONS = (
    "시장 규모·성장",
    "고객 문제·Use Case",
    "고객검증·배치·계약",
    "WTP·도입 경제성",
    "사업모델·가격",
    "팀·Founder",
    "제조 파트너·원가·Fleet 운영",
    "사업 확장성",
    "시장·상용화 Risk",
)

# 기업 고유 근거가 필요한 분석축: 공통 시장자료만으로는 서술할 수 없다 (Market Evidence ≠ Company Evidence)
COMPANY_SPECIFIC_DIMENSIONS = (
    "고객검증·배치·계약",
    "사업모델·가격",
    "팀·Founder",
    "제조 파트너·원가·Fleet 운영",
)

MARKET_RULES = """[Market & Traction 규칙]
1. 공통 시장자료(source_type=industry_report)는 시장 기회 판단에만 쓴다. 산업 전망을 이 기업의 성과로 서술하지 않는다.
2. 고객·계약·배치·사업모델·팀·제조 파트너는 이 기업의 자료로만 서술한다.
3. 공통 시장자료에 다른 기업(다른 후보 포함)의 사례가 나와도 이 기업의 근거로 쓰지 않는다.
4. 제품 성능·기술 수준은 평가하지 않는다 (Technology & Product 담당).
5. 팀·Founder·가격·계약 조건이 근거에 없으면 추정하지 말고 missing_information에 '공개 근거 없음'으로 기록한다.
6. 계획·목표·예정(will, plan, goal, aim, expected, capable of)은 '계획'으로 서술하고 달성한 사실로 쓰지 않는다.
7. statement에는 인용한 근거에 적힌 내용만 쓴다. 근거에 없는 해석·전망(예: 가격 인하, 소비자 시장 진출)을 덧붙이지 않는다."""


def sufficiency_request(company: str, questions: list[dict]) -> str:
    """하위 질문별 근거 충분성 판단 + 부족 시 보완 검색어 요청."""
    lines = [
        f"평가 대상 기업: {company}",
        "아래 하위 질문마다, 해당 질문에 검색된 근거(chunk_id 목록)만 보고 답할 수 있는지 판단하라.",
        "- sufficient: 근거에 질문에 대한 직접적인 답(구체적 사실·수치·이름)이 있으면 true.",
        "  주제만 관련 있고 질문에 답하지 못하면 false. 예: 경영진을 묻는데 제품 소개만 있으면 false.",
        "- answer_quote: sufficient=true일 때, 질문에 직접 답하는 원문 문장 1개를 근거에서 그대로 옮겨 적는다 (영어 원문, 번역 금지).",
        "- missing_aspects: 부족한 측면 (sufficient=true면 빈 목록)",
        "- refined_query: 부족할 때 쓸 영어 검색어 1개 (짧고 구체적으로). sufficient 여부와 관계없이 항상 적는다.",
        "",
        "[하위 질문]",
    ]
    for q in questions:
        lines.append(f"- {q['id']} ({q['scope']}): {q['query']} → 근거 {', '.join(q['chunk_ids']) or '없음'}")
    return "\n".join(lines)


def analysis_request(company: str, target_market: str | None, questions: list[dict], gaps: list[str]) -> str:
    """시장·Traction 분석 요청. 하위 질문별 근거 대응표를 함께 준다."""
    gap_text = "\n".join(f"- {g}" for g in gaps) or "- 없음"
    q_text = "\n".join(
        f"- {q['id']} ({q['scope']}, {'근거 충분' if q.get('sufficient') or q['scope'] == 'market' else '근거 부족'}) [분석축: {' / '.join(q['dimensions'])}]: "
        f"{q['query']} → {', '.join(q['chunk_ids']) or '근거 없음'}"
        for q in questions
    )
    return f"""평가 대상 기업: {company}
Target Market (사전 입력): {target_market or '미입력'}

[하위 질문별 근거]
{q_text}

위 근거로 {company}의 시장·고객·사업을 분석하라.

- findings: [하위 질문별 근거]에서 '근거 충분' 하위 질문마다 최소 1개의 finding을 쓰고 question_id에 그 질문 ID를 적는다.
  '근거 부족' 하위 질문은 근거에 질문에 대한 직접적인 답이 있을 때만 쓰고, 없으면 missing_information에 기록한다.
  dimension은 그 하위 질문의 [분석축] 중 하나만 쓴다. statement는 그 하위 질문에 답하는 내용이어야 한다.
  예: 사업모델·가격 질문에는 가격·판매 방식·수익구조만 쓰고, 생산능력·공급망은 제조 질문에 쓴다.
  특히 기업 자료(company) 하위 질문의 근거를 빠뜨리지 않는다. 한 질문에 서로 다른 판단이 있으면 finding을 나눠 쓴다.
  statement는 구체적 사실(수치·고객명·파트너명·조건)을 담은 1~2문장이며, 기업 자료의 내용은 기업 주장임을 드러낸다.
  statement 안에는 chunk_id를 쓰지 않는다 (근거는 evidence_chunk_ids에만).
  evidence_chunk_ids에는 그 판단의 근거 chunk_id를 넣는다 (위 근거 목록에 있는 것만).
  finding은 근거에 있는 사실만 쓴다. '확인되지 않음·정보 부족'은 finding이 아니라 missing_information에 쓴다.
  시장(market) 질문은 공통 시장자료로 답하고, Target Market과 다른 시장을 다룬 자료면 그 점을 문장에 밝힌다.
  질문에 답하지 못하는 근거로 finding을 채우지 않는다 (주제만 관련 있는 내용을 다른 분석축에 옮겨 쓰지 않는다).
- evidence: findings에서 사용한 근거(chunk_id)마다 evidence_level과 fact(근거가 말하는 사실 1문장)를 적는다.
  evidence_level: E1 기업 주장·계획 / E2 시연·제한 환경 / E3 외부 확인 / E4 외부 확인된 실제 운영·계약 / E5 반복·규모화
- risks: 시장·상용화·사업 측면 Risk
- missing_information: 판단에 필요하지만 근거가 없는 정보
- summary: 2~3문장 요약

검색 후에도 근거가 부족했던 측면:
{gap_text}"""


def grounding_request(company: str, findings: list[dict]) -> str:
    """분석 문장을 인용 원문과 대조하는 요청 (Self-RAG의 환각 검사와 같은 역할)."""
    lines = [
        f"평가 대상 기업: {company}",
        "아래 분석 문장마다, 위 근거 중 해당 문장이 인용한 chunk만 보고 판정하라.",
        "- supported: 문장의 모든 사실이 인용 근거에 있고, 기업 자료(source_type=company)의 내용은 기업 주장임이 드러난다.",
        "- overstated: 근거는 있으나 과장됐다. 계획·목표를 달성한 사실로 썼거나, 근거에 없는 해석·전망을 덧붙였거나,",
        "  기업 자료의 내용을 확인된 사실처럼 썼다. → revised_statement에 근거에 적힌 내용만으로 고친 문장을 쓴다",
        "  (계획은 '계획'으로, 기업 자료는 '○○ 발표에 따르면'처럼 주장임을 드러낸다).",
        "- unsupported: 문장의 핵심 내용이 인용 근거에 전혀 없다.",
        "  근거에 일부라도 있으면 unsupported가 아니라 overstated로 판정하고 근거에 있는 내용만으로 고쳐 쓴다.",
        "- off_topic: 문장이 [분석축]의 주제와 맞지 않는다. 예: [사업모델·가격]에 가격·수익구조가 아닌 생산·공급망 내용,",
        "  [팀·Founder]에 경영진이 아닌 제품 내용. 근거가 맞아도 분석축이 틀리면 off_topic이다.",
        "",
        "[분석 문장]",
    ]
    for i, f in enumerate(findings, start=1):
        refs = ", ".join(r["chunk_id"] for r in f["evidence_refs"])
        lines.append(f"- F{i} [{f['dimension']}] {f['statement']} (인용: {refs})")
    return "\n".join(lines)
