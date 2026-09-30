"""모든 Agent 프롬프트에 공통으로 넣는 규칙과 근거 표기 형식

규칙 출처: 설계 산출물 2.1.2(LLM/Python 역할 구분·공통 원칙), 2.2.4(근거 안전 규칙)
각 Agent 프롬프트는 이 파일의 규칙을 가져다 쓰고, Agent별 지시만 자기 파일에 작성한다.

    from prompts.common import EVIDENCE_RULES, CITATION_RULES, build_system_prompt, build_user_prompt, format_documents

    system = build_system_prompt("휴머노이드 스타트업의 기술·제품 역량을 분석하는 애널리스트")
    user = build_user_prompt(format_documents(docs), "Figure AI의 실제 고객 현장 배치 수준을 평가하라.",
                             EVIDENCE_RULES, CITATION_RULES)
    get_llm().invoke([("system", system), ("user", user)])

규칙은 system이 아니라 user 메시지의 요청 바로 앞에 둔다.
gpt-4o-mini로 확인한 결과, system에만 둔 규칙은 자유 서술에서 잘 지켜지지 않았고
(기업 자체 발표를 확인된 사실·상용 운영으로 서술), 요청 바로 앞에 두면 기업 주장과 외부 확인을 구분했다.
프롬프트만으로는 완전히 보장되지 않으므로, Evidence Level 같은 판정은 구조화 출력 + Python 검증을 함께 쓴다.
"""

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# 근거 안전 규칙 (설계 2.2.4) - 분석·비교·평가·보고 모든 LLM 호출에 넣는다
# ---------------------------------------------------------------------------
EVIDENCE_RULES = """[근거 안전 규칙]
1. Claim ≠ Achievement: 기업의 계획·목표·주장을 달성한 성과로 표현하지 않는다.
2. Demo ≠ Commercial Deployment: 시연, 제한 환경 테스트, Pilot, 상용 운영을 구분한다.
3. Market Evidence ≠ Company Evidence: 산업·시장 전망을 특정 기업의 성과로 확대하지 않는다.
4. Missing Evidence ≠ Negative Evidence: 정보가 없으면 '확인되지 않음'으로 기록하고, 낮은 평가의 근거로 쓰지 않는다.
5. 제공된 근거에 없는 사실·수치·날짜·고객명·파트너명을 만들지 않는다."""

# ---------------------------------------------------------------------------
# 인용 규칙 (설계 2.2.3 추적 경로: Report Statement → Evidence → chunk_id → source_id + page)
# ---------------------------------------------------------------------------
CITATION_RULES = """[인용 규칙]
1. 판단 문장마다 근거의 chunk_id와 source_id를 붙인다. 예: [F4_P02_C01 | F4]
2. 제공된 근거 목록에 있는 chunk_id만 인용한다. 목록에 없는 ID를 만들지 않는다.
3. 기업 자체 자료(source_type=company)의 내용은 확인된 사실처럼 쓰지 않는다.
   "Figure AI 발표에 따르면"처럼 기업 주장임을 문장에 드러내고, 외부 자료로 확인된 경우에만 사실로 서술한다.
   기업 단독 발표의 운영·계약 주장은 외부 확인 전까지 기업 주장(E1~E2)으로 다룬다."""

# ---------------------------------------------------------------------------
# 공통 원칙 (설계 2.1.2) - 해당 Agent에만 넣는다
# ---------------------------------------------------------------------------
NO_NEW_EVIDENCE_RULE = """[신규 근거 금지] Competition·Judge·Report 공통
- 새로 검색하지 않는다. 앞선 Agent가 전달한 구조화 결과와 근거만 사용한다.
- 앞선 결과에 없는 사실·점수·Risk·판단을 새로 만들지 않는다."""

TARGET_MARKET_RULE = """[Target Market 규칙] Competition·Judge
- 기업마다 진입 시장(산업용·물류·가정용 등)이 다를 수 있다.
- 시장이 다르다는 사실만으로 우열을 판단하거나 같은 시장에서 경쟁한다고 해석하지 않는다."""

OUTPUT_LANGUAGE_RULE = """[출력 언어]
- 분석 문장은 한국어로 쓴다. 기업명·제품명·기술명·수치는 원문 표기를 그대로 쓴다."""


def build_system_prompt(role: str) -> str:
    """역할과 출력 언어만 담은 system 프롬프트. 규칙은 build_user_prompt로 요청 바로 앞에 넣는다."""
    return f"당신은 {role}입니다.\n\n{OUTPUT_LANGUAGE_RULE}"


def build_user_prompt(evidence: str, request: str, *rules: str) -> str:
    """근거 → 규칙 → 요청 순서로 user 프롬프트를 만든다."""
    blocks = [f"[근거]\n{evidence}", *rules, f"[요청] 위 규칙을 지켜 답하라.\n{request}"]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 근거 표기 형식 - LLM이 chunk_id로 인용할 수 있게 메타데이터와 함께 넣는다
# ---------------------------------------------------------------------------
def format_documents(docs: list[Document]) -> str:
    """검색 결과(rag.retriever)를 프롬프트용 XML 형식으로 바꾼다 (Technology·Market Agent)."""
    return "\n".join(
        "<document>"
        f"<chunk_id>{d.metadata['chunk_id']}</chunk_id>"
        f"<source_id>{d.metadata['source_id']}</source_id>"
        f"<source_type>{d.metadata['source_type']}</source_type>"
        f"<title>{d.metadata['title']}</title>"
        f"<publisher>{d.metadata['publisher']}</publisher>"
        f"<published_date>{d.metadata['published_date']}</published_date>"
        f"<page>{d.metadata['page']}</page>"
        f"<content>{d.page_content}</content>"
        "</document>"
        for d in docs
    )


def format_evidence_items(items: list[dict]) -> str:
    """EvidenceItem 목록을 프롬프트용 XML 형식으로 바꾼다 (Competition·Judge·Report Agent)."""
    return "\n".join(
        "<evidence>"
        f"<chunk_id>{e['chunk_id']}</chunk_id>"
        f"<source_id>{e['source_id']}</source_id>"
        f"<source_type>{e['source_type']}</source_type>"
        f"<title>{e['title']}</title>"
        f"<page>{e['page']}</page>"
        f"<evidence_level>{e.get('evidence_level') or '미분류'}</evidence_level>"
        f"<fact>{e.get('fact') or ''}</fact>"
        "</evidence>"
        for e in items
    )
