"""문서 설정: 경로, 기업 목록, 문서 분류, 페이지 선택, 기업별 필수 근거 체크리스트

문서는 두 종류다. 현재는 둘 다 RAG로 검색한다 (설계 산출물 3.2절).
- 기업 자료 (data/technology/{기업}/): company 메타데이터로 현재 평가 기업의 자료만 검색
- 산업 자료 (data/market/): 모든 기업을 평가할 때 공통 기준으로 사용

기업 자료를 검색 없이 전부 읽는 방식은 documents.company에 옵션으로 둔다.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

COMPANY_DOCS_PREFIX = "data/technology/"  # 기업 자료: data/technology/{기업}/
INDUSTRY_DOCS_PREFIX = "data/market/"  # 산업 자료

# 폴더명 → 기업명
COMPANIES = {
    "figure": "Figure AI",
    "apptronik": "Apptronik",
    "1x": "1X Technologies",
}

# 문서 성격
DOCUMENT_TYPES = {
    "F1": "hardware",
    "F2": "ai",
    "F3": "ai",
    "F4": "deployment",
    "F5": "deployment",
    "A1": "hardware",
    "A2": "deployment",
    "A3": "ai",
    "A4": "hardware",
    "A5": "manufacturing",
    "X1": "ai",
    "X2": "ai",
    "X3": "ai",
    "X4": "hardware",
    "X5": "product",
    "M1": "market_report",
    "M2": "market_report",
    "M3": "market_report",
    "M4": "market_report",
    "M5": "market_report",
    "M6": "market_report",
    "M7": "capability_report",
}

# 출처 성격: 기업 자체 자료 / 파트너 자료 / 산업 보고서
PARTNER_SOURCES = {"A3", "A4", "A5"}

# ---------------------------------------------------------------------------
# 페이지 선택 / 잡음 제거 (기업 자료, 산업 자료 공통)
# ---------------------------------------------------------------------------
# 필요한 페이지만 사용한다 (지정하지 않은 문서는 전체 페이지)
# M7 OECD: 요약, 현재 AI 역량 비교표·해설, 5단계 개요, Vision / Manipulation / Robotic intelligence 척도
INCLUDE_PAGES = {
    "M7": [10, 11, 14, 15, 16, 17, 24, 42, 43, 44, 45, 46, 47],
}

# 본문이 아닌 페이지 (연락처, 면책 조항, 저자 명단 등)
EXCLUDE_PAGES = {
    "M1": [2, 3],  # 발행 정보·저작권 페이지
    "M2": [2, 3],  # 발행 정보·저작권 페이지
    "M3": [2, 18, 19, 20, 21, 22],  # 애널리스트 연락처, Disclosure Appendix
    "A3": [4],  # 감사 인사·저자 명단
}

# 이 문구부터 해당 페이지 끝까지 버린다
CUT_FROM_MARKER = {
    "A3": "Acknowledgements",
}

# RAG 문서 + 기업 문서 총 페이지 한도 (과제 조건)
PAGE_BUDGET = 200

# 기업 1곳의 문서를 한 번에 LLM에 넣을 수 있는지 점검하는 기준 (대략 2만 토큰)
MAX_COMPANY_CONTEXT_CHARS = 80_000

# ---------------------------------------------------------------------------
# 기업별 필수 문서 체크리스트
# ---------------------------------------------------------------------------
# 새 기업을 추가할 때도 같은 종류의 근거를 갖추도록 해서 기업 간 동일 기준 비교를 보장한다.
# 팀·창업자·투자 정보(B05, B10)는 company_profiles 입력으로 제공한다.
EVIDENCE_CHECKLIST = {
    "product_tech": "제품·기술 (B02, B04, B11)",
    "deployment": "고객·배치 사례 (B03, B06)",
    "manufacturing": "양산·제조 (B12)",
    "business_model": "가격·수익 모델 (B07)",
    "third_party": "외부 제3자 자료 (Evidence Level E3 이상)",
}

# 문서별로 채워 주는 근거 종류 (문서 내용을 보고 태깅)
DOC_EVIDENCE = {
    "F1": ["product_tech", "manufacturing"],
    "F2": ["product_tech"],
    "F3": ["product_tech"],
    "F4": ["deployment"],
    "F5": ["deployment"],
    "A1": ["product_tech"],
    "A2": ["deployment"],
    "A3": ["product_tech", "third_party"],
    "A4": ["product_tech", "third_party"],
    "A5": ["manufacturing", "deployment", "third_party"],
    "X1": ["product_tech"],
    "X2": ["product_tech"],
    "X3": ["product_tech"],
    "X4": ["product_tech"],
    "X5": ["product_tech", "business_model"],
}
