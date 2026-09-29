"""RAG 설정값 (설계 산출물 3.5 ~ 3.6절 기준)

기업 자료(data/technology/)와 산업 자료(data/market/)를 모두 하나의 Chroma 컬렉션에 넣고,
metadata(domain, company)로 Agent별 검색 범위를 제한한다.
Chunk Size, Overlap, Top-K는 실제 Retrieval 실험으로 확정한 값이 아니라 초기 설계값이다.
"""

from documents.config import ROOT

VECTORSTORE_DIR = ROOT / "vectorstore" / "chroma"
COLLECTION_NAME = "humanoid_rag"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 60
TOP_K = 5
MIN_CHUNK_CHARS = 50

# Split 우선순위: Section → Paragraph → Sentence → Token
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# 연락처가 몰린 청크는 본문이 아닌 것으로 보고 버린다 (새 문서에도 적용되는 공통 규칙)
MAX_CONTACTS_PER_CHUNK = 2
