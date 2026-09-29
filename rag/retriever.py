"""Metadata Filter + Semantic Similarity Search

Agent별 Retrieval Scope (설계 산출물 3.2절):
- Technology & Product Agent: 현재 기업의 기술자료만 검색
- Market & Traction Agent: 공통 시장자료 + 현재 기업 자료 (다른 후보 기업 자료는 섞지 않음)

한국어 질의는 LLM으로 영어로 번역한 뒤 검색한다 (문서가 대부분 영어라 검색 정확도가 높아짐).
영어 질의는 그대로 검색하므로 추가 비용이 없다. translate=False로 끌 수 있다.

빠른 확인:
    uv run python -m rag.retriever "humanoid market size 2035" --scope market --company figure
    uv run python -m rag.retriever "factory deployment results" --scope technology --company apptronik
    uv run python -m rag.retriever "manipulation capability level" --scope industry --type capability_report
"""

import argparse
import json
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from documents.company import company_key
from documents.config import COMPANIES
from rag.config import COLLECTION_NAME, TOP_K, VECTORSTORE_DIR
from rag.embeddings import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    if not VECTORSTORE_DIR.exists():
        raise SystemExit("벡터DB가 없습니다. 먼저 `uv run python -m rag.build_index`를 실행하세요.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )


HANGUL = re.compile(r"[가-힣]")

TRANSLATE_PROMPT = """Translate the following Korean search query into a concise English search query for retrieving passages from English industry reports and company documents about humanoid robots.
Keep company names, product names, numbers, and technical terms exactly. Output only the English query.

Korean query: {query}"""


@lru_cache(maxsize=1024)
def to_search_query(query: str) -> str:
    """한국어가 섞인 질의는 영어로 번역하고, 영어 질의는 그대로 돌려준다. 같은 질의는 한 번만 번역한다."""
    if not HANGUL.search(query):
        return query
    from core.llm import get_llm

    return get_llm().invoke(TRANSLATE_PROMPT.format(query=query)).content.strip().strip('"')


def _where(**conditions: str | None) -> dict | None:
    clauses = [{k: v} for k, v in conditions.items() if v is not None]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(
    query: str,
    *,
    domain: str | None = None,
    company: str | None = None,
    document_type: str | None = None,
    k: int = TOP_K,
    translate: bool = True,
) -> list[Document]:
    """유사도 검색. 각 Document의 metadata에 score(코사인 유사도 0~1)와 search_query(실제 검색어)를 넣어 반환한다."""
    company = company_key(company) if company else None
    search_query = to_search_query(query) if translate else query
    results = get_vectorstore().similarity_search_with_relevance_scores(
        search_query, k=k, filter=_where(domain=domain, company=company, document_type=document_type)
    )
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(score, 4)
        doc.metadata["search_query"] = search_query
        docs.append(doc)
    return docs


def search_technology(query: str, company: str, k: int = TOP_K, translate: bool = True) -> list[Document]:
    """Technology & Product Agent: 현재 기업의 기술자료만."""
    return search(query, domain="technology", company=company, k=k, translate=translate)


def search_industry(
    query: str, k: int = TOP_K, document_type: str | None = None, translate: bool = True
) -> list[Document]:
    """공통 산업 자료만 (market_report, capability_report)."""
    return search(query, domain="market", document_type=document_type, k=k, translate=translate)


def search_market(query: str, company: str, k: int = TOP_K, translate: bool = True) -> dict[str, list[Document]]:
    """Market & Traction Agent: 공통 시장자료와 현재 기업 자료를 따로 반환한다.

    Market Evidence와 Company Evidence를 섞지 않기 위해 결과를 분리한다.
    """
    return {
        "market": search_industry(query, k=k, translate=translate),
        "company": search(query, company=company, k=k, translate=translate),
    }


def to_evidence(doc: Document) -> dict:
    """검색 결과 → EvidenceItem (evidence_level, fact는 분석 Agent가 채운다)."""
    m = doc.metadata
    return {
        "source_id": m["source_id"],
        "chunk_id": m["chunk_id"],
        "page": m["page"],
        "title": m["title"],
        "publisher": m["publisher"],
        "published_date": m["published_date"],
        "source_type": m["source_type"],
        "reference_metadata": json.loads(m["reference_metadata"]),
        "evidence_level": None,
        "fact": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 검색 확인")
    parser.add_argument("query")
    parser.add_argument("--scope", choices=["technology", "market", "industry", "all"], default="all")
    parser.add_argument("--company", help=f"{', '.join(COMPANIES)} 또는 기업명")
    parser.add_argument("--type", dest="document_type", help="market_report, capability_report 등")
    parser.add_argument("-k", type=int, default=TOP_K)
    parser.add_argument("--no-translate", action="store_true", help="한국어 질의를 번역하지 않고 그대로 검색")
    args = parser.parse_args()

    if args.scope in ("technology", "market") and not args.company:
        parser.error(f"--scope {args.scope}에는 --company가 필요합니다.")

    tr = not args.no_translate
    if args.scope == "technology":
        groups = {"technology": search_technology(args.query, args.company, args.k, translate=tr)}
    elif args.scope == "market":
        groups = search_market(args.query, args.company, args.k, translate=tr)
    elif args.scope == "industry":
        groups = {"industry": search_industry(args.query, args.k, args.document_type, translate=tr)}
    else:
        groups = {"all": search(args.query, document_type=args.document_type, k=args.k, translate=tr)}

    first = next((docs[0] for docs in groups.values() if docs), None)
    if first and first.metadata["search_query"] != args.query:
        print(f"검색어 (번역): {first.metadata['search_query']}")

    for name, docs in groups.items():
        print(f"\n### {name}")
        for d in docs:
            m = d.metadata
            preview = d.page_content[:160].replace("\n", " ")
            print(f"[{m['score']:.3f}] {m['chunk_id']} ({m['company']}, {m['document_type']}) {preview}…")


if __name__ == "__main__":
    main()
