"""Metadata Filter + Semantic Similarity Search

Agent별 Retrieval Scope (설계 산출물 3.2절):
- Technology & Product Agent: 현재 기업의 기술자료만 검색
- Market & Traction Agent: 공통 시장자료 + 현재 기업 자료 (다른 후보 기업 자료는 섞지 않음)

빠른 확인:
    uv run python -m rag.retriever "humanoid market size 2035" --scope market --company figure
    uv run python -m rag.retriever "factory deployment results" --scope technology --company apptronik
    uv run python -m rag.retriever "manipulation capability level" --scope industry --type capability_report
"""

import argparse
import json
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
) -> list[Document]:
    """유사도 검색. 각 Document의 metadata["score"]에 코사인 유사도(0~1)를 넣어 반환한다."""
    company = company_key(company) if company else None
    results = get_vectorstore().similarity_search_with_relevance_scores(
        query, k=k, filter=_where(domain=domain, company=company, document_type=document_type)
    )
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(score, 4)
        docs.append(doc)
    return docs


def search_technology(query: str, company: str, k: int = TOP_K) -> list[Document]:
    """Technology & Product Agent: 현재 기업의 기술자료만."""
    return search(query, domain="technology", company=company, k=k)


def search_industry(query: str, k: int = TOP_K, document_type: str | None = None) -> list[Document]:
    """공통 산업 자료만 (market_report, capability_report)."""
    return search(query, domain="market", document_type=document_type, k=k)


def search_market(query: str, company: str, k: int = TOP_K) -> dict[str, list[Document]]:
    """Market & Traction Agent: 공통 시장자료와 현재 기업 자료를 따로 반환한다.

    Market Evidence와 Company Evidence를 섞지 않기 위해 결과를 분리한다.
    """
    return {
        "market": search_industry(query, k=k),
        "company": search(query, company=company, k=k),
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
    args = parser.parse_args()

    if args.scope in ("technology", "market") and not args.company:
        parser.error(f"--scope {args.scope}에는 --company가 필요합니다.")

    if args.scope == "technology":
        groups = {"technology": search_technology(args.query, args.company, args.k)}
    elif args.scope == "market":
        groups = search_market(args.query, args.company, args.k)
    elif args.scope == "industry":
        groups = {"industry": search_industry(args.query, args.k, args.document_type)}
    else:
        groups = {"all": search(args.query, document_type=args.document_type, k=args.k)}

    for name, docs in groups.items():
        print(f"\n### {name}")
        for d in docs:
            m = d.metadata
            preview = d.page_content[:160].replace("\n", " ")
            print(f"[{m['score']:.3f}] {m['chunk_id']} ({m['company']}, {m['document_type']}) {preview}…")


if __name__ == "__main__":
    main()
