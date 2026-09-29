"""기업 자료 직접 읽기 (옵션: 현재 기본 방식은 RAG 검색 → rag.retriever)

평가 중인 기업의 문서를 검색 없이 전부 읽어 Agent 프롬프트에 넣는다.
페이지마다 [F4 p.2] 태그를 붙여서 Agent가 근거를 source_id + page로 인용하게 한다.

    uv run python -m documents.company figure
"""

import argparse
from functools import lru_cache

from documents.config import COMPANIES
from documents.loader import base_metadata, company_of, load_manifest, load_pages

_NAME_TO_KEY = {name.lower(): key for key, name in COMPANIES.items()}


def company_key(company: str) -> str:
    """'Figure AI' 같은 기업명이나 'figure' 같은 키를 받아 폴더명 키로 바꾼다."""
    key = _NAME_TO_KEY.get(company.lower(), company.lower())
    if key not in COMPANIES:
        raise ValueError(f"알 수 없는 기업: {company}. 사용 가능: {list(COMPANIES.values())}")
    return key


@lru_cache
def load_company_documents(company: str) -> tuple[dict, ...]:
    """기업 문서 전체. 각 항목은 문서 메타데이터 + pages([{page, text}])."""
    key = company_key(company)
    docs = []
    for row in load_manifest():
        if company_of(row) == key:
            docs.append({**base_metadata(row), "pages": load_pages(row)})
    if not docs:
        raise ValueError(f"{COMPANIES[key]}의 문서가 manifest에 없습니다.")
    return tuple(docs)


def format_company_context(company: str) -> str:
    """Agent 프롬프트에 넣을 기업 문서 전문."""
    blocks = []
    for doc in load_company_documents(company):
        header = (
            f"=== [{doc['source_id']}] {doc['title']} | {doc['publisher']} | "
            f"{doc['published_date']} | source_type={doc['source_type']} ==="
        )
        pages = "\n\n".join(f"[{doc['source_id']} p.{p['page']}]\n{p['text']}" for p in doc["pages"])
        blocks.append(f"{header}\n{pages}")
    return "\n\n".join(blocks)


def to_evidence(company: str, source_id: str, page: int, fact: str | None = None) -> dict:
    """Agent가 인용한 [source_id p.page]를 EvidenceItem으로 바꾼다.

    기업 자료는 청크로 나누지 않으므로 chunk_id는 페이지 단위(F4_P02)로 둔다.
    """
    doc = next((d for d in load_company_documents(company) if d["source_id"] == source_id), None)
    if doc is None or all(p["page"] != page for p in doc["pages"]):
        raise ValueError(f"{company}의 문서에 {source_id} p.{page}가 없습니다.")
    return {
        "source_id": source_id,
        "chunk_id": f"{source_id}_P{page:02d}",
        "page": page,
        "title": doc["title"],
        "publisher": doc["publisher"],
        "published_date": doc["published_date"],
        "source_type": doc["source_type"],
        "reference_metadata": doc["reference_metadata"],
        "evidence_level": None,
        "fact": fact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="기업 문서 전문 확인")
    parser.add_argument("company", help=f"{', '.join(COMPANIES)} 또는 기업명")
    parser.add_argument("--full", action="store_true", help="전문 출력")
    args = parser.parse_args()

    context = format_company_context(args.company)
    if args.full:
        print(context)
        return
    for doc in load_company_documents(args.company):
        print(f"[{doc['source_id']}] {doc['title']} ({doc['source_type']}, {len(doc['pages'])}쪽)")
    print(f"\n전체 {len(context):,}자 (약 {len(context) // 4:,} 토큰)")


if __name__ == "__main__":
    main()
