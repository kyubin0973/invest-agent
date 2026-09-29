"""RAG 인덱스 생성 (최초 1회 또는 문서 변경 시 실행)

Documents(기업 + 산업) → Loading → Cleaning / Page Tracking → Chunking → Embedding → Chroma

    uv run python -m rag.build_index
"""

import json
import re
import shutil
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from documents.config import ROOT
from documents.loader import base_metadata, load_manifest, load_pages
from rag.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CONTACTS_PER_CHUNK,
    MIN_CHUNK_CHARS,
    SEPARATORS,
    VECTORSTORE_DIR,
)
from rag.embeddings import get_embeddings

CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+\d{1,3}[\s(-]*\d")


def is_noise(chunk: str) -> bool:
    return len(chunk) < MIN_CHUNK_CHARS or len(CONTACT.findall(chunk)) > MAX_CONTACTS_PER_CHUNK


def build_documents() -> list[Document]:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=SEPARATORS,
    )

    docs = []
    for row in load_manifest():
        meta = base_metadata(row)
        # Chroma 메타데이터는 str/int/float/bool만 가능 → 중첩 정보는 JSON 문자열로 저장
        meta["reference_metadata"] = json.dumps(meta["reference_metadata"], ensure_ascii=False)
        for page in load_pages(row):
            chunks = [c for c in splitter.split_text(page["text"]) if not is_noise(c)]
            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = f"{meta['source_id']}_P{page['page']:02d}_C{idx:02d}"
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={**meta, "page": page["page"], "chunk_id": chunk_id},
                        id=chunk_id,
                    )
                )
    return docs


def main() -> None:
    docs = build_documents()

    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    Chroma.from_documents(
        docs,
        get_embeddings(),
        ids=[d.id for d in docs],
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"청크 {len(docs)}개 저장 → {VECTORSTORE_DIR.relative_to(ROOT)}")
    groups = Counter(f"{d.metadata['domain']}/{d.metadata['company']}" for d in docs)
    for key, n in sorted(groups.items()):
        print(f"  {key}: {n}")


if __name__ == "__main__":
    main()
