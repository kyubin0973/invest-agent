"""manifest 기반 문서 로딩: 페이지 추출, 머리말 정리, 서지 메타데이터

기업 자료(직접 읽기)와 산업 자료(RAG) 모두 이 모듈로 읽어서 같은 메타데이터 형식을 쓴다.
"""

import csv
import re

from pypdf import PdfReader

from documents.config import (
    COMPANY_DOCS_PREFIX,
    CUT_FROM_MARKER,
    DOCUMENT_TYPES,
    EXCLUDE_PAGES,
    INCLUDE_PAGES,
    INDUSTRY_DOCS_PREFIX,
    MANIFEST_PATH,
    PARTNER_SOURCES,
    ROOT,
)

# 웹페이지를 변환한 PDF의 머리말 (출처 정보는 manifest 메타데이터로 따로 보존)
CONVERTED_HEADER = re.compile(
    r"^(Publisher:|Source:|Source format:|Multimedia demonstrations remain at the source URL)"
)


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing = [r["path"] for r in rows if not (ROOT / r["path"]).exists()]
    if missing:
        raise SystemExit(f"manifest에 있는 파일이 없습니다: {missing}")
    return rows


def is_company_doc(row: dict) -> bool:
    return row["path"].startswith(COMPANY_DOCS_PREFIX)


def is_industry_doc(row: dict) -> bool:
    return row["path"].startswith(INDUSTRY_DOCS_PREFIX)


def company_of(row: dict) -> str | None:
    """기업 자료면 폴더명(figure, apptronik, 1x), 산업 자료면 None."""
    if not is_company_doc(row):
        return None
    return row["path"].removeprefix(COMPANY_DOCS_PREFIX).split("/")[0]


def selected_pages(source_id: str, num_pages: int) -> list[int]:
    """사용할 페이지 번호 (1부터). INCLUDE_PAGES가 있으면 그 페이지만, EXCLUDE_PAGES는 제외."""
    pages = INCLUDE_PAGES.get(source_id, range(1, num_pages + 1))
    excluded = set(EXCLUDE_PAGES.get(source_id, []))
    return [p for p in pages if p not in excluded and 1 <= p <= num_pages]


def clean_page(text: str, row: dict) -> str:
    lines = [line.strip() for line in text.splitlines()]
    if lines and lines[0].startswith(f"{row['id']} |") and "converted to PDF" in lines[0]:
        # 1~2번째 줄: "F1 | Figure AI | official source converted to PDF" / 페이지 번호
        lines = lines[2:] if len(lines) > 1 and lines[1].isdigit() else lines[1:]
        lines = [line for line in lines if not CONVERTED_HEADER.match(line)]
    text = "\n".join(line for line in lines if line)
    marker = CUT_FROM_MARKER.get(row["id"])
    if marker and marker in text:
        text = text[: text.index(marker)]
    return re.sub(r"[ \t]+", " ", text).strip()


def base_metadata(row: dict) -> dict:
    """설계 산출물 3.7절 Metadata (page, chunk_id 제외)."""
    source_id = row["id"]
    if is_industry_doc(row):
        source_type = "industry_report"
    elif source_id in PARTNER_SOURCES:
        source_type = "partner"
    else:
        source_type = "company"

    is_web = "webpage" in row["format"]
    return {
        "source_id": source_id,
        "company": company_of(row) or "common",
        "domain": "technology" if is_company_doc(row) else "market",
        "document_type": DOCUMENT_TYPES.get(source_id, "unknown"),
        "title": row["title"],
        "publisher": row["publisher"],
        "published_date": row["publication_date"],
        "source_type": source_type,
        "source_path": row["path"],
        "reference_metadata": {
            "reference_type": "web" if is_web else "report",
            "url": row["source_url"] or None,
            "site_name": row["publisher"] if is_web else None,
            "source_format": row["format"],
        },
    }


def load_pages(row: dict) -> list[dict]:
    """문서 1개를 정리된 페이지 목록으로 읽는다. 빈 페이지는 뺀다."""
    reader = PdfReader(ROOT / row["path"])
    pages = []
    for page_no in selected_pages(row["id"], len(reader.pages)):
        text = clean_page(reader.pages[page_no - 1].extract_text() or "", row)
        if text:
            pages.append({"page": page_no, "text": text})
    return pages
