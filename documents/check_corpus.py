"""문서 구성 점검: 페이지 예산 + 기업 문서 크기 + 기업별 필수 근거 체크리스트

새 기업이나 문서를 추가한 뒤 실행한다.

    uv run python -m documents.check_corpus
"""

from pypdf import PdfReader

from documents.company import format_company_context
from documents.config import (
    COMPANIES,
    DOC_EVIDENCE,
    EVIDENCE_CHECKLIST,
    MAX_COMPANY_CONTEXT_CHARS,
    PAGE_BUDGET,
    ROOT,
)
from documents.loader import company_of, is_industry_doc, load_manifest, selected_pages


def main() -> None:
    rows = load_manifest()
    problems = 0

    # 1) 페이지 예산 (산업 자료 + 기업 자료)
    used = {"산업 자료": 0, "기업 자료": 0}
    print("## 페이지 예산")
    for row in rows:
        n = len(PdfReader(ROOT / row["path"]).pages)
        pages = len(selected_pages(row["id"], n))
        used["산업 자료" if is_industry_doc(row) else "기업 자료"] += pages
        if pages != n:
            print(f"  {row['id']:>3}: {pages}/{n}쪽 사용")
    total = sum(used.values())
    for label, pages in used.items():
        print(f"  {label}: {pages}쪽")
    print(f"  합계 {total}쪽 / 한도 {PAGE_BUDGET}쪽 ({'OK' if total <= PAGE_BUDGET else '초과'})")
    problems += total > PAGE_BUDGET

    # 2) 기업 문서 크기 (한 번에 LLM에 넣을 수 있는지)
    print("\n## 기업 문서 크기 (직접 읽기 옵션 사용 시 참고)")
    for key, name in COMPANIES.items():
        size = len(format_company_context(key))
        ok = size <= MAX_COMPANY_CONTEXT_CHARS
        print(f"  {name}: {size:,}자 (약 {size // 4:,} 토큰) {'OK' if ok else '초과 → 문서를 줄이거나 나눠 읽어야 함'}")
        problems += not ok

    # 3) 기업별 필수 근거
    coverage = {c: {k: [] for k in EVIDENCE_CHECKLIST} for c in COMPANIES}
    for row in rows:
        company = company_of(row)
        if company is None:
            continue
        if row["id"] not in DOC_EVIDENCE:
            print(f"\n[경고] {row['id']}의 근거 종류가 documents/config.py의 DOC_EVIDENCE에 없습니다.")
            problems += 1
            continue
        for kind in DOC_EVIDENCE[row["id"]]:
            coverage[company][kind].append(row["id"])

    print("\n## 기업별 필수 근거 체크리스트")
    header = ["근거 종류", *COMPANIES.values()]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for kind, label in EVIDENCE_CHECKLIST.items():
        cells = []
        for company in COMPANIES:
            ids = coverage[company][kind]
            cells.append(", ".join(ids) if ids else "❌ 없음")
            problems += not ids
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n팀·창업자·투자 정보(B05, B10)는 company_profiles 입력으로 제공합니다.")
    print(f"\n점검 결과: {'문제 없음' if problems == 0 else f'보완 필요 {problems}건'}")


if __name__ == "__main__":
    main()
