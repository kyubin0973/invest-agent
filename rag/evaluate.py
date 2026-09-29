"""RAG 검색 성능 평가 (Hit Rate@K, MRR)

실제 파이프라인(Chroma + e5-base + 설계 청크 설정 + Agent별 검색 범위)으로 측정한다.
정답은 질문을 만든 원문 페이지(source_id + page)이며, 그 페이지의 청크가 검색되면 적중으로 본다.

    # 1) 질문셋 생성 (OpenAI 사용, 1회) → 사람이 검토
    uv run python -m rag.evaluate make-qa

    # 2) 측정
    uv run python -m rag.evaluate run

    # 3) 임베딩 모델 비교 (같은 청크, 같은 질문셋, 같은 검색 범위에서 모델만 바꿔 측정)
    uv run python -m rag.evaluate compare
"""

import argparse
import json
import random
import time
from collections import defaultdict

from pydantic import BaseModel, Field

from documents.config import COMPANIES, ROOT
from documents.loader import base_metadata, company_of, load_manifest, load_pages
from rag.retriever import search, search_industry, search_technology

EVAL_DIR = ROOT / "rag" / "eval"
QA_PATH = EVAL_DIR / "qa_set.json"
RESULTS_PATH = EVAL_DIR / "results.json"

PAGES_PER_COMPANY = 6
INDUSTRY_PAGES = 15
MIN_PAGE_CHARS = 400
KS = (1, 3, 5)


# ---------------------------------------------------------------------------
# 질문셋 생성
# ---------------------------------------------------------------------------
class GeneratedQuestion(BaseModel):
    question_ko: str = Field(description="한국어 질문")
    question_en: str = Field(description="같은 의미의 영어 질문")


QA_PROMPT = """다음은 {scope} 문서의 한 페이지다.
투자 분석가가 휴머노이드 스타트업을 평가하면서 실제로 검색할 법한 질문을 1개 만들어라.

규칙:
- 이 페이지를 읽어야 답할 수 있는 구체적인 질문 (수치, 사례, 기술 특성, 시장 동향 등)
- 본문 표현을 그대로 베끼지 말고 자연스럽게 바꿔 말할 것
- {company_rule}
- 한국어 질문과 같은 의미의 영어 질문을 함께 작성

[문서] {title} ({publisher})
[본문]
{text}"""


def sample_pages(seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in load_manifest():
        meta = base_metadata(row)
        for page in load_pages(row):
            if len(page["text"]) >= MIN_PAGE_CHARS:
                by_group[company_of(row) or "industry"].append({**meta, **page})

    picked = []
    for group, pages in sorted(by_group.items()):
        n = INDUSTRY_PAGES if group == "industry" else PAGES_PER_COMPANY
        picked += rng.sample(pages, min(n, len(pages)))
    return picked


def cmd_make_qa(args: argparse.Namespace) -> None:
    from core.llm import get_structured_llm

    llm = get_structured_llm(GeneratedQuestion)

    qa_set = []
    pages = sample_pages(args.seed)
    for i, p in enumerate(pages, start=1):
        is_company = p["company"] != "common"
        name = COMPANIES.get(p["company"])
        q = llm.invoke(
            QA_PROMPT.format(
                scope=f"{name} 기업" if is_company else "휴머노이드 산업",
                company_rule=f"질문에 기업명 '{name}'을 자연스럽게 포함할 것"
                if is_company
                else "특정 보고서명이나 발행기관 이름은 넣지 말 것",
                title=p["title"],
                publisher=p["publisher"],
                text=p["text"][:6000],
            )
        )
        base = {
            "scope": "technology" if is_company else "industry",
            "company": p["company"] if is_company else None,
            "answer": {"source_id": p["source_id"], "page": p["page"]},
        }
        qa_set.append({**base, "lang": "ko", "question": q.question_ko})
        qa_set.append({**base, "lang": "en", "question": q.question_en})
        print(f"[{i}/{len(pages)}] {p['source_id']} p.{p['page']} → {q.question_ko}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa_set, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n질문 {len(qa_set)}개 저장: {QA_PATH.relative_to(ROOT)} (검토 후 run 실행)")


# ---------------------------------------------------------------------------
# 측정
# ---------------------------------------------------------------------------
def retrieve(qa: dict, mode: str, k: int, translate: bool = True):
    """mode=scoped: Agent가 실제로 쓰는 범위 / mode=unfiltered: 메타데이터 필터 없이 전체 검색."""
    if mode == "unfiltered":
        return search(qa["question"], k=k, translate=translate)
    if qa["scope"] == "technology":
        return search_technology(qa["question"], qa["company"], k=k, translate=translate)
    return search_industry(qa["question"], k=k, translate=translate)


def rank_of(qa: dict, docs) -> int | None:
    gold = qa["answer"]
    for rank, d in enumerate(docs, start=1):
        if d.metadata["source_id"] == gold["source_id"] and d.metadata["page"] == gold["page"]:
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        out[f"hit@{k}"] = round(sum(r is not None and r <= k for r in ranks) / n, 3)
    out[f"mrr@{max(KS)}"] = round(sum(1 / r for r in ranks if r is not None) / n, 3)
    return out


def cmd_run(args: argparse.Namespace) -> None:
    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    k = max(KS)
    translate = not args.no_translate
    results_path = RESULTS_PATH if translate else RESULTS_PATH.with_name("results_no_translate.json")
    print(f"한국어 질의 자동 번역: {'켜짐' if translate else '꺼짐'}")

    per_question = []
    for qa in qa_set:
        row = {**qa}
        for mode in ("scoped", "unfiltered"):
            docs = retrieve(qa, mode, k, translate)
            row["search_query"] = docs[0].metadata["search_query"] if docs else qa["question"]
            row[f"rank_{mode}"] = rank_of(qa, docs)
            row[f"top1_{mode}"] = docs[0].metadata["chunk_id"] if docs else None
        per_question.append(row)

    groups = {
        "전체": per_question,
        "기업 자료 (technology)": [q for q in per_question if q["scope"] == "technology"],
        "산업 자료 (industry)": [q for q in per_question if q["scope"] == "industry"],
        "한국어 질의": [q for q in per_question if q["lang"] == "ko"],
        "영어 질의": [q for q in per_question if q["lang"] == "en"],
    }

    summary = {}
    for mode, title in (("scoped", "Agent 검색 범위 적용 (실제 사용 방식)"), ("unfiltered", "필터 없이 전체 검색 (참고)")):
        print(f"\n### {title}")
        print("| 구분 | n | " + " | ".join(f"Hit@{k}" for k in KS) + f" | MRR@{max(KS)} |")
        print("|---|---|" + "---|" * (len(KS) + 1))
        summary[mode] = {}
        for name, qs in groups.items():
            s = summarize([q[f"rank_{mode}"] for q in qs])
            summary[mode][name] = s
            cells = " | ".join(f"{s[f'hit@{k}']:.3f}" for k in KS)
            print(f"| {name} | {s['n']} | {cells} | {s[f'mrr@{max(KS)}']:.3f} |")

    misses = [q for q in per_question if q["rank_scoped"] is None]
    print(f"\nTop-{k} 안에 정답 페이지가 없는 질문: {len(misses)}개")
    for q in misses:
        print(f"  - [{q['answer']['source_id']} p.{q['answer']['page']}] ({q['lang']}) {q['question']}  → top1 {q['top1_scoped']}")

    results_path.write_text(
        json.dumps(
            {
                "measured_at": time.strftime("%Y-%m-%d %H:%M"),
                "translate_korean_query": translate,
                "summary": summary,
                "per_question": per_question,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n상세 결과: {results_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 임베딩 모델 비교
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = {
    "e5-base": {
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    "gte-base": {
        "name": "Alibaba-NLP/gte-multilingual-base",
        "trust_remote_code": True,  # 모델 저장소의 커스텀 코드로 로드
    },
}
COMPARE_PATH = EVAL_DIR / "embedding_comparison.json"


def _scope_mask(chunks: list, qa: dict):
    import numpy as np

    if qa["scope"] == "technology":
        return np.array([c.metadata["domain"] == "technology" and c.metadata["company"] == qa["company"] for c in chunks])
    return np.array([c.metadata["domain"] == "market" for c in chunks])


def cmd_compare(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from rag.build_index import build_documents

    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    chunks = build_documents()  # 실제 인덱스와 같은 청크
    masks = [_scope_mask(chunks, qa) for qa in qa_set]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    k = max(KS)
    print(f"청크 {len(chunks)}개 / 질문 {len(qa_set)}개 / device={device}")

    report = {}
    for key in args.models.split(","):
        cfg = CANDIDATE_MODELS[key]
        t0 = time.perf_counter()
        model = SentenceTransformer(cfg["name"], device=device, trust_remote_code=cfg.get("trust_remote_code", False))
        load_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        doc_emb = model.encode(
            [cfg.get("doc_prefix", "") + c.page_content for c in chunks],
            batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        )
        doc_sec = time.perf_counter() - t0
        q_emb = model.encode(
            [cfg.get("query_prefix", "") + q["question"] for q in qa_set],
            batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        )

        per_question = []
        for qa, qv, mask in zip(qa_set, q_emb, masks):
            sims = np.where(mask, doc_emb @ qv, -np.inf)  # Agent 검색 범위 밖은 제외
            top = np.argsort(-sims)[:k]
            gold = qa["answer"]
            rank = next(
                (r for r, i in enumerate(top, start=1)
                 if chunks[i].metadata["source_id"] == gold["source_id"] and chunks[i].metadata["page"] == gold["page"]),
                None,
            )
            per_question.append({"question": qa["question"], "scope": qa["scope"], "lang": qa["lang"], "rank": rank})

        groups = {
            "전체": per_question,
            "기업 자료": [q for q in per_question if q["scope"] == "technology"],
            "산업 자료": [q for q in per_question if q["scope"] == "industry"],
            "한국어 질의": [q for q in per_question if q["lang"] == "ko"],
            "영어 질의": [q for q in per_question if q["lang"] == "en"],
        }
        report[key] = {
            "model": cfg["name"],
            "dimension": int(doc_emb.shape[1]),
            "load_sec": round(load_sec, 1),
            "doc_embed_sec": round(doc_sec, 1),
            "summary": {g: summarize([q["rank"] for q in qs]) for g, qs in groups.items()},
            "per_question": per_question,
        }
        del model
        if device == "mps":
            torch.mps.empty_cache()

    keys = list(report)
    for group in report[keys[0]]["summary"]:
        print(f"\n### {group}")
        print("| 모델 | n | " + " | ".join(f"Hit@{k}" for k in KS) + f" | MRR@{max(KS)} |")
        print("|---|---|" + "---|" * (len(KS) + 1))
        for key in keys:
            s = report[key]["summary"][group]
            print(f"| {key} | {s['n']} | " + " | ".join(f"{s[f'hit@{k}']:.3f}" for k in KS) + f" | {s[f'mrr@{max(KS)}']:.3f} |")

    print("\n### 실행 비용")
    print("| 모델 | 차원 | 모델 로드(초) | 청크 임베딩(초) |")
    print("|---|---|---|---|")
    for key in keys:
        r = report[key]
        print(f"| {key} | {r['dimension']} | {r['load_sec']} | {r['doc_embed_sec']} |")

    if len(keys) == 2:
        a, b = (report[x]["per_question"] for x in keys)
        better = sum((x["rank"] or 99) < (y["rank"] or 99) for x, y in zip(a, b))
        worse = sum((x["rank"] or 99) > (y["rank"] or 99) for x, y in zip(a, b))
        print(f"\n질문별 순위: {keys[0]}가 더 높은 질문 {better}개, {keys[1]}가 더 높은 질문 {worse}개, 같음 {len(a) - better - worse}개")

    COMPARE_PATH.write_text(
        json.dumps({"measured_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "results": report},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n상세 결과: {COMPARE_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 검색 성능 평가 (Hit Rate@K, MRR)")
    sub = parser.add_subparsers(dest="command", required=True)

    mq = sub.add_parser("make-qa", help="문서 페이지에서 질문셋 생성 (core/llm.py의 공통 모델 사용)")
    mq.add_argument("--seed", type=int, default=42)
    mq.set_defaults(func=cmd_make_qa)

    run = sub.add_parser("run", help="Hit Rate@K, MRR 측정")
    run.add_argument("--no-translate", action="store_true", help="한국어 질의 자동 번역 없이 측정")
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="임베딩 모델 비교")
    cmp.add_argument("--models", default="e5-base,gte-base", help=f"사용 가능: {','.join(CANDIDATE_MODELS)}")
    cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
