"""기존 근거 ID로 QA 원문을 일괄 조회한다. 검색·번역·새 근거 생성은 하지 않는다."""
from evaluation.report_quality import CITATION, QualityJudgeError


def load_original_documents(state: dict, *, fetch=None) -> dict:
    report = state.get("final_report")
    if not isinstance(report, str):
        return {}
    cited = {(source, chunk) for chunk, source in CITATION.findall(report)}
    available = set()
    for key in ("technology_results", "market_traction_results"):
        for result in (state.get(key) or {}).values():
            available.update((e.get("source_id"), e.get("chunk_id")) for e in result.get("evidence", []))
    for result in (state.get("investment_results") or {}).values():
        for criterion in result.get("criteria", []):
            available.update((e.get("source_id"), e.get("chunk_id")) for e in criterion.get("evidence", []))
    requested = cited & available
    if not requested:
        return {}
    ids = sorted({chunk for _, chunk in requested})
    try:
        if fetch is None:
            from rag.retriever import get_vectorstore
            fetch = get_vectorstore().get
        result = fetch(ids=ids, include=["documents", "metadatas"])
        returned_ids, texts, metadata = result["ids"], result["documents"], result["metadatas"]
        if not (len(returned_ids) == len(texts) == len(metadata)) or len(set(returned_ids)) != len(returned_ids):
            raise ValueError("invalid source response")
        originals = {}
        for chunk, text, meta in zip(returned_ids, texts, metadata):
            if not isinstance(meta, dict) or chunk != meta.get("chunk_id"):
                raise ValueError("source metadata mismatch")
            if (meta.get("source_id"), chunk) not in requested:
                raise ValueError("unexpected source")
            if type(meta.get("page")) is not int or meta["page"] < 1 or not isinstance(text, str) or not text.strip():
                raise ValueError("invalid original")
            originals[chunk] = {"source_id": meta["source_id"], "chunk_id": chunk,
                                "page": meta["page"], "text": text}
        if set(originals) != set(ids):
            raise ValueError("missing indexed originals")
        # 저장소의 원문 누락을 분석 근거 부족으로 오인해 재조사시키지 않는다.
        return originals
    except (Exception, SystemExit):
        # 조회 실패는 QA 실행 오류다. 분석 담당 재검색 요청으로 바꾸지 않는다.
        raise QualityJudgeError("QA 원문 저장소 조회 실패; 기존 인덱스와 조회 경로를 확인하세요.") from None
