"""E5 계열 임베딩: 문서에는 "passage: ", 질의에는 "query: " 접두어를 붙인다."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from rag.config import EMBEDDING_MODEL


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=_pick_device())
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]


@lru_cache
def get_embeddings() -> E5Embeddings:
    return E5Embeddings()
