

#Model: BAAI/bge-m3


from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    s = get_settings()
    return SentenceTransformer(s.embedding_model, device=s.embedding_device)


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = _model()
    vecs = model.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    model = _model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()
