"""Local embedding + reranking models (PyTorch / sentence-transformers).

Bi-encoder for retrieval embeddings; cross-encoder for reranking. Both run
locally — no API, no key. Lazy-loaded (first call downloads weights, then cached).
Swappable: this module is the only place models live, so an API backend can
replace it without touching retrieve/ingest.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (normalized — cosine == dot product)."""
    vecs = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def rerank(query: str, candidates: list[str]) -> list[float]:
    """Cross-encoder relevance scores for (query, candidate) pairs.

    The cross-encoder reads query+candidate JOINTLY (unlike the bi-encoder),
    so it judges true relevance — the precision stage of the funnel.
    """
    if not candidates:
        return []
    pairs = [(query, c) for c in candidates]
    scores = _reranker().predict(pairs, show_progress_bar=False)
    return [float(s) for s in scores]
