"""Hybrid retrieval funnel: dense + sparse → RRF → cross-encoder rerank.

Wide net cheaply (bi-encoder dense + BM25 sparse, fused by Reciprocal Rank
Fusion), then a precise cross-encoder rerank over the small candidate set.
"""
from __future__ import annotations

import json
import re

from . import config, embed
from .ingest import index_text


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9#]+", text.lower())


class Retriever:
    def __init__(self):
        import chromadb
        from rank_bm25 import BM25Okapi

        self.chunks = json.loads(config.CHUNKS_PATH.read_text())
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([_tokenize(index_text(c)) for c in self.chunks])
        self._ids = [c["id"] for c in self.chunks]
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.col = client.get_collection(config.COLLECTION)

    def _dense(self, query: str, k: int) -> list[str]:
        res = self.col.query(query_embeddings=[embed.embed_query(query)], n_results=k)
        return res["ids"][0]

    def _sparse(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked[:k]]

    @staticmethod
    def _rrf(*ranked_lists: list[str]) -> list[str]:
        """Reciprocal Rank Fusion — scale-free merge by rank."""
        scores: dict[str, float] = {}
        for lst in ranked_lists:
            for rank, cid in enumerate(lst):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K
        dense = self._dense(query, config.DENSE_K)
        sparse = self._sparse(query, config.SPARSE_K)
        fused = self._rrf(dense, sparse)[: config.RERANK_CANDIDATES]
        if not fused:
            return []
        texts = [index_text(self.by_id[cid]) for cid in fused]
        scores = embed.rerank(query, texts)
        order = sorted(range(len(fused)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order[:top_k]:
            c = dict(self.by_id[fused[i]])
            c["rerank_score"] = round(scores[i], 4)
            out.append(c)
        return out
