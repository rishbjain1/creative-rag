"""Augment → generate → citation-verify.

Grounds the answer in retrieved chunks (rerank order, lost-in-the-middle aware),
then a second LLM pass checks every claim is supported by the notes — the eval
layer that makes the answer trustworthy.
"""
from __future__ import annotations

from . import config, llm
from .retrieve import Retriever

ANSWER_SYSTEM = """You are a cinematography + AI-filmmaking assistant. Answer the
question USING ONLY the numbered notes provided. Cite the note number(s) you used
inline like [3]. If the notes do not cover the question, say "Not in the corpus."
Do not use outside knowledge. Be concrete and practical."""

VERIFY_SYSTEM = """You verify a RAG answer against its source notes. For each claim
in the answer, check whether the notes actually support it. Return ONLY JSON:
{"supported": true|false, "unsupported_claims": ["..."], "notes": "<one line>"}
supported=false if ANY substantive claim is not grounded in the notes."""

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        src = c["source"] + (f" §{c['heading']}" if c.get("heading") else "")
        blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query: str, top_k: int | None = None, verify: bool = True) -> dict:
    chunks = _get_retriever().retrieve(query, top_k)
    if not chunks:
        return {"query": query, "answer": "Not in the corpus.", "sources": [], "verification": None}

    ctx = _context(chunks)
    reply = llm.chat([
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"NOTES:\n{ctx}\n\nQUESTION: {query}"},
    ])

    verification = None
    if verify:
        try:
            verification = llm.chat_json([
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": f"NOTES:\n{ctx}\n\nANSWER:\n{reply}"},
            ])
        except Exception as e:  # verification is best-effort, never blocks the answer
            verification = {"supported": None, "error": str(e)[:200]}

    sources = [
        {"n": i + 1, "source": c["source"], "heading": c.get("heading", ""),
         "rerank_score": c.get("rerank_score")}
        for i, c in enumerate(chunks)
    ]
    return {"query": query, "answer": reply, "sources": sources, "verification": verification}
