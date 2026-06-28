"""Ingest the craft corpus → chunks → embeddings → Chroma + chunk store.

Markdown-aware chunking: split on headings (semantic seams), then window any
oversized section with overlap. Each chunk keeps its source + heading so answers
can cite. Builds the Chroma vector index and a chunks.json the BM25 sparse
retriever + display layer read.
"""
from __future__ import annotations

import json
import re

from . import config, embed


def discover_docs() -> list:
    """All craft .md/.txt under CORPUS_ROOT, minus noise/personal/derived files."""
    docs = []
    root = config.CORPUS_ROOT
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in config.INCLUDE_SUFFIXES:
            continue
        rel = p.relative_to(root).as_posix()
        if any(rel.startswith(d + "/") or f"/{d}/" in f"/{rel}" for d in config.EXCLUDE_DIRS):
            continue
        if p.name in config.EXCLUDE_NAMES:
            continue
        if any(s in p.name for s in config.EXCLUDE_NAME_SUBSTR):
            continue
        docs.append(p)
    return docs


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections on #/##/### lines."""
    lines = text.splitlines()
    sections, heading, buf = [], "", []
    for ln in lines:
        if re.match(r"^#{1,3}\s+", ln):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = ln.lstrip("# ").strip()
        else:
            buf.append(ln)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def _window(body: str) -> list[str]:
    """Window an oversized body with overlap."""
    if len(body) <= config.CHUNK_CHARS:
        return [body]
    out, start = [], 0
    step = config.CHUNK_CHARS - config.CHUNK_OVERLAP
    while start < len(body):
        out.append(body[start : start + config.CHUNK_CHARS])
        start += step
    return out


def index_text(c: dict) -> str:
    """Text used for retrieval (dense embedding + BM25) — the heading prepended to
    the body. The heading is often the most informative line ("AUDIO RULE", "TOOL
    ORDER — Soul 2.0 is primary"); indexing body-only misses it. Display/citation
    still use the raw body (`text`); only the retrieval signal changes."""
    heading = c.get("heading", "")
    return f"{heading}\n{c['text']}" if heading else c["text"]


def chunk_doc(source: str, text: str) -> list[dict]:
    chunks = []
    for heading, body in _split_sections(text):
        for i, piece in enumerate(_window(body)):
            chunks.append({"source": source, "heading": heading, "text": piece})
    # Fallback: a doc with no headings still gets windowed.
    if not chunks and text.strip():
        for piece in _window(text.strip()):
            chunks.append({"source": source, "heading": "", "text": piece})
    return chunks


def build() -> dict:
    """Full ingest: discover → chunk → embed → write Chroma + chunks.json."""
    import chromadb

    docs = discover_docs()
    all_chunks: list[dict] = []
    for p in docs:
        text = p.read_text(errors="ignore")
        all_chunks.extend(chunk_doc(p.relative_to(config.CORPUS_ROOT).as_posix(), text))

    for i, c in enumerate(all_chunks):
        c["id"] = f"c{i}"

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2))

    vectors = embed.embed_texts([index_text(c) for c in all_chunks])
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=vectors,
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in all_chunks],
    )
    return {"docs": len(docs), "chunks": len(all_chunks), "index": str(config.INDEX_DIR)}
