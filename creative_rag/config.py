"""Configuration — corpus source, models, paths, retrieval knobs.

Everything tunable lives here or in env. The corpus defaults to the local
AI-content knowledge base; override with CRAG_CORPUS_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Corpus ---
CORPUS_ROOT = Path(os.environ.get("CRAG_CORPUS_ROOT", str(Path.home() / "AI content")))
INCLUDE_SUFFIXES = {".md", ".txt"}
# Exclude noise + non-craft + personal docs (see scoping decision).
EXCLUDE_DIRS = {"graphify-out/cache", "MAREA", ".claude"}
EXCLUDE_NAMES = {
    "CAREER_CONTEXT.md",      # personal/career, not craft
    "SESSION_MEMORY.md",      # session/meta, not craft
    ".DS_Store",
}
EXCLUDE_SUFFIXES = {".json"}  # graph index dumps, manifests
EXCLUDE_NAME_SUBSTR = ("_manifest",)

# --- Index storage ---
INDEX_DIR = Path(os.environ.get("CRAG_INDEX_DIR", str(Path(__file__).resolve().parent.parent / "index")))
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
COLLECTION = "craft"

# --- Models (local, PyTorch via sentence-transformers) ---
EMBED_MODEL = os.environ.get("CRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("CRAG_RERANK_MODEL", "BAAI/bge-reranker-base")

# --- Chunking ---
CHUNK_CHARS = int(os.environ.get("CRAG_CHUNK_CHARS", "2200"))   # ~550 tokens
CHUNK_OVERLAP = int(os.environ.get("CRAG_CHUNK_OVERLAP", "200"))

# --- Retrieval ---
DENSE_K = int(os.environ.get("CRAG_DENSE_K", "20"))
SPARSE_K = int(os.environ.get("CRAG_SPARSE_K", "20"))
RRF_K = 60                                                      # RRF constant
RERANK_CANDIDATES = int(os.environ.get("CRAG_RERANK_CANDIDATES", "30"))
TOP_K = int(os.environ.get("CRAG_TOP_K", "6"))                  # final chunks to the LLM

# --- LLM (provider-agnostic, OpenAI-compatible; defaults to Anthropic) ---
LLM_BASE_URL = os.environ.get("CRAG_LLM_BASE_URL", "https://api.anthropic.com/v1")
LLM_MODEL = os.environ.get("CRAG_LLM_MODEL", "claude-opus-4-8")
LLM_MAX_TOKENS = int(os.environ.get("CRAG_LLM_MAX_TOKENS", "2048"))

# --- API auth ---
API_KEY = os.environ.get("CRAG_API_KEY", "")  # if set, /query requires X-API-Key
