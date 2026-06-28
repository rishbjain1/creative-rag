"""Offline unit tests — chunking, fusion, tokenization, JSON parse. No models/network."""
from creative_rag import ingest, llm
from creative_rag.retrieve import Retriever, _tokenize


# --- chunking ---
def test_split_sections_on_headings():
    md = "# Title\nintro\n## Stocks\n500T is warm\n## Lenses\n75mm for faces"
    secs = ingest._split_sections(md)
    headings = [h for h, _ in secs]
    assert "Stocks" in headings and "Lenses" in headings


def test_window_overlaps_oversized():
    body = "x" * (ingest.config.CHUNK_CHARS + 500)
    pieces = ingest._window(body)
    assert len(pieces) >= 2
    assert all(len(p) <= ingest.config.CHUNK_CHARS for p in pieces)


def test_chunk_doc_carries_source_and_heading():
    chunks = ingest.chunk_doc("lib.md", "## Lighting\nbacklight at dusk")
    assert chunks and chunks[0]["source"] == "lib.md"
    assert chunks[0]["heading"] == "Lighting"


def test_chunk_doc_fallback_no_headings():
    chunks = ingest.chunk_doc("flat.md", "just prose, no headings here")
    assert len(chunks) == 1 and chunks[0]["text"]


def test_index_text_prepends_heading():
    # the heading must be in the retrieval text so heading-only terms are findable
    assert ingest.index_text({"heading": "AUDIO RULE", "text": "no music, diegetic SFX"}) \
        == "AUDIO RULE\nno music, diegetic SFX"


def test_index_text_body_only_when_no_heading():
    assert ingest.index_text({"heading": "", "text": "body only"}) == "body only"


# --- fusion + tokenization ---
def test_rrf_rewards_top_ranks():
    a = ["x", "y", "z"]
    b = ["y", "w"]
    fused = Retriever._rrf(a, b)
    assert fused[0] == "y"  # ranked high in both
    assert set(fused) == {"x", "y", "z", "w"}


def test_tokenize_keeps_hex_and_numbers():
    toks = _tokenize("Shot on 500T, palette #c8a15a, 21:9")
    assert "500t" in toks and "#c8a15a" in toks and "21" in toks


# --- llm json extraction ---
def test_extract_json_fenced():
    assert llm._extract_json('```json\n{"supported": true}\n```') == {"supported": True}


def test_extract_json_with_prose():
    assert llm._extract_json('verdict: {"supported": false} done')["supported"] is False
