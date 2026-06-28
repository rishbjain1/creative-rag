"""Lightweight observability — per-LLM-call cost, latency, and a request trace.

Production LLM systems need to answer "what did this request cost, how slow was
it, and what ran" — so each LLM call is timed, priced from its token usage, and
emitted as a structured trace line. A `request_scope` accumulates the calls made
while serving one request and exposes an aggregate (n_calls, tokens, cost_usd,
latency_ms) the API returns alongside the answer.

No external deps, no vendor agent — just enough to make cost/latency legible.
"""
from __future__ import annotations

import contextvars
import json
import sys
import time
import uuid
from contextlib import contextmanager

# --- Pricing: USD per 1M tokens (input, output). Source: claude-api reference,
# 2026-06. Cache reads bill at ~0.1x input; cache writes at 1.25x (5m TTL).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}
DEFAULT_PRICE = (5.0, 25.0)  # unknown model → assume Opus-tier (conservative)
CACHE_READ_MULT = 0.1


def price_for(model: str) -> tuple[float, float]:
    return PRICES.get(model, DEFAULT_PRICE)


def cost_usd(model: str, in_tok: int, out_tok: int, cache_read_tok: int = 0) -> float:
    """Cost of one call. Cache-read tokens bill at 0.1x input."""
    p_in, p_out = price_for(model)
    fresh_in = max(0, in_tok - cache_read_tok)
    dollars = (fresh_in * p_in + cache_read_tok * p_in * CACHE_READ_MULT + out_tok * p_out) / 1_000_000
    return round(dollars, 6)


# --- Per-request span accumulation (contextvar = thread/async-safe, no globals) ---

_spans: contextvars.ContextVar[list | None] = contextvars.ContextVar("crag_spans", default=None)


def record(span: dict) -> None:
    """Add a span to the active request scope (no-op if none is open)."""
    spans = _spans.get()
    if spans is not None:
        spans.append(span)


def _emit(event: dict) -> None:
    """Structured trace line to stderr (one JSON object per line)."""
    print(json.dumps(event), file=sys.stderr, flush=True)


@contextmanager
def request_scope(name: str = "request"):
    """Open a trace scope; on exit emit an aggregated trace event. Yields the trace_id."""
    trace_id = uuid.uuid4().hex[:12]
    token = _spans.set([])
    t0 = time.perf_counter()
    try:
        yield trace_id
    finally:
        spans = _spans.get() or []
        agg = _aggregate(spans)
        _emit({
            "event": "request",
            "trace_id": trace_id,
            "name": name,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            **agg,
        })
        _spans.reset(token)


def _aggregate(spans: list[dict]) -> dict:
    return {
        "llm_calls": len(spans),
        "input_tokens": sum(s.get("input_tokens", 0) for s in spans),
        "output_tokens": sum(s.get("output_tokens", 0) for s in spans),
        "cost_usd": round(sum(s.get("cost_usd", 0.0) for s in spans), 6),
        "llm_latency_ms": round(sum(s.get("latency_ms", 0.0) for s in spans), 1),
    }


def summary() -> dict:
    """Aggregate of the spans recorded so far in the active scope (for API responses)."""
    return _aggregate(_spans.get() or [])
