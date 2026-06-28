"""Provider-agnostic LLM layer (OpenAI-compatible Chat Completions).

One code path over any OpenAI-compatible endpoint — defaults to Anthropic.
Used by generation + citation-verify. Key from CRAG_LLM_API_KEY or ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

from . import config, obs


def _key() -> str:
    key = os.environ.get("CRAG_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No LLM key. Set CRAG_LLM_API_KEY or ANTHROPIC_API_KEY.")
    return key.strip()


def chat(messages: list[dict], model: str | None = None) -> str:
    model = model or config.LLM_MODEL
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
        },
        timeout=120,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    # Observability: price the call from its token usage and record a span.
    usage = body.get("usage") or {}
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    obs.record({
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": obs.cost_usd(model, in_tok, out_tok),
        "latency_ms": latency_ms,
    })
    return body["choices"][0]["message"]["content"]


def chat_json(messages: list[dict], model: str | None = None) -> dict:
    text = chat(messages, model)
    parsed = _extract_json(text)
    if parsed is None:
        raise ValueError(f"Model did not return valid JSON:\n{text[:400]}")
    return parsed


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
