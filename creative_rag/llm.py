"""Provider-agnostic LLM layer (OpenAI-compatible Chat Completions).

One code path over any OpenAI-compatible endpoint — defaults to Anthropic.
Used by generation + citation-verify. Key from CRAG_LLM_API_KEY or ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import os
import re

import httpx

from . import config


def _key() -> str:
    key = os.environ.get("CRAG_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No LLM key. Set CRAG_LLM_API_KEY or ANTHROPIC_API_KEY.")
    return key.strip()


def chat(messages: list[dict], model: str | None = None) -> str:
    resp = httpx.post(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
        json={
            "model": model or config.LLM_MODEL,
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:500]}")
    return resp.json()["choices"][0]["message"]["content"]


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
