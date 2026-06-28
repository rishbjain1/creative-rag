"""Offline unit tests for observability — cost math + span aggregation. No network."""
from creative_rag import obs


def test_cost_opus_known_rate():
    # Opus 4.8: $5/M in, $25/M out → 1M in + 1M out = $30
    assert obs.cost_usd("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0


def test_cost_sonnet_known_rate():
    # Sonnet 4.6: $3/M in, $15/M out
    assert obs.cost_usd("claude-sonnet-4-6", 1_000_000, 0) == 3.0
    assert obs.cost_usd("claude-sonnet-4-6", 0, 1_000_000) == 15.0


def test_cost_unknown_model_uses_default():
    assert obs.cost_usd("some-future-model", 1_000_000, 0) == obs.DEFAULT_PRICE[0]


def test_cost_cache_read_discounted():
    # 1M input, all cache-read → 0.1x → $0.50 on Opus
    assert obs.cost_usd("claude-opus-4-8", 1_000_000, 0, cache_read_tok=1_000_000) == 0.5


def test_record_is_noop_outside_scope():
    obs.record({"input_tokens": 5})  # must not raise
    assert obs.summary() == {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                             "cost_usd": 0.0, "llm_latency_ms": 0.0}


def test_request_scope_aggregates_spans():
    with obs.request_scope("t") as trace_id:
        assert len(trace_id) == 12
        obs.record({"model": "claude-opus-4-8", "input_tokens": 100, "output_tokens": 50,
                    "cost_usd": 0.00175, "latency_ms": 12.0})
        obs.record({"model": "claude-opus-4-8", "input_tokens": 200, "output_tokens": 10,
                    "cost_usd": 0.00125, "latency_ms": 8.0})
        s = obs.summary()
    assert s["llm_calls"] == 2
    assert s["input_tokens"] == 300
    assert s["output_tokens"] == 60
    assert s["cost_usd"] == 0.003
    assert s["llm_latency_ms"] == 20.0


def test_scope_resets_after_exit():
    with obs.request_scope():
        obs.record({"input_tokens": 1})
    assert obs.summary()["llm_calls"] == 0  # scope closed
