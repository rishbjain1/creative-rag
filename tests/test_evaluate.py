"""Offline unit tests for the eval metrics — pure ranking math + gold resolution.

No models, no network, no index. Toy fixtures with hand-computed expected values.
"""
import math

from creative_rag import evaluate as ev


# --- recall@k ---
def test_recall_at_k_partial():
    # gold = {a, c, e}; top-3 retrieved = a, b, c → 2 of 3 found
    assert ev.recall_at_k(["a", "b", "c", "d"], ["a", "c", "e"], 3) == 2 / 3


def test_recall_at_k_respects_cutoff():
    # the only gold hit sits at rank 4, outside k=3 → 0 recall
    assert ev.recall_at_k(["b", "d", "f", "a"], ["a"], 3) == 0.0


def test_recall_at_k_none_when_no_gold():
    assert ev.recall_at_k(["a", "b"], [], 3) is None


# --- reciprocal rank ---
def test_reciprocal_rank_first_hit_at_two():
    assert ev.reciprocal_rank(["x", "a", "y"], ["a"]) == 0.5


def test_reciprocal_rank_zero_when_absent():
    assert ev.reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank_none_when_no_gold():
    assert ev.reciprocal_rank(["a", "b"], []) is None


# --- nDCG ---
def test_ndcg_perfect_is_one():
    # both gold at the top → DCG == IDCG
    assert ev.ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_ndcg_demoted_hit():
    # single gold at rank 3 (index 2): DCG = 1/log2(4); IDCG = 1/log2(2)=1
    expected = (1 / math.log2(4)) / 1.0
    assert abs(ev.ndcg_at_k(["x", "y", "a"], ["a"], 3) - expected) < 1e-9


def test_ndcg_none_when_no_gold():
    assert ev.ndcg_at_k(["a"], [], 3) is None


# --- gold resolution (source + phrase, reindex-robust) ---
def _chunks():
    return [
        {"id": "c0", "source": "lib.md", "text": "Cinestill 800T is the night stock"},
        {"id": "c1", "source": "lib.md", "text": "Portra 400 is the day stock"},
        {"id": "c2", "source": "other.md", "text": "Cinestill 800T mentioned elsewhere"},
    ]


def test_gold_matches_source_and_phrase():
    gold = ev.gold_chunk_ids(_chunks(), ["lib.md"], ["Cinestill 800T"])
    assert gold == ["c0"]  # c2 has the phrase but wrong source


def test_gold_is_case_insensitive():
    gold = ev.gold_chunk_ids(_chunks(), ["lib.md"], ["cinestill 800t"])
    assert gold == ["c0"]


def test_gold_empty_when_phrase_absent():
    assert ev.gold_chunk_ids(_chunks(), ["lib.md"], ["Vision3 500T"]) == []


# --- gate logic ---
def test_gate_flags_below_threshold_and_unmatched():
    result = {"aggregate": {"recall@k": 0.5, "mrr": 0.9, "ndcg@k": 0.9, "unmatched_labels": 1, "k": 6}}
    fails = ev._gate(result, {"recall@k": 0.8, "mrr": 0.8, "ndcg@k": 0.8})
    assert any("recall@k" in f for f in fails)
    assert any("matched no gold" in f for f in fails)


def test_gate_passes_when_all_met():
    result = {"aggregate": {"recall@k": 0.95, "mrr": 0.95, "ndcg@k": 0.95, "unmatched_labels": 0, "k": 6}}
    assert ev._gate(result, {"recall@k": 0.8, "mrr": 0.8, "ndcg@k": 0.8}) == []
