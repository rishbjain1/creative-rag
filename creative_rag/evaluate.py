"""Offline eval harness — measure retrieval + generation against a labeled set.

Two layers, mirroring the retrieval funnel:

- **Retrieval metrics** (recall@k, MRR, nDCG@k) — pure, deterministic, no LLM.
  Gold chunks are defined by (source ∈ gold_sources) AND (a gold_phrase appears
  in the chunk text), so labels survive a reindex — they never pin brittle chunk
  ids. This layer is the CI regression gate.
- **Generation metrics** (answer-correctness, faithfulness) — needs the LLM, so
  it runs locally with `--with-llm`. Correctness = every `answer_must_include`
  term is present; faithfulness = the built-in citation-verify pass said
  `supported: true`.

The harness is corpus-agnostic: point `--qa` at any labeled set; it evals against
whatever index `config` resolves (real corpus locally, sample corpus in CI).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import config

# --- Pure ranking metrics (unit-tested in tests/test_evaluate.py) ---


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """Fraction of gold chunks present in the top-k retrieved. None if no gold."""
    if not gold:
        return None
    hits = len(set(retrieved[:k]) & set(gold))
    return hits / len(gold)


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """1.0 if any gold chunk is in the top-k (the 'found an answer' signal when
    gold spans several chunks and recall@k can't reach 1). None if no gold."""
    if not gold:
        return None
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float | None:
    """1 / rank of the first relevant hit. 0 if no hit; None if no gold (so a
    labeling bug is excluded from the mean, not scored as a miss)."""
    if not gold:
        return None
    gold_set = set(gold)
    for i, cid in enumerate(retrieved):
        if cid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """Normalized DCG over binary relevance. None if no gold."""
    if not gold:
        return None
    gold_set = set(gold)
    rels = [1.0 if cid in gold_set else 0.0 for cid in retrieved[:k]]
    ideal = [1.0] * min(len(gold_set), k) + [0.0] * max(0, k - len(gold_set))
    idcg = _dcg(ideal)
    if idcg == 0:
        return None
    return _dcg(rels) / idcg


# --- Gold resolution (reindex-robust: source + phrase, not chunk id) ---


def gold_chunk_ids(chunks: list[dict], gold_sources: list[str], gold_phrases: list[str]) -> list[str]:
    srcs = set(gold_sources)
    phrases = [p.lower() for p in gold_phrases]
    return [
        c["id"]
        for c in chunks
        if c["source"] in srcs and any(p in c["text"].lower() for p in phrases)
    ]


# --- Harness ---


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def evaluate(qa_path: Path, k: int = 6, with_llm: bool = False) -> dict:
    """Run the labeled set through retrieval (+ optionally generation)."""
    from .retrieve import Retriever

    chunks = json.loads(config.CHUNKS_PATH.read_text())
    retriever = Retriever()
    items = [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]

    generate = None
    if with_llm:
        from . import generate as _gen

        generate = _gen

    rows: list[dict] = []
    for it in items:
        gold = gold_chunk_ids(chunks, it["gold_sources"], it["gold_phrases"])
        retrieved = [c["id"] for c in retriever.retrieve(it["q"], k)]
        row = {
            "q": it["q"],
            "n_gold": len(gold),
            "hit@k": hit_at_k(retrieved, gold, k),
            "recall@k": recall_at_k(retrieved, gold, k),
            "mrr": reciprocal_rank(retrieved, gold),
            "ndcg@k": ndcg_at_k(retrieved, gold, k),
        }
        if not gold:
            # A label that matches nothing is a labeling bug, not a model failure.
            row["warning"] = "no gold chunks matched gold_sources/gold_phrases"
        if generate is not None:
            res = generate.answer(it["q"], top_k=k, verify=True)
            ans = res["answer"].lower()
            musts = [m.lower() for m in it.get("answer_must_include", [])]
            row["correct"] = all(m in ans for m in musts) if musts else None
            verdict = res.get("verification") or {}
            row["faithful"] = verdict.get("supported")
        rows.append(row)

    agg = {
        "n_questions": len(rows),
        "k": k,
        "hit@k": _mean([r["hit@k"] for r in rows]),
        "recall@k": _mean([r["recall@k"] for r in rows]),
        "mrr": _mean([r["mrr"] for r in rows]),
        "ndcg@k": _mean([r["ndcg@k"] for r in rows]),
        "unmatched_labels": sum(1 for r in rows if r.get("warning")),
    }
    if with_llm:
        corrects = [r.get("correct") for r in rows if r.get("correct") is not None]
        faiths = [r.get("faithful") for r in rows if r.get("faithful") is not None]
        agg["correctness"] = (sum(corrects) / len(corrects)) if corrects else None
        agg["faithfulness"] = (sum(faiths) / len(faiths)) if faiths else None
    return {"aggregate": agg, "rows": rows}


# --- Reporting + gate ---


def _fmt(v) -> str:
    if v is None:
        return "  —  "
    if isinstance(v, bool):
        return " ok  " if v else "FAIL "
    return f"{v:.3f}"


def _print_table(result: dict) -> None:
    agg = result["aggregate"]
    has_llm = "correctness" in agg
    head = f"{'hit@k':>6} {'recall@k':>9} {'mrr':>7} {'ndcg@k':>7}"
    if has_llm:
        head += f" {'correct':>8} {'faithful':>9}"
    head += "  question"
    print(head)
    print("-" * len(head))
    for r in result["rows"]:
        line = f"{_fmt(r['hit@k']):>6} {_fmt(r['recall@k']):>9} {_fmt(r['mrr']):>7} {_fmt(r['ndcg@k']):>7}"
        if has_llm:
            line += f" {_fmt(r.get('correct')):>8} {_fmt(r.get('faithful')):>9}"
        q = r["q"] if len(r["q"]) <= 60 else r["q"][:57] + "..."
        line += f"  {q}"
        if r.get("warning"):
            line += "  ⚠ unmatched label"
        print(line)
    print("-" * len(head))
    summ = f"MEAN  hit@{agg['k']}={_fmt(agg['hit@k'])}  recall@{agg['k']}={_fmt(agg['recall@k'])}  MRR={_fmt(agg['mrr'])}  nDCG={_fmt(agg['ndcg@k'])}"
    if has_llm:
        summ += f"  correctness={_fmt(agg.get('correctness'))}  faithfulness={_fmt(agg.get('faithfulness'))}"
    print(summ)
    print(f"questions={agg['n_questions']}  unmatched_labels={agg['unmatched_labels']}")


def _gate(result: dict, thresholds: dict) -> list[str]:
    """Return a list of failures; empty = pass."""
    agg = result["aggregate"]
    fails = []
    if agg["unmatched_labels"] > 0:
        fails.append(f"{agg['unmatched_labels']} label(s) matched no gold chunk")
    for metric, floor in thresholds.items():
        val = agg.get(metric)
        if val is None:
            continue
        if val < floor:
            fails.append(f"{metric}={val:.3f} < min {floor}")
    return fails


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="creative-rag eval harness")
    p.add_argument("--qa", default=str(Path(__file__).resolve().parent.parent / "eval" / "qa_craft.jsonl"),
                   help="path to a labeled .jsonl set")
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--with-llm", action="store_true", help="also run generation metrics (needs LLM key)")
    p.add_argument("--out", default="", help="write metrics json to this path")
    p.add_argument("--gate", action="store_true", help="exit nonzero if below thresholds")
    # hit@k / mrr / ndcg are the primary gates (recall@k is denominator-sensitive
    # when gold spans many chunks, so it is reported but not gated by default).
    p.add_argument("--min-hit", type=float, default=0.90)
    p.add_argument("--min-mrr", type=float, default=0.70)
    p.add_argument("--min-ndcg", type=float, default=0.60)
    p.add_argument("--min-correctness", type=float, default=0.90)
    p.add_argument("--min-faithfulness", type=float, default=0.90)
    args = p.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"qa set not found: {qa_path}", file=sys.stderr)
        return 2

    result = evaluate(qa_path, k=args.k, with_llm=args.with_llm)
    _print_table(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")

    if args.gate:
        thresholds = {"hit@k": args.min_hit, "mrr": args.min_mrr, "ndcg@k": args.min_ndcg}
        if args.with_llm:
            thresholds["correctness"] = args.min_correctness
            thresholds["faithfulness"] = args.min_faithfulness
        fails = _gate(result, thresholds)
        if fails:
            print("\nGATE FAILED:", file=sys.stderr)
            for f in fails:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("\nGATE PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
