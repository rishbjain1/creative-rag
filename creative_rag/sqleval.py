"""Execution-accuracy eval for the text→SQL agent.

For each labeled question, run the agent's predicted SQL and a reference SQL and
compare result sets (order- and float-normalized) — the standard text2SQL
**execution match** metric, which credits any query that returns the right rows
regardless of phrasing. Also tracks a hard safety signal: the agent must never
emit a non-SELECT / write query.

- Without `--with-llm`: validates every reference query is a safe, runnable
  SELECT (the CI-safe gate — no model needed).
- With `--with-llm`: runs the agent and reports execution accuracy + cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import obs, sqlagent


def _norm_cell(x):
    if isinstance(x, float):
        return round(x, 4)
    return x


def _norm(rows: list[tuple]) -> list[tuple]:
    return sorted((tuple(_norm_cell(c) for c in r) for r in rows),
                  key=lambda r: tuple(str(c) for c in r))


def result_match(pred: list[tuple] | None, gold: list[tuple]) -> bool:
    if pred is None:
        return False
    return _norm(pred) == _norm(gold)


def _load(qa_path: Path) -> list[dict]:
    return [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]


def validate_gold(qa_path: Path) -> dict:
    """Offline: every reference SQL must be a safe SELECT that runs (CI gate)."""
    conn = sqlagent.build_db()
    rows = _load(qa_path)
    problems = []
    for it in rows:
        sql = it["gold_sql"]
        if not sqlagent.is_safe_select(sql):
            problems.append(f"unsafe gold SQL: {it['q']}")
            continue
        try:
            sqlagent.run_sql(conn, sql)
        except Exception as e:  # a broken reference query is a labeling bug
            problems.append(f"gold SQL error ({it['q']}): {e}")
    return {"n": len(rows), "problems": problems}


def evaluate(qa_path: Path, model: str | None = None) -> dict:
    """Run the agent on every question; score execution match + safety."""
    conn = sqlagent.build_db()
    items = _load(qa_path)
    rows = []
    with obs.request_scope("sql_eval"):
        for it in items:
            gold = sqlagent.run_sql(conn, it["gold_sql"])
            res = sqlagent.answer(it["q"], conn, model=model)
            rows.append({
                "q": it["q"],
                "safe": res["safe"],
                "ran": res["rows"] is not None,
                "match": result_match(res["rows"], gold),
                "pred_sql": res["sql"],
                "error": res["error"],
            })
        cost = obs.summary()
    n = len(rows)
    agg = {
        "n_questions": n,
        "execution_accuracy": round(sum(r["match"] for r in rows) / n, 3) if n else None,
        "safety_violations": sum(1 for r in rows if not r["safe"]),
        "ran_ok": sum(1 for r in rows if r["ran"]),
        "cost_usd": cost["cost_usd"],
        "llm_calls": cost["llm_calls"],
    }
    return {"aggregate": agg, "rows": rows}


def _print(result: dict) -> None:
    print(f"{'match':>6} {'safe':>5}  question")
    print("-" * 60)
    for r in result["rows"]:
        m = " ok  " if r["match"] else "MISS "
        s = " ok " if r["safe"] else "BAD "
        q = r["q"] if len(r["q"]) <= 45 else r["q"][:42] + "..."
        print(f"{m:>6} {s:>5}  {q}")
    a = result["aggregate"]
    print("-" * 60)
    print(f"execution_accuracy={a['execution_accuracy']}  safety_violations={a['safety_violations']}  "
          f"ran_ok={a['ran_ok']}/{a['n_questions']}  cost=${a['cost_usd']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="text→SQL agent execution-accuracy eval")
    p.add_argument("--qa", default=str(Path(__file__).resolve().parent.parent / "eval" / "sql" / "qa_sql.jsonl"))
    p.add_argument("--with-llm", action="store_true", help="run the agent (needs LLM key); else validate gold only")
    p.add_argument("--out", default="")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--min-accuracy", type=float, default=0.80)
    args = p.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"qa set not found: {qa_path}", file=sys.stderr)
        return 2

    if not args.with_llm:
        v = validate_gold(qa_path)
        print(f"gold validation: {v['n']} queries, {len(v['problems'])} problems")
        for pr in v["problems"]:
            print(f"  - {pr}", file=sys.stderr)
        if args.gate and v["problems"]:
            print("\nGATE FAILED: invalid reference SQL", file=sys.stderr)
            return 1
        if args.gate:
            print("\nGATE PASSED ✓ (gold SQL all safe + runnable)")
        return 0

    result = evaluate(qa_path)
    _print(result)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    if args.gate:
        a = result["aggregate"]
        fails = []
        if a["safety_violations"] > 0:
            fails.append(f"{a['safety_violations']} safety violation(s) — agent emitted non-SELECT SQL")
        if a["execution_accuracy"] is not None and a["execution_accuracy"] < args.min_accuracy:
            fails.append(f"execution_accuracy={a['execution_accuracy']} < min {args.min_accuracy}")
        if fails:
            print("\nGATE FAILED:", file=sys.stderr)
            for f in fails:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("\nGATE PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
