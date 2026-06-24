"""CLI — ingest the corpus, or query it from the terminal."""
from __future__ import annotations

import argparse
import json

from . import ingest


def main() -> None:
    ap = argparse.ArgumentParser(prog="crag", description="creative-rag CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="build the index from the corpus")
    q = sub.add_parser("query", help="ask the corpus")
    q.add_argument("question")
    q.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.cmd == "ingest":
        stats = ingest.build()
        print(json.dumps(stats, indent=2))
    elif args.cmd == "query":
        from . import generate

        r = generate.answer(args.question, verify=not args.no_verify)
        print("\nANSWER:\n" + r["answer"])
        print("\nSOURCES:")
        for s in r["sources"]:
            print(f"  [{s['n']}] {s['source']} §{s['heading']}  (rerank {s['rerank_score']})")
        if r["verification"] is not None:
            print("\nVERIFY:", json.dumps(r["verification"]))
