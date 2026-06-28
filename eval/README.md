# Eval harness

Offline evaluation for the retrieval funnel and the generated answer. Two layers,
mirroring the system:

| Layer | Metrics | Needs LLM? | Where it runs |
|---|---|---|---|
| **Retrieval** | `hit@k`, `recall@k`, `MRR`, `nDCG@k` | no | local + CI |
| **Generation** | `correctness`, `faithfulness` | yes | local only |

`citation-verify` (in `generate.py`) is the *runtime* guardrail — per query, online.
This harness is the *offline* eval: a labeled set with ground truth, so you can tell
whether a change **helps or hurts** and put a number on quality.

## Labels (reindex-robust)

A labeled item (`*.jsonl`):

```json
{"q": "...", "gold_sources": ["Cinematic_Prompt_Library.md"],
 "gold_phrases": ["Cinestill 800T"], "answer_must_include": ["Cinestill 800T"]}
```

Gold chunks are resolved at eval time as: **chunk source ∈ `gold_sources` AND a
`gold_phrase` appears in the chunk text.** This pins the *answer*, not a chunk id, so
labels survive a full reingest. `gold_sources` is restricted to craft **explanation**
docs (not prompt/usage docs that merely mention a term), so a hit means retrieval
surfaced a chunk that actually *answers* the question.

Metric notes:
- **hit@k** — did any gold chunk land in the top-k. The primary "found an answer"
  signal, since a fact is often explained in several chunks.
- **recall@k** — fraction of all gold chunks in the top-k. Denominator-sensitive
  (large gold sets cap it below 1), so it is reported but not gated by default.
- **MRR / nDCG@k** — how *high* the first / all relevant chunks rank.
- **correctness** — every `answer_must_include` term is present in the answer.
- **faithfulness** — the citation-verify pass returned `supported: true`.

## Run

```bash
# retrieval only (fast, no key) — against the real local index
crag-eval --qa eval/qa_craft.jsonl --out eval/metrics_craft.json

# + generation metrics (needs CRAG_LLM_API_KEY / ANTHROPIC_API_KEY)
set -a; source .env; set +a
crag-eval --qa eval/qa_craft.jsonl --with-llm

# regression gate (nonzero exit if below thresholds)
crag-eval --qa eval/qa_craft.jsonl --gate
```

## Sample corpus (CI)

The real corpus is personal + gitignored, so CI can't rebuild it. `eval/sample_corpus/`
is a tiny self-contained craft corpus with `eval/qa_sample.jsonl`; CI ingests it and
gates retrieval on it (`.github/workflows/eval.yml`). The harness is corpus-agnostic —
point `--qa` and the index env at any (corpus, set) pair.

## Baseline (real corpus, 24 Q, k=6)

| layer | metric | value |
|---|---|---|
| retrieval | hit@6 | 1.000 |
| retrieval | MRR | 0.938 |
| retrieval | nDCG@6 | 0.690 |
| retrieval | recall@6 | 0.644 |
| generation | correctness | 1.000 |
| generation | faithfulness | 1.000 |

**Eval-driven improvement (worked example).** The first run scored hit@6=0.917 with
two misses (audio-rule, primary-generator). Both were heading-vs-body gaps: the
answering term lived in the section *heading* ("AUDIO RULE", "Soul 2.0 is primary")
but only the body was indexed. The fix — index `heading + body` across all three
stages (dense, sparse, **and** the cross-encoder rerank) — took hit@6 to 1.000,
lifted MRR 0.868→0.938 and nDCG 0.626→0.690, and carried generation correctness and
faithfulness from 0.958 to 1.000 (better-ranked chunks gave the LLM cleaner grounding).
This is exactly the loop the harness exists for: measure → find the gap → fix → re-measure.
