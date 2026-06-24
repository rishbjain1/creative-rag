# creative-rag

**Hybrid-retrieval, citation-verified RAG over a craft/style corpus.**

Ask a knowledge base of cinematography + AI-filmmaking notes a real question and
get a **grounded, cited, verified** answer — not a hallucination. Built to be
trustworthy: every claim is checked against the retrieved sources.

```
$ crag query "what film stock and lens for a moody dusk beach scene?"

ANSWER: ... Cinestill 800T (tungsten-balanced, teal shadows) at the day→night
transition [5][6]; 85mm f/1.4 for faces, 14–35mm wide [5]. The notes don't give a
beach-specific lighting recipe — fill direction/sources yourself [2].

SOURCES: [5] Cinematic_Prompt_Library.md §Campaign Lock ... [6] §Film Stocks ...
VERIFY:  {"supported": true, "unsupported_claims": []}
```

## The retrieval funnel

Each stage is cheaper-but-coarser early, expensive-but-sharper late:

```
corpus → chunk (heading-aware) → embed (bi-encoder) → Chroma + BM25

query → dense (vector) ─┐
        sparse (BM25)  ─┴→ RRF fuse → top-30   (fast, approximate, high recall)
                        → cross-encoder rerank → top-6   (precise, joint attention)
                        → augment (rerank order, grounding instruction)
                        → generate (LLM, answer only from notes, cite)
                        → citation-verify (LLM-judge entailment per claim)
```

- **Dense** catches *meaning*, **sparse (BM25)** catches *exact terms* (stock names, `#hex`, `21:9`); **RRF** fuses them scale-free.
- **Cross-encoder rerank** reads query+chunk *together* — disambiguates near-identical-embedding opposites ("use 500T" vs "avoid 500T").
- **Citation-verify** is the eval layer: a second pass flags any claim not grounded in the notes.

## Local models (PyTorch)

Embeddings + reranker run **locally** via `sentence-transformers` — no API, no key,
fully reproducible:
- embed: `BAAI/bge-small-en-v1.5` (bi-encoder)
- rerank: `BAAI/bge-reranker-base` (cross-encoder)

The embedder/reranker live in one module (`embed.py`) behind a clean interface, so
an API backend can swap in without touching retrieval.

## Provider-agnostic generation

Generation + verification go through any **OpenAI-compatible** endpoint —
Anthropic (default), OpenRouter, OpenAI, local — chosen by env. No provider hard-coded.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # set CRAG_LLM_API_KEY (or ANTHROPIC_API_KEY)
```

## Use

```bash
crag ingest                                  # build the index from the corpus
crag query "what lens for a face close-up?"  # ask it (cited + verified)
creative-rag                                 # serve the FastAPI app (:8000)
```

API:
```
GET  /health                       # liveness + index status
POST /query  {query, top_k, verify}  # grounded answer + sources + verification
```
Set `CRAG_API_KEY` to require an `X-API-Key` header on `/query`.

## Corpus

Defaults to a local craft knowledge base (`CRAG_CORPUS_ROOT`). Ingest is
markdown-aware (chunks on headings) and skips non-craft/derived files. The index
is gitignored — only code is tracked.

## Test

```bash
pytest tests/ -q
```

## Deploy (Docker + Terraform on AWS Fargate)

The service containerizes (PyTorch + bge models + a sample index baked in, so the
image is self-contained) and deploys to ECS Fargate via Terraform (`deploy/`):
ECR + ECS cluster + task + service on the default VPC, with CloudWatch logs.

```bash
# 1. build + push the image
aws ecr create-repository --repository-name creative-rag    # or let terraform make it
docker build -t creative-rag .
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag creative-rag <acct>.dkr.ecr.<region>.amazonaws.com/creative-rag:latest
docker push <acct>.dkr.ecr.<region>.amazonaws.com/creative-rag:latest

# 2. provision + run
cd deploy
terraform init
terraform apply -var image_tag=latest        # ANTHROPIC_API_KEY via task secret in prod

# 3. tear down (no ongoing cost)
terraform destroy
```

Fargate task CPU/memory default to 1 vCPU / 4GB (torch needs the headroom). The
config is `terraform validate`-clean.

## License

MIT
