---
title: creative-rag
emoji: 🎬
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# creative-rag

Hybrid RAG over a cinematography craft corpus: bge-small dense retrieval + BM25 → RRF fusion → cross-encoder reranking → grounded generation with citation verification.

## Endpoints

- `GET /health`
- `POST /query` with JSON `{"query": "...", "top_k": 6}`

```bash
curl -X POST "https://YOUR-SPACE.hf.space/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"How should I light a night interior?","top_k":6}'
```

## Evaluation

hit@6=1.0 · MRR=0.94 · nDCG=0.69 · faithfulness=1.0

Source: [github.com/rishbjain1/creative-rag](https://github.com/rishbjain1/creative-rag)
