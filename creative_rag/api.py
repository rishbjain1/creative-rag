"""FastAPI service over the craft RAG.

POST /query  → grounded, cited, verified answer
GET  /health → liveness + index status
Auth: if CRAG_API_KEY is set, /query requires header X-API-Key to match.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import config, generate

app = FastAPI(title="creative-rag", version="0.1.0")


class QueryIn(BaseModel):
    query: str
    top_k: int | None = None
    verify: bool = True


def _auth(x_api_key: str | None) -> None:
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
def health() -> dict:
    exists = config.CHUNKS_PATH.exists()
    n = 0
    if exists:
        import json
        n = len(json.loads(config.CHUNKS_PATH.read_text()))
    return {"status": "ok", "indexed": exists, "chunks": n, "model": config.LLM_MODEL}


@app.post("/query")
def query(body: QueryIn, x_api_key: str | None = Header(default=None)) -> dict:
    _auth(x_api_key)
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query required")
    return generate.answer(body.query, top_k=body.top_k, verify=body.verify)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
