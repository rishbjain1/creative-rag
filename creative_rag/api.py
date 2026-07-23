"""FastAPI service over the craft RAG.

POST /query  → grounded, cited, verified answer
GET  /health → liveness + index status
Auth: if CRAG_API_KEY is set, /query requires header X-API-Key to match.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config, generate, obs

app = FastAPI(title="creative-rag", version="0.1.0")


LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>creative-rag — hybrid RAG with verified citations</title>
<style>
  :root{--bg:#0f1216;--card:#171b21;--line:#2a313b;--ink:#e7edf3;--mut:#9aa7b4;--acc:#4cc3a6}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  main{max-width:760px;margin:0 auto;padding:44px 20px 80px}
  h1{font-size:1.7rem;margin:0 0 4px}
  .sub{color:var(--mut);margin:0 0 20px}
  .pills{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}
  .pill{background:var(--card);border:1px solid var(--line);border-radius:99px;padding:5px 12px;font-size:.82rem;color:var(--mut)}
  .pill b{color:var(--acc)}
  .box{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
  textarea{width:100%;background:#0f1216;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:11px;font:inherit;resize:vertical;min-height:60px}
  .row{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}
  button{background:var(--acc);color:#08120e;border:0;border-radius:8px;padding:10px 18px;font-weight:650;cursor:pointer;font:inherit}
  button:disabled{opacity:.5;cursor:default}
  .ex{background:none;border:1px solid var(--line);color:var(--mut);font-size:.82rem;padding:6px 11px;border-radius:99px;cursor:pointer}
  #out{margin-top:18px;white-space:pre-wrap}
  .ans{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:14px;margin-top:12px}
  .meta{color:var(--mut);font-size:.85rem;margin-top:10px}
  a{color:var(--acc)}
  code{background:#0f1216;border:1px solid var(--line);padding:1px 5px;border-radius:4px;font-size:.85em}
</style></head><body><main>
  <h1>creative-rag</h1>
  <p class="sub">Hybrid-retrieval RAG over a film-craft corpus. Answers only from the notes, cites every claim, and verifies its own citations. Says "not in the corpus" instead of inventing.</p>
  <div class="pills">
    <span class="pill">retrieval hit@6 <b>1.0</b></span>
    <span class="pill">MRR <b>0.94</b></span>
    <span class="pill">answer faithfulness <b>1.0</b></span>
    <span class="pill">text&rarr;SQL exec-acc <b>0.917</b></span>
    <span class="pill">dense + BM25 &rarr; RRF &rarr; cross-encoder rerank &rarr; cited gen</span>
  </div>
  <div class="box">
    <textarea id="q" placeholder="Ask the craft corpus…">What film stock for a moody dusk scene?</textarea>
    <div class="row">
      <button id="go" onclick="ask()">Ask</button>
      <button class="ex" onclick="setq('What lens separates a subject from the background?')">example: lens choice</button>
      <button class="ex" onclick="setq('What feeling does a dolly-in create?')">example: camera move</button>
    </div>
    <div id="out"></div>
  </div>
  <p class="meta">Public demo runs on a lightweight model over a sample corpus; the private corpus stays local. Source + eval numbers: <a href="https://github.com/rishbjain1/creative-rag">github.com/rishbjain1/creative-rag</a> · API: <code>POST /query</code>, <code>GET /health</code></p>
<script>
function setq(t){document.getElementById('q').value=t;ask()}
async function ask(){
  const q=document.getElementById('q').value.trim();if(!q)return;
  const out=document.getElementById('out'),btn=document.getElementById('go');
  btn.disabled=true;out.innerHTML='<div class="meta">thinking… (first call wakes the model, ~10s)</div>';
  try{
    const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,top_k:4})});
    const d=await r.json();
    if(!r.ok){out.innerHTML='<div class="ans">'+(d.detail||'error')+'</div>';return}
    let srcs=(d.sources||[]).map(s=>'['+s.n+'] '+s.source+(s.heading?' §'+s.heading:'')).join('   ');
    let v=d.verification&&d.verification.supported;
    out.innerHTML='<div class="ans">'+(d.answer||'').replace(/</g,'&lt;')+'</div>'+
      '<div class="meta"><b>sources:</b> '+srcs+'<br><b>citation-verify:</b> '+(v===true?'✓ supported':v===false?'⚠ unsupported claim flagged':'—')+'</div>';
  }catch(e){out.innerHTML='<div class="ans">request failed: '+e+'</div>'}
  finally{btn.disabled=false}
}
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    return LANDING


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
    with obs.request_scope("query") as trace_id:
        result = generate.answer(body.query, top_k=body.top_k, verify=body.verify)
        result["usage"] = {"trace_id": trace_id, **obs.summary()}
    return result


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
