# creative-rag — containerized RAG API.
# Bakes the local embedding/reranker models + a sample index into the image so
# the service is self-contained (no runtime model download, no external corpus).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CRAG_CORPUS_ROOT=/app/sample_corpus \
    CRAG_INDEX_DIR=/app/index

WORKDIR /app

# System deps for some wheels; keep the layer slim.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache-friendly).
COPY pyproject.toml README.md ./
COPY creative_rag ./creative_rag
RUN pip install --no-cache-dir -e .

# Pre-download the bge models at build (no cold-start download at runtime).
RUN python -c "from creative_rag import embed; embed.embed_texts(['warm up']); embed.rerank('q', ['c'])"

# Bake a sample index so the container answers out of the box.
COPY sample_corpus ./sample_corpus
RUN crag ingest

EXPOSE 7860
CMD uvicorn creative_rag.api:app --host 0.0.0.0 --port ${PORT:-7860}
