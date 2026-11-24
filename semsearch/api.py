"""FastAPI service.

    GET /health                 -> liveness + index stats
    GET /search?q=oil prices&k=5 -> top-k results with cosine scores
    GET /documents/doc-0106     -> full document by id

Run:  uvicorn semsearch.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException, Query

from .config import SearchConfig
from .search import SearchService

app = FastAPI(
    title="semsearch",
    description="Semantic search over a real news corpus — embeddings, exact cosine "
                "index, evaluated with recall@k / MRR.",
    version="0.1.0",
)

_service: SearchService | None = None
_lock = threading.Lock()


def get_service() -> SearchService:
    """Singleton service, built on first request (keeps test/CLI imports cheap)."""
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = SearchService.from_config(SearchConfig.from_env())
    return _service


@app.get("/health")
def health() -> dict:
    svc = get_service()
    return {
        "status": "ok",
        "documents": svc.index.size,
        "embedder": svc.embedder.name,
        "embedding_dim": svc.embedder.dim,
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="free-text query"),
    k: int = Query(5, ge=1, le=50, description="number of results"),
) -> dict:
    svc = get_service()
    hits = svc.search(q, k=k)
    return {
        "query": q,
        "embedder": svc.embedder.name,
        "results": [
            {
                "id": r.doc_id,
                "title": r.title,
                "label": r.label,
                "score": round(r.score, 4),
                "snippet": r.text[:220],
            }
            for r in hits
        ],
    }


@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    svc = get_service()
    doc = next((d for d in svc.corpus.docs if d.id == doc_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"unknown document: {doc_id}")
    return {"id": doc.id, "title": doc.title, "text": doc.text, "label": doc.label}
