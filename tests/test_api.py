import pytest
from fastapi.testclient import TestClient

import semsearch.api as api


@pytest.fixture()
def client(mini_csv, tmp_path, monkeypatch):
    """Point the API at the mini corpus + hashing embedder via env overrides."""
    monkeypatch.setenv("SEMSEARCH_CORPUS", str(mini_csv))
    monkeypatch.setenv("SEMSEARCH_EMBEDDER", "hashing")
    monkeypatch.setenv("SEMSEARCH_HASHING_DIM", "128")
    monkeypatch.setenv("SEMSEARCH_CACHE_DIR", str(tmp_path / "cache"))
    api._service = None  # reset singleton between tests
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents"] == 5
    assert body["embedder"] == "hashing-128"
    assert body["embedding_dim"] == 128


def test_search_ok(client):
    r = client.get("/search", params={"q": "oil prices", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "oil prices"
    assert len(body["results"]) == 2
    top = body["results"][0]
    assert set(top) == {"id", "title", "label", "score", "snippet"}
    scores = [item["score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_default_k_is_five(client):
    r = client.get("/search", params={"q": "earnings"})
    assert len(r.json()["results"]) == 5


def test_search_requires_q(client):
    assert client.get("/search").status_code == 422


def test_search_rejects_blank_and_bad_k(client):
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 999}).status_code == 422


def test_get_document_and_404(client):
    r = client.get("/documents/doc-0002")
    assert r.status_code == 200
    assert r.json()["title"] == "Tennis star wins opening match"
    assert client.get("/documents/doc-9999").status_code == 404
