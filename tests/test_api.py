from pathlib import Path

from fastapi.testclient import TestClient

from coderag.api import app

SAMPLE_REPO = str(Path(__file__).parent.parent / "examples" / "sample_repo")

client = TestClient(app)


def test_health_before_ingest():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_before_ingest_returns_400():
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 400


def test_ingest_then_ask():
    ingest_resp = client.post("/ingest", json={"path": SAMPLE_REPO})
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["indexed_chunks"] > 0

    ask_resp = client.post("/ask", json={"question": "how is a session token generated"})
    assert ask_resp.status_code == 200
    body = ask_resp.json()
    assert body["mode"] == "extractive"  # no ANTHROPIC_API_KEY in test env
    assert body["sources"]
    assert body["sources"][0]["file_path"] == "auth.py"


def test_ingest_bad_path_returns_400():
    resp = client.post("/ingest", json={"path": "/no/such/path"})
    assert resp.status_code == 400
