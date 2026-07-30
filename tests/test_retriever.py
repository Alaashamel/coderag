import pytest

from coderag.chunker import chunk_repository
from coderag.retriever import CodeRetriever
from pathlib import Path

SAMPLE_REPO = Path(__file__).parent.parent / "examples" / "sample_repo"


@pytest.fixture
def retriever():
    r = CodeRetriever()
    r.index(chunk_repository(SAMPLE_REPO))
    return r


def test_query_returns_relevant_chunk_for_password_hashing(retriever):
    results = retriever.query("how do I hash a password", top_k=3)
    assert results, "expected at least one result"
    top_chunk, _score = results[0]
    assert top_chunk.file_path == "auth.py"


def test_query_returns_relevant_chunk_for_inventory(retriever):
    results = retriever.query("fulfill a customer order and deduct stock", top_k=3)
    assert results
    top_chunk, _score = results[0]
    assert top_chunk.file_path == "inventory.py"


def test_query_on_empty_index_raises():
    r = CodeRetriever()
    with pytest.raises(RuntimeError):
        r.query("anything")


def test_index_with_no_chunks_raises():
    r = CodeRetriever()
    with pytest.raises(ValueError):
        r.index([])


def test_save_and_load_roundtrip(retriever, tmp_path):
    path = tmp_path / "index.pkl"
    retriever.save(path)
    loaded = CodeRetriever.load(path)
    assert len(loaded.chunks) == len(retriever.chunks)
    results = loaded.query("password hash")
    assert results
