from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chunker import chunk_repository
from .generator import answer_question
from .retriever import CodeRetriever

app = FastAPI(
    title="coderag",
    description="Retrieval-augmented Q&A over a codebase.",
    version="0.1.0",
)

_state: dict[str, CodeRetriever] = {}


class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to a local repository to index.")
    chunk_lines: int = 60
    overlap_lines: int = 10


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceRef(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    relevance: float


class AskResponse(BaseModel):
    answer: str
    mode: str
    sources: list[SourceRef]


@app.get("/health")
def health():
    return {"status": "ok", "indexed": "index" in _state}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not Path(req.path).exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {req.path}")

    chunks = chunk_repository(req.path, chunk_lines=req.chunk_lines, overlap_lines=req.overlap_lines)
    if not chunks:
        raise HTTPException(status_code=400, detail="No indexable source files found at that path.")

    retriever = CodeRetriever()
    retriever.index(chunks)
    _state["index"] = retriever
    return {"indexed_chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retriever = _state.get("index")
    if retriever is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet — call POST /ingest first.")

    results = retriever.query(req.question, top_k=req.top_k)
    answer = answer_question(req.question, results)
    return AskResponse(
        answer=answer.text,
        mode=answer.mode,
        sources=[
            SourceRef(file_path=c.file_path, start_line=c.start_line, end_line=c.end_line, relevance=score)
            for c, score in results
        ],
    )
