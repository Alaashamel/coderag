"""Turns retrieved chunks + a question into an answer.

Two modes:
  * "generative" — if ANTHROPIC_API_KEY is set, sends the question and
    retrieved context to Claude and returns a synthesized natural
    language answer with citations back to file:line.
  * "extractive" — no API key configured. Returns the raw retrieved
    chunks, clearly labeled, so the tool is still useful standalone
    (this is the default and needs zero external services).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .chunker import Chunk

ANTHROPIC_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are a codebase assistant. Answer the user's question using ONLY the "
    "provided code context. Cite the specific file and line range for every "
    "claim you make, in the form (file.py:12-30). If the context doesn't "
    "contain enough information to answer confidently, say so explicitly "
    "rather than guessing."
)


@dataclass
class Answer:
    text: str
    mode: str  # "generative" | "extractive"
    sources: list[Chunk]


def _build_context_block(results: list[tuple[Chunk, float]]) -> str:
    return "\n\n---\n\n".join(chunk.as_context_block() for chunk, _score in results)


def _extractive_answer(question: str, results: list[tuple[Chunk, float]]) -> Answer:
    if not results:
        return Answer(
            text="No relevant code was found for that question. Try rephrasing, "
            "or make sure you've run `coderag ingest <path>` first.",
            mode="extractive",
            sources=[],
        )
    lines = [f"Here's the most relevant code I found for: \"{question}\"\n"]
    for chunk, score in results:
        lines.append(f"### {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})  [relevance {score:.2f}]")
        lines.append(f"```\n{chunk.text}\n```")
    return Answer(text="\n".join(lines), mode="extractive", sources=[c for c, _ in results])


def _generative_answer(question: str, results: list[tuple[Chunk, float]], api_key: str) -> Answer:
    import urllib.request
    import json as _json

    context = _build_context_block(results)
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Question: {question}\n\nCode context:\n\n{context}",
            }
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=_json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read().decode())
    text = "".join(block.get("text", "") for block in data.get("content", []))
    return Answer(text=text.strip(), mode="generative", sources=[c for c, _ in results])


def answer_question(question: str, results: list[tuple[Chunk, float]]) -> Answer:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _generative_answer(question, results, api_key)
        except Exception:
            # Network/API issues shouldn't crash the tool — fall back gracefully.
            pass
    return _extractive_answer(question, results)
