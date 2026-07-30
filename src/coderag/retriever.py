"""A lightweight retrieval index over code chunks.

Uses TF-IDF + cosine similarity (scikit-learn) rather than a neural
embedding model. This is a deliberate tradeoff for a portfolio-scale
tool: it needs no GPU, no multi-hundred-megabyte model download, and no
API key to run — `pip install` and go. Code identifiers (function/class
names, imports) are exactly the kind of sparse, high-signal tokens TF-IDF
is good at matching, which is why lexical retrieval remains a strong
baseline for *code* search specifically (unlike prose search, where
semantic embeddings usually win by a wider margin).

The `EmbeddingBackend` protocol below makes it straightforward to swap in
a real embedding model (sentence-transformers, OpenAI, Voyage, ...) later
without touching the rest of the pipeline.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk

_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _code_tokenizer(text: str) -> list[str]:
    """Splits camelCase / snake_case identifiers into sub-tokens too,
    so a query for "get user" can match `getUser` or `get_user`."""
    tokens: list[str] = []
    for word in _CODE_TOKEN_RE.findall(text):
        tokens.append(word.lower())
        # split snake_case
        if "_" in word:
            tokens.extend(p.lower() for p in word.split("_") if p)
        # split camelCase / PascalCase
        camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", word)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts)
    return tokens


class EmbeddingBackend(Protocol):
    def fit_transform(self, texts: list[str]): ...
    def transform(self, texts: list[str]): ...


class CodeRetriever:
    """Fits a TF-IDF index over a set of chunks and answers similarity queries."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=_code_tokenizer,
            token_pattern=None,
            lowercase=False,
            max_features=50_000,
            sublinear_tf=True,
        )
        self.chunks: list[Chunk] = []
        self._matrix = None

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        self.chunks = chunks
        self._matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def query(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self._matrix is None:
            raise RuntimeError("Index is empty — call .index(chunks) first.")
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], float(scores[i])) for i in ranked[:top_k] if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "chunks": self.chunks, "matrix": self._matrix}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CodeRetriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        retriever = cls()
        retriever.vectorizer = data["vectorizer"]
        retriever.chunks = data["chunks"]
        retriever._matrix = data["matrix"]
        return retriever
