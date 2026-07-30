from __future__ import annotations

import sys
from pathlib import Path

import click

from .chunker import chunk_repository
from .generator import answer_question
from .retriever import CodeRetriever

DEFAULT_INDEX_PATH = ".coderag_index.pkl"


@click.group()
@click.version_option()
def main():
    """coderag — a retrieval-augmented Q&A assistant for a codebase."""


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False), default=".")
@click.option("--index", "index_path", default=DEFAULT_INDEX_PATH, show_default=True, help="Where to save the index.")
@click.option("--chunk-lines", default=60, show_default=True)
@click.option("--overlap-lines", default=10, show_default=True)
def ingest(path: str, index_path: str, chunk_lines: int, overlap_lines: int):
    """Chunk and index every source file under PATH."""
    click.echo(f"Scanning {path} ...")
    chunks = chunk_repository(path, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    if not chunks:
        click.echo("No source files found — check the path and file extensions.", err=True)
        sys.exit(1)

    retriever = CodeRetriever()
    retriever.index(chunks)
    retriever.save(index_path)
    click.echo(f"Indexed {len(chunks)} chunks from {path} -> {index_path}")


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--index", "index_path", default=DEFAULT_INDEX_PATH, show_default=True)
@click.option("--top-k", default=5, show_default=True)
def ask(question: tuple[str, ...], index_path: str, top_k: int):
    """Ask a question about the ingested codebase."""
    if not Path(index_path).exists():
        click.echo(f"No index found at {index_path}. Run `coderag ingest <path>` first.", err=True)
        sys.exit(1)

    retriever = CodeRetriever.load(index_path)
    q = " ".join(question)
    results = retriever.query(q, top_k=top_k)
    answer = answer_question(q, results)

    click.echo(answer.text)
    if answer.mode == "extractive":
        click.echo("\n(tip: set ANTHROPIC_API_KEY for a synthesized natural-language answer instead of raw excerpts)", err=True)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host: str, port: int):
    """Run the HTTP API (see api.py) with uvicorn."""
    import uvicorn

    uvicorn.run("coderag.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
