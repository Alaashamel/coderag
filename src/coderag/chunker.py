"""Splits a codebase into overlapping, line-numbered chunks suitable for
embedding and retrieval.

Design notes
------------
We chunk by *lines* rather than trying to parse every possible language's
AST. This keeps the chunker language-agnostic (works for Python, JS, Dart,
config files, docs, ...) at the cost of occasionally splitting a function
in half — the overlap between consecutive chunks mitigates that in
practice, since the retriever returns neighbouring chunks together often
enough for the model (or the human) to still have full context.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INCLUDE = (
    "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.rs", "*.java",
    "*.rb", "*.php", "*.c", "*.h", "*.cpp", "*.hpp", "*.cs", "*.dart",
    "*.md", "*.mdx", "*.yaml", "*.yml", "*.json", "*.toml", "*.cfg",
    "*.html", "*.css",
)

DEFAULT_EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
    "build", ".next", ".cache", "coverage", ".pytest_cache", ".mypy_cache",
}

MAX_FILE_BYTES = 500_000  # skip anything larger (generated bundles, data dumps, ...)


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of code."""

    id: str
    file_path: str
    start_line: int
    end_line: int
    text: str

    def as_context_block(self) -> str:
        return f"# {self.file_path} (lines {self.start_line}-{self.end_line})\n{self.text}"


def iter_source_files(
    root: Path,
    include: tuple[str, ...] = DEFAULT_INCLUDE,
    exclude_dirs: set[str] = None,
):
    exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in exclude_dirs for part in path.parts):
            continue
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in include):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def chunk_file(
    path: Path,
    root: Path,
    chunk_lines: int = 60,
    overlap_lines: int = 10,
) -> list[Chunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    lines = text.splitlines()
    if not lines:
        return []

    rel_path = str(path.relative_to(root))
    chunks: list[Chunk] = []
    step = max(chunk_lines - overlap_lines, 1)

    for start in range(0, len(lines), step):
        end = min(start + chunk_lines, len(lines))
        body = "\n".join(lines[start:end]).strip()
        if not body:
            if end == len(lines):
                break
            continue
        chunk_id = f"{rel_path}:{start + 1}-{end}"
        chunks.append(
            Chunk(
                id=chunk_id,
                file_path=rel_path,
                start_line=start + 1,
                end_line=end,
                text=body,
            )
        )
        if end == len(lines):
            break

    return chunks


def chunk_repository(
    root: str | Path,
    chunk_lines: int = 60,
    overlap_lines: int = 10,
) -> list[Chunk]:
    """Walk `root` and return chunks for every included source file."""
    root = Path(root).resolve()
    all_chunks: list[Chunk] = []
    for file_path in iter_source_files(root):
        all_chunks.extend(chunk_file(file_path, root, chunk_lines, overlap_lines))
    return all_chunks
