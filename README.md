# coderag

![CI](https://github.com/Alaashamel/coderag/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

A retrieval-augmented Q&A assistant for codebases. Point it at a repo, ask
questions in plain English, and get back the exact lines of code that
answer them — with an optional LLM-generated summary on top.

```
$ coderag ingest ./my-project
Indexed 412 chunks from ./my-project -> .coderag_index.pkl

$ coderag ask "how does the app authenticate a user?"
auth.py (lines 1-26)  [relevance 0.41]
...
```

## Why TF-IDF instead of a neural embedding model?

This is a deliberate design choice, not a shortcut. TF-IDF + cosine
similarity needs no GPU, no API key, and no multi-hundred-megabyte model
download — `pip install` and it works immediately. Code identifiers
(function names, class names, imports) are exactly the kind of sparse,
high-signal tokens lexical search is good at matching, which is why it
remains a strong baseline for *code* search specifically. The
`EmbeddingBackend` interface in `retriever.py` makes it straightforward to
swap in a real embedding model later if you need semantic (not just
lexical) matching.

## Features

- **Language-agnostic chunking** — line-based with overlap, works across
  Python, JS/TS, Go, Rust, Dart, Markdown, and more
- **Identifier-aware tokenization** — splits `camelCase` and `snake_case`
  so a query for "get user" matches `getUser()` too
- **Two answer modes** — extractive (zero config, always works) or
  generative (set `ANTHROPIC_API_KEY` for a synthesized, cited answer)
- **CLI + HTTP API** — use it from the terminal or embed it in another
  service via FastAPI
- **Fully tested** — chunking, retrieval, and API layers all covered

## Installation

```bash
git clone https://github.com/Alaashamel/coderag.git
cd coderag
pip install -e .
```

## Usage

### CLI

```bash
coderag ingest /path/to/repo          # build the index
coderag ask "where is rate limiting implemented?"
```

### HTTP API

```bash
coderag serve
# POST http://127.0.0.1:8000/ingest   {"path": "/path/to/repo"}
# POST http://127.0.0.1:8000/ask      {"question": "..."}
```

### Better answers with Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
coderag ask "how does the checkout flow handle failed payments?"
```

Without a key, `coderag` returns the raw retrieved excerpts (still useful
on its own). With a key, it synthesizes a cited natural-language answer.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](./LICENSE).
