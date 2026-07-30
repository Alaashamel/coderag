from pathlib import Path

from coderag.chunker import chunk_file, chunk_repository

SAMPLE_REPO = Path(__file__).parent.parent / "examples" / "sample_repo"


def test_chunk_repository_finds_all_sample_files():
    chunks = chunk_repository(SAMPLE_REPO)
    files = {c.file_path for c in chunks}
    assert "auth.py" in files
    assert "inventory.py" in files


def test_chunk_file_respects_line_bounds(tmp_path):
    content = "\n".join(f"line {i}" for i in range(1, 101))
    f = tmp_path / "sample.py"
    f.write_text(content)

    chunks = chunk_file(f, tmp_path, chunk_lines=20, overlap_lines=5)

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 20
    assert chunks[-1].end_line == 100
    # consecutive chunks should overlap by the configured amount
    assert chunks[1].start_line == chunks[0].end_line - 5 + 1


def test_chunk_file_skips_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    assert chunk_file(f, tmp_path) == []
