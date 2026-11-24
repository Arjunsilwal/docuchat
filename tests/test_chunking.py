import pytest

from ragbot.chunking import chunk_document, recursive_split, split_markdown_sections
from ragbot.loaders import Document

FENCE = "`" * 3  # a Markdown code fence, built so this file has no literal fence

MARKDOWN = f"""# Handbook

Intro text.

## Leave

### Parental Leave

Sixteen weeks of paid leave.

{FENCE}python
# this is a comment, not a heading
{FENCE}

## Expenses

Submit within 30 days.
"""


def test_markdown_sections_keep_heading_path():
    sections = dict(split_markdown_sections(MARKDOWN))
    assert sections["Handbook"] == "Intro text."
    assert "Sixteen weeks" in sections["Handbook > Leave > Parental Leave"]
    assert sections["Handbook > Expenses"] == "Submit within 30 days."


def test_hash_inside_code_block_is_not_a_heading():
    sections = dict(split_markdown_sections(MARKDOWN))
    assert "# this is a comment" in sections["Handbook > Leave > Parental Leave"]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_neighbouring_chunks_overlap():
    text = "\n".join(f"Line {i}: some content about topic {i}." for i in range(60))
    chunks = recursive_split(text, chunk_size=300, chunk_overlap=80)
    for first, second in zip(chunks, chunks[1:]):
        last_line_of_first = first.splitlines()[-1]
        assert last_line_of_first in second


def test_zero_overlap_loses_nothing():
    text = "\n".join(f"Line {i}." for i in range(100))
    chunks = recursive_split(text, chunk_size=120, chunk_overlap=0)
    assert "\n".join(chunks).split() == text.split()


def test_short_text_is_one_chunk():
    assert recursive_split("Hello world.", chunk_size=100, chunk_overlap=10) == ["Hello world."]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        recursive_split("text", chunk_size=100, chunk_overlap=100)


def test_chunk_document_prefixes_heading_and_sets_metadata():
    doc = Document(text=MARKDOWN, source="handbook.md")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    leave = next(c for c in chunks if c.section.endswith("Parental Leave"))
    assert leave.text.startswith("Handbook > Leave > Parental Leave\n")
    assert leave.source == "handbook.md"
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique