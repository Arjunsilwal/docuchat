"""Split documents into chunks for embedding.

Two strategies are combined:

1. Structure-aware: Markdown is first split on headings, so a chunk never mixes
   two sections, and every chunk remembers its heading path
   (for example "Employee Handbook > Leave and Time Off > Parental Leave").
2. Recursive with overlap: a section that is still too long is split on the
   largest natural boundary that works (paragraph, then line, then sentence,
   then word), and neighbouring chunks share some overlapping text so that
   facts near a boundary are not cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragbot.loaders import Document

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
CODE_FENCE = "`" * 3  # three backticks: opens or closes a Markdown code block
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    id: str
    text: str  # what gets embedded and shown to the LLM (heading path + body)
    source: str
    section: str = ""
    page: int = 0


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading_path, body) pairs. Ignores '#' lines inside code blocks."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    buffer: list[str] = []
    in_code_block = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(path), body))
        buffer.clear()

    for line in text.splitlines():
        if line.lstrip().startswith(CODE_FENCE):
            in_code_block = not in_code_block
        match = None if in_code_block else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2)]
        else:
            buffer.append(line)
    flush()
    return sections


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters, about `chunk_overlap` overlapping.

    Overlap is built from whole pieces (sentences, lines), so it can be a little
    smaller than `chunk_overlap`, but a chunk never starts in the middle of a word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    pieces = _split_into_pieces(text, chunk_size, SEPARATORS)
    return _merge_pieces(pieces, chunk_size, chunk_overlap)


def _split_into_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than chunk_size, preferring the earliest separator."""
    if len(text) <= chunk_size:
        return [text]
    separator, remaining = separators[0], separators[1:]
    if separator == "":  # last resort: hard cut by characters
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # keep the separator attached so joining pieces restores the original text
        piece = part + separator if i < len(parts) - 1 else part
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_into_pieces(piece, chunk_size, remaining))
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, carrying the tail of each chunk into the next."""
    chunks: list[str] = []
    window: list[str] = []
    length = 0
    for piece in pieces:
        if window and length + len(piece) > chunk_size:
            chunks.append("".join(window).strip())
            # drop pieces from the front until only the overlap tail is left
            # and the next piece fits
            while window and (length > chunk_overlap or length + len(piece) > chunk_size):
                length -= len(window.pop(0))
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Turn one Document into Chunks with stable ids and metadata."""
    if document.source.lower().endswith(".md"):
        sections = split_markdown_sections(document.text)
    else:
        sections = [("", document.text)]

    chunks: list[Chunk] = []
    for section, body in sections:
        for piece in recursive_split(body, chunk_size, chunk_overlap):
            # Prefixing the heading path gives each chunk context on its own,
            # e.g. a bare "16 weeks" becomes findable under "Parental Leave".
            text = f"{section}\n{piece}" if section else piece
            chunk_id = f"{document.source}#p{document.page}-c{len(chunks)}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=document.source,
                    section=section,
                    page=document.page,
                )
            )
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    return [c for doc in documents for c in chunk_document(doc, chunk_size, chunk_overlap)]