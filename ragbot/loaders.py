"""Load raw documents (Markdown, text, PDF) from a folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass
class Document:
    """One loaded document. PDFs become one Document per page so we can cite page numbers."""

    text: str
    source: str  # file name, used in citations
    page: int = 0  # 0 means "not a paged document"


def load_documents(data_dir: str | Path) -> list[Document]:
    """Read every supported file under `data_dir` (recursively), sorted for stable output."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path.resolve()}")

    documents: list[Document] = []
    for path in sorted(data_path.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(Document(text=text, source=path.name))
    return documents


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Document(text=text, source=path.name, page=number))
    return pages