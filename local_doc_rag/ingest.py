"""Разбор документов: .md, .txt, .pdf (через pypdf)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Document:
    doc_id: str  # путь относительно корня загрузки, с прямыми слешами
    title: str
    text: str


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    # utf-8-sig снимает BOM, который любит Блокнот Windows
    return path.read_text(encoding="utf-8-sig", errors="replace")


def guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s and not s.startswith(">"):
            return s[:200]
    return fallback


def load_documents(root: str | Path) -> list[Document]:
    """Рекурсивно читает поддерживаемые файлы. Пустые и нечитаемые пропускает с предупреждением."""
    root = Path(root)
    if root.is_file():
        files, base = [root], root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
        base = root
    docs: list[Document] = []
    for p in files:
        try:
            text = read_file(p)
        except Exception as e:  # битый PDF не должен ронять всю загрузку
            log.warning("Пропущен %s: %s", p, e)
            continue
        if not text.strip():
            log.warning("Пропущен пустой файл (возможно, скан без текстового слоя): %s", p)
            continue
        doc_id = p.relative_to(base).as_posix()
        docs.append(Document(doc_id=doc_id, title=guess_title(text, p.stem), text=text))
    return docs
