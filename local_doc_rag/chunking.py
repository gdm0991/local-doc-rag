"""Нарезка текста на фрагменты с перекрытием.

Режем по строкам, а не по символам вслепую: строка таблицы или пункт инструкции
не должны разрываться посередине. Если фрагмент начинается внутри markdown-таблицы,
к нему приписывается шапка таблицы — иначе строка «| P_OIL | ... | 0,26 |»
теряет смысл столбцов.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str

    @property
    def index_text(self) -> str:
        """Текст для индексации: заголовок документа + фрагмент.

        Заголовок нужен потому, что во фрагменте из середины документа часто нет
        названия стенда, а вопрос его содержит.
        """
        return f"{self.title}\n{self.text}"


def _split_long(line: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [line[i:i + size] for i in range(0, len(line), step) if line[i:i + size].strip()]


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and set(s) <= set("|-: ")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Режет текст на фрагменты длиной не более size символов (кроме шапки таблицы),
    соседние фрагменты перекрываются хвостом не длиннее overlap символов."""
    if size <= 0:
        raise ValueError("size должен быть > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap должен быть в диапазоне [0, size)")

    units: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        units.extend(_split_long(line, size, overlap) if len(line) > size else [line])
    if not units:
        return []

    # Для каждой строки запоминаем шапку таблицы, внутри которой она стоит.
    headers: list[str | None] = []
    current_header: str | None = None
    for i, u in enumerate(units):
        is_row = u.strip().startswith("|")
        if is_row and i + 1 < len(units) and _is_table_sep(units[i + 1]):
            current_header = u + "\n" + units[i + 1]
        elif not is_row:
            current_header = None
        headers.append(current_header)

    chunks: list[str] = []
    cur: list[int] = []
    cur_len = 0
    for i, u in enumerate(units):
        if cur and cur_len + len(u) + 1 > size:
            chunks.append(_render(units, headers, cur))
            # Перекрытие: хвост из целых строк общей длиной не больше overlap.
            tail: list[int] = []
            tail_len = 0
            for j in reversed(cur):
                if tail_len + len(units[j]) + 1 > overlap:
                    break
                tail.insert(0, j)
                tail_len += len(units[j]) + 1
            cur, cur_len = tail, tail_len
        cur.append(i)
        cur_len += len(u) + 1
    if cur:
        chunks.append(_render(units, headers, cur))
    return chunks


def _render(units: list[str], headers: list[str | None], idx: list[int]) -> str:
    body = "\n".join(units[i] for i in idx)
    first = idx[0]
    h = headers[first]
    starts_inside_table = (h is not None and units[first].strip().startswith("|")
                           and not body.startswith(h.split("\n")[0]) and not _is_table_sep(units[first]))
    return f"{h}\n{body}" if starts_inside_table else body


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        for i, t in enumerate(chunk_text(d.text, size, overlap)):
            out.append(Chunk(chunk_id=f"{d.doc_id}#{i}", doc_id=d.doc_id, title=d.title, text=t))
    return out
