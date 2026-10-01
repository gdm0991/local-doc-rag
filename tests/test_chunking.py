import pytest

from local_doc_rag.chunking import chunk_documents, chunk_text
from local_doc_rag.ingest import Document


def test_short_text_is_one_chunk():
    assert chunk_text("строка 1\nстрока 2", size=100, overlap=10) == ["строка 1\nстрока 2"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("\n\n   \n", size=100, overlap=10) == []


def test_chunks_respect_size_and_overlap():
    lines = [f"Пункт {i}: проверить канал {i} модуля МВА-8." for i in range(40)]
    chunks = chunk_text("\n".join(lines), size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Соседние фрагменты делят хотя бы одну строку (перекрытие)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.splitlines()) & set(b.splitlines())
    # Ни одна строка не потеряна
    joined = set("\n".join(chunks).splitlines())
    assert set(lines) <= joined


def test_zero_overlap_has_no_shared_lines():
    lines = [f"строка номер {i:03d}" for i in range(30)]
    chunks = chunk_text("\n".join(lines), size=100, overlap=0)
    seen = []
    for c in chunks:
        seen.extend(c.splitlines())
    assert seen == lines


def test_long_line_is_split():
    text = "А" * 1000
    chunks = chunk_text(text, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) >= 4


def test_table_header_is_carried_to_next_chunk():
    header = "| Тег | Параметр | Аварийная |\n|---|---|---|"
    rows = "\n".join(f"| T{i} | Параметр номер {i} | {i},5 |" for i in range(30))
    chunks = chunk_text(header + "\n" + rows, size=250, overlap=0)
    assert len(chunks) > 2
    for c in chunks:
        assert c.startswith("| Тег | Параметр | Аварийная |")


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_bad_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("текст", size=size, overlap=overlap)


def test_chunk_ids_and_title():
    docs = [Document("a/b.md", "Карта уставок стенда ИС-7", "строка\n" * 300)]
    chunks = chunk_documents(docs, size=200, overlap=20)
    assert [c.chunk_id for c in chunks] == [f"a/b.md#{i}" for i in range(len(chunks))]
    assert chunks[0].index_text.startswith("Карта уставок стенда ИС-7\n")
