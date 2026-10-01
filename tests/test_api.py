"""Тесты HTTP API через TestClient. LLM не используется, векторный индекс — TF-IDF."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_doc_rag.api import create_app
from local_doc_rag.pipeline import RagPipeline

CORPUS = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "docs"


@pytest.fixture()
def mini_corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "karta.md").write_text(
        "# Карта уставок стенда ТЕСТ-1\n\n| Тег | Параметр | Аварийная |\n|---|---|---|\n"
        "| P_OIL | Давление масла | 0,25 |\n| T_OIL | Температура масла | 95 |\n", encoding="utf-8")
    (d / "pusk.txt").write_text("Инструкция по пуску стенда ТЕСТ-1\nПрогрев 15 минут.\n", encoding="utf-8")
    # Ещё три документа: на корпусе из двух у BM25Okapi IDF вырождается в ноль
    (d / "reglament.md").write_text("# Регламент ТО стенда ТЕСТ-1\nЗамена фильтра каждые 500 моточасов.\n",
                                    encoding="utf-8")
    (d / "glossariy.md").write_text("# Глоссарий\nКвитирование — подтверждение аварии оператором.\n",
                                    encoding="utf-8")
    (d / "modbus.md").write_text("# Карта Modbus стенда ТЕСТ-1\nP_OIL — регистр 30001.\n", encoding="utf-8")
    (d / "ignored.docx").write_bytes(b"not supported")
    return tmp_path


@pytest.fixture()
def client(mini_corpus):
    pipe = RagPipeline(index_dir=mini_corpus / "index")
    return TestClient(create_app(pipe, ingest_root=mini_corpus))


def test_health_before_ingest(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["ready"] is False and body["llm"] == "none"


def test_search_requires_index(client):
    assert client.post("/search", json={"query": "давление"}).status_code == 409


def test_ingest_search_ask(client):
    r = client.post("/ingest", json={"path": "docs"})
    assert r.status_code == 200, r.text
    assert r.json()["documents"] == 5  # .docx пропущен

    health = client.get("/health").json()
    assert health["ready"] is True
    assert health["embed_backend"].startswith("tfidf")

    r = client.post("/search", json={"query": "аварийная уставка давления масла ТЕСТ-1", "k": 3})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["doc_id"] == "karta.md"
    assert "0,25" in res[0]["text"]

    # Формулировка «время прогрева» здесь не годится: Snowball режет «Прогрев» до «прогр»,
    # а «прогрева» — до «прогрев», и BM25 их не сопоставляет (ограничение описано в README).
    r = client.post("/ask", json={"question": "Сколько минут длится прогрев на стенде ТЕСТ-1?", "use_llm": False})
    body = r.json()
    assert r.status_code == 200
    assert body["llm"] == "none"
    assert body["passages"][0]["doc_id"] == "pusk.txt"


def test_ask_unknown_object_refuses_without_llm(client):
    client.post("/ingest", json={"path": "docs"})
    body = client.post("/ask", json={"question": "Сколько стоит ремонт насоса НШ-99Х?"}).json()
    assert body["refused"] is True
    assert body["answer"].startswith("Не знаю")


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "vector"])
def test_search_modes(client, mode):
    client.post("/ingest", json={"path": "docs"})
    r = client.post("/search", json={"query": "температура масла", "mode": mode})
    assert r.status_code == 200 and r.json()["results"]


def test_validation_and_path_guard(client, tmp_path_factory):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422
    outside = tmp_path_factory.mktemp("outside")
    assert client.post("/ingest", json={"path": str(outside)}).status_code == 403
    assert client.post("/ingest", json={"path": "../.."}).status_code == 403
    assert client.post("/ingest", json={"path": "nope"}).status_code == 404


@pytest.mark.skipif(not CORPUS.exists(), reason="синтетический корпус не сгенерирован")
def test_synthetic_corpus_end_to_end(tmp_path):
    """Весь синтетический корпус, включая PDF: факт из PDF-протокола находится поиском."""
    pipe = RagPipeline(index_dir=tmp_path / "idx")
    info = pipe.ingest(CORPUS)
    assert 60 <= info["documents"] <= 100
    res = pipe.search("заключение протокол испытаний ПИ-ИС-7-2025-106", k=5)
    assert any(r["doc_id"] == "is7/protokol_2_is7.pdf" for r in res)
