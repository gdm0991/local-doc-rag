"""Командная строка: ldr ingest | search | ask | serve."""

from __future__ import annotations

import json
import os
from typing import Optional

import typer

from .llm import make_generator
from .pipeline import RagPipeline

app = typer.Typer(add_completion=False, help="local-doc-rag: офлайн-поиск по документам")

IndexOpt = typer.Option("index", "--index", "-i", help="Папка индекса")


@app.command()
def ingest(path: str = typer.Argument(..., help="Папка с .md/.txt/.pdf"), index: str = IndexOpt,
           chunk_size: int = 800, overlap: int = 150) -> None:
    """Проиндексировать документы."""
    p = RagPipeline(index, chunk_size=chunk_size, overlap=overlap)
    typer.echo(json.dumps(p.ingest(path), ensure_ascii=False))


@app.command()
def search(query: str, index: str = IndexOpt, k: int = 5, mode: str = "hybrid") -> None:
    """Найти фрагменты без генерации ответа."""
    p = RagPipeline(index)
    if not p.load():
        raise typer.Exit("Индекс не найден: сначала ldr ingest")
    for r in p.search(query, k=k, mode=mode):
        typer.echo(f"[{r['score']:.4f}] {r['chunk_id']}  bm25#{r['bm25_rank']} vec#{r['vector_rank']}")
        typer.echo("    " + r["text"][:200].replace("\n", " "))


@app.command()
def ask(question: str, index: str = IndexOpt, k: int = 5,
        no_llm: bool = typer.Option(False, "--no-llm", help="Только поиск и цитаты"),
        llm: str = typer.Option("llamacpp", help="llamacpp | ollama"),
        model: Optional[str] = typer.Option(None, help="Путь к GGUF или имя модели Ollama")) -> None:
    """Ответить на вопрос с цитатами."""
    gen = make_generator("none" if no_llm else llm, model)
    p = RagPipeline(index, generator=gen)
    if not p.load():
        raise typer.Exit("Индекс не найден: сначала ldr ingest")
    r = p.ask(question, k=k)
    typer.echo(f"Ответ: {r['answer'] if r['answer'] else '(без LLM — смотрите цитаты)'}")
    typer.echo(f"Отказ: {r['refused']}, покрытие запроса: {r['coverage']}, время: {r['timings_ms']} мс")
    for i, ps in enumerate(r["passages"], 1):
        typer.echo(f"[{i}] {ps['chunk_id']}: {ps['text'][:160].replace(chr(10), ' ')}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, index: str = IndexOpt,
          no_llm: bool = typer.Option(False, "--no-llm"), llm: str = "llamacpp",
          model: Optional[str] = None) -> None:
    """Запустить HTTP API."""
    import uvicorn

    os.environ["LDR_INDEX_DIR"] = index
    os.environ["LDR_LLM"] = "none" if no_llm else llm
    if model:
        os.environ["LDR_LLM_MODEL"] = model
    uvicorn.run("local_doc_rag.api:get_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
