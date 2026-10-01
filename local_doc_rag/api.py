"""HTTP API на FastAPI.

POST /ingest  — проиндексировать папку на сервере (только внутри LDR_INGEST_ROOT)
POST /search  — гибридный поиск, возвращает фрагменты
POST /ask     — ответ LLM с цитатами или, без LLM, цитаты/отказ
GET  /health  — состояние индекса и какой векторный бэкенд реально работает
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm import make_generator
from .pipeline import RagPipeline


class IngestRequest(BaseModel):
    path: str = Field(..., description="Папка или файл на сервере")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    use_llm: bool = True


def create_app(pipeline: RagPipeline | None = None, ingest_root: str | Path | None = None) -> FastAPI:
    if pipeline is None:
        pipeline = RagPipeline(index_dir=os.environ.get("LDR_INDEX_DIR", "index"), generator=make_generator())
        pipeline.load()
    root = Path(ingest_root or os.environ.get("LDR_INGEST_ROOT", ".")).resolve()

    app = FastAPI(title="local-doc-rag", version="0.1.0",
                  description="Офлайн-поиск по документам: BM25 + векторы + RRF, ответ локальной моделью.")
    app.state.pipeline = pipeline

    def require_index() -> None:
        if not pipeline.ready:
            raise HTTPException(409, "Индекс пуст: сначала POST /ingest")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **pipeline.info()}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        p = Path(req.path)
        p = (p if p.is_absolute() else root / p).resolve()
        if root != p and root not in p.parents:
            raise HTTPException(403, f"Путь вне разрешённого корня {root}")
        if not p.exists():
            raise HTTPException(404, "Путь не найден")
        return pipeline.ingest(p)

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        require_index()
        return {"query": req.query, "mode": req.mode, "results": pipeline.search(req.query, req.k, req.mode)}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        require_index()
        try:
            return pipeline.ask(req.question, k=req.k, use_llm=req.use_llm)
        except OSError as e:  # Ollama не запущена и т.п.
            raise HTTPException(502, f"Генератор недоступен: {e}") from e

    return app


def get_app() -> FastAPI:
    """Фабрика для uvicorn: uvicorn local_doc_rag.api:get_app --factory"""
    return create_app()
