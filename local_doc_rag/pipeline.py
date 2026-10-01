"""Сборка: загрузка → нарезка → индексы → гибридный поиск → ответ."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from .chunking import Chunk, chunk_documents
from .hybrid import rrf
from .index_bm25 import BM25Index
from .index_vec import DEFAULT_MODEL, VectorIndex
from .ingest import load_documents
from .llm import REFUSAL_TEXT, Generator, NoLLM, is_refusal
from .text import tokenize

log = logging.getLogger(__name__)

MODES = ("hybrid", "bm25", "vector")

# Порог покрытия запроса для отказа без LLM. Подобран `python eval/run_eval.py --calibrate`
# на eval/calibration.jsonl (10 вопросов, не пересекается с eval/questions.jsonl):
# 8 верных решений из 10. Набор маленький, порог грубый — см. README, «Ограничения».
DEFAULT_COVERAGE_THRESHOLD = 0.424


class RagPipeline:
    def __init__(self, index_dir: str | Path = "index", chunk_size: int = 800, overlap: int = 150,
                 model_name: str = DEFAULT_MODEL, generator: Generator | None = None,
                 coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD):
        self.index_dir = Path(index_dir)
        self.chunk_size, self.overlap = chunk_size, overlap
        self.model_name = model_name
        self.generator: Generator = generator or NoLLM()
        self.coverage_threshold = coverage_threshold
        self.chunks: list[Chunk] = []
        self.bm25: BM25Index | None = None
        self.vec: VectorIndex | None = None

    # ------------------------------------------------------------------ индекс
    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.bm25 is not None and self.vec is not None

    def ingest(self, path: str | Path) -> dict:
        t0 = time.perf_counter()
        docs = load_documents(path)
        chunks = chunk_documents(docs, self.chunk_size, self.overlap)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (self.index_dir / "meta.json").write_text(json.dumps({
            "source": str(path), "documents": len(docs), "chunks": len(chunks),
            "chunk_size": self.chunk_size, "overlap": self.overlap, "model": self.model_name,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        self._build(chunks)
        return {"documents": len(docs), "chunks": len(chunks), "embed_backend": self.vec.backend_name,
                "seconds": round(time.perf_counter() - t0, 2)}

    def load(self) -> bool:
        p = self.index_dir / "chunks.jsonl"
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        self._build(chunks)
        return True

    def _build(self, chunks: list[Chunk]) -> None:
        texts = [c.index_text for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Index(texts)
        self.vec = VectorIndex(texts, cache_dir=self.index_dir, model_name=self.model_name)

    # ------------------------------------------------------------------- поиск
    def search(self, query: str, k: int = 5, mode: str = "hybrid", depth: int = 50) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode должен быть одним из {MODES}")
        if not self.ready:
            return []
        bm = self.bm25.search(query, depth) if mode in ("hybrid", "bm25") else []
        ve = self.vec.search(query, depth) if mode in ("hybrid", "vector") else []
        bm_rank = {i: r for r, (i, _) in enumerate(bm, 1)}
        ve_rank = {i: r for r, (i, _) in enumerate(ve, 1)}
        if mode == "hybrid":
            fused = rrf([[i for i, _ in bm], [i for i, _ in ve]])
        elif mode == "bm25":
            fused = bm
        else:
            fused = ve
        out = []
        for i, score in fused[:k]:
            c = self.chunks[i]
            out.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "text": c.text,
                        "score": round(float(score), 6), "bm25_rank": bm_rank.get(i),
                        "vector_rank": ve_rank.get(i)})
        return out

    # ---------------------------------------------------------- отказ без LLM
    def coverage(self, query: str, passage: dict) -> float:
        """Доля «веса» запроса (по IDF), чьи термины встречаются во фрагменте.

        Слова и обозначения, которых нет во всём корпусе («ИС-11»), получают максимальный IDF,
        поэтому вопрос про несуществующий стенд даёт низкое покрытие.
        """
        q = set(tokenize(query))
        if not q:
            return 0.0
        have = set(tokenize(passage["title"] + "\n" + passage["text"]))
        total = sum(self.bm25.idf(t) for t in q)
        hit = sum(self.bm25.idf(t) for t in q if t in have)
        return hit / total if total else 0.0

    # ------------------------------------------------------------------- ответ
    def ask(self, question: str, k: int = 5, mode: str = "hybrid", use_llm: bool = True) -> dict:
        t0 = time.perf_counter()
        passages = self.search(question, k=k, mode=mode)
        t_search = time.perf_counter() - t0
        cov = self.coverage(question, passages[0]) if passages else 0.0
        llm_on = use_llm and not isinstance(self.generator, NoLLM)
        result = {"question": question, "passages": passages, "coverage": round(cov, 3),
                  "llm": self.generator.name if llm_on else "none"}
        if not passages:
            result.update(answer=REFUSAL_TEXT, refused=True)
        elif llm_on:
            text = self.generator.generate(question, passages)
            result.update(answer=text, refused=is_refusal(text))
        else:
            # Без LLM ответа «словами» нет: отдаём фрагменты-цитаты либо отказ по порогу покрытия.
            refused = cov < self.coverage_threshold
            result.update(answer=REFUSAL_TEXT if refused else None, refused=refused)
        result["timings_ms"] = {"search": round(t_search * 1000, 1),
                                "total": round((time.perf_counter() - t0) * 1000, 1)}
        return result

    def info(self) -> dict:
        return {"ready": self.ready, "chunks": len(self.chunks),
                "documents": len({c.doc_id for c in self.chunks}),
                "embed_backend": self.vec.backend_name if self.vec else None,
                "embed_fallback_reason": self.vec.fallback_reason if self.vec else None,
                "llm": self.generator.name, "index_dir": str(self.index_dir)}
