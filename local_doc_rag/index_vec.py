"""Векторный индекс.

Основной вариант — fastembed (ONNX Runtime, без PyTorch) с моделью
intfloat/multilingual-e5-small (384 измерения). Эмбеддинги фрагментов кешируются
на диске: ключ кеша — хеш имени модели и всех текстов, так что повторный запуск
на том же корпусе не пересчитывает векторы.

Запасной вариант — TF-IDF по символьным n-граммам (scikit-learn). Он включается,
если fastembed не установлен или модель не скачалась, либо принудительно через
LDR_EMBED_BACKEND=tfidf. Это НЕ семантический поиск: он ловит общие подстроки,
а не смысл. Какой вариант реально работает, видно в GET /health (поле embed_backend).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def default_model_cache() -> Path:
    return Path(os.environ.get("LDR_MODEL_CACHE", Path.home() / ".cache" / "local-doc-rag" / "models"))


def _l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class _FastEmbedBackend:
    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding

        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        if model_name == DEFAULT_MODEL and model_name not in supported:
            # В fastembed 0.8 e5-small нет во встроенном списке — регистрируем ONNX-файл из репозитория модели.
            from fastembed.common.model_description import ModelSource, PoolingType

            try:
                TextEmbedding.add_custom_model(
                    model=model_name, pooling=PoolingType.MEAN, normalization=True,
                    sources=ModelSource(hf=model_name), dim=384, model_file="onnx/model.onnx")
            except ValueError:
                pass  # уже зарегистрирована в этом процессе
        self.model = TextEmbedding(model_name, cache_dir=str(cache_dir))
        self.model_name = model_name
        # e5 обучалась с префиксами query:/passage:, без них качество падает
        self.is_e5 = "e5" in model_name.lower()
        self.name = f"fastembed:{model_name}"

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self.is_e5:
            texts = [f"passage: {t}" for t in texts]
        return _l2norm(np.array(list(self.model.embed(texts, batch_size=32)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        t = f"query: {text}" if self.is_e5 else text
        return _l2norm(np.array(list(self.model.embed([t])), dtype=np.float32))[0]


class _TfidfBackend:
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, sublinear_tf=True)
        self.name = "tfidf-fallback (char 2-4, не семантический)"
        self.fitted = False

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        m = self.vec.fit_transform(texts)
        self.fitted = True
        return _l2norm(m.toarray().astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2norm(self.vec.transform([text]).toarray().astype(np.float32))[0]


def make_backend(model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None):
    """Возвращает (backend, причина_fallback или None)."""
    forced = os.environ.get("LDR_EMBED_BACKEND", "").lower()
    if forced == "tfidf":
        return _TfidfBackend(), "задано LDR_EMBED_BACKEND=tfidf"
    try:
        return _FastEmbedBackend(model_name, cache_dir or default_model_cache()), None
    except Exception as e:  # нет пакета, нет сети, битый кеш модели
        reason = f"fastembed недоступен: {type(e).__name__}: {e}"
        log.warning("%s — включён TF-IDF fallback", reason)
        return _TfidfBackend(), reason


class VectorIndex:
    def __init__(self, texts: list[str], cache_dir: Path | None = None, model_name: str = DEFAULT_MODEL,
                 backend=None):
        self.backend, self.fallback_reason = (backend, None) if backend else make_backend(model_name)
        self.n = len(texts)
        self._qcache: dict[str, np.ndarray] = {}
        if not texts:
            self.matrix = np.zeros((0, 1), dtype=np.float32)
            return
        if isinstance(self.backend, _TfidfBackend):
            self.matrix = self.backend.embed_passages(texts)  # считается за доли секунды, не кешируем
            return
        key = hashlib.sha1((self.backend.name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = Path(cache_dir) / f"vectors_{key}.npy" if cache_dir else None
        if path and path.exists():
            self.matrix = np.load(path)
            log.info("Эмбеддинги взяты из кеша %s", path)
        else:
            self.matrix = self.backend.embed_passages(texts)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.matrix)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def query_vector(self, query: str) -> np.ndarray:
        v = self._qcache.get(query)
        if v is None:
            v = self.backend.embed_query(query)
            if len(self._qcache) > 1000:
                self._qcache.clear()
            self._qcache[query] = v
        return v

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if self.n == 0:
            return []
        sims = self.matrix @ self.query_vector(query)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
