"""Лексический индекс BM25 (rank_bm25.BM25Okapi)."""

from __future__ import annotations

import math

import numpy as np
from rank_bm25 import BM25Okapi

from .text import tokenize


class BM25Index:
    def __init__(self, texts: list[str]):
        self.tokens = [tokenize(t) for t in texts]
        # BM25Okapi падает на пустом корпусе — подстрахуемся пустым документом
        self.bm25 = BM25Okapi(self.tokens if self.tokens else [[""]])
        self.n_docs = len(self.tokens)
        df: dict[str, int] = {}
        for toks in self.tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.df = df

    def idf(self, term: str) -> float:
        """Сглаженный IDF. Для слова, которого нет в корпусе, — максимальный."""
        n = max(self.n_docs, 1)
        return math.log(1 + (n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if self.n_docs == 0:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        # Кандидат — фрагмент, где есть хотя бы один термин запроса. Фильтр «score > 0» не годится:
        # в BM25Okapi на маленьком корпусе IDF термина, встречающегося в половине документов, равен 0,
        # и совпадение пропадало из выдачи (нашёл тест test_search_modes[bm25]).
        qs = set(q)
        cand = [i for i, toks in enumerate(self.tokens) if qs.intersection(toks)]
        cand.sort(key=lambda i: (-scores[i], i))
        return [(i, float(scores[i])) for i in cand[:k]]
