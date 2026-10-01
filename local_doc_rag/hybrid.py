"""Слияние выдач методом Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = сумма по выдачам 1 / (k + rank(d)), rank с 1.
RRF смотрит только на места в выдаче, а не на сырые оценки, поэтому не нужно
приводить к одной шкале оценки BM25 (десятки) и косинусы (0..1).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

RRF_K = 60


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K,
        weights: Sequence[float] | None = None) -> list[tuple[Hashable, float]]:
    """Сливает несколько ранжированных списков. Возвращает [(id, score)] по убыванию score.

    При равенстве оценок порядок — по первому появлению id (стабильно и воспроизводимо).
    """
    if k < 0:
        raise ValueError("k должен быть >= 0")
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("число весов не совпадает с числом выдач")
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking, w in zip(rankings, weights):
        seen_here = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen_here:  # дубль в одной выдаче учитываем один раз
                continue
            seen_here.add(item)
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda x: (-x[1], first_seen[x[0]]))
