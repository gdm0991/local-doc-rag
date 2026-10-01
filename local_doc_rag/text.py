"""Токенизация для BM25 и оценки покрытия запроса.

Обозначения вида «ИС-7», «К-200», «0,25», «192.168.11.10» сохраняются целым токеном:
по ним чаще всего и ищут. Русские слова приводятся к основе стеммером Snowball.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("russian")
except ImportError:  # без стеммера поиск работает, но хуже на словоформах
    _STEMMER = None

TOKEN_RE = re.compile(r"\w+(?:[-.,/]\w+)*", re.UNICODE)
DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")

STOPWORDS = {
    "а", "в", "во", "и", "к", "ко", "на", "о", "об", "от", "по", "под", "при", "с", "со", "у", "из", "за",
    "до", "для", "не", "ни", "но", "или", "ли", "же", "то", "как", "что", "это", "этот", "эта", "эти",
    "какой", "какая", "какое", "какие", "каков", "какова", "каково", "каковы", "какую", "каким", "каких",
    "сколько", "где", "когда", "кто", "чем", "чему", "чего", "ли", "есть", "был", "была", "было", "были",
    "его", "её", "ее", "их", "он", "она", "оно", "они", "мы", "вы", "я", "ты", "нужно", "надо", "можно",
    "ли", "бы", "уже", "все", "всё", "всех", "весь", "также", "так", "там", "тут", "через",
}


@lru_cache(maxsize=100_000)
def _stem(word: str) -> str:
    if _STEMMER is None or not word.isalpha():
        return word
    return _STEMMER.stemWord(word)


def normalize(text: str) -> str:
    # «0.25» и «0,25» должны совпадать: в документах десятичная запятая
    return DECIMAL_DOT_RE.sub(",", text.lower().replace("ё", "е"))


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for tok in TOKEN_RE.findall(normalize(text)):
        if not keep_stopwords and tok in STOPWORDS:
            continue
        if any(ch in tok for ch in "-/") and not tok.replace("-", "").replace("/", "").isdigit():
            out.append(tok)  # составное обозначение целиком: «ис-7»
            out.extend(_stem(p) for p in re.split(r"[-/]", tok) if len(p) > 1 and p.isalpha())
        else:
            out.append(_stem(tok))
    return out
