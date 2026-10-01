"""Генератор ответа. Три варианта:

- none      — без языковой модели: только поиск и цитаты (режим --no-llm);
- llamacpp  — локальный GGUF-файл через llama-cpp-python;
- ollama    — HTTP API Ollama (по умолчанию http://localhost:11434).

Модель получает только найденные фрагменты и должна ответить «Не знаю»,
если ответа в них нет. Это просьба, а не гарантия: маленькие модели её нарушают,
доля нарушений измеряется в eval/run_eval.py.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Protocol

REFUSAL_TEXT = "Не знаю: в найденных документах ответа нет."

# Два варианта системного промпта, оба измерены на eval (Qwen2.5-3B Q4_K_M, 30 + 10 вопросов):
#   v1 — точность 18/30, отказ на ловушках 8/10   (eval/results_2026-10-01.md)
#   v2 — точность 16/30, отказ на ловушках 9/10   (eval/results_2026-10-01_run2_prompt-v2.md)
# v2 писался, чтобы модель не отвечала голым «[1]», но на этом наборе стал хуже по точности.
# Разница — 1–2 вопроса, на 30 вопросах это в пределах шума. По умолчанию v1, выбор — LDR_PROMPT=v2.
PROMPTS = {
    "v1": (
        "Ты отвечаешь на вопросы по технической документации испытательных стендов. "
        "Используй только пронумерованные фрагменты из сообщения пользователя. "
        "Если во фрагментах нет ответа на вопрос, ответь ровно: «" + REFUSAL_TEXT + "» "
        "Не додумывай значения. После ответа укажи номера использованных фрагментов в квадратных скобках, "
        "например [2]. Отвечай кратко, одним-двумя предложениями, на русском языке."
    ),
    "v2": (
        "Ты отвечаешь на вопросы по технической документации испытательных стендов. "
        "Используй только пронумерованные фрагменты из сообщения пользователя. "
        "Сначала найди фрагмент про тот самый стенд или документ, о котором спрашивают, и выпиши из него "
        "значение. Ответ — одно предложение на русском: значение с единицей измерения, затем номер "
        "фрагмента в квадратных скобках. Ответ без значения не допускается. "
        "Если ни в одном фрагменте нет ответа для указанного стенда или документа, ответь ровно: "
        "«" + REFUSAL_TEXT + "» Не додумывай и не бери значения другого стенда."
    ),
}


def system_prompt() -> str:
    return PROMPTS[os.environ.get("LDR_PROMPT", "v1")]


_REFUSAL_RE = re.compile(r"не\s+знаю|нет\s+(?:такой\s+|этой\s+)?информации|информаци\w*\s+нет|"
                         r"не\s+(?:указан|приведен|содерж)\w*", re.IGNORECASE)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text.replace("ё", "е")))


def build_prompt(question: str, passages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        # Заголовок обязателен: во фрагменте из середины документа часто нет названия стенда,
        # и без заголовка модель отказывалась отвечать при правильном фрагменте (замечено на eval, Q01).
        title = p.get("title", "")
        parts.append(f"[{i}] Документ: {p['doc_id']} — «{title}»\n{p['text']}")
    return "Фрагменты:\n\n" + "\n\n".join(parts) + f"\n\nВопрос: {question}"


class Generator(Protocol):
    name: str

    def generate(self, question: str, passages: list[dict]) -> str: ...


class NoLLM:
    name = "none"

    def generate(self, question: str, passages: list[dict]) -> str:
        raise RuntimeError("Генератор отключён (режим --no-llm)")


class LlamaCppGenerator:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int | None = None,
                 max_tokens: int = 160):
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(),
                         verbose=False)
        self.max_tokens = max_tokens
        self.name = f"llamacpp:{os.path.basename(model_path)}"

    def generate(self, question: str, passages: list[dict]) -> str:
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system_prompt()},
                      {"role": "user", "content": build_prompt(question, passages)}],
            max_tokens=self.max_tokens, temperature=0.0)
        return r["choices"][0]["message"]["content"].strip()


class OllamaGenerator:
    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: float = 120.0):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.name = f"ollama:{model}"

    def generate(self, question: str, passages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system_prompt()},
                         {"role": "user", "content": build_prompt(question, passages)}],
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.url}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["message"]["content"].strip()


def make_generator(kind: str | None = None, model: str | None = None, url: str | None = None) -> Generator:
    """Создаёт генератор по аргументам или переменным окружения LDR_LLM, LDR_LLM_MODEL, LDR_OLLAMA_URL."""
    kind = (kind or os.environ.get("LDR_LLM", "none")).lower()
    model = model or os.environ.get("LDR_LLM_MODEL", "")
    if kind in ("none", "no", "off", ""):
        return NoLLM()
    if kind == "llamacpp":
        if not model:
            raise ValueError("Для llamacpp укажите путь к GGUF: --model или LDR_LLM_MODEL")
        return LlamaCppGenerator(model)
    if kind == "ollama":
        return OllamaGenerator(model or "qwen2.5:1.5b-instruct",
                               url or os.environ.get("LDR_OLLAMA_URL", "http://localhost:11434"))
    raise ValueError(f"Неизвестный генератор: {kind}")
