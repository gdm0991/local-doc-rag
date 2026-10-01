"""Проверка ответов в eval/run_eval.py: ссылки [n] не засчитываются как значение."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parents[1] / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


@pytest.mark.parametrize("answer,keys,expected", [
    ("[5]", ["5"], False),                      # прогон 1, Q07: номер фрагмента принят за ответ
    ("Задержка — 5 с [2]", ["5"], True),
    ("0.24 МПа", ["0,24"], True),               # точка и запятая равнозначны
    ("10,24 МПа", ["0,24"], False),
    ("0,245 МПа", ["0,24"], False),
    ("Слот 2, канал 3 [1]", ["слот 2", "канал 3"], True),
    ("192.168.17.100", ["192.168.17.10"], False),
    ("изделие не соответствует требованиям", ["не соответствует"], True),
])
def test_answer_ok(answer, keys, expected):
    assert run_eval.answer_ok(answer, keys) is expected


def test_percentile_nearest_rank():
    assert run_eval.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert run_eval.pct(list(range(1, 31)), 95) == 29
