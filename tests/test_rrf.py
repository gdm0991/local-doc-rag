import pytest

from local_doc_rag.hybrid import rrf
from local_doc_rag.text import tokenize


def test_rrf_formula():
    res = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert res["a"] == pytest.approx(1 / 61)
    assert res["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert res["c"] == pytest.approx(1 / 62)


def test_rrf_item_in_both_lists_wins():
    order = [x for x, _ in rrf([["a", "b", "c"], ["c", "d", "a"]])]
    # «a»: 1/61 + 1/63; «c»: 1/63 + 1/61 — равны, порядок по первому появлению
    assert order[:2] == ["a", "c"]
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_empty_and_single():
    assert rrf([]) == []
    assert [x for x, _ in rrf([["x", "y"]])] == ["x", "y"]


def test_rrf_duplicates_in_one_list_counted_once():
    res = dict(rrf([["a", "a", "b"]], k=0))
    assert res["a"] == pytest.approx(1.0)
    assert res["b"] == pytest.approx(1 / 3)


def test_rrf_weights():
    res = dict(rrf([["a"], ["b"]], k=0, weights=[2.0, 1.0]))
    assert res["a"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        rrf([["a"]], weights=[1.0, 2.0])


def test_tokenize_keeps_identifiers_and_decimals():
    toks = tokenize("Уставка стенда ИС-7: 0.25 МПа")
    assert "ис-7" in toks
    assert "0,25" in toks  # десятичная точка приводится к запятой
    assert tokenize("уставки") == tokenize("уставка")  # стемминг
