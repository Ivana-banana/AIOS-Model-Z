"""Инструмент мутационного прогона: план и применение обязаны совпадать.

Инструмент, который врёт о себе, хуже отсутствующего: «выжило 0» из сбитой
нумерации читается как доказательство качества тестов, а означает, что мутации
применялись не туда, куда отчёт их записал.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mutate import apply_one, plan  # noqa: E402

SOURCE = "\n".join([
    "def f(a, b):",
    "    if a < 10 and b:",
    "        return a + 1",
    "    return True",
    "",
])


def test_every_planned_mutation_lands_where_the_plan_says():
    """Мутация номер N меняет ровно то, что план назвал под номером N.

    Первая редакция инструмента обходила дерево в разном порядке: `_Planner`
    сверху вниз, `_Applier` снизу вверх (`generic_visit` стоял первым). План
    называл под нулём булеву связку, а применялась подмена числа. Поймано на
    самом инструменте до первого настоящего прогона.
    """
    planned = plan(SOURCE)
    assert len(planned) == 6, [str(one) for one in planned]

    expected = {
        "булева связка": "or b",
        "сравнение": "a <= 10",
        "число": "a < 0",
        "арифметика": "a - 1",
    }
    for index, mutation in enumerate(planned):
        body = apply_one(SOURCE, index)
        if mutation.kind == "число" and mutation.line == 3:
            assert "a + 0" in body, body
        elif mutation.kind == "булева константа":
            assert "return False" in body, body
        else:
            assert expected[mutation.kind] in body, f"{mutation}:\n{body}"


def test_one_mutation_at_a_time():
    """Применяется ровно одна подмена: иначе вердикт относится к смеси, а не к ней."""
    body = apply_one(SOURCE, 1)  # сравнение Lt → LtE

    assert "a <= 10" in body
    assert "and b" in body, "булева связка не должна была измениться"
    assert "a + 1" in body, "арифметика не должна была измениться"
    assert "return True" in body, "константа не должна была измениться"


def test_booleans_are_not_mistaken_for_numbers():
    """`True` мутирует в `False`, а не в ноль.

    В Python `isinstance(True, int)` истинно, и без явной ветки булева константа
    попала бы в правило чисел — подмена вышла бы бессмысленной.
    """
    kinds = [one.kind for one in plan("x = True\ny = 5\n")]

    assert kinds == ["булева константа", "число"], kinds
    assert "x = False" in apply_one("x = True\ny = 5\n", 0)


def test_zero_is_left_alone():
    """Ноль в ноль — не мутация: такая «подмена» ничего не меняет и лишь шумит."""
    assert plan("x = 0\ny = 0.0\n") == []


def test_an_index_beyond_the_plan_changes_nothing():
    """Номер вне плана оставляет исходник как есть, а не роняет прогон."""
    assert apply_one(SOURCE, 999).strip() == apply_one(SOURCE, -1).strip()


@pytest.mark.parametrize("source", [
    "a = 1 < 2 < 3\n",          # цепочка сравнений
    "if a and b and c:\n    pass\n",
    "x = (1 + 2) * 3 - 4\n",
])
def test_generated_code_stays_valid_python(source: str):
    """Каждая мутация обязана быть разбираемым Python — иначе вердикт «сломана»."""
    import ast

    for index in range(len(plan(source))):
        ast.parse(apply_one(source, index))
