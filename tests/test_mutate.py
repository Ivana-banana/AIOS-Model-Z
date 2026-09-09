"""Инструмент мутационного прогона: план и применение обязаны совпадать.

Инструмент, который врёт о себе, хуже отсутствующего: «выжило 0» из сбитой
нумерации читается как доказательство качества тестов, а означает, что мутации
применялись не туда, куда отчёт их записал.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts"))

from mutate import apply_one, plan  # noqa: E402

#: Тесты для подопытных прогонов ниже — **чужие**, а не свои.
#: `--tests tests/test_mutate.py` означал бы, что инструмент запускает тот самый
#: файл, который его и запустил: каждая мутация порождала вложенный `pytest`,
#: тот — новый `mutate.py`, и так вглубь. Пока `mutate.py` падал на пути к
#: замеру, этого не было видно; как только путь починили, рекурсия оставила
#: подопытный модуль мутированным. Нужен дешёвый файл, не зовущий инструмент.
NEUTRAL_TESTS = "tests/test_config.py"

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


def test_two_mutations_on_one_line_are_told_apart():
    """Две подмены в одной строке различимы в отчёте.

    `if a < b or c < d:` даёт два сравнения с одинаковым описанием. Пока в
    описании не было номера, отчёт называл выжившей мутацию, которую тест на
    самом деле ловил, — выживала её тёзка, и найти её было нельзя. Поймано
    05.09 на `aios/policy.py`.
    """
    source = "def f(a, b, c, d):" + chr(10) + "    return a < b or c < d" + chr(10)

    planned = [one for one in plan(source) if one.kind == "сравнение"]

    assert len(planned) == 2, [str(one) for one in planned]
    assert planned[0].line == planned[1].line, "обе в одной строке — иначе пример не тот"
    assert str(planned[0]) != str(planned[1]), (
        f"описания совпали, выжившую не опознать: {planned[0]}"
    )
    assert planned[0].number != planned[1].number

    # и номер действительно адресует свою подмену
    assert "a <= b" in apply_one(source, planned[0].number)
    assert "c <= d" in apply_one(source, planned[1].number)


def test_an_interrupted_run_restores_the_module_on_the_next_start(tmp_path):
    """Прогон, убитый снаружи, не оставляет модуль мутированным навсегда.

    `finally` восстанавливает файл при обычном выходе, но при убийстве процесса
    (таймаут команды, Ctrl-C) до него дело не доходит. 06.09 так и вышло:
    `aios/run.py` десять минут лежал переписанным `ast.unparse`, то есть без
    единого комментария, и заметил это `ruff`, а не человек.

    Поэтому восстановление стоит и на входе: увидев рядом бэкап прерванного
    прогона, инструмент возвращает файл прежде, чем что-либо мерить.
    """
    import subprocess
    import sys as system

    module = tmp_path / "жертва.py"
    module.write_text("# важный комментарий" + chr(10) + "def f(a):" + chr(10)
                      + "    return a < 10" + chr(10), encoding="utf-8")
    original = module.read_text(encoding="utf-8")

    # так выглядит дерево после убитого прогона: рядом лежит бэкап, а сам
    # модуль мутирован
    backup = module.with_suffix(".py.mutation-backup")
    backup.write_text(original, encoding="utf-8")
    module.write_text("def f(a):" + chr(10) + "    return a <= 10" + chr(10),
                      encoding="utf-8")

    root = ROOT
    done = subprocess.run(
        [system.executable, str(root / "scripts" / "mutate.py"),
         "--module", str(module), "--tests", NEUTRAL_TESTS, "--limit", "1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=root, timeout=600,
    )

    assert module.read_text(encoding="utf-8") == original, (
        "модуль не восстановлен из бэкапа прерванного прогона: "
        + module.read_text(encoding="utf-8")
    )
    assert not backup.exists(), "бэкап остался лежать рядом"
    assert "восстановлен" in (done.stderr or ""), done.stderr[:300]


def test_a_run_killed_by_sigterm_restores_the_module_itself(tmp_path):
    """`SIGTERM` тоже восстанавливает модуль — не дожидаясь следующего прогона.

    Восстановление «на входе» (тест выше) спасает лишь того, кто запустит
    мутацию по тому же модулю снова. Всё остальное — `ruff`, `pytest`,
    коммит — видит переписанный `ast.unparse` файл: без комментариев, без
    docstring, с перекроенными импортами. Так и вышло 09.09 на `aios/run.py`,
    вторично после 06.09: прогон убил таймаут вызывающей команды, `finally`
    при `SIGTERM` не выполняется, и файл пролежал мутированным, пока его не
    заметил `ruff`.

    Проверяется делом: настоящий подпроцесс, настоящий сигнал.

    **На Windows проверяется другое, и тест от этого не пропускается.**
    `Popen.send_signal(SIGTERM)` там зовёт `TerminateProcess`: процесс снимает
    ядро, обработчик Python не выполняется никогда — перехватить нечего, и
    никакая правка `mutate.py` этого не изменит. Значит, на Windows защита
    другая: рядом с модулем обязан остаться бэкап, по которому прерванный
    прогон виден и человеку, и сторожу
    `test_no_module_of_the_repository_is_left_mutated`. Пропуск здесь был бы
    хуже вдвойне — он и не проверяет ничего, и делает число пропусков разным
    на двух машинах проекта, а его сверяют строго (Р-247).
    """
    import signal
    import subprocess
    import sys as system
    import time

    module = tmp_path / "жертва.py"
    original = ("# важный комментарий" + chr(10) + "def f(a, b):" + chr(10)
                + "    if a < 10 and b:" + chr(10) + "        return a + 1" + chr(10)
                + "    return a - 1" + chr(10))
    module.write_text(original, encoding="utf-8")

    root = ROOT
    # Тесты нарочно медленные: нужно успеть послать сигнал в середине прогона.
    proc = subprocess.Popen(
        [system.executable, str(root / "scripts" / "mutate.py"),
         "--module", str(module), "--tests", NEUTRAL_TESTS, "--limit", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=root)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and module.read_text(encoding="utf-8") == original:
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    mutated = module.read_text(encoding="utf-8") != original
    proc.send_signal(signal.SIGTERM)
    _, errors = proc.communicate(timeout=120)

    if not mutated:
        pytest.skip("прогон закончился раньше, чем удалось прервать его на середине")

    backup = module.with_suffix(".py.mutation-backup")

    if system.platform == "win32":
        # Модуль остаётся мутированным — иначе и быть не может; проверяется, что
        # прерванный прогон не исчезает бесследно: бэкап рядом и хранит оригинал.
        assert backup.exists(), (
            "снятый прогон не оставил бэкапа: прерывание Windows-машины "
            "становится невидимым"
        )
        assert backup.read_text(encoding="utf-8") == original
        return

    assert module.read_text(encoding="utf-8") == original, (
        "убитый сигналом прогон оставил модуль мутированным: "
        + module.read_text(encoding="utf-8")
    )
    assert not backup.exists(), "бэкап остался лежать рядом"
    assert "прерван сигналом" in (errors or ""), errors[-300:]


def test_a_run_started_from_inside_a_run_refuses(tmp_path):
    """Замок против форк-бомбы: прогон изнутри прогона не запускается.

    `--tests tests/test_mutate.py` означает, что на каждую мутацию запускается
    `pytest` того самого файла, который запускает `mutate.py`, — и так вглубь
    без дна. В ночь на 09.09 это дважды уронило графическую сессию машины: 884
    живых процесса python по 41 МБ, 19,6 ГБ RSS при 15,9 ГБ памяти, а `earlyoom`
    убивал не виновников (каждый процесс мелкий, oom_score низкий), а
    пользовательскую шину dbus вместе со всем рабочим столом — снаружи это
    выглядело как «терминал закрылся сам». Нашла и измерила соседняя сессия.

    Развязать рекурсию в самих тестах мало: аргумент передаёт человек, и
    ошибиться он может снова. Замок держит независимо от аргументов.
    """
    import os
    import subprocess
    import sys as system

    module = tmp_path / "жертва.py"
    module.write_text("def f(a):" + chr(10) + "    return a < 10" + chr(10), encoding="utf-8")

    root = ROOT
    done = subprocess.run(
        [system.executable, str(root / "scripts" / "mutate.py"),
         "--module", str(module), "--tests", NEUTRAL_TESTS, "--limit", "1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=root, timeout=600, env={**os.environ, "AIOS_MUTATE_CHILD": "1"},
    )

    assert done.returncode == 1, done.stdout[-300:]
    assert "изнутри мутационного прогона" in done.stderr, done.stderr[-300:]
    assert not (tmp_path / "жертва.py.mutation-backup").exists(), (
        "отказ случился после снятия бэкапа — замок стоит слишком поздно"
    )


def test_a_red_suite_refuses_instead_of_reporting_every_mutation_killed(tmp_path):
    """Красный набор — отказ, а не отчёт «убито всё, выжило 0».

    Без базового прогона на неизменённом модуле любая причина, по которой набор
    красен сам по себе — сломанный импорт, отсутствующая зависимость, один
    упавший тест, неверный `--tests`, замок против рекурсии, — давала «убито
    всё, выжило 0» и код 0. Это худший из возможных отчётов: он неотличим от
    настоящего успеха и читается как доказательство качества тестов. Ровно ту же
    ложь первый абзац этого файла называет недопустимой, только там источником
    была сбитая нумерация.

    Нашла соседняя сессия — по замеру, который сделал я сам: прогон с заведомо
    неверным `--tests` напечатал «убито 7, выжило 0» и вернул 0.
    """
    import subprocess
    import sys as system

    module = tmp_path / "жертва.py"
    module.write_text("def f(a):" + chr(10) + "    return a < 10" + chr(10), encoding="utf-8")
    red = tmp_path / "test_красный.py"
    red.write_text("def test_red():" + chr(10) + "    assert False" + chr(10), encoding="utf-8")

    root = ROOT
    done = subprocess.run(
        [system.executable, str(root / "scripts" / "mutate.py"),
         "--module", str(module), "--tests", str(red), "--limit", "1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=root, timeout=600,
    )

    assert done.returncode == 1, done.stdout[-400:]
    assert "красные на неизменённом модуле" in done.stderr, done.stderr[-400:]
    assert "выжило" not in done.stdout, (
        "инструмент всё-таки выдал вердикт по красному набору: " + done.stdout[-400:]
    )


def test_no_module_of_the_repository_is_left_mutated():
    """В дереве не лежит ни одного бэкапа прерванного мутационного прогона.

    Это защита, которая работает **на любой платформе**, в отличие от
    восстановления по сигналу: Windows снимает процесс `TerminateProcess`, и
    перехватить там нечего. Бэкап рядом с модулем означает ровно одно — прогон
    убили на середине, и модуль сейчас в мутированном виде.

    Поймать это иначе нечем: пережившая мутация тем и определяется, что тесты
    от неё зелёные. 06.09 и 09.09 такой файл дважды доживал до `ruff`, и оба
    раза его замечал человек, а не прогон.
    """
    left = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.mutation-backup")
        if ".venv" not in path.parts and ".git" not in path.parts
    )

    assert not left, (
        "мутационный прогон прерван, модуль остался переписанным — "
        "восстановите его из бэкапа и удалите бэкап: " + ", ".join(left)
    )
