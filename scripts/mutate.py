#!/usr/bin/env python3
"""Мутационный прогон: сколько наших тестов держат подмену смысла, а не строк.

Зачем это отдельно от `pytest`. Зелёная сюита говорит, что код проходит тесты,
и ничего не говорит о том, поймали бы тесты **изменение поведения**. Разница не
теоретическая: за два дня ручных мутаций она вскрыла шесть слабых тестов —
порог страницы, принимавший что угодно ниже 150 КБ; проверку «есть хоть какое-то
предупреждение»; вырожденную фикстуру из одних добывающих; тест, дававший
одинаковый ответ в обеих системах координат.

Ручные мутации не воспроизводятся и не считаются. Этот скрипт делает то же
систематически и одной командой:

    python scripts/mutate.py --module aios/balance.py --tests tests/test_balance.py
    python scripts/mutate.py --module aios/trust.py --limit 40 --seed 20260817

Что считается мутацией. Только подмены, меняющие **смысл**, а не форму:
сравнения (`<` и `<=`, `>` и `>=`), границы (`+` и `-`), булевы константы,
`and` и `or`, обнуление числовых литералов. Переименования и перестановки не
трогаются — они смысла не меняют, и «выжившая» такая мутация ничего не сообщает.

Вердикт по каждой:

* **убита** — тесты покраснели: поведение под сторожем;
* **выжила** — тесты зелёные при изменённом поведении: сторожа нет либо он
  смотрит не туда. Это находка, а не статистика: каждую выжившую надо либо
  закрыть тестом, либо объяснить, почему подмена смысла не меняет;
* **сломана** — модуль перестал импортироваться (синтаксис). В счёт не идёт.

Случайность только через `--seed`: тот же seed даёт тот же набор мутаций, иначе
цифру «выжило N» нельзя повторить и нельзя предъявить.

Код возврата: `0` — выживших нет, `1` — команда не выполнилась, `2` — есть
выжившие. Ненулевой код при выживших намеренный: прогон стоит ставить в тот же
ряд, что и `ruff` с `pytest`, а не читать глазами.
"""

from __future__ import annotations

import argparse
import ast
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Подмены сравнений: каждая меняет границу включения, то есть ровно тот класс
#: ошибок, из-за которого пишут тесты на «ровно на пределе».
COMPARISONS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

BINARY: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}

BOOLEAN: dict[type[ast.boolop], type[ast.boolop]] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


@dataclass(frozen=True)
class Mutation:
    """Одна подмена смысла: где и что заменено.

    `number` — порядковый номер в плане, и без него описание не опознаёт
    мутацию однозначно: в строке `if a < b or c < d:` два разных сравнения
    выглядят одинаково («строка 221: сравнение Lt -> LtE»). Поймано 05.09 на
    `aios/policy.py`: отчёт назвал выжившей мутацию, которую тест на самом деле
    ловил, — выжила её тёзка из той же строки, и найти её по описанию было
    нельзя. Номер даёт и воспроизведение: `--only N` повторяет ровно её.
    """

    number: int
    line: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"#{self.number} строка {self.line}: {self.kind} — {self.detail}"


class _Planner(ast.NodeVisitor):
    """Собрать все возможные подмены, не применяя ни одной."""

    def __init__(self) -> None:
        self.found: list[Mutation] = []

    def _add(self, line: int, kind: str, detail: str) -> None:
        self.found.append(Mutation(len(self.found), line, kind, detail))

    def visit_Compare(self, node: ast.Compare) -> None:
        for index, operator in enumerate(node.ops):
            replacement = COMPARISONS.get(type(operator))
            if replacement is not None:
                self._add(node.lineno, "сравнение",
                          f"{type(operator).__name__} -> {replacement.__name__} (#{index})")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        replacement = BINARY.get(type(node.op))
        if replacement is not None:
            self._add(node.lineno, "арифметика",
                      f"{type(node.op).__name__} -> {replacement.__name__}")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        replacement = BOOLEAN.get(type(node.op))
        if replacement is not None:
            self._add(node.lineno, "булева связка",
                      f"{type(node.op).__name__} -> {replacement.__name__}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # `True`/`False` — сначала, иначе они попадут в ветку чисел: в Python
        # `isinstance(True, int)` истинно, и булева константа мутировала бы в 0.
        if isinstance(node.value, bool):
            self._add(node.lineno, "булева константа", f"{node.value} -> {not node.value}")
        elif isinstance(node.value, (int, float)) and node.value not in (0, 0.0):
            self._add(node.lineno, "число", f"{node.value} -> 0")
        self.generic_visit(node)


class _Applier(ast.NodeTransformer):
    """Применить ровно одну подмену — ту, что стоит под номером `target`."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.seen = -1

    def _take(self) -> bool:
        self.seen += 1
        return self.seen == self.target

    # Во всех обходчиках свой узел проверяется ДО спуска к детям — тем же
    # порядком, что и в `_Planner`. Обратный порядок (`generic_visit` первым)
    # даёт обход снизу вверх, номера расходятся с планом, и мутация номер N
    # применяется не там, где обещано. Поймано на своём же инструменте до
    # первого прогона: `plan()[0]` называл булеву связку, а `apply_one(src, 0)`
    # менял число.
    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        ops = list(node.ops)
        for index, operator in enumerate(ops):
            replacement = COMPARISONS.get(type(operator))
            if replacement is not None and self._take():
                ops[index] = replacement()
                return ast.Compare(left=node.left, ops=ops, comparators=node.comparators)
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        replacement = BINARY.get(type(node.op))
        if replacement is not None and self._take():
            return ast.BinOp(left=node.left, op=replacement(), right=node.right)
        self.generic_visit(node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        replacement = BOOLEAN.get(type(node.op))
        if replacement is not None and self._take():
            return ast.BoolOp(op=replacement(), values=node.values)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool):
            if self._take():
                return ast.Constant(value=not node.value)
        elif (isinstance(node.value, (int, float))
                and node.value not in (0, 0.0) and self._take()):
            return ast.Constant(value=type(node.value)(0))
        return node


def plan(source: str) -> list[Mutation]:
    """Все подмены смысла, какие в этом файле вообще возможны."""
    planner = _Planner()
    planner.visit(ast.parse(source))
    return planner.found


def apply_one(source: str, index: int) -> str:
    """Текст модуля с применённой подменой номер `index`."""
    tree = _Applier(index).visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _run_tests(tests: list[str], timeout: float) -> bool:
    """Зелёные ли тесты. Падение по таймауту считаем убийством, а не выживанием.

    Мутация нередко превращает выход в бесконечный цикл; ждать её вечно нельзя,
    а записать «выжила» было бы прямой ложью: поведение изменилось так, что
    прогон не заканчивается.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:warnings",
             "--no-header", *tests],
            capture_output=True, text=True, cwd=ROOT, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", type=Path, required=True, help="файл, который мутируем")
    parser.add_argument("--tests", action="append", default=None,
                        help="какие тесты гонять; по умолчанию tests/test_<имя>.py")
    parser.add_argument("--limit", type=int, default=30,
                        help="сколько мутаций проверить (0 — все)")
    parser.add_argument("--seed", type=int, default=20260817,
                        help="seed отбора мутаций; тот же seed даёт тот же набор")
    parser.add_argument("--only", type=int, action="append", default=None,
                        help="проверить только эти номера мутаций (из прошлого отчёта)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="секунд на один прогон тестов")
    arguments = parser.parse_args(argv)

    module = arguments.module if arguments.module.is_absolute() else ROOT / arguments.module
    if not module.is_file():
        print(f"нет файла: {module}", file=sys.stderr)
        return 1

    tests = arguments.tests or [f"tests/test_{module.stem}.py"]
    for one in tests:
        if not (ROOT / one).exists():
            print(f"нет тестов: {one} — укажите --tests явно", file=sys.stderr)
            return 1

    # Прогон, убитый снаружи (таймаут команды, Ctrl-C по процессу), до `finally`
    # не доходит — и оставляет в рабочем дереве **мутированный** модуль. Случилось
    # 06.09 на `aios/run.py`: файл десять минут лежал переписанным `ast.unparse`,
    # то есть без единого комментария, и заметил это `ruff`, а не человек.
    # Поэтому восстановление стоит и на входе, а не только на выходе.
    backup = module.with_suffix(module.suffix + ".mutation-backup")
    if backup.is_file():
        shutil.copy2(backup, module)
        backup.unlink()
        print(f"восстановлен {module.name} из бэкапа прерванного прогона",
              file=sys.stderr)

    source = module.read_text(encoding="utf-8")
    every = plan(source)
    if not every:
        print(f"в {module.name} нечего мутировать")
        return 0

    if arguments.only:
        chosen = [one for one in arguments.only if 0 <= one < len(every)]
        if not chosen:
            print(f"номера вне плана: в {module.name} мутаций {len(every)}", file=sys.stderr)
            return 1
    else:
        order = list(range(len(every)))
        random.Random(arguments.seed).shuffle(order)
        chosen = order if arguments.limit == 0 else order[:arguments.limit]

    # Модуль бывает и вне репозитория (так его гоняют тесты самого инструмента),
    # и `relative_to` на таком пути бросает ValueError — трассировкой, сразу
    # после того, как бэкап уже снят.
    inside = module.is_relative_to(ROOT)
    print(f"модуль:  {module.relative_to(ROOT) if inside else module}")
    print(f"тесты:   {' '.join(tests)}")
    print(f"мутаций: {len(chosen)} из {len(every)} возможных, seed {arguments.seed}")
    print()

    shutil.copy2(module, backup)

    # `finally` спасает от исключения и от Ctrl-C, но не от `SIGTERM`: так
    # прогон убивает таймаут вызывающей команды или диспетчер задач, и модуль
    # остаётся мутированным. Ровно это случилось 09.09 — `aios/run.py` пролежал
    # переписанным `ast.unparse`, и заметил снова `ruff`, а не человек.
    # Обработчик восстанавливает файл и выходит тем же кодом, что и без него.
    def _restore_and_exit(number: int, _frame: object) -> None:
        shutil.copy2(backup, module)
        backup.unlink(missing_ok=True)
        print(f"\nпрогон прерван сигналом {number}: {module.name} восстановлен",
              file=sys.stderr)
        raise SystemExit(128 + number)

    previous = {
        number: signal.signal(number, _restore_and_exit)
        for number in (signal.SIGTERM, signal.SIGHUP)
        if hasattr(signal, number.name)
    }

    survived: list[Mutation] = []
    broken = killed = 0
    started = time.monotonic()
    try:
        for position, index in enumerate(chosen, start=1):
            mutation = every[index]
            try:
                module.write_text(apply_one(source, index), encoding="utf-8")
            except SyntaxError:
                broken += 1
                continue
            green = _run_tests(tests, arguments.timeout)
            mark = "ВЫЖИЛА" if green else "убита"
            if green:
                survived.append(mutation)
            else:
                killed += 1
            print(f"  [{position:3d}/{len(chosen)}] {mark:7s} {mutation} "
                  f"({time.monotonic() - started:.0f} с)", flush=True)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        # Обработчик сигнала мог восстановить и убрать бэкап раньше нас.
        if backup.is_file():
            shutil.copy2(backup, module)
            backup.unlink()

    print()
    print(f"убито {killed}, выжило {len(survived)}, не собралось {broken}")
    if survived:
        print()
        print("ВЫЖИВШИЕ — тесты не заметили изменения поведения:")
        for mutation in survived:
            print(f"  {mutation}")
        print()
        print("Каждую надо либо закрыть тестом, либо объяснить, почему подмена "
              "смысла не меняет (эквивалентная мутация — законный ответ, "
              "молчание — нет).")
        return 2
    return 0


if __name__ == "__main__":
    # Как и остальные скрипты пакета: вывод не зависит от локали консоли
    # (п. 3.5 ТЗ). Умолчание русской Windows — cp1251, и без этого падало бы
    # даже `--help`.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
