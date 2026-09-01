"""Журнал решений: номера уникальны, ссылки из кода ведут в существующее.

За 22.08 номера дважды столкнулись между машинами и дважды перенумеровывались
(Р-67…Р-69 → Р-71…Р-73, Р-80 → Р-82). Столкновение само по себе не страшно —
машины работают параллельно, — но после перенумерации ссылка в коде может
остаться от старой нумерации и указывать в чужое решение. Это тихо и заметно
только при чтении.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "docs" / "decisions.md"
SOURCES = ("aios", "scripts", "tests")
SUFFIXES = (".py", ".sh", ".toml")

REFERENCE = re.compile(r"Р-\d+")
#: Решение объявляется двумя способами: `**Р-42.** …` (так писались первые сто с
#: лишним) и заголовком `## Р-118. …` (так пишутся с 30.08). Пока шаблон знал
#: только первый, девять последних решений были невидимы обеим проверкам ниже:
#: ссылка на них из кода считалась бы висячей, а столкновение номеров — не нашлось
#: бы вовсе. Найдено 01.09 ссылкой на Р-120 из `scripts/export_rows.py`.
DEFINITION = re.compile(r"^(?:\*\*|##\s+)(Р-\d+)[.\s]", re.MULTILINE)


def defined() -> list[str]:
    return DEFINITION.findall(DECISIONS.read_text(encoding="utf-8"))


def referenced() -> set[str]:
    found: set[str] = set()
    for name in SOURCES:
        for path in (ROOT / name).rglob("*"):
            if path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                found |= set(REFERENCE.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_decision_number_is_used_once():
    """Два решения под одним номером — это потерянное решение, а не опечатка."""
    numbers = defined()
    duplicates = sorted({one for one in numbers if numbers.count(one) > 1})

    assert not duplicates, f"номера заняты дважды: {duplicates}"


def test_every_reference_from_the_code_resolves():
    missing = sorted(referenced() - set(defined()), key=lambda one: int(one.split("-")[1]))

    assert not missing, (
        f"код ссылается на решения, которых нет в decisions.md: {missing} — "
        "скорее всего ссылка осталась от нумерации до перенумерации"
    )


def test_both_ways_of_declaring_a_decision_are_seen():
    """Решения, объявленные заголовком, тоже считаются объявленными."""
    numbers = set(defined())
    text = DECISIONS.read_text(encoding="utf-8")

    headed = set(re.findall(r"^##\s+(Р-\d+)[.\s]", text, re.MULTILINE))
    bolded = set(re.findall(r"^\*\*(Р-\d+)[.\s]", text, re.MULTILINE))

    assert headed, "в журнале не осталось решений-заголовков — проверка потеряла смысл"
    assert bolded, "в журнале не осталось решений в прежнем формате"
    assert headed <= numbers and bolded <= numbers


def test_the_journal_is_not_empty_so_the_check_means_something():
    """Пустой журнал прошёл бы обе проверки выше и ничего бы не доказал."""
    assert len(defined()) > 50
    assert referenced(), "код обязан ссылаться на решения, иначе журнал живёт отдельно"
