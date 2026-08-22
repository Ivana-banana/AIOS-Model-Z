"""Переносимость входного контроля: деки, которые он **обязан** отвергнуть.

`tests/test_deckcheck.py` доказывает вторую половину утверждения — что рабочий
кейс (Модель Z) проходит. Первую половину — что чужой дек не проходит **молча** —
на самой Модели Z доказать нечем: на ней предположения генератора выполняются, и
контроль на ней всегда возвращает пустой список. Контроль, который ничего не
отвергает, неотличим от отсутствующего; на этом проект уже обжигался с детектором
обучающего диапазона (`docs/report.md`, п. 6 незакрытого).

Поэтому здесь два слоя:

* **синтетика в `tmp_path`** — маленькие деки, каждый воспроизводит одну ловушку
  каталога. Гоняются всегда, в том числе на машине без сети;
* **настоящие открытые деки** (`vendor/portability/`, ODbL/DBCL) — под
  `skipif`. Они подтверждают, что синтетика не выдумана: та же ловушка живёт в
  чужом деке, который считают другие люди.

Как забрать деки и что именно на них получилось — `docs/portability.md`.

Чего эти тесты не доказывают: что мы **посчитаем** чужой дек. Только то, что не
посчитаем его молча неправильно.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios import deckcheck

DECKS = Path(__file__).resolve().parent.parent / "vendor" / "portability" / "opm-data"
SPE9 = DECKS / "spe9" / "SPE9.DATA"
SPE9_GROUP = DECKS / "spe9" / "SPE9_CP_GROUP.DATA"
NORNE = DECKS / "norne" / "NORNE_ATW2013.DATA"

#: Три ложных срабатывания, найденных задачей Б20 (`INCLUDE` без кавычек,
#: хвостовой пробел в имени скважины, `PVTNUM` внутри `EQUALS`), исправлены
#: машиной А 22.08 (Р-83, коммит `0074616`). Тесты ниже проверяли, что до
#: ожидаемых ловушек дело не доходит; теперь они проверяют, что доходит.
#: `xfail(strict=True)` сработал как задумано — прогон покраснел в день правки.


def codes(found) -> list[str]:
    return sorted({refusal.code for refusal in found})


def deck(tmp_path: Path, text: str, name: str = "CASE.DATA") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return path


# ── синтетика: одна ловушка — один дек ───────────────────────────────────────

FIELD_TSTEP = """RUNSPEC
FIELD

START
  1 'JAN' 2007 /

GRID
SCHEDULE
TSTEP
  31 28 31 /
"""

SUBDAY_DATES = """RUNSPEC
METRIC

START
  1 'JAN' 2007 /

GRID
SCHEDULE
DATES
  14 'NOV' 2007 00:00:00.001 /
/
"""

GROUP_CONTROL = """RUNSPEC
METRIC

START
  1 'JAN' 2007 /

GRID
SCHEDULE
GRUPTREE
 'PLAT-A' 'FIELD' /
/

GCONPROD
 'PLAT-A' 'ORAT' 20000 4* 'RATE' /
/
"""

HISTORY_CONTROL = """RUNSPEC
METRIC

START
  1 'JAN' 2007 /

GRID
SCHEDULE
WELSPECS
 'B-1H' 'PLAT-A' 10 10 1* 'OIL' /
/

WCONHIST
 'B-1H' 'OPEN' 'ORAT' 1000 200 50 /
/
"""

WELL_PATTERN = """RUNSPEC
METRIC

START
  1 'JAN' 2007 /

GRID
SCHEDULE
WELSPECS
 'PRODU1' 'G1' 10 10 1* 'OIL' /
/

WCONPROD
 'PRODU*' 'OPEN' 'LRAT' 3* 1000 /
/
"""


@pytest.mark.parametrize("body, expected, trap", [
    (FIELD_TSTEP, "ECL-005", "промысловые единицы FIELD"),
    (FIELD_TSTEP, "ECL-007", "время идёт шагами TSTEP, а не датами"),
    (SUBDAY_DATES, "ECL-007", "дата внутри суток"),
    (GROUP_CONTROL, "ECL-011", "групповое управление"),
    (HISTORY_CONTROL, "ECL-010", "исторический режим WCONHIST"),
    (WELL_PATTERN, "ECL-014", "шаблон имени скважины"),
])
def test_a_synthetic_trap_is_refused(tmp_path, body: str, expected: str, trap: str):
    assert expected in codes(deckcheck.check(deck(tmp_path, body))), f"пропущена ловушка: {trap}"


def test_the_control_is_not_a_rubber_stamp(tmp_path):
    """Контрольный отрицательный: METRIC-дек с месячными датами проходит.

    Без него набор выше доказывал бы только то, что контроль ругается на всё
    подряд — а такой контроль тоже бесполезен, просто иначе.
    """
    clean = """RUNSPEC
METRIC

START
  1 'JAN' 2007 /

GRID
SCHEDULE
WELSPECS
 'W1' 'G1' 10 10 1* 'OIL' /
/

DATES
  1 'FEB' 2007 /
/
"""
    assert deckcheck.check(deck(tmp_path, clean)) == []


# ── настоящие открытые деки: подтверждают, что синтетика не выдумана ─────────

@pytest.mark.skipif(not SPE9.exists(), reason="открытые деки не скачаны (docs/portability.md)")
@pytest.mark.parametrize("path", [SPE9, SPE9_GROUP, NORNE])
def test_a_foreign_deck_is_refused_at_all(path: Path):
    """Главное утверждение задачи: ни один чужой дек не проходит молча.

    Проверяется именно это, а не конкретный код: каким кодом контроль остановится
    первым, зависит от порядка проверок, а вот «остановился» — не зависит.
    """
    assert deckcheck.check(path), f"чужой дек прошёл контроль без единого отказа: {path.name}"


@pytest.mark.skipif(not NORNE.exists(), reason="открытые деки не скачаны (docs/portability.md)")
def test_norne_history_and_group_control_are_seen():
    """Norne: METRIC, но реальная история — `WCONHIST`, `WELOPEN`, дерево групп.

    Это ближайший к Модели Z дек по устройству, и именно поэтому он опасен: по
    единицам и виду он «наш», а управление у него не сводится к поскважинным
    уставкам вовсе.
    """
    found = codes(deckcheck.check(NORNE))

    assert "ECL-010" in found, "историческое управление не замечено"
    assert "ECL-011" in found, "групповое управление не замечено"
    assert "ECL-007" in found, "внутрисуточная дата не замечена"
    # `PVTNUM` у Norne задан внутри `EQUALS`, и до правки Р-83 массив «не
    # находился»: два PVT-региона объявлены, отнести скважину не по чему
    assert "ECL-006" not in found, "PVTNUM внутри EQUALS снова не виден"


@pytest.mark.skipif(not SPE9.exists(), reason="открытые деки не скачаны (docs/portability.md)")
def test_spe9_units_and_timeline_traps_are_reached():
    """SPE9: `FIELD` + `TSTEP` — ровно то, чего наш генератор не представляет.

    До 22.08 тест стоял здесь падающим (`xfail(strict=True)`): на настоящем SPE9
    до этих проверок дело не доходило, потому что раньше срабатывал ложный
    `ECL-002` на `INCLUDE` без кавычек, а `check()` выходит на первом отказе
    включений. Тогда обе ловушки были подтверждены только на копии дека с
    проставленными кавычками.

    Будильник сработал: машина А поправила `ECL-002` (Р-83), прогон покраснел на
    `XPASS(strict)` в тот же день, и проверка переписана на то, что теперь
    происходит на самом деле. Замер 22.08 после правки: 5 отказов —
    `ECL-005`, `ECL-007` и трижды `ECL-014` на шаблоне `PRODU*`.
    """
    found = codes(deckcheck.check(SPE9))

    assert "ECL-005" in found, "промысловые единицы FIELD не замечены"
    assert "ECL-007" in found, "время через TSTEP не замечено"
    assert "ECL-002" not in found, "ложное срабатывание на INCLUDE без кавычек вернулось"


@pytest.mark.skipif(not SPE9_GROUP.exists(), reason="открытые деки не скачаны (docs/portability.md)")
def test_spe9_group_control_trap_is_reached():
    """SPE9group: `GRUPTREE`/`GCONPROD` — та же история, что и у SPE9.

    Здесь ушли сразу два ложных срабатывания: `ECL-002` и десять `ECL-009` на
    именах вида `'INJE1 '` с хвостовым пробелом. Отказов было 17 на копии с
    кавычками, стало 7 — и все семь по делу.
    """
    found = codes(deckcheck.check(SPE9_GROUP))

    assert "ECL-011" in found, "групповое управление не замечено"
    assert "ECL-009" not in found, "хвостовой пробел в имени скважины снова считается ошибкой"
