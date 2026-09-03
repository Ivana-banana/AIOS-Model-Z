"""Генерация и валидация `wells_schedule.inc`.

Это единственный артефакт, который уходит организаторам и считается на их ГГДМ.
Файл обязан читаться моделью **без ручного редактирования**, а расчёт по нему —
ровно один. Поэтому валидатор здесь строгий и падает, а не предупреждает.

Семантика Eclipse: режим, назначенный скважине, действует до следующего явного
изменения. Не перечислить скважину на шаге — значит оставить ей прежний режим, а не
остановить её.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from aios.deck import InjectorControl, ProducerControl, ScheduleStep, format_date_record

#: Дебит жидкости добывающей скважины выше этого значения методикой не допускается.
#: Число не наше и стоит в двух местах их поставки сразу: верхняя граница старшего
#: ЭЦН в `Нормативы_ЧДД.xlsx` и **жёсткий литерал** в эталонном расчётчике
#: (`vendor/CHDD_PYTHON/chdd_model.py`, `if row["WLPR"] > 500` → `ValueError`).
#: Второе делает его потолком при любой таблице: расчёт с превышением не «считает
#: иначе», а падает целиком. На кейсе таблица приходит от организаторов и предел
#: берётся из неё — но не выше этого потолка (`liquid_limit()`).
MAX_LIQUID_RATE_M3_PER_DAY = 500.0

#: Предельное давление нагнетания, атм. Ровно это значение стоит во всей истории
#: дека, и организаторы на разборе объяснили почему: выше него в жизни раскрываются
#: трещины (в некоторых коллекторах уже со 180–190 атм), то есть закачка идёт мимо
#: пласта. Модель такой режим воспроизведёт «отлично» и покажет красивую добычу —
#: но это физически нереализуемое решение, а их проверяют отдельным критерием.
MAX_INJECTION_BHP_ATM = 300.0

PRODUCER_STATUSES = frozenset({"OPEN", "SHUT", "STOP"})
INJECTOR_STATUSES = frozenset({"OPEN", "SHUT", "STOP"})
PRODUCER_MODES = frozenset({"LRAT", "ORAT", "WRAT", "GRAT", "RESV", "BHP", "THP", "GRUP"})
INJECTOR_MODES = frozenset({"RATE", "RESV", "BHP", "THP", "GRUP"})
INJECTED_FLUIDS = frozenset({"WATER", "GAS", "OIL"})

HEADER = """\
-- wells_schedule.inc
-- Управляющие последовательности, сформированные МАС AIOS (трек 2, Модель Z).
-- Сгенерировано автоматически, вручную не править.
"""


class ScheduleError(ValueError):
    """Расписание нарушает ограничения и не должно уходить организаторам."""


@dataclass(frozen=True)
class ScheduleProblem:
    """Одно нарушение с привязкой к шагу и скважине."""

    step_date: date
    well: str
    message: str

    def __str__(self) -> str:
        return f"{self.step_date.isoformat()} скв. {self.well}: {self.message}"


def liquid_limit(pumps: list[dict[str, Any]]) -> float:
    """Предел дебита жидкости по таблице ЭЦН организаторов, м³/сут.

    Верхний насос закрывает верхний интервал, и его правая граница и есть тот
    предел, который методика запрещает превышать. Своего числа у нас здесь нет:
    на валидационном кейсе (ТЗ п. 4.3) таблица придёт от организаторов, и предел
    обязан приехать вместе с ней, а не остаться литералом 500 от Модели Z.

    **Но выше 500 предел не поднимается ни при какой таблице.** У эталонного
    расчётчика 500 зашито литералом и не читается из нормативов:
    `vendor/CHDD_PYTHON/chdd_model.py`, «`if row["WLPR"] > 500`» — и это не
    предупреждение, а `ValueError`, роняющий весь расчёт. Таблица кейса с верхним
    насосом на 700 означала бы, что наш валидатор пропускает режим, на котором
    расчёт организаторов **падает целиком**, а не «считает чуть иначе». Поэтому
    берётся меньшее из двух: их таблица и их же потолок. То, что эти два числа у
    организаторов могут разойтись, — их противоречие, и оно закреплено тестом,
    который читает литерал прямо из вендоренного файла.
    """
    if not pumps:
        raise ValueError("таблица ЭЦН пуста: предел дебита жидкости брать не из чего")
    return min(max(float(pump["max"]) for pump in pumps), MAX_LIQUID_RATE_M3_PER_DAY)


def validate(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
             known_wells: frozenset[str] | None = None,
             *,
             max_liquid_rate: float = MAX_LIQUID_RATE_M3_PER_DAY,
             max_injection_bhp: float = MAX_INJECTION_BHP_ATM) -> list[ScheduleProblem]:
    """Собрать все нарушения. Пустой список — расписание пригодно к расчёту.

    `max_injection_bhp` — предел давления нагнетания. 300 атм здесь умолчание
    Модели Z, а не закон: `[limits].max_injector_bhp_atm` читается конфигурацией и
    исполняется планировщиком, но до 03.09 не доходил сюда — предполётная проверка
    судила чужой кейс по числу нашего объекта. Кейс с пределом 250 атм проходил бы
    её с режимом 290, кейс с 350 — получал бы отказ на законном режиме."""
    problems: list[ScheduleProblem] = []
    previous_date: date | None = None

    for step in steps:
        if previous_date is not None and step.date <= previous_date:
            problems.append(
                ScheduleProblem(step.date, "-", f"дата не возрастает (предыдущая {previous_date})")
            )
        previous_date = step.date

        records: dict[tuple[str, str], int] = {}
        working: dict[str, set[str]] = {}

        for control in step.producers:
            problems.extend(_check_producer(step.date, control, known_wells,
                                            max_liquid_rate))
            _note_role(problems, records, working, step.date, control.well, "добыча",
                       control.status == "OPEN")
        for control in step.injectors:
            problems.extend(_check_injector(step.date, control, known_wells,
                                            max_injection_bhp))
            _note_role(problems, records, working, step.date, control.well, "закачка",
                       control.status == "OPEN")

        for well, roles in working.items():
            if len(roles) > 1:
                problems.append(
                    ScheduleProblem(step.date, well, "одновременно открыта на добычу и на закачку")
                )

    return problems


def assert_valid(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                 known_wells: frozenset[str] | None = None,
                 *,
                 max_liquid_rate: float = MAX_LIQUID_RATE_M3_PER_DAY,
                 max_injection_bhp: float = MAX_INJECTION_BHP_ATM) -> None:
    """Упасть, если расписание нарушает ограничения."""
    problems = validate(steps, known_wells, max_liquid_rate=max_liquid_rate,
                        max_injection_bhp=max_injection_bhp)
    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems[:50])
        more = "" if len(problems) <= 50 else f"\n  ... ещё {len(problems) - 50}"
        raise ScheduleError(f"Расписание некорректно ({len(problems)} нарушений):\n{listed}{more}")


def render(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
           header: str = HEADER,
           carried: dict[date, str] | None = None) -> str:
    """Собрать текст `wells_schedule.inc`.

    `carried` — блоки исходного дека, которые не являются режимами и потому не
    разбираются в `ScheduleStep`: вскрытия `COMPDAT`, `WPIMULT` и подобное. Они
    переносятся дословно на свою дату.

    Забыть их — значит получить скважину, объявленную в `WELSPECS` и получающую
    режим, но не имеющую ни одного соединения с сеткой. Симулятор на этом не
    падает: он пишет `has no connections to grid and will remain SHUT` и считает
    дальше, а скважина молча даёт ноль весь горизонт.
    """
    carried = carried or {}
    lines: list[str] = [header.rstrip("\n"), ""]
    for step in steps:
        lines.append("DATES")
        lines.append(f" {format_date_record(step.date)} /")
        lines.append("/")
        lines.append("")
        extra = carried.get(step.date)
        if extra:
            lines.append(extra.rstrip("\n"))
            lines.append("")
        if step.producers:
            lines.append("WCONPROD")
            lines.extend(_producer_record(control) for control in step.producers)
            lines.append("/")
            lines.append("")
        if step.injectors:
            lines.append("WCONINJE")
            lines.extend(_injector_record(control) for control in step.injectors)
            lines.append("/")
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _producer_record(control: ProducerControl) -> str:
    rate = _rate_value(control.liquid_rate, control.status)
    bhp = _value(control.bhp)
    return (
        f" '{control.well}' '{control.status}' '{control.mode}'"
        f" 1* 1* 1* {rate} 1* {bhp} 1* 1* /"
    )


def _injector_record(control: InjectorControl) -> str:
    rate = _rate_value(control.rate, control.status)
    bhp = _value(control.bhp)
    return (
        f" '{control.well}' '{control.fluid}' '{control.status}' '{control.mode}'"
        f" {rate} 1* {bhp} 1* 1* /"
    )


def _value(number: float | None) -> str:
    return "1*" if number is None else f"{number:g}"


def _rate_value(number: float | None, status: str) -> str:
    """Дебит (или закачка) в записи режима.

    У остановленной скважины значения обычно нет, но пропуск `1*` здесь — не
    «значение по умолчанию», а отсутствие уставки: OPM Flow отвергает такую
    запись целиком («Trying to set CMODE to: LRAT - no value has been specified
    for this control») и валит прогон на разборе входа. Дек организаторов пишет
    остановку с явным нулём, и мы пишем так же.
    """
    if number is None and status != "OPEN":
        return "0.0"
    return _value(number)


def _check_producer(step_date: date, control: ProducerControl,
                    known_wells: frozenset[str] | None,
                    max_liquid_rate: float = MAX_LIQUID_RATE_M3_PER_DAY,
                    ) -> list[ScheduleProblem]:
    problems: list[ScheduleProblem] = []

    def add(message: str) -> None:
        problems.append(ScheduleProblem(step_date, control.well, message))

    if known_wells is not None and control.well not in known_wells:
        add("скважины нет в деке")
    if control.status not in PRODUCER_STATUSES:
        add(f"неизвестный статус {control.status!r}")
    if control.mode not in PRODUCER_MODES:
        add(f"неизвестный режим {control.mode!r}")

    rate = control.liquid_rate
    if control.status == "OPEN" and control.mode == "LRAT":
        if rate is None:
            add("режим LRAT без дебита жидкости")
        elif rate <= 0:
            add(f"дебит жидкости {rate:g} м3/сут: открытая скважина не может иметь нулевой дебит")
        elif rate > max_liquid_rate:
            add(f"дебит жидкости {rate:g} > {max_liquid_rate:g} м3/сут — запрещено методикой")
    elif rate is not None and rate < 0:
        add(f"отрицательный дебит жидкости {rate:g}")

    if control.bhp is not None and control.bhp <= 0:
        add(f"забойное давление {control.bhp:g} должно быть положительным")
    return problems


def _check_injector(step_date: date, control: InjectorControl,
                    known_wells: frozenset[str] | None,
                    max_injection_bhp: float = MAX_INJECTION_BHP_ATM) -> list[ScheduleProblem]:
    problems: list[ScheduleProblem] = []

    def add(message: str) -> None:
        problems.append(ScheduleProblem(step_date, control.well, message))

    if known_wells is not None and control.well not in known_wells:
        add("скважины нет в деке")
    if control.status not in INJECTOR_STATUSES:
        add(f"неизвестный статус {control.status!r}")
    if control.mode not in INJECTOR_MODES:
        add(f"неизвестный режим {control.mode!r}")
    if control.fluid not in INJECTED_FLUIDS:
        add(f"неизвестный агент закачки {control.fluid!r}")

    rate = control.rate
    if control.status == "OPEN" and control.mode == "RATE":
        if rate is None:
            add("режим RATE без приёмистости")
        elif rate <= 0:
            add(f"приёмистость {rate:g} м3/сут: открытая скважина не может закачивать ноль")
    elif rate is not None and rate < 0:
        add(f"отрицательная приёмистость {rate:g}")

    if control.bhp is not None:
        if control.bhp <= 0:
            add(f"давление нагнетания {control.bhp:g} должно быть положительным")
        elif control.bhp > max_injection_bhp:
            add(
                f"давление нагнетания {control.bhp:g} > {max_injection_bhp:g} атм — "
                f"физически нереализуемо, раскрытие трещин"
            )
    return problems


def _note_role(problems: list[ScheduleProblem],
               records: dict[tuple[str, str], int],
               working: dict[str, set[str]],
               step_date: date, well: str, role: str, is_open: bool) -> None:
    """Учесть запись режима на шаге.

    Скважина может законно иметь на одном шаге запись и в `WCONPROD`, и в
    `WCONINJE` — именно так в Eclipse записывается перевод под закачку: старый
    режим закрывается `SHUT`, новый открывается. Ровно эту пару делает и дек
    организаторов (например, скважина 102 на 01.01.2007). Нарушение — не сам
    факт двух записей, а две **открытые**: добывать и закачивать одновременно
    скважина не может.
    """
    key = (well, role)
    records[key] = records.get(key, 0) + 1
    if records[key] > 1:
        problems.append(ScheduleProblem(step_date, well, f"дублируется на шаге ({role})"))
    if is_open:
        working.setdefault(well, set()).add(role)
