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
from pathlib import Path

from aios.deck import InjectorControl, ProducerControl, ScheduleStep, format_date_record

#: Дебит жидкости добывающей скважины выше этого значения методикой не допускается.
MAX_LIQUID_RATE_M3_PER_DAY = 500.0

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


def validate(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
             known_wells: frozenset[str] | None = None) -> list[ScheduleProblem]:
    """Собрать все нарушения. Пустой список — расписание пригодно к расчёту."""
    problems: list[ScheduleProblem] = []
    previous_date: date | None = None

    for step in steps:
        if previous_date is not None and step.date <= previous_date:
            problems.append(
                ScheduleProblem(step.date, "-", f"дата не возрастает (предыдущая {previous_date})")
            )
        previous_date = step.date

        seen: dict[str, str] = {}
        for control in step.producers:
            problems.extend(_check_producer(step.date, control, known_wells))
            _note_role(problems, seen, step.date, control.well, "добыча")
        for control in step.injectors:
            problems.extend(_check_injector(step.date, control, known_wells))
            _note_role(problems, seen, step.date, control.well, "закачка")

    return problems


def assert_valid(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                 known_wells: frozenset[str] | None = None) -> None:
    """Упасть, если расписание нарушает ограничения."""
    problems = validate(steps, known_wells)
    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems[:50])
        more = "" if len(problems) <= 50 else f"\n  ... ещё {len(problems) - 50}"
        raise ScheduleError(f"Расписание некорректно ({len(problems)} нарушений):\n{listed}{more}")


def render(steps: list[ScheduleStep] | tuple[ScheduleStep, ...], header: str = HEADER) -> str:
    """Собрать текст `wells_schedule.inc`."""
    lines: list[str] = [header.rstrip("\n"), ""]
    for step in steps:
        lines.append("DATES")
        lines.append(f" {format_date_record(step.date)} /")
        lines.append("/")
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


def write(path: str | Path,
          steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
          known_wells: frozenset[str] | None = None,
          header: str = HEADER) -> Path:
    """Проверить и записать `wells_schedule.inc`. Невалидное расписание не пишется."""
    assert_valid(steps, known_wells)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(steps, header), encoding="utf-8", newline="\n")
    return target


def _producer_record(control: ProducerControl) -> str:
    rate = _value(control.liquid_rate)
    bhp = _value(control.bhp)
    return (
        f" '{control.well}' '{control.status}' '{control.mode}'"
        f" 1* 1* 1* {rate} 1* {bhp} 1* 1* /"
    )


def _injector_record(control: InjectorControl) -> str:
    rate = _value(control.rate)
    bhp = _value(control.bhp)
    return (
        f" '{control.well}' '{control.fluid}' '{control.status}' '{control.mode}'"
        f" {rate} 1* {bhp} 1* 1* /"
    )


def _value(number: float | None) -> str:
    return "1*" if number is None else f"{number:g}"


def _check_producer(step_date: date, control: ProducerControl,
                    known_wells: frozenset[str] | None) -> list[ScheduleProblem]:
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
        elif rate > MAX_LIQUID_RATE_M3_PER_DAY:
            add(f"дебит жидкости {rate:g} > {MAX_LIQUID_RATE_M3_PER_DAY:g} м3/сут — запрещено методикой")
    elif rate is not None and rate < 0:
        add(f"отрицательный дебит жидкости {rate:g}")

    if control.bhp is not None and control.bhp <= 0:
        add(f"забойное давление {control.bhp:g} должно быть положительным")
    return problems


def _check_injector(step_date: date, control: InjectorControl,
                    known_wells: frozenset[str] | None) -> list[ScheduleProblem]:
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

    if control.bhp is not None and control.bhp <= 0:
        add(f"давление нагнетания {control.bhp:g} должно быть положительным")
    return problems


def _note_role(problems: list[ScheduleProblem], seen: dict[str, str],
               step_date: date, well: str, role: str) -> None:
    previous = seen.get(well)
    if previous is None:
        seen[well] = role
    elif previous != role:
        problems.append(
            ScheduleProblem(step_date, well, "на одном шаге назначены и добыча, и закачка")
        )
    else:
        problems.append(ScheduleProblem(step_date, well, f"дублируется на шаге ({role})"))
