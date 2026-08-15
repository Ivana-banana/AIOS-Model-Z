"""Сборка расчётного кейса: наш SCHEDULE и наш SUMMARY поверх дека Модели Z.

Дек организаторов не редактируем. Собираем рядом отдельный `.DATA`, который
ссылается на их включаемые файлы по абсолютным путям, а SUMMARY и SCHEDULE
подменяет нашими. Так любой прогон воспроизводится из чистой поставки, а сетку
на 74 МБ никто не копирует.

Запуск симулятора — не здесь: он идёт на Linux (`scripts/run_flow.sh`,
`docs/setup-simulator.md`). Здесь только подготовка входа и разбор пути к выходу.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aios import summary as summary_module
from aios.deck import ScheduleStep, parse_date_record, parse_schedule
from aios.schedule import render as render_steps

_INCLUDE_RE = re.compile(r"(INCLUDE\s*\n\s*)'([^']+)'", re.IGNORECASE)
_DATES_LINE_RE = re.compile(r"^\s*DATES\s*$", re.IGNORECASE)
_RECORD_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{3})\s+(\d{4})\s*/")

SUMMARY_INCLUDE = "Model_Z_summary.inc"
SCHEDULE_INCLUDE = "Model_Z_sch.inc"


@dataclass(frozen=True)
class Case:
    """Собранный кейс: что запускать и где искать результат."""

    data_file: Path
    schedule_file: Path
    summary_file: Path

    @property
    def summary_case(self) -> Path:
        """Путь к результату без расширения — его ждёт `aios.summary.monthly_rows`."""
        return self.data_file.with_suffix("")


def build_case(run_dir: str | Path,
               model_dir: str | Path,
               steps: list[ScheduleStep] | tuple[ScheduleStep, ...] | None = None,
               control_start: date | None = None,
               wells: tuple[str, ...] | list[str] | None = None,
               case_name: str = "RUN") -> Case:
    """Собрать расчётный кейс в `run_dir`.

    `steps` — наши управляющие шаги. Если заданы, история дека обрезается по
    `control_start` и дальше идут наши шаги; без `steps` собирается базовый
    сценарий — история как есть.

    История обрезается по тексту, а не пересобирается из разобранных структур:
    внутри шагов лежат `COMPDATMD` и `WELLTRACK`, задающие вскрытие пласта во
    времени. Потерять их — значит получить скважины без соединений и нулевую
    добычу, причём симулятор об этом не скажет.
    """
    run_path = Path(run_dir)
    model_path = Path(model_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    source_schedule = model_path / SCHEDULE_INCLUDE
    schedule_text = source_schedule.read_text(encoding="utf-8", errors="replace")

    if steps is not None:
        if control_start is None:
            raise ValueError("Для своих шагов нужна дата начала управления (control_start).")
        schedule_text = history_prefix(schedule_text, control_start) + "\n" + render_steps(steps)

    if wells is None:
        wells = parse_schedule(source_schedule).wells

    schedule_file = run_path / "wells_schedule.inc"
    schedule_file.write_text(schedule_text, encoding="utf-8", newline="\n")

    summary_file = run_path / "summary.inc"
    summary_file.write_text(
        summary_module.render_summary_section(wells), encoding="utf-8", newline="\n"
    )

    data_file = run_path / f"{case_name}.DATA"
    data_file.write_text(
        _render_data(model_path, schedule_file, summary_file), encoding="utf-8", newline="\n"
    )
    return Case(data_file=data_file, schedule_file=schedule_file, summary_file=summary_file)


def history_prefix(schedule_text: str, cut_date: date) -> str:
    """Часть SCHEDULE до первого шага с датой `cut_date` или позже.

    Всё, что раньше, сохраняется дословно — включая вскрытия и траектории.
    """
    lines = schedule_text.splitlines()
    for index, line in enumerate(lines):
        if not _DATES_LINE_RE.match(line):
            continue
        step_date = _first_date_after(lines, index)
        if step_date is not None and step_date >= cut_date:
            return "\n".join(lines[:index]).rstrip() + "\n"
    return schedule_text


def _first_date_after(lines: list[str], index: int) -> date | None:
    for line in lines[index + 1: index + 6]:
        stripped = line.split("--", 1)[0].strip()
        if not stripped or stripped == "/":
            continue
        match = _RECORD_RE.match(line.split("--", 1)[0])
        if match:
            return parse_date_record(match.groups())
        return None
    return None


def _render_data(model_path: Path, schedule_file: Path, summary_file: Path) -> str:
    """`Model_Z.data` с подменёнными SUMMARY и SCHEDULE и абсолютными путями."""
    text = (model_path / "Model_Z.data").read_text(encoding="utf-8", errors="replace")

    def replace(match: re.Match[str]) -> str:
        prefix, name = match.group(1), match.group(2)
        if name == SUMMARY_INCLUDE:
            target = summary_file
        elif name == SCHEDULE_INCLUDE:
            target = schedule_file
        else:
            target = model_path / name
        return f"{prefix}'{target.resolve().as_posix()}'"

    header = (
        "-- Расчётный кейс, собранный aios/run.py.\n"
        "-- Дек организаторов не изменён: включаемые файлы берутся из поставки,\n"
        "-- подменены только SUMMARY и SCHEDULE.\n\n"
    )
    return header + _INCLUDE_RE.sub(replace, text)
