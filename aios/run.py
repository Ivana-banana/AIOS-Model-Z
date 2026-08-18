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
from aios.deck import (
    REPLACING_CONTROL_KEYWORDS,
    ScheduleStep,
    parse_date_record,
    parse_schedule,
    parse_schedule_text,
)
from aios.deckfiles import DeckLayout, layout
from aios.schedule import render as render_steps

_INCLUDE_RE = re.compile(r"(INCLUDE\s*\n\s*)'([^']+)'", re.IGNORECASE)
_DATES_LINE_RE = re.compile(r"^\s*DATES\s*$", re.IGNORECASE)
_RECORD_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z]{3})\s+(\d{4})\s*/")



@dataclass(frozen=True)
class Case:
    """Собранный кейс: что запускать и где искать результат."""

    data_file: Path
    schedule_file: Path
    summary_file: Path

    unsupported_controls: tuple[str, ...] = ()
    """Управляющие слова расписания, которых наш разбор не понимает.

    Пустой кортеж — режимы прочитаны полностью. Непустой означает, что в дек они
    попадут дословно и симулятор их применит, а вот наши маржа, DoE и подбор
    режимов их не видят: оценка приблизительна ровно на их величину.
    """

    wells_without_completions: tuple[str, ...] = ()
    """Скважины из `WELSPECS`, у которых нет ни одного вскрытия.

    Такая скважина получает режим, но не соединена с пластом: симулятор пишет
    `has no connections to grid and will remain SHUT` и считает дальше, а
    скважина молча даёт ноль весь горизонт. Так проект уже потерял 22 скважины и
    все сценарные прогоны 16.08 — поэтому список едет вместе с кейсом, а не
    остаётся в логе симулятора.
    """

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

    # имена файлов дека не зашиты: на кейсе организаторов они будут другими
    deck = layout(model_path)
    source_schedule = deck.one("SCHEDULE")
    schedule_text = source_schedule.read_text(encoding="utf-8", errors="replace")

    if steps is not None:
        if control_start is None:
            raise ValueError("Для своих шагов нужна дата начала управления (control_start).")
        schedule_text = (
            history_prefix(schedule_text, control_start)
            + "\n"
            + render_steps(steps, carried=carried_keywords(
                source_schedule.read_text(encoding="utf-8", errors="replace"), control_start
            ))
        )
        unsupported = _assert_assembled_schedule_is_sound(schedule_text, control_start)

    if wells is None:
        wells = parse_schedule(source_schedule).wells
    if not wells:
        # диагноз обязан называть причину: пустой фонд и неразобранный ход
        # времени выглядят одинаково, но искать надо совсем разное
        unparsed = parse_schedule_text(schedule_text).unsupported_controls
        raise ValueError(
            f"в {source_schedule.name} не нашлось ни одной работающей скважины. "
            + (f"Причина, вероятно, в том, что мы не разбираем {', '.join(unparsed)}: "
               "шаги расписания не прочитаны, и режимы вместе с ними"
               if unparsed else
               "WELSPECS пуст или ни одной скважине не назначен режим — считать нечего, "
               "и выгрузка вышла бы пустой уже после прогона")
        )

    # шаблон Eclipse (`'P*'`) — не имя скважины: разобранный как имя, он даёт
    # фонд из одной скважины-призрака, SUMMARY выписывается на неё, а настоящие
    # остаются без векторов, и выгрузка приходит пустой уже после полного прогона
    templates = tuple(well for well in wells if "*" in well or "?" in well)
    if templates:
        raise ValueError(
            f"в {source_schedule.name} режим задан по шаблону имени: "
            + ", ".join(templates)
            + ". Мы разбираем шаблоны как имена скважин, поэтому фонд получится "
            "неверным; такие записи надо раскрыть в явные имена"
        )

    uncompleted = wells_without_completions(schedule_text, wells)
    if steps is None:
        unsupported = parse_schedule_text(schedule_text).unsupported_controls

    schedule_file = run_path / "wells_schedule.inc"
    schedule_file.write_text(schedule_text, encoding="utf-8", newline="\n")

    summary_file = run_path / "summary.inc"
    summary_file.write_text(
        summary_module.render_summary_section(wells), encoding="utf-8", newline="\n"
    )

    data_file = run_path / f"{case_name}.DATA"
    data_file.write_text(
        _render_data(deck, schedule_file, summary_file), encoding="utf-8", newline="\n"
    )
    return Case(data_file=data_file, schedule_file=schedule_file, summary_file=summary_file,
                unsupported_controls=unsupported, wells_without_completions=uncompleted)


def _assert_assembled_schedule_is_sound(schedule_text: str,
                                        control_start: date) -> tuple[str, ...]:
    """Проверить склейку истории с нашими шагами, а не только сами шаги.

    `schedule.validate` следит за возрастанием дат, но видит объект `steps`.
    Дубль возникает уровнем выше — при склейке текста истории с отрисованными
    шагами, и туда валидатор не заглядывает. Самый простой способ это получить:
    `control_start` за концом дека. Тогда история отдаётся целиком, наши шаги
    дописываются следом и повторяют её последнюю дату, а для симулятора шаг, не
    превосходящий текущего времени, — ошибка.
    """
    previous: date | None = None
    for step in parse_schedule_text(schedule_text).steps:
        if previous is not None and step.date <= previous:
            raise ValueError(
                f"в собранном расписании дата не возрастает: {step.date} после {previous}. "
                f"Проверьте control_start ({control_start}): за концом дека история "
                "отдаётся целиком, и наши шаги дублируют её последний шаг"
            )
        previous = step.date

    unsupported = parse_schedule_text(schedule_text).unsupported_controls
    replacing = tuple(name for name in unsupported if name in REPLACING_CONTROL_KEYWORDS)
    if replacing:
        raise ValueError(
            "в SCHEDULE есть управление, заменяющее генерируемое нами: "
            + ", ".join(replacing)
            + ". Разобранные шаги остаются без режимов, а в файл попадёт и оно, и наш "
            "WCONPROD — поддержку надо дописать явно, а не надеяться, что «как-нибудь "
            "разберётся»"
        )
    return unsupported


def wells_without_completions(schedule_text: str,
                              wells: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Скважины, объявленные в `WELSPECS`, но ни разу не вскрытые.

    Вскрытием считается запись и в `COMPDAT`, и в `COMPDATMD`: первый формат
    приходит из OPM-конвертации, второй — из поставки тНавигатора, и на
    валидационном кейсе (ТЗ п. 4.3) заранее неизвестно, какой будет.
    """
    completed: set[str] = set()
    for match in re.finditer(r"^[ 	]*COMPDAT(?:MD)?[ 	]*$", schedule_text, re.MULTILINE):
        for line in schedule_text[match.end():].splitlines():
            stripped = line.split("--", 1)[0].strip()
            if not stripped:
                continue
            if stripped == "/" or not stripped.endswith("/"):
                break
            completed.add(stripped.split()[0].strip("'\""))
    return tuple(well for well in wells if well not in completed)


#: Ключевые слова, которые мы генерируем сами. Всё остальное внутри шага —
#: события модели (вскрытия, множители проводимости), и они обязаны пережить
#: пересборку расписания.
CONTROL_KEYWORDS = frozenset({"DATES", "WCONPROD", "WCONINJE"})

_KEYWORD_LINE_RE = re.compile(r"^([A-Z][A-Z0-9]{1,7})\s*$")


def carried_keywords(schedule_text: str, control_start: date) -> dict[date, str]:
    """Неуправляющие блоки SCHEDULE начиная с `control_start`, по датам.

    После даты начала управления в деке остаются не только режимы: там бурятся
    новые скважины, и их `COMPDAT` лежит ровно там же, между `DATES`. Наш
    генератор пишет только режимы, поэтому при пересборке расписания эти блоки
    надо перенести дословно — иначе скважина получит режим, но не получит ни
    одного соединения с пластом.

    Обнаружено 16.08: в Модели Z после 2007 года бурится 22 скважины, и без
    переноса они дают ноль за весь горизонт, а симулятор сообщает об этом лишь
    предупреждением (`docs/experiments.md`).
    """
    carried: dict[date, list[str]] = {}
    current: date | None = None
    lines = schedule_text.splitlines()
    index = 0

    while index < len(lines):
        match = _KEYWORD_LINE_RE.match(lines[index].split("--", 1)[0].strip())
        if match is None:
            index += 1
            continue

        name = match.group(1)
        block, index = _read_block(lines, index)

        if name == "DATES":
            for line in block[1:]:
                record = _RECORD_RE.match(line.split("--", 1)[0])
                if record:
                    current = parse_date_record(record.groups())
            continue

        if name in CONTROL_KEYWORDS or current is None or current < control_start:
            continue

        carried.setdefault(current, []).append("\n".join(block))

    return {step_date: "\n\n".join(blocks) for step_date, blocks in carried.items()}


def _read_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Ключевое слово со всеми его записями до терминатора `/`."""
    block = [lines[start]]
    index = start + 1
    while index < len(lines):
        stripped = lines[index].split("--", 1)[0].strip()
        block.append(lines[index])
        index += 1
        if stripped == "/":
            break
    return block, index


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


def _render_data(deck: DeckLayout, schedule_file: Path, summary_file: Path) -> str:
    """Корневой `.DATA` дека с подменёнными SUMMARY и SCHEDULE и абсолютными путями.

    Какой `INCLUDE` относится к SUMMARY, а какой к SCHEDULE, определяет секция, в
    которой он записан, а не его имя (`aios.deckfiles`).
    """
    text = deck.data_file.read_text(encoding="utf-8", errors="replace")
    model_path = deck.data_file.parent
    ours = {deck.one("SUMMARY"): summary_file, deck.one("SCHEDULE"): schedule_file}

    def replace(match: re.Match[str]) -> str:
        prefix, name = match.group(1), match.group(2)
        target = ours.get((model_path / name).resolve(), model_path / name)
        return f"{prefix}'{target.resolve().as_posix()}'"

    header = (
        "-- Расчётный кейс, собранный aios/run.py.\n"
        "-- Дек организаторов не изменён: включаемые файлы берутся из поставки,\n"
        "-- подменены только SUMMARY и SCHEDULE.\n\n"
    )
    return header + _INCLUDE_RE.sub(replace, text)
