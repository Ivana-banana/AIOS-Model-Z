"""Чтение SCHEDULE-секции дека Модели Z (Eclipse-совместимый формат тНавигатора).

Нужно для двух вещей: узнать фактический фонд и историю управления (базовый
сценарий) и понять формат, в котором организаторы ждут `wells_schedule.inc`.
Запись расписания — в `aios/schedule.py`.

Формат: ключевое слово на своей строке, дальше записи, каждая заканчивается `/`,
блок закрывается одиночным `/`. Комментарий — `--` до конца строки. `1*` — одно
значение по умолчанию, `3*` — три подряд, `3*50` — значение 50 три раза.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Позиции полей внутри записи (после раскрытия повторов), нумерация с нуля.
_WCONPROD_WELL = 0
_WCONPROD_STATUS = 1
_WCONPROD_MODE = 2
_WCONPROD_LRAT = 6
_WCONPROD_BHP = 8

_WCONINJE_WELL = 0
_WCONINJE_FLUID = 1
_WCONINJE_STATUS = 2
_WCONINJE_MODE = 3
_WCONINJE_RATE = 4
_WCONINJE_BHP = 6

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MONTH_NAMES = {number: name for name, number in _MONTHS.items()}

_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|/|\S+")
_REPEAT_RE = re.compile(r"^(\d+)\*(.*)$")
# Eclipse ограничивает ключевые слова восемью символами, тНавигатор выпускает и длиннее
# (WELLTRACK, COMPDATMD — по девять), поэтому берём с запасом.
_KEYWORD_RE = re.compile(r"[A-Z][A-Z0-9]{1,11}")


@dataclass(frozen=True)
class Keyword:
    """Ключевое слово с его записями. Значение по умолчанию (`1*`) — None."""

    name: str
    records: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class ProducerControl:
    """Запись WCONPROD: режим добывающей скважины."""

    well: str
    status: str
    mode: str
    liquid_rate: float | None
    bhp: float | None


@dataclass(frozen=True)
class InjectorControl:
    """Запись WCONINJE: режим нагнетательной скважины."""

    well: str
    fluid: str
    status: str
    mode: str
    rate: float | None
    bhp: float | None


@dataclass(frozen=True)
class ScheduleStep:
    """Один шаг управления: дата и режимы, назначенные с этой даты."""

    date: date
    producers: tuple[ProducerControl, ...]
    injectors: tuple[InjectorControl, ...]


@dataclass(frozen=True)
class Schedule:
    """Разобранная SCHEDULE-секция."""

    preamble: tuple[Keyword, ...]
    """Всё до первой DATES: RPTSCHED, WELLTRACK, WELSPECS, COMPDATMD и стартовые режимы."""

    steps: tuple[ScheduleStep, ...]

    unsupported_controls: tuple[str, ...] = ()
    """Ключевые слова внутри шагов, которые задают режим, но нами не разбираются.

    Пустой кортеж — весь режим дека прочитан. Непустой означает, что часть
    управления мы не увидели: разобранные шаги неполны, и строить на них базовый
    сценарий, DoE или маржу нельзя. Молчать здесь опаснее всего, потому что
    отсутствие добычи выглядит как «скважины стояли», а не как «мы не прочли».
    """

    @property
    def producer_wells(self) -> tuple[str, ...]:
        return _sorted_wells({c.well for s in self.steps for c in s.producers})

    @property
    def injector_wells(self) -> tuple[str, ...]:
        return _sorted_wells({c.well for s in self.steps for c in s.injectors})

    @property
    def wells(self) -> tuple[str, ...]:
        return _sorted_wells(set(self.producer_wells) | set(self.injector_wells))


def parse_keywords(text: str) -> list[Keyword]:
    """Разобрать текст в последовательность ключевых слов с записями."""
    tokens = _tokenize(text)
    keywords: list[Keyword] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not _is_keyword(token):
            continue  # мусор вне блока: пропускаем, а не угадываем
        name = token
        records: list[tuple[str | None, ...]] = []
        current: list[str] = []
        while index < len(tokens):
            token = tokens[index]
            if not current and _is_keyword(token):
                break  # блок без закрывающего '/' закрывается следующим ключевым словом
            index += 1
            if token == "/":
                if not current:
                    break  # одиночный '/' закрывает блок
                records.append(_expand_defaults(current))
                current = []
            else:
                current.append(token)
        if current:
            records.append(_expand_defaults(current))
        keywords.append(Keyword(name=name, records=tuple(records)))
    return keywords


#: Ключевые слова, которые задают режим, но нашим разбором не поддержаны.
#:
#: Р-03 разрешает пропускать всё лишнее, и для `COMPDATMD` или `WPIMULT` это
#: верно — они не режимы. Но эти задают именно управление, и молчаливый пропуск
#: даёт шаги без добычи вовсе: базовый сценарий, DoE и маржа посчитаются от
#: пустоты. В Модели Z их нет ни одного, а `WCONHIST` в исторических деках
#: встречается сплошь — то есть риск ровно на валидационном кейсе (ТЗ п. 4.3).
UNSUPPORTED_CONTROL_KEYWORDS = frozenset({
    "WCONHIST",   # исторический режим добычи вместо WCONPROD
    "WCONINJH",   # исторический режим закачки
    "WELTARG",    # смена одной уставки у действующего режима
    "WEFAC",      # коэффициент эксплуатации — множитель к дебиту
    "WELOPEN",    # открытие и закрытие скважины отдельным ключевым словом
    "GCONPROD",   # групповое ограничение добычи
    "GCONINJE",   # групповое ограничение закачки
})


def parse_schedule_text(text: str) -> Schedule:
    """Разобрать SCHEDULE-секцию из строки."""
    keywords = parse_keywords(text)

    preamble: list[Keyword] = []
    steps: list[ScheduleStep] = []
    unsupported: dict[str, None] = {}
    current_date: date | None = None
    producers: list[ProducerControl] = []
    injectors: list[InjectorControl] = []

    for keyword in keywords:
        if keyword.name == "DATES":
            for record in keyword.records:
                if current_date is not None:
                    steps.append(
                        ScheduleStep(current_date, tuple(producers), tuple(injectors))
                    )
                producers, injectors = [], []
                current_date = parse_date_record(record)
        elif keyword.name == "WCONPROD" and current_date is not None:
            producers.extend(_producer(record) for record in keyword.records)
        elif keyword.name == "WCONINJE" and current_date is not None:
            injectors.extend(_injector(record) for record in keyword.records)
        elif current_date is None:
            preamble.append(keyword)
        elif keyword.name in UNSUPPORTED_CONTROL_KEYWORDS:
            unsupported[keyword.name] = None
        # прочие ключевые слова внутри шага (COMPDATMD, WPIMULT) режимами не являются

    if current_date is not None:
        steps.append(ScheduleStep(current_date, tuple(producers), tuple(injectors)))
    return Schedule(preamble=tuple(preamble), steps=tuple(steps),
                    unsupported_controls=tuple(sorted(unsupported)))


def parse_schedule(path: str | Path) -> Schedule:
    """Разобрать SCHEDULE-файл, например `vendor/Model_Z/Model_Z_sch.inc`."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_schedule_text(text)


def parse_date_record(record: tuple[str | None, ...]) -> date:
    """`01 NOV 1994` → date(1994, 11, 1)."""
    if len(record) < 3 or any(part is None for part in record[:3]):
        raise ValueError(f"Некорректная запись DATES: {record!r}")
    day, month, year = record[0], record[1], record[2]
    return date(int(str(year)), _MONTHS[str(month).upper()[:3]], int(str(day)))


def format_date_record(value: date) -> str:
    """date(1994, 11, 1) → `01 NOV 1994`."""
    return f"{value.day:02d} {_MONTH_NAMES[value.month]} {value.year}"


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.split("--", 1)[0]
        if line.strip():
            tokens.extend(_TOKEN_RE.findall(line))
    return tokens


def _is_keyword(token: str) -> bool:
    return bool(_KEYWORD_RE.fullmatch(token))


def _expand_defaults(items: list[str]) -> tuple[str | None, ...]:
    """Раскрыть повторы: `3*` → три None, `3*50` → три '50', `1*` → один None."""
    expanded: list[str | None] = []
    for item in items:
        match = _REPEAT_RE.match(item)
        if match:
            count = int(match.group(1))
            value = match.group(2)
            expanded.extend([_unquote(value) if value else None] * count)
        else:
            expanded.append(_unquote(item))
    return tuple(expanded)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _sorted_wells(names: set[str]) -> tuple[str, ...]:
    return tuple(sorted(names, key=lambda name: (0, int(name), "") if name.isdigit() else (1, 0, name)))


def _number(value: str | None) -> float | None:
    return None if value is None else float(value)


def _field(record: tuple[str | None, ...], index: int) -> str | None:
    return record[index] if index < len(record) else None


def _producer(record: tuple[str | None, ...]) -> ProducerControl:
    return ProducerControl(
        well=str(_field(record, _WCONPROD_WELL)),
        status=str(_field(record, _WCONPROD_STATUS) or "OPEN"),
        mode=str(_field(record, _WCONPROD_MODE) or ""),
        liquid_rate=_number(_field(record, _WCONPROD_LRAT)),
        bhp=_number(_field(record, _WCONPROD_BHP)),
    )


def _injector(record: tuple[str | None, ...]) -> InjectorControl:
    return InjectorControl(
        well=str(_field(record, _WCONINJE_WELL)),
        fluid=str(_field(record, _WCONINJE_FLUID) or "WATER"),
        status=str(_field(record, _WCONINJE_STATUS) or "OPEN"),
        mode=str(_field(record, _WCONINJE_MODE) or "RATE"),
        rate=_number(_field(record, _WCONINJE_RATE)),
        bhp=_number(_field(record, _WCONINJE_BHP)),
    )
