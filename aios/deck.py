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

    preamble_controls: tuple[str, ...] = ()
    """Ключевые слова, задающие режим **до первого `DATES`**.

    Разбор их не теряет, но и в шаги не превращает: дата у такого режима — начало
    расчёта из `START` (RUNSPEC), а этот файл сюда не приходит. На Модели Z список
    пуст (замер 20.08: в преамбуле только `RPTSCHED` и `WELSPECS`), но заявляемое
    число считается на тестовом кейсе организаторов, дека которого мы не видели.
    Решение, что с этим делать, принимает `aios.run.build_case` — см. Р-51.
    """
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


#: Ключевые слова, которые действительно начинают блок. Всё остальное, что
#: похоже на ключевое слово, — данные записи.
#:
#: Зачем нужен список, хотя раньше хватало шаблона `[A-Z][A-Z0-9]{1,11}`. Этот
#: шаблон совпадает и с именем скважины: дек, где имена не закавычены (`P1`
#: вместо `'P1'`), разбирался в **пустой** блок, потому что первый же токен
#: записи закрывал только что открытое ключевое слово. Молча: `WCONPROD` с
#: незакавыченным именем давал ноль добывающих скважин и ни одной ошибки, то
#: есть на деке валидационного кейса скважина исчезла бы из фонда вместе со своей
#: экономикой (ревизия 03.09, 2.3 #1, воспроизведено на `parse_schedule_text`).
#:
#: Полного списка Eclipse тут нет и не нужно: нам важны те, что встречаются в
#: SCHEDULE и в шапке дека. Неизвестное ключевое слово в **незакрытом** блоке
#: по-прежнему сольётся с ним — это малформед-дек, и о нём говорит входной
#: контроль (`aios.deckcheck`), а не парсер.
BLOCK_KEYWORDS = frozenset({
    # режим скважин и время
    "WCONPROD", "WCONINJE", "WCONHIST", "WCONINJH", "WELOPEN", "WELTARG", "WEFAC",
    "WELSPECS", "WELSPECL", "COMPDAT", "COMPDATL", "COMPDATMD", "WELLTRACK",
    "WPIMULT", "WELSEGS", "COMPSEGS", "WSEGVALV", "WLIST", "WTEST", "WECON",
    "WPOLYMER", "WSALT", "WTRACER", "WGRUPCON", "WDFAC", "WFRICTN",
    "DATES", "TSTEP", "TIME", "SKIPREST", "RESTART",
    # группы и объект
    "GCONPROD", "GCONINJE", "GRUPTREE", "GRUPNET", "GEFAC", "GECON", "GCONSUMP",
    "NODEPROP", "BRANPROP",
    # управление расчётом и вывод
    "RPTSCHED", "RPTRST", "RPTPRINT", "TUNING", "TUNINGDP", "NEXTSTEP", "NUPCOL",
    "DRSDT", "DRVDT", "VAPPARS", "AQUCT", "AQUFETP", "AQUANCON", "MULTFLT",
    "SUMMARY", "SCHEDULE", "END", "ENDBOX", "BOX", "INCLUDE", "SAVE", "TERMINATE",
    "WSEGITER", "ACTIONX", "ENDACTIO", "UDQ", "UDADIMS", "PYACTION",
    "ACTION", "ACTIONG", "ACTIONR", "ACTIONS", "ACTIONW", "WELCNTL",
    # секции и шапка: их читает входной контроль дека и сборка кейса
    "RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "START",
    "TABDIMS", "DIMENS", "WELLDIMS", "EQLDIMS", "REGDIMS", "AQUDIMS", "GRIDOPTS",
    "METRIC", "FIELD", "LAB", "OIL", "WATER", "GAS", "DISGAS", "VAPOIL",
    "UNIFOUT", "UNIFIN", "NOSIM", "ENDSCALE", "RUNSUM", "MESSAGES", "TITLE",
    "PVTO", "PVTW", "PVTG", "DENSITY", "ROCK", "PVTNUM", "SATNUM", "EQLNUM",
    "FIPNUM", "MULTNUM", "FLUXNUM", "OPERNUM", "ACTNUM", "COORD", "ZCORN",
    "PERMX", "PERMY", "PERMZ", "PORO", "NTG", "MULTX", "MULTY", "MULTZ",
    "MULTPV", "EQUALS", "COPY", "ADD", "MULTIPLY", "OPERATE", "EDITNNC",
    "MULTREGT", "FAULTS", "PINCH", "PINCHREG", "LGR", "CARFIN", "RADFIN",
    "RADFIN4", "ENDFIN", "EQUIL", "PRESSURE", "SWAT", "SGAS", "RS", "RV",
})


def parse_keywords(text: str) -> list[Keyword]:
    """Разобрать текст в последовательность ключевых слов с записями."""
    tokens = _tokenize(text)
    keywords: list[Keyword] = []
    index = 0
    while index < len(tokens):
        token, at_line_start = tokens[index]
        index += 1
        if not _starts_block(token, at_line_start):
            continue  # мусор вне блока: пропускаем, а не угадываем
        name = token
        records: list[tuple[str | None, ...]] = []
        current: list[str] = []
        while index < len(tokens):
            token, at_line_start = tokens[index]
            if not current and _starts_block(token, at_line_start):
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


#: Ключевые слова, назначающие скважине режим. Нужны отдельно от
#: `UNSUPPORTED_CONTROL_KEYWORDS`: там рядом лежат `TSTEP` и `INCLUDE`, которые
#: режима не задают. Список тот же, по которому машина А разбирала преамбулу
#: настоящего `Model_Z_sch.inc` (20.08): там их ноль.
#:
#: Имя намеренно длиннее очевидного `CONTROL_KEYWORDS`: такое имя уже занято в
#: `aios/run.py` под другой смысл — «что наш генератор перепишет сам, а значит
#: переносить дословно не надо», и туда входит `DATES`. Совпадение имён при
#: перекрёстных импортах этих двух модулей означало бы, что `carried_keywords`
#: начнёт переносить блоки дат дословно и продублирует шаги.
WELL_CONTROL_KEYWORDS = frozenset({
    "WCONPROD", "WCONINJE", "WCONHIST", "WCONINJH", "WELOPEN", "WELTARG",
})

#: Из них те, что **заменяют** генерируемые нами `WCONPROD`/`WCONINJE`.
#:
#: Разница принципиальная и решает, отказывать или предупреждать. `WCONHIST`
#: занимает место нашего ключевого слова: разобранные шаги остаются без добычи
#: вовсе, а в собранный файл попадут и он, и наш `WCONPROD` — конфликт. Остальные
#: лишь домножают или переключают уже заданный режим, доходят до симулятора
#: дословно, и неточен только наш собственный взгляд на режимы. Отказывать из-за
#: них значило бы превратить работоспособный кейс организаторов в отказ.
#:
#: Отказ печатает `aios.run.build_case`, а `scripts/build_scaled_case.py` ловит
#: его и называет ценой: «дек такого вида мы не считаем» вместо трассировки на
#: середине сборки (Б29, 23.08).
REPLACING_CONTROL_KEYWORDS = frozenset({"WCONHIST", "WCONINJH", "TSTEP", "INCLUDE"})

#: Ключевые слова, которые меняют смысл расписания, но нами не разбираются.
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
    "TSTEP",      # шаг времени в сутках вместо DATES — законный способ вести время
    "INCLUDE",    # расписание, разбитое по файлам: включение мы не раскрываем
})


#: Ключевые слова, у которых пустой блок означает не «нечего назначать», а
#: непрочитанную запись: `WCONPROD` без записей не назначает режим никому, `DATES`
#: без записей не заводит шаг. В настоящем деке таких блоков не бывает.
_KEYWORDS_THAT_MUST_HAVE_RECORDS = WELL_CONTROL_KEYWORDS | {"DATES", "WELSPECS"}


def parse_schedule_text(text: str) -> Schedule:
    """Разобрать SCHEDULE-секцию из строки.

    Отказывает, если блок режима разобрался **пустым**. Это вторая сетка под
    `BLOCK_KEYWORDS` (Р-177), а не замена ей: список закрывает известную причину
    — имя скважины, принятое за ключевое слово, — а проверка ловит **последствие**
    любой причины, включая ту, которой в списке ещё нет. Ключевое слово чужого
    дека, не названное в `BLOCK_KEYWORDS`, разбирается как данные и попадает в
    записи соседа; если при этом блок режима остаётся пустым, скважина исчезает из
    фонда, SUMMARY-секции и ЧДД молча. Отказ превращает молчание в отказ.
    """
    keywords = parse_keywords(text)

    empty = [one.name for one in keywords
             if one.name in _KEYWORDS_THAT_MUST_HAVE_RECORDS and not one.records]
    if empty:
        raise ValueError(
            f"блок {empty[0]} разобрался без записей — запись прочитана не до "
            f"конца. Проверить, все ли ключевые слова дека названы в BLOCK_KEYWORDS: "
            f"не названное разбирается как данные и уходит в записи соседнего блока"
        )

    preamble: list[Keyword] = []
    steps: list[ScheduleStep] = []
    unsupported: dict[str, None] = {}
    preamble_controls: dict[str, None] = {}
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
        elif keyword.name in WELL_CONTROL_KEYWORDS and current_date is None:
            # Режим до первой даты действует с начала расчёта (`START` в RUNSPEC),
            # а этого файла мы не читаем — датировать его здесь нечем. Молча
            # потерять нельзя: скважина, чей единственный режим стоит тут, не
            # попадёт ни в `wells`, ни в SUMMARY-секцию, и её выручка не дойдёт до
            # ЧДД. Поэтому запоминаем и даём отказать тому, кто собирает кейс.
            preamble_controls[keyword.name] = None
            preamble.append(keyword)
        elif keyword.name in UNSUPPORTED_CONTROL_KEYWORDS:
            # и до первой DATES тоже: дек, ведущий время через TSTEP, шагов не
            # даёт вовсе, и всё его содержимое оказывается в преамбуле
            unsupported[keyword.name] = None
            if current_date is None:
                preamble.append(keyword)
        elif current_date is None:
            preamble.append(keyword)
        # прочие ключевые слова внутри шага (COMPDATMD, WPIMULT) режимами не являются

    if current_date is not None:
        steps.append(ScheduleStep(current_date, tuple(producers), tuple(injectors)))
    return Schedule(preamble=tuple(preamble), steps=tuple(steps),
                    unsupported_controls=tuple(sorted(unsupported)),
                    preamble_controls=tuple(sorted(preamble_controls)))


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


def _tokenize(text: str) -> list[tuple[str, bool]]:
    """Токены с пометкой «первый в своей строке».

    Пометка нужна разбору: ключевое слово в Eclipse стоит отдельной строкой, а
    данные записи — после него. Без пометки токен `P1` в середине записи ничем не
    отличается от ключевого слова, и один шаблон обслуживает оба смысла.
    """
    tokens: list[tuple[str, bool]] = []
    for line in text.splitlines():
        line = _strip_comment(line)
        if not line.strip():
            continue
        for position, token in enumerate(_TOKEN_RE.findall(line)):
            tokens.append((token, position == 0))
    return tokens


def _strip_comment(line: str) -> str:
    """Убрать комментарий, не тронув `--` внутри кавычек.

    Простое `split("--")` режет запись посреди строки в кавычках: путь
    `'model--z/sch.inc'` превращается в `'model`, и `INCLUDE` начинает указывать
    в никуда — молча, потому что токен остаётся синтаксически годным. Каталог
    ловушек Eclipse (`docs/manus/round4.md`, ответ 4-4) называет это отдельной
    конструкцией: вне кавычек всё от `--` до конца строки удаляется **до**
    разбора, внутри одиночных или двойных кавычек комментарием не является.
    """
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "-" and line.startswith("--", index):
            return line[:index]
    return line


def _is_keyword(token: str) -> bool:
    return bool(_KEYWORD_RE.fullmatch(token))


def _starts_block(token: str, at_line_start: bool) -> bool:
    """Открывает ли токен новый блок.

    Три условия сразу: токен стоит первым в своей строке, выглядит как ключевое
    слово и **назван** в `BLOCK_KEYWORDS`. Последнее и отличает ключевое слово от
    имени скважины: `P1` проходит первые два условия и раньше закрывал только что
    открытый блок (ревизия 03.09, 2.3 #1).
    """
    return at_line_start and _is_keyword(token) and token.upper() in BLOCK_KEYWORDS


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
