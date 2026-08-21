"""Входной контроль чужого дека: отказать явно вместо тихо неверного расписания.

По ТЗ п. 4.3 заявляемый результат считается на **валидационном кейсе**, который
придёт с другими условиями. Наш генератор `SCHEDULE` умеет узкое подмножество
Eclipse: METRIC, один PVT-регион, месячные `DATES`, поскважинные
`WCONPROD`/`WCONINJE`, буквальный перенос `COMPDAT`. На Модели Z эти
предположения выполняются, и потому не видны. На чужом деке любое из них может
не выполниться — и сегодня генератор в таком случае не отказывается, а пишет
физически другой график **без сообщения**.

Проверки и коды отказов взяты из каталога ловушек Eclipse
(`manus-answered/Каталог ловушек Eclipse SCHEDULE…`, разбор — `docs/manus/round4.md`,
ответ 4-4), версионная база — OPM Flow Reference Manual 2025.10 и `opm-common`
release/2026.04. Порядок проверок значим: нельзя судить о единицах, не раскрыв
включения, и нельзя раскрывать включения, не убедившись, что файлы читаются.

Что здесь **не** делается и почему:

* **ECL-004 (статический разбор `opm-common`) и ECL-015 (dry-run Flow)** требуют
  самого Flow и здесь не выполняются: модуль обязан работать на машине без
  симулятора. Оба шага возвращаются отдельными отметками `pending`, а не молча
  считаются пройденными — иначе «дек проверен» означало бы меньше, чем звучит.
* **Нормализация `COMPDAT`** (ECL-012 в части «разобрать, а не скопировать»)
  здесь только диагностируется. Сама нормализация — задача генератора, а не
  входного контроля.

Отказ — это `Refusal` с кодом, местом и готовым текстом. Пустой список означает
«наши предположения на этом деке выполняются», а не «дек хорош»: физику модели
входной контроль не оценивает вовсе.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from aios.deck import parse_keywords

MAX_PORTABLE_WELL_NAME = 8
"""Документированный предел имени скважины в `WELSPECS`/`WCON*`/`COMPDAT`.

Явного `max_length` в JSON-схеме `opm-common` не найдено, поэтому это граница
переносимости формата, а не проверка парсера: длинное имя разные симуляторы и
утилиты могут обрезать по-разному и разойтись в том, о какой скважине речь.
"""

SUPPORTED_UNITS = "METRIC"

#: Что ломает геометрию перфораций: буквальный `COMPDAT` их не представляет.
GEOMETRY_KEYWORDS = frozenset({
    "WELSPECL", "COMPDATL", "COMPDATMD", "CARFIN", "ENDFIN", "LGR",
    "RADFIN", "RADFIN4", "WELSEGS", "COMPSEGS", "WSEGVALV",
})

#: Условная логика времени исполнения: статическая замена вырезает её целиком.
DYNAMIC_KEYWORDS = frozenset({
    "ACTION", "ACTIONG", "ACTIONR", "ACTIONS", "ACTIONW", "ACTIONX",
    "UDQ", "UDADIMS", "PYACTION",
})

#: Групповое распределение и экономика: поскважинная перезапись меняет модель.
GROUP_KEYWORDS = frozenset({
    "GRUPTREE", "GCONPROD", "GCONINJE", "GECON", "GEFAC", "GRUPNET",
})

#: Управление, не сводимое к нашим `WCONPROD`/`WCONINJE` без потери смысла.
CONTROL_KEYWORDS = frozenset({
    "WCONHIST", "WCONINJH", "WELTARG", "WELOPEN", "WECON", "WEFAC", "WELCNTL",
})

#: Ключевые слова, ссылающиеся на скважину по имени в первом поле записи.
WELL_REFERENCING = frozenset({
    "COMPDAT", "WCONPROD", "WCONINJE", "WCONHIST", "WCONINJH",
    "WELTARG", "WELOPEN", "WECON", "WEFAC",
})

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    "JLY": 7,
}

_INCLUDE_PATH = re.compile(r"""['"]([^'"]+)['"]""")
# до 15 символов, а не до 8: COMPDATMD длиннее формального предела, и
# ловить его надо именно потому, что Flow смотрит только первые восемь
_INCLUDE_LINE = re.compile(r"^\s*INCLUDE\s*$", re.IGNORECASE)
_KEYWORD_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_-]{1,15})\s*$", re.MULTILINE)

#: Ключевые слова, у которых мы смотрим **записи**, а не только присутствие.
#: Только файлы с ними разбираются целиком — см. комментарий в `check`.
_RECORD_KEYWORDS = frozenset({
    "TABDIMS", "DATES", "WELSPECS", *WELL_REFERENCING,
})


@dataclass(frozen=True)
class Refusal:
    """Отказ входного контроля: код каталога, место и готовое сообщение."""

    code: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"ОТКАЗ {self.code}: {self.message} [{self.where}]"


@dataclass(frozen=True)
class Pending:
    """Проверка, которую здесь выполнить нельзя, и чем её выполняют.

    Существует отдельным типом намеренно: «не проверяли» и «проверили, всё
    хорошо» — разные утверждения, и смешивать их в пустом списке отказов значит
    заявлять больше, чем сделано.
    """

    code: str
    reason: str
    command: str


PENDING = (
    Pending(
        "ECL-004",
        "статический разбор opm-common здесь не выполняется: нужен сам парсер",
        "flow --parsing-strictness=normal --enable-dry-run=true <DATA>",
    ),
    Pending(
        "ECL-015",
        "семантическая dry-run проверка требует OPM Flow",
        "flow --enable-dry-run=true <собранный DATA>",
    ),
)


def check(data_file: str | Path) -> list[Refusal]:
    """Проверить дек целиком. Пустой список — наши предположения выполняются.

    Проверки идут в порядке каталога: следующая имеет смысл только когда
    предыдущая прошла. Поэтому при отказе на раскрытии включений мы не судим о
    единицах и расписании — судить было бы не по чему.
    """
    root = Path(data_file)
    found: list[Refusal] = []

    tree, broken = _include_tree(root)
    found.extend(broken)
    found.extend(_non_ascii(tree))
    found.extend(_unbalanced_quotes(tree))
    if found:
        return found

    # Имена ключевых слов ищутся во **всех** файлах: LGR и CARFIN живут в GRID, а
    # не в SCHEDULE, и проверять только расписание значит их пропустить. А вот
    # разбор на записи идёт лишь по файлам, где есть что разбирать: Model_Z_grid.inc
    # это 74,6 МБ массивов COORD/ZCORN, и полный разбор всего дерева занимал 34,8 с
    # против 1,4 с сейчас — при том же результате.
    names = {name for path in tree for name in _keyword_names(_read(path))}
    keywords = parse_keywords("\n".join(
        text for text in (_read(path) for path in tree)
        if _RECORD_KEYWORDS & _keyword_names(text)
    ))

    found.extend(_units(keywords, names, root))
    found.extend(_pvt_regions(keywords, names, root))
    found.extend(_timeline(keywords, names, root))
    found.extend(_restart(names, root))
    found.extend(_well_definitions(keywords, root))
    found.extend(_controls(keywords, names, root))
    found.extend(_groups(names, root))
    found.extend(_geometry(names, root))
    found.extend(_dynamic(names, root))
    found.extend(_patterns(keywords, root))
    return found


def report(data_file: str | Path) -> str:
    """Человеко-читаемый вывод: отказы плюс честный список невыполненных проверок."""
    found = check(data_file)
    lines = [f"дек: {Path(data_file)}"]
    if found:
        lines.append(f"отказов: {len(found)}")
        lines.extend(f"  {refusal}" for refusal in found)
    else:
        lines.append("отказов нет: предположения генератора на этом деке выполняются")
    observed = notes(data_file)
    if observed:
        lines.append(f"заметки ({len(observed)}), сборке не мешают:")
        lines.extend(f"  {one}" for one in observed)
    lines.append("не выполнено здесь (нужен симулятор):")
    lines.extend(f"  {one.code}: {one.reason} — {one.command}" for one in PENDING)
    return "\n".join(lines)


# ── ECL-001…003: файлы, включения, лексика ───────────────────────────────────

def _include_tree(root: Path) -> tuple[list[Path], list[Refusal]]:
    """Раскрыть `INCLUDE` рекурсивно. Относительные пути — от каталога `.DATA`.

    Это не произвол, а совместимостная особенность `opm-common`: вложенный
    относительный `INCLUDE` разрешается относительно корня `.DATA`, а не файла,
    в котором записан. Разрешать «как удобнее» значит собрать другой дек.
    """
    base = root.parent
    seen: list[Path] = []
    visiting: set[Path] = set()
    problems: list[Refusal] = []

    def walk(path: Path, source: str) -> None:
        resolved = path.resolve()
        if resolved in visiting:
            problems.append(Refusal(
                "ECL-002", source,
                f"циклическое включение {path.name}; генерация по такому деку запрещена",
            ))
            return
        if not path.exists():
            problems.append(Refusal(
                "ECL-002", source,
                f"не удалось раскрыть INCLUDE {path.name}; "
                "генерация по неполному деку запрещена",
            ))
            return
        visiting.add(resolved)
        seen.append(path)
        text = _read(path)
        if re.search(r"^\s*PATHS\s*$", text, re.MULTILINE | re.IGNORECASE):
            problems.append(Refusal(
                "ECL-002", str(path),
                "дек использует PATHS-алиасы; без таблицы алиасов "
                "внешние файлы разрешить нельзя",
            ))
        if "INCLUDE" not in text:
            visiting.discard(resolved)
            return
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not _INCLUDE_LINE.match(line):
                continue
            target = _include_target(text.splitlines(), line_number)
            if target is None:
                problems.append(Refusal(
                    "ECL-002", f"{path}:{line_number}",
                    "за INCLUDE не нашлось имени файла в кавычках",
                ))
                continue
            walk(base / target, f"{path}:{line_number}")
        visiting.discard(resolved)

    walk(root, str(root))
    return seen, problems


def _include_target(lines: list[str], after: int) -> str | None:
    for line in lines[after:after + 5]:
        match = _INCLUDE_PATH.search(line)
        if match:
            return match.group(1)
    return None


def _non_ascii(tree: list[Path]) -> list[Refusal]:
    """Не-ASCII **вне комментария**. `opm-common` документирует поддержку только ASCII.

    Различение комментария здесь не послабление, а результат замера. Дек Модели Z
    несёт кириллицу в четырёх файлах — заголовки «Copyright» и «Этот файл был
    сгенерирован тНавигатор v25.1» — и Flow 2026.04 посчитал его целиком
    (базовый прогон 18 мин 43 с, ЧДД 11 851,8). Отказ на этом деке был бы ложным,
    а входной контроль, отвергающий работающий кейс, стоит ровно столько же,
    сколько контроль, пропускающий сломанный.

    Каталог ловушек формулирует так же: «особенно опасны non-ASCII в именах,
    путях и строках, а не только в комментариях». Не-ASCII в комментарии остаётся
    заметкой (`notes`), не-ASCII в разбираемой части — отказом.
    """
    found = []
    for path in tree:
        text = _read(path)
        if text.isascii():
            continue  # 74,6 МБ сетки отсеиваются одной проверкой на уровне C
        for number, line in enumerate(text.splitlines(), start=1):
            if line.isascii():
                continue
            payload = _without_comment(line)
            if not payload.isascii():
                found.append(Refusal(
                    "ECL-001", f"{path}:{number}",
                    "не-ASCII символы вне комментария; opm-common документирует "
                    "поддержку только ASCII, и имена, пути и строки такого дека "
                    "разбираются непредсказуемо",
                ))
                break
    return found


def notes(data_file: str | Path) -> list[str]:
    """Наблюдения, которые не являются отказом, но которые надо видеть.

    Отдельно от отказов намеренно: заметка не мешает сборке, а отказ её
    запрещает. Смешивать значит либо шуметь, либо прятать.
    """
    tree, problems = _include_tree(Path(data_file))
    if problems:
        return []
    found = []
    for path in tree:
        text = _read(path)
        if text.isascii():
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.isascii() and _without_comment(line).isascii():
                found.append(
                    f"{path.name}:{number}: не-ASCII в комментарии — Flow это принимает, "
                    "но при переносе дека кодировка может поехать"
                )
                break
    return found


def _unbalanced_quotes(tree: list[Path]) -> list[Refusal]:
    """Незакрытая кавычка делает строковую подмену SCHEDULE небезопасной."""
    found = []
    for path in tree:
        text = _read(path)
        if "'" not in text and '"' not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if "'" not in line and '"' not in line:
                continue
            stripped = _without_comment(line)
            if stripped.count("'") % 2 or stripped.count('"') % 2:
                found.append(Refusal(
                    "ECL-003", f"{path}:{number}",
                    "незакрытая кавычка: лексическая структура Eclipse неразбираема",
                ))
    return found


# ── ECL-005…007: единицы, PVT, время ─────────────────────────────────────────

def _units(keywords, names: set[str], root: Path) -> list[Refusal]:
    declared = [name for name in ("METRIC", "FIELD", "LAB", "PVT-M") if name in names]
    if not declared:
        return [Refusal("ECL-005", str(root),
                        "система единиц не объявлена; генератор работает только в METRIC")]
    if len(declared) > 1:
        return [Refusal("ECL-005", str(root),
                        f"объявлено несколько систем единиц {declared}; "
                        "какая активна, из дека не следует")]
    if declared[0] != SUPPORTED_UNITS:
        return [Refusal("ECL-005", str(root),
                        f"система единиц {declared[0]} несовместима с генератором METRIC; "
                        "требуется полное размерностное преобразование каждого поля")]
    return []


def _pvt_regions(keywords, names: set[str], root: Path) -> list[Refusal]:
    """Несколько PVT-регионов допустимы, но только когда регион скважины выводим.

    Каталог ловушек требует отказывать при `NTPVT > 1`, потому что описывал
    генератор с одним регионом. Наш генератор с тех пор умеет больше: `aios.pvt`
    относит скважину к региону по её вскрытым ячейкам (`COMPDAT` → `PVTNUM`), и
    именно поэтому плотности Модели Z берутся поскважинно, а не по первой строке
    `DENSITY`.

    Замер, из-за которого проверка и переписана: сама Модель Z объявляет
    `TABDIMS 2 2 …`, то есть **два** PVT-региона, и мы её считаем. Отказ по
    буквальному прочтению каталога отверг бы наш собственный рабочий кейс.

    Остаётся отказ там, где регион вывести действительно не из чего: регионов
    больше одного, а массива `PVTNUM` в деке нет. Тогда любое сопоставление
    скважины с плотностью — угадывание, а плотность входит в ЧДД множителем.
    """
    declared = 1
    for keyword in keywords:
        if keyword.name.upper() != "TABDIMS":
            continue
        for record in keyword.records:
            ntpvt = record[1] if len(record) > 1 else None
            if ntpvt and ntpvt.isdigit():
                declared = max(declared, int(ntpvt))
    if declared > 1 and "PVTNUM" not in names:
        return [Refusal(
            "ECL-006", str(root),
            f"TABDIMS объявляет {declared} PVT-региона, а массива PVTNUM в деке нет: "
            "отнести скважину к плотности не по чему, а плотность входит в ЧДД множителем",
        )]
    return []


def _timeline(keywords, names: set[str], root: Path) -> list[Refusal]:
    """`START` обязателен при `DATES`; шаги — только первые числа месяцев."""
    found: list[Refusal] = []
    if "DATES" in names and "START" not in names:
        found.append(Refusal(
            "ECL-007", str(root),
            "есть DATES, но нет START: календарная дата шагов не восстанавливается",
        ))
    if "TSTEP" in names:
        found.append(Refusal(
            "ECL-007", str(root),
            "расписание ведёт время через TSTEP; генератор умеет только месячные DATES, "
            "а 365.25/4 не означает календарную дату",
        ))
    for keyword in keywords:
        if keyword.name.upper() != "DATES":
            continue
        for record in keyword.records:
            # день сравнивается числом, а не строкой: тНавигатор пишет «01 NOV 1994»,
            # и текстовое сравнение с «1» отвергло бы расписание самой Модели Z
            day = record[0] if record else None
            month = record[1].upper() if len(record) > 1 and record[1] else None
            if day is None or not day.isdigit() or int(day) != 1 or month not in _MONTHS:
                found.append(Refusal(
                    "ECL-007", str(root),
                    f"шаг {' '.join(str(item) for item in record if item)} не на месячной "
                    "сетке; внутримесячные события генератор не представляет",
                ))
                break
        if found and found[-1].code == "ECL-007":
            break
    return found


def _restart(names: set[str], root: Path) -> list[Refusal]:
    if "RESTART" in names or "SKIPREST" in names:
        return [Refusal(
            "ECL-008", str(root),
            "restart-дек нельзя развернуть без restart-файла и времени точки RSNUM; "
            "стартовое состояние принадлежит расчёту, которого у нас нет",
        )]
    return []


# ── ECL-009…014: скважины, управление, геометрия, логика ─────────────────────

def _well_definitions(keywords, root: Path) -> list[Refusal]:
    """Каждая ссылка на скважину обязана иметь предшествующий `WELSPECS`."""
    found: list[Refusal] = []
    declared: set[str] = set()
    for keyword in keywords:
        name = keyword.name.upper()
        if name == "WELSPECS":
            for record in keyword.records:
                if record and record[0]:
                    declared.add(record[0])
                    if len(record[0]) > MAX_PORTABLE_WELL_NAME:
                        found.append(Refusal(
                            "ECL-009", str(root),
                            f"имя скважины {record[0]!r} длиннее переносимых "
                            f"{MAX_PORTABLE_WELL_NAME} символов",
                        ))
        elif name in WELL_REFERENCING:
            for record in keyword.records:
                well = record[0] if record else None
                if well is None or _is_pattern(well):
                    continue
                if well not in declared:
                    found.append(Refusal(
                        "ECL-009", str(root),
                        f"{name} ссылается на скважину {well!r} без предшествующей WELSPECS",
                    ))
    return found


def _controls(keywords, names: set[str], root: Path) -> list[Refusal]:
    found = [
        Refusal("ECL-010", str(root),
                f"управление {name} не сводится к поддерживаемому WCONPROD/WCONINJE "
                "без потери семантики")
        for name in sorted(CONTROL_KEYWORDS & names)
    ]
    for keyword in keywords:
        if keyword.name.upper() not in ("WCONPROD", "WCONINJE"):
            continue
        for record in keyword.records:
            mode = _mode_of(keyword.name.upper(), record)
            if mode in ("GRUP", "THP"):
                found.append(Refusal(
                    "ECL-010", str(root),
                    f"{keyword.name.upper()} задаёт режим {mode}: "
                    + ("скважина управляется группой, а не собственной уставкой"
                       if mode == "GRUP" else "режим THP требует VFP-таблиц"),
                ))
                break
    return found


def _groups(names: set[str], root: Path) -> list[Refusal]:
    return [
        Refusal("ECL-011", str(root),
                f"обнаружено {name}: переписывание только поскважинных режимов "
                "меняет аллокацию и ограничения объекта")
        for name in sorted(GROUP_KEYWORDS & names)
    ]


def _geometry(names: set[str], root: Path) -> list[Refusal]:
    return [
        Refusal("ECL-012", str(root),
                f"геометрия/перфорация {name} не представима буквальным COMPDAT "
                "в глобальной сетке")
        for name in sorted(GEOMETRY_KEYWORDS & names)
    ]


def _dynamic(names: set[str], root: Path) -> list[Refusal]:
    return [
        Refusal("ECL-013", str(root),
                f"дек содержит динамическую логику {name}; статическая замена SCHEDULE "
                "удалит исполняемое условие")
        for name in sorted(DYNAMIC_KEYWORDS & names)
    ]


def _patterns(keywords, root: Path) -> list[Refusal]:
    """Шаблон имени и коллизия регистра: точное множество скважин не определено."""
    found: list[Refusal] = []
    seen: dict[str, str] = {}
    for keyword in keywords:
        if keyword.name.upper() not in WELL_REFERENCING | {"WELSPECS"}:
            continue
        for record in keyword.records:
            well = record[0] if record else None
            if not well:
                continue
            if _is_pattern(well):
                found.append(Refusal(
                    "ECL-014", str(root),
                    f"идентификатор {well!r} является шаблоном: "
                    "точное множество скважин не определено",
                ))
                continue
            earlier = seen.setdefault(well.upper(), well)
            if earlier != well:
                found.append(Refusal(
                    "ECL-014", str(root),
                    f"имена {earlier!r} и {well!r} различаются только регистром; "
                    "opm-common ищет скважину точным ключом и не сливает их",
                ))
    return found


# ── мелочь ───────────────────────────────────────────────────────────────────

def _mode_of(keyword: str, record: tuple[str | None, ...]) -> str | None:
    """Режим управления: у `WCONPROD` это третье поле, у `WCONINJE` — четвёртое."""
    index = 2 if keyword == "WCONPROD" else 3
    return record[index] if len(record) > index and record[index] else None


def _is_pattern(name: str) -> bool:
    return "*" in name or "?" in name


def _keyword_names(text: str) -> set[str]:
    """Имена ключевых слов — дешёвым проходом, без разбора записей.

    Ключевое слово Eclipse стоит на своей строке и не длиннее восьми символов;
    этого достаточно, чтобы узнать о присутствии `ACTIONX`, `CARFIN` или
    `GCONPROD`, не превращая гигабайты массивов сетки в токены.
    """
    return {match.group(1).upper() for match in _KEYWORD_LINE.finditer(text)}


@lru_cache(maxsize=64)
def _read_cached(path: Path, stamp: float, size: int) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read(path: Path) -> str:
    """Прочитать файл дека один раз на прогон.

    Кэш здесь не преждевременная оптимизация: `Model_Z_grid.inc` весит 74,6 МБ, а
    проверки обходят дерево несколько раз (байты, кавычки, имена, заметки). Без
    кэша прогон занимал 23,9 с против 1,5 с. Ключ включает время и размер файла,
    поэтому изменённый на диске дек перечитывается.
    """
    info = path.stat()
    return _read_cached(path, info.st_mtime, info.st_size)


def _without_comment(line: str) -> str:
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
