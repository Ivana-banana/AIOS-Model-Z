"""Состав дека: какой файл за что отвечает и какая у модели сетка.

Зачем отдельный разбор. До 18.08 имена файлов дека были зашиты в коде
(`Model_Z.data`, `Model_Z_sch.inc`, `Model_Z_summary.inc`), а размеры сетки
передавались вызывающим числом. На Модели Z это работало, но по ТЗ п. 4.3
заявляемый ЧДД считается **на тестовом кейсе организаторов**, где дек будет
называться иначе. Зашитое имя — это не стилевая придирка, а отказ пайплайна на
том единственном прогоне, который оценивают.

Поэтому состав дека выясняется из самого дека: в корневом `.DATA` секции идут
подряд (`RUNSPEC`, `GRID`, `EDIT`, `PROPS`, `REGIONS`, `SOLUTION`, `SUMMARY`,
`SCHEDULE`), и каждый `INCLUDE` принадлежит той секции, внутри которой он
записан. Это свойство формата, а не соглашение организаторов.

Ошибки здесь громкие. Ноль или несколько `.DATA` в каталоге, отсутствие
`DIMENS`, секция без `INCLUDE` — всё это отказ с указанием, что именно не найдено.
Тихо подставить умолчание значит посчитать не ту модель и не узнать об этом.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Секции дека в порядке следования. `INCLUDE` относится к последней открытой.
SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")

_SECTION_RE = re.compile(rf"^\s*({'|'.join(SECTIONS)})\s*$", re.IGNORECASE)
_INCLUDE_RE = re.compile(r"^\s*INCLUDE\s*$", re.IGNORECASE)
_QUOTED_RE = re.compile(r"'([^']+)'")
_DIMENS_RE = re.compile(r"^\s*DIMENS\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class DeckLayout:
    """Что нашлось в деке. Пути абсолютные, размеры — из `DIMENS`."""

    data_file: Path
    includes: dict[str, tuple[Path, ...]]
    """Секция → включаемые файлы в порядке появления."""

    dimensions: tuple[int, int, int] | None
    """`NX, NY, NZ` из `DIMENS`, если оно есть в корневом файле.

    Может быть `None`: `DIMENS` иногда лежит во включаемом файле RUNSPEC. Размеры
    нужны только тем, кто адресует ячейку (`aios.pvt`), — там отсутствие и
    становится ошибкой, а сборке кейса они не нужны вовсе.
    """

    def one(self, section: str) -> Path:
        """Единственный `INCLUDE` секции. Ноль или несколько — ошибка, а не выбор.

        Именно так подключены SCHEDULE и SUMMARY в деках обоих виденных нами
        форматов; если в кейсе организаторов их окажется несколько, мы об этом
        узнаем отказом, а не неверным числом.
        """
        found = self.includes.get(section.upper(), ())
        if len(found) != 1:
            raise ValueError(
                f"в секции {section.upper()} дека {self.data_file.name} ожидался ровно один "
                f"INCLUDE, найдено {len(found)}: {[p.name for p in found]}"
            )
        return found[0]

    def grid(self) -> tuple[int, int, int]:
        """Размеры сетки или внятный отказ. Для тех, кому они обязательны."""
        if self.dimensions is None:
            raise ValueError(
                f"в {self.data_file.name} нет DIMENS — размеры сетки неизвестны, "
                "адресовать ячейку невозможно"
            )
        return self.dimensions

    @property
    def cells(self) -> int:
        nx, ny, nz = self.grid()
        return nx * ny * nz


def find_data_file(model_dir: str | Path) -> Path:
    """Корневой `.DATA` каталога. Регистр расширения у поставок разный."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ValueError(f"нет каталога дека: {directory}")

    found = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".data"
    )
    if not found:
        raise ValueError(f"в {directory} нет ни одного .DATA — это не каталог дека")
    if len(found) > 1:
        raise ValueError(
            f"в {directory} несколько .DATA ({[p.name for p in found]}); "
            "какой из них считать — должен решить человек, а не порядок сортировки"
        )
    return found[0]


def parse_layout(data_file: str | Path) -> DeckLayout:
    """Разобрать корневой `.DATA`: включаемые файлы по секциям и размеры сетки."""
    path = Path(data_file)
    text = path.read_text(encoding="utf-8", errors="replace")

    includes: dict[str, list[Path]] = {}
    dimensions: tuple[int, int, int] | None = None
    section: str | None = None
    expect_include = False
    expect_dimens = False

    for raw in text.splitlines():
        line = raw.split("--", 1)[0].strip()
        if not line:
            continue

        if expect_include:
            quoted = _QUOTED_RE.search(line)
            if quoted:
                target = (path.parent / quoted.group(1)).resolve()
                includes.setdefault(section or "", []).append(target)
            expect_include = False
            continue

        if expect_dimens:
            numbers = [int(token) for token in re.findall(r"\d+", line)[:3]]
            if len(numbers) == 3:
                dimensions = (numbers[0], numbers[1], numbers[2])
            expect_dimens = False
            continue

        matched = _SECTION_RE.match(line)
        if matched:
            section = matched.group(1).upper()
            continue
        if _INCLUDE_RE.match(line):
            expect_include = True
            continue
        if _DIMENS_RE.match(line):
            expect_dimens = True
            continue

    return DeckLayout(
        data_file=path,
        includes={name: tuple(paths) for name, paths in includes.items()},
        dimensions=dimensions,
    )


def layout(model_dir: str | Path) -> DeckLayout:
    """Состав дека по каталогу поставки — обычная точка входа."""
    return parse_layout(find_data_file(model_dir))
