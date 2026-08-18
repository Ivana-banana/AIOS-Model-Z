"""PVT-регионы дека: какая плотность нефти относится к какой скважине.

Зачем. Плотность переводит объём симулятора в тонны, а тонны идут в выручку —
это множитель прямо в ЧДД, а не уточнение. В Модели Z два PVT-региона с
плотностью нефти 913,08 и 928,16 кг/м³, разница 1,6 %. До 18.08 весь фонд считался
по первому региону просто потому, что он первый в `DENSITY`.

Как определяется регион скважины. У скважины есть вскрытия (`COMPDAT`: ячейки
`I J K1..K2`), у каждой ячейки сетки есть номер PVT-региона (`PVTNUM`). Регион
скважины — тот, в котором лежит большинство её вскрытых ячеек.

Два ограничения этого способа, которые обязаны быть в отчёте:

* **Скважина может вскрывать оба региона.** Тогда выбирается регион с бо́льшим
  числом вскрытых ячеек, а сам факт помечается (`WellRegion.mixed`) и виден в
  отчёте — молча усреднять физически разные флюиды нельзя.
* **Счёт идёт по ячейкам, а не по объёму или продуктивности.** Честное взвешивание
  требует толщин и проницаемостей из `Model_Z_grid.inc` (74 МБ), и это отдельная
  задача. Пока способ приближённый, и так он и подаётся.

Ячейки с `PVTNUM = 0` — неактивные, вне пласта: в Модели Z их 306 974 из 547 638.
В регион скважины они не идут.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from aios.summary import Fluids


@dataclass(frozen=True)
class PvtRegion:
    """Строка `DENSITY`: поверхностные плотности одного PVT-региона, кг/м³."""

    number: int
    oil_density: float
    water_density: float
    gas_density: float


@dataclass(frozen=True)
class WellRegion:
    """К какому PVT-региону отнесена скважина и насколько уверенно."""

    well: str
    region: int
    counts: dict[int, int] = field(default_factory=dict)
    """Сколько вскрытых ячеек пришлось на каждый регион."""

    @property
    def mixed(self) -> bool:
        """Скважина вскрывает больше одного региона — выбор приближённый."""
        return len([n for n, count in self.counts.items() if count > 0]) > 1

    @property
    def cells(self) -> int:
        return sum(self.counts.values())

    @property
    def share(self) -> float:
        """Доля вскрытых ячеек в выбранном регионе. 1.0 — скважина целиком в нём."""
        return self.counts.get(self.region, 0) / self.cells if self.cells else 0.0


def parse_density(text: str) -> tuple[PvtRegion, ...]:
    """Разобрать `DENSITY`: по строке на регион, нумерация с единицы."""
    body = _keyword_body(text, "DENSITY")
    if body is None:
        raise ValueError("в тексте нет ключевого слова DENSITY")

    regions: list[PvtRegion] = []
    for record in body:
        values = [float(token) for token in record]
        if len(values) != 3:
            raise ValueError(f"строка DENSITY должна нести три значения, получено: {record}")
        regions.append(
            PvtRegion(
                number=len(regions) + 1,
                oil_density=values[0],
                water_density=values[1],
                gas_density=values[2],
            )
        )
    if not regions:
        raise ValueError("DENSITY не содержит ни одной строки")
    return tuple(regions)


def parse_int_array(text: str, keyword: str) -> list[int]:
    """Развернуть целочисленный массив по сетке (`PVTNUM`, `SATNUM`, `FIPNUM`).

    Формат Eclipse сжимает повторы как `N*значение`; разворачиваем полностью —
    массив на 547 638 ячеек занимает единицы мегабайт, а поиск по индексу нужен
    прямой.
    """
    start = _keyword_start(text, keyword)
    if start is None:
        raise ValueError(f"в тексте нет ключевого слова {keyword}")
    end = text.index("/", start)

    values: list[int] = []
    for line in text[start:end].splitlines()[1:]:
        line = line.split("--", 1)[0]
        for token in line.split():
            if "*" in token:
                repeat, _, value = token.partition("*")
                values.extend([int(value)] * int(repeat))
            else:
                values.append(int(token))
    return values


def parse_completions(text: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """Вскрытия из `COMPDAT`: скважина → список `(I, J, K1, K2)`.

    Скважины, описанные через `COMPDATMD` (по измеренной глубине), сюда не
    попадают: там нет координат ячеек. Их отсутствие видно вызывающему, а не
    подменяется первым регионом.
    """
    completions: dict[str, list[tuple[int, int, int, int]]] = {}
    for body in _keyword_bodies(text, "COMPDAT"):
        for record in body:
            if len(record) < 5:
                continue
            well = record[0].strip("'\"")
            try:
                i, j, k1, k2 = (int(record[index]) for index in range(1, 5))
            except ValueError:
                continue  # запись с умолчаниями (1*) в координатах — пропускаем
            completions.setdefault(well, []).append((i, j, k1, k2))
    return completions


def assign_regions(completions: dict[str, list[tuple[int, int, int, int]]],
                   region_of_cell: list[int],
                   dimensions: tuple[int, int, int]) -> dict[str, WellRegion]:
    """Отнести каждую скважину к PVT-региону по её вскрытым ячейкам.

    Порядок ячеек в массиве — натуральный для Eclipse: быстрее всего меняется `I`,
    затем `J`, затем `K`. Ячейки с регионом 0 (неактивные) не учитываются.
    """
    nx, ny, nz = dimensions
    expected = nx * ny * nz
    if len(region_of_cell) != expected:
        raise ValueError(
            f"массив регионов на {len(region_of_cell)} ячеек не совпадает с сеткой "
            f"{nx}×{ny}×{nz} = {expected}"
        )

    assigned: dict[str, WellRegion] = {}
    for well, blocks in completions.items():
        counts: dict[int, int] = {}
        for i, j, k1, k2 in blocks:
            for k in range(min(k1, k2), max(k1, k2) + 1):
                if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                    continue
                region = region_of_cell[(k - 1) * nx * ny + (j - 1) * nx + (i - 1)]
                if region <= 0:  # неактивная ячейка
                    continue
                counts[region] = counts.get(region, 0) + 1
        if not counts:
            continue
        # большинство вскрытых ячеек; при равенстве — меньший номер региона,
        # чтобы выбор был воспроизводим, а не зависел от порядка словаря
        best = max(sorted(counts), key=lambda number: counts[number])
        assigned[well] = WellRegion(well=well, region=best, counts=counts)
    return assigned


def well_fluids(regions: tuple[PvtRegion, ...],
                assigned: dict[str, WellRegion]) -> dict[str, Fluids]:
    """Плотности по скважинам — то, что уходит в `summary.monthly_rows`."""
    by_number = {region.number: region for region in regions}
    fluids: dict[str, Fluids] = {}
    for well, placement in assigned.items():
        region = by_number.get(placement.region)
        if region is None:
            continue
        fluids[well] = Fluids(
            oil_density=region.oil_density, water_density=region.water_density
        )
    return fluids


def from_deck(model_dir: str | Path,
              *,
              dimensions: tuple[int, int, int],
              props: str = "Model_Z_props.inc",
              regions: str = "Model_Z_regs.inc",
              schedule: str = "Model_Z_sch.inc") -> tuple[dict[str, Fluids], dict[str, WellRegion]]:
    """Собрать плотности по скважинам прямо из дека.

    Имена файлов — параметры, а не константы: на валидационном кейсе организаторов
    дек будет называться иначе (ТЗ п. 4.3).
    """
    directory = Path(model_dir)
    pvt = parse_density(_read(directory / props))
    array = parse_int_array(_read(directory / regions), "PVTNUM")
    completions = parse_completions(_read(directory / schedule))
    assigned = assign_regions(completions, array, dimensions)
    return well_fluids(pvt, assigned), assigned


def report(regions: tuple[PvtRegion, ...], assigned: dict[str, WellRegion]) -> str:
    """Человеко-читаемая сводка: куда попал фонд и где выбор приближённый."""
    by_region: dict[int, list[WellRegion]] = {}
    for placement in assigned.values():
        by_region.setdefault(placement.region, []).append(placement)

    lines = [f"PVT-регионов в деке: {len(regions)}"]
    for region in regions:
        wells = by_region.get(region.number, [])
        lines.append(
            f"  регион {region.number}: нефть {region.oil_density:.2f} кг/м³, "
            f"вода {region.water_density:.1f} — скважин {len(wells)}"
        )
    mixed = sorted(p.well for p in assigned.values() if p.mixed)
    lines.append(f"Скважин, вскрывающих несколько регионов: {len(mixed)}")
    if mixed:
        lines.append("  " + ", ".join(mixed[:20]) + (" …" if len(mixed) > 20 else ""))
    return "\n".join(lines)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _keyword_start(text: str, keyword: str) -> int | None:
    match = re.search(rf"^[ \t]*{keyword}[ \t]*$", text, re.MULTILINE)
    return match.start() if match else None


def _keyword_bodies(text: str, keyword: str) -> list[list[list[str]]]:
    """Все блоки ключевого слова: список блоков, в каждом — список записей."""
    bodies: list[list[list[str]]] = []
    for match in re.finditer(rf"^[ \t]*{keyword}[ \t]*$", text, re.MULTILINE):
        bodies.append(_records(text, match.end()))
    return bodies


def _keyword_body(text: str, keyword: str) -> list[list[str]] | None:
    start = _keyword_start(text, keyword)
    return None if start is None else _records(text, start + len(keyword))


def _records(text: str, position: int) -> list[list[str]]:
    """Записи блока: до пустой строки-терминатора `/` или следующего ключевого слова."""
    records: list[list[str]] = []
    for line in text[position:].splitlines():
        stripped = line.split("--", 1)[0].strip()
        if not stripped:
            continue
        if stripped == "/":
            break
        if re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", stripped):
            break  # началось следующее ключевое слово — блок кончился
        if not stripped.endswith("/"):
            break
        records.append(stripped[:-1].split())
    return records
