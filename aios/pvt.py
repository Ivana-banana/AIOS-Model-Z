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


@dataclass(frozen=True)
class VolumeFactors:
    """Объёмные коэффициенты: во сколько раз пластовый объём больше поверхностного.

    Нужны для компенсации отбора закачкой: закачка возмещает **отобранный объём в
    пластовых условиях**, а выгрузка даёт поверхностные. Пересчёт —
    `Qотб,пласт ≈ Qн·Bн + Qв·Bв`.
    """

    oil_min: float
    oil_max: float
    water: float

    @property
    def oil_typical(self) -> float:
        """Середина диапазона `Bн` по таблице."""
        return (self.oil_min + self.oil_max) / 2

    @property
    def oil_spread(self) -> float:
        """Насколько `Bн` вообще меняется по таблице, доля."""
        return (self.oil_max - self.oil_min) / self.oil_min if self.oil_min else 0.0


def parse_volume_factors(text: str) -> VolumeFactors:
    """Диапазон `Bн` из `PVTO` и `Bв` из `PVTW`.

    `PVTO` — таблица с продолжениями строк: первая строка узла несёт газосодержание,
    давление, `Bн` и вязкость, следующие — только давление, `Bн`, вязкость.
    Поэтому берутся третье с конца и второе с конца числа соответственно.
    """
    oil: list[float] = []
    start = _keyword_start(text, "PVTO")
    if start is None:
        raise ValueError("в тексте нет ключевого слова PVTO")
    for line in text[start:].splitlines()[1:]:
        stripped = line.split("--", 1)[0].strip()
        if not stripped:
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", stripped):
            break
        numbers = [float(token) for token in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", stripped)]
        if len(numbers) == 4:
            oil.append(numbers[2])
        elif len(numbers) == 3:
            oil.append(numbers[1])
    if not oil:
        raise ValueError("в PVTO не нашлось ни одного значения объёмного коэффициента")

    water_body = _keyword_body(text, "PVTW")
    if not water_body:
        raise ValueError("в тексте нет ключевого слова PVTW")
    water = float(water_body[0][1])

    return VolumeFactors(oil_min=min(oil), oil_max=max(oil), water=water)


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
    попадают: там нет координат ячеек. Так устроена поставка тНавигатора —
    `materials/Model_Z.zip` целиком на `COMPDATMD`, а `COMPDAT` есть только в
    OPM-конвертации. Пустой результат превращает в отказ `from_deck`: молча он
    означал бы первый регион на весь фонд.
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


@dataclass(frozen=True)
class DeckFluids:
    """Плотности по скважинам плюс честная пометка о том, что получилось.

    Пометка обязательна. Дек может не дать плотностей вовсе — например, если
    вскрытия описаны через `COMPDATMD` (по измеренной глубине), где координат
    ячеек нет. Тогда считать приходится по общей плотности из конфигурации, и об
    этом надо сказать вслух: падать посреди прогона нельзя, но и молчать нельзя —
    на чужом пласте общая плотность будет просто неверной (ТЗ п. 4.3).
    """

    by_well: dict[str, Fluids]
    note: str
    """Одна строка для лога, отчёта и UI: откуда взяты плотности."""

    without_region: tuple[str, ...] = ()
    """Скважины, для которых регион определить не удалось."""

    @property
    def complete(self) -> bool:
        """Все ли скважины получили плотность из дека."""
        return bool(self.by_well) and not self.without_region


def fluids_by_well(model_dir: str | Path,
                   wells: tuple[str, ...] | list[str] | None = None) -> DeckFluids:
    """Плотности по скважинам из дека — точка входа для рабочих путей.

    Не бросает: разобрать дек может не получиться по множеству причин, а прогон
    из-за этого падать не должен. Всё, что пошло не так, попадает в `note`.
    """
    try:
        by_well, assigned = from_deck(model_dir)
    except (ValueError, OSError) as error:
        return DeckFluids({}, f"плотности из дека недоступны ({error}); взята общая из конфигурации")

    if not by_well:
        return DeckFluids(
            {}, "в деке не нашлось вскрытий COMPDAT — плотности по скважинам определить "
                "нельзя, взята общая из конфигурации"
        )

    missing = tuple(sorted(set(wells) - set(by_well))) if wells else ()
    regions = sorted({placement.region for placement in assigned.values()})
    mixed = sum(1 for placement in assigned.values() if placement.mixed)

    note = (
        f"плотности из дека: скважин {len(by_well)}, "
        f"PVT-регион{'ы' if len(regions) > 1 else ''} {', '.join(map(str, regions))}"
    )
    if mixed:
        note += f"; вскрывают несколько регионов: {mixed}"
    if missing:
        note += f"; без региона осталось {len(missing)} — им общая из конфигурации"
    return DeckFluids(by_well, note, missing)


def from_deck(model_dir: str | Path) -> tuple[dict[str, Fluids], dict[str, WellRegion]]:
    """Собрать плотности по скважинам прямо из дека.

    Ни имена файлов, ни размеры сетки не задаются: и то и другое читается из
    корневого `.DATA` (`aios.deckfiles`). На валидационном кейсе организаторов
    дек будет называться иначе, а сетка будет другой (ТЗ п. 4.3).
    """
    from aios.deckfiles import layout  # локально: разбор состава дека нужен только здесь

    deck = layout(model_dir)
    pvt = parse_density(_read(deck.one("PROPS")))
    array = parse_int_array(_read(deck.one("REGIONS")), "PVTNUM")
    schedule = _read(deck.one("SCHEDULE"))
    completions = parse_completions(schedule)
    assigned = assign_regions(completions, array, deck.grid())
    if not assigned:
        raise ValueError(_nothing_assigned(deck.one("SCHEDULE").name, schedule))
    return well_fluids(pvt, assigned), assigned


def _nothing_assigned(schedule_name: str, schedule: str) -> str:
    """Почему ни одна скважина не попала в регион — с указанием причины.

    Пустой результат нельзя возвращать молча: `summary.monthly_rows` не находит
    скважину в `well_fluids` и берёт общую плотность, то есть весь фонд уезжает
    на первый регион и узнать об этом неоткуда. Плотность — множитель в выручке.
    """
    head = f"ни одна скважина {schedule_name} не сопоставлена с PVT-регионом: "
    if _keyword_start(schedule, "COMPDATMD") is not None:
        return head + (
            "вскрытия заданы через COMPDATMD, по измеренной глубине, и координат "
            "ячеек I J K в них нет. Так устроена поставка тНавигатора; чтобы "
            "получить ячейки, нужны траектории скважин либо дек после конвертации "
            "в COMPDAT"
        )
    return head + (
        f"в SCHEDULE не найдено ни одного COMPDAT с координатами ячеек "
        f"({schedule_name})"
    )


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
