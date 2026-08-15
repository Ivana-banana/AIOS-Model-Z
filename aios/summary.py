"""Выгрузка результатов симулятора в контракт эталонного расчётчика ЧДД.

Два дела:

1. Собрать SUMMARY-секцию для дека. В поставке `Model_Z_summary.inc` пустой, то есть
   симулятор по умолчанию не выгрузит ничего — векторы задаём мы.
2. Прочитать посчитанный `.UNSMRY` и превратить его в строки «скважина × месяц»
   с теми самыми 14 колонками, которые ждёт расчётчик (`aios.economics`).

Раскладка строк повторяет экспортёр организаторов из тНавигатора
(`vendor/Model_Z_OPM/USER/Model_Z/script_1.py`): их набор величин —
`wlpt, wlpr, womt, womr, wwir, wwit, wbp9, wbhp, weff`, и он ложится в колонки
`WLPT, WLPR, WOMT, WOMR, WWIR, WWIT, THP, BHP, WEFF` именно в этом порядке.
То есть `THP` — это **пластовое давление** (`WBP9`), а не устьевое, а `BHP` —
забойное (`WBHP`); имена колонок здесь вводят в заблуждение, и полагаться надо на
скрипт, а не на них.

Единицы у организаторов смешанные, и это не опечатка: накопленные величины и
приращения — в массе (тонны), дебиты — в объёме (м³/сут). `WOMT_Diff` — нефть, т;
`WLPT_Diff` — жидкость, т; `WWIT_Diff` — закачка, м³; `WLPR`, `WWIR` — м³/сут.
Симулятор считает объёмы, перевод в массу — через поверхностные плотности из
`DENSITY` дека. Плотность нефти отличается по PVT-регионам (913,08 против
928,16 кг/м³, разница 1,6 %), поэтому она параметр, а не константа: ошибка здесь
идёт прямо в выручку.

`WEFF` (коэффициент эксплуатации) ставим 1.0: в деке нет `WEFAC`, а простои
выражены самими режимами. На ЧДД колонка не влияет.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

#: Векторы по добывающим скважинам.
PRODUCER_VECTORS = ("WOPR", "WOPT", "WWPR", "WWPT", "WLPR", "WLPT", "WBHP")

#: Векторы по нагнетательным скважинам.
INJECTOR_VECTORS = ("WWIR", "WWIT")

#: Пластовое давление в зоне дренирования скважины. Может не поддерживаться
#: симулятором — тогда колонка THP заполняется нулями, на ЧДД это не влияет.
PRESSURE_VECTORS = ("WBP9",)

#: Векторы по объекту — для санити-чеков и графиков, в экономику не идут.
FIELD_VECTORS = ("FOPR", "FOPT", "FWPR", "FWPT", "FWIR", "FWIT", "FLPR", "FLPT", "FPR", "FWCT")


@dataclass(frozen=True)
class Fluids:
    """Плотности в поверхностных условиях, кг/м³ (ключевое слово DENSITY дека).

    Значения по умолчанию — первый PVT-регион Модели Z.
    """

    oil_density: float = 913.0765883224138
    water_density: float = 1172.0


def render_summary_section(wells: tuple[str, ...] | list[str],
                           include_field: bool = True) -> str:
    """Собрать SUMMARY-секцию: перечисленные векторы по каждой скважине.

    Список скважин передаётся явно (из `deck.Schedule.wells`), потому что `*`
    поддерживают не все симуляторы одинаково, а молча недовыгруженный вектор —
    это потерянный прогон.
    """
    lines: list[str] = [
        "-- SUMMARY-секция под выгрузку в расчётчик ЧДД.",
        "-- Сгенерировано автоматически (aios/summary.py), вручную не править.",
        "",
    ]
    if include_field:
        for vector in FIELD_VECTORS:
            lines.append(vector)
        lines.append("")

    for vector in (*PRODUCER_VECTORS, *INJECTOR_VECTORS, *PRESSURE_VECTORS):
        lines.append(vector)
        lines.extend(f" '{well}'" for well in wells)
        lines.append("/")
        lines.append("")

    lines.append("RUNSUM")
    lines.append("")
    return "\n".join(lines)


def monthly_rows(case: str | Path,
                 wells: tuple[str, ...] | list[str] | None = None,
                 fluids: Fluids | None = None) -> list[dict[str, Any]]:
    """Прочитать результат расчёта и собрать строки «скважина × месяц».

    `case` — путь к расчёту без расширения (или к `.UNSMRY`/`.SMSPEC`).

    Раскладка повторяет экспортёр организаторов
    (`vendor/Model_Z_OPM/USER/Model_Z/script_1.py`) — это не наша договорённость, а
    их. Там `df.diff()` и следом `df.shift(-1)`, то есть строка с датой `t` несёт:

    * накопленные и мгновенные величины на конец месяца, то есть на шаге `t+1`;
    * приращения `*_Diff` за месяц: значение на `t+1` минус значение на `t`.

    Отсюда важное следствие: **дебиты берутся мгновенные на конец месяца**, а не
    средние за месяц. Разница не косметическая — по `WLPR` методика подбирает
    типоразмер ЭЦН и определяет действующий фонд.
    """
    from resdata.summary import Summary  # тяжёлый импорт — только когда реально читаем

    fluids = fluids or Fluids()
    summary = Summary(str(_case_path(case)))
    steps = list(summary.report_index_list())
    report_dates = _report_dates(summary, steps)
    if len(report_dates) < 2:
        raise ValueError(f"В {case} меньше двух отчётных шагов — считать нечего.")

    known = set(summary.keys())
    names = list(wells) if wells is not None else _wells_in_summary(summary)

    oil_to_t = fluids.oil_density / 1000.0
    water_to_t = fluids.water_density / 1000.0

    rows: list[dict[str, Any]] = []
    for well in names:
        oil_total = _vector(summary, f"WOPT:{well}", known, steps)
        water_total = _vector(summary, f"WWPT:{well}", known, steps)
        injection_total = _vector(summary, f"WWIT:{well}", known, steps)
        oil_rate = _vector(summary, f"WOPR:{well}", known, steps)
        liquid_rate = _vector(summary, f"WLPR:{well}", known, steps)
        injection_rate = _vector(summary, f"WWIR:{well}", known, steps)
        bhp = _vector(summary, f"WBHP:{well}", known, steps)
        reservoir_pressure = _vector(summary, f"WBP9:{well}", known, steps)

        # накопленные массы: объём симулятора × поверхностная плотность
        oil_mass = [value * oil_to_t for value in oil_total]
        liquid_mass = [
            oil * oil_to_t + water * water_to_t
            for oil, water in zip(oil_total, water_total, strict=True)
        ]

        for index in range(len(report_dates) - 1):
            end = index + 1
            rows.append(
                {
                    "DATA": report_dates[index].isoformat(),
                    "well": well,
                    "WLPT": liquid_mass[end],
                    "WLPR": liquid_rate[end],
                    "WOMT": oil_mass[end],
                    "WOMR": oil_rate[end] * oil_to_t,
                    "WWIR": injection_rate[end],
                    "WWIT": injection_total[end],
                    "THP": reservoir_pressure[end],
                    "BHP": bhp[end],
                    "WEFF": 1.0,
                    "WLPT_Diff": _increment(liquid_mass, index),
                    "WOMT_Diff": _increment(oil_mass, index),
                    "WWIT_Diff": _increment(injection_total, index),
                }
            )

    rows.sort(key=lambda row: (row["DATA"], _well_key(row["well"])))
    return rows


def sanity_problems(rows: list[dict[str, Any]],
                    max_liquid_rate: float = 500.0) -> list[str]:
    """Санити-чеки на границе «симулятор → экономика».

    Симулятор и суррогат не падают, а тихо врут, поэтому физический диапазон
    проверяется явно, до того как числа попадут в ЧДД.
    """
    problems: list[str] = []
    previous: dict[str, dict[str, float]] = {}

    for row in rows:
        well = row["well"]
        where = f"{row['DATA']} скв. {well}"

        for column in ("WLPT_Diff", "WOMT_Diff", "WWIT_Diff"):
            if row[column] < 0:
                problems.append(f"{where}: отрицательное {column} = {row[column]:.6g}")

        if row["WOMT_Diff"] > row["WLPT_Diff"] + 1e-9:
            problems.append(
                f"{where}: нефти больше, чем жидкости "
                f"({row['WOMT_Diff']:.6g} > {row['WLPT_Diff']:.6g} т)"
            )
        if row["WLPR"] > max_liquid_rate:
            problems.append(
                f"{where}: дебит жидкости {row['WLPR']:.6g} > {max_liquid_rate:g} м3/сут"
            )
        if row["WLPR"] > 0 and row["WWIR"] > 0:
            problems.append(f"{where}: одновременно добыча и закачка")
        if row["BHP"] < 0 or row["THP"] < 0:
            problems.append(f"{where}: отрицательное давление")

        last = previous.get(well)
        if last is not None:
            for column in ("WLPT", "WOMT", "WWIT"):
                if row[column] + 1e-6 < last[column]:
                    problems.append(f"{where}: накопленное {column} уменьшилось")
        previous[well] = {column: row[column] for column in ("WLPT", "WOMT", "WWIT")}

    return problems


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Записать строки в CSV, который читает эталонный расчётчик как есть."""
    import csv

    from aios.economics import REQUIRED_COLUMNS

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _case_path(case: str | Path) -> Path:
    path = Path(case)
    return path.with_suffix("") if path.suffix.upper() in {".UNSMRY", ".SMSPEC", ".DATA"} else path


def _wells_in_summary(case_summary: Any) -> list[str]:
    # .keys() здесь — метод resdata, а не словарь
    names = {key.split(":", 1)[1] for key in case_summary.keys() if ":" in key}  # noqa: SIM118
    return sorted(names, key=_well_key)


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)


def _report_dates(summary: Any, steps: list[int]) -> list[date]:
    """Даты отчётных шагов, округлённые до суток.

    Берём именно записанные значения на шагах, а не `report_only=True`: тот
    интерполирует по времени и на файле с ненулевым смещением суток даёт
    приращения, отличающиеся на доли процента, — молча и в нашу пользу.
    """
    stamps = list(summary.numpy_dates)
    return [_as_date(stamps[index]) for index in steps]


def _vector(summary: Any, key: str, known: set[str], steps: list[int]) -> list[float]:
    """Значения вектора на отчётных шагах; отсутствующий вектор — нули, а не падение."""
    if key not in known:
        return [0.0] * len(steps)
    values = summary.numpy_vector(key)
    return [float(values[index]) for index in steps]


def _increment(values: list[float], index: int) -> float:
    """Приращение накопленной величины за шаг — как есть, включая отрицательное.

    Соблазн обнулить отрицательные приращения здесь надо давить: методика
    исключает такие строки из экономики **целиком** (`clampNegativeDiffs` в
    эталоне заблокирован в False), и именно так посчитают организаторы. Обнулив
    их у себя, мы завысили бы свой ЧДД ровно на те строки, которые у них выпадут,
    и разошлись бы с их пересчётом. Отрицательные приращения ловит
    `sanity_problems` — их надо увидеть, а не спрятать.
    """
    return values[index + 1] - values[index]


def _as_date(value: Any) -> date:
    """Привести отметку времени к дате, округляя к ближайшим суткам."""
    if isinstance(value, datetime):
        return (value + timedelta(hours=12)).date()
    if isinstance(value, date):
        return value
    stamp = datetime.fromisoformat(str(value).replace("T", " ").split(".")[0])
    return (stamp + timedelta(hours=12)).date()
