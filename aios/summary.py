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

import warnings
from collections.abc import Mapping
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

#: Векторы, которые monthly_rows реально читает по каждой скважине (WBP9 сюда не
#: входит — его отсутствие законно, см. PRESSURE_VECTORS). Если по скважине есть
#: хоть один из них, обязаны быть все: частичный набор — тот же силент-нулевой
#: подлог, что и полностью пустая секция, только на уровне одной скважины.
REQUIRED_WELL_VECTORS = ("WOPT", "WWPT", "WWIT", "WOPR", "WLPR", "WWIR", "WBHP")

#: Всё, что секция пишет по скважинам. Отсюда же проверяется, что выгрузка её
#: вообще несёт: списывать этот набор третьим местом нельзя — разойдётся.
SECTION_VECTORS = (*PRODUCER_VECTORS, *INJECTOR_VECTORS, *PRESSURE_VECTORS)

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
                 fluids: Fluids | None = None,
                 well_fluids: Mapping[str, Fluids] | None = None) -> list[dict[str, Any]]:
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
    # `report_index_list` внутри resdata гасит `DeprecationWarning` на создании
    # `IntVector`, но не на `append` в цикле сразу под ним — и печатает по
    # предупреждению на отчётный шаг. На Модели Z это 371 предупреждение по две
    # строки каждое: единственная команда прогона выдаёт 742 строки шума на три
    # строки собственного вывода (замер 05.09). Гасим здесь, у своего вызова, а
    # не глобально: остальные предупреждения библиотек нам нужны.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        steps = list(summary.report_index_list())
    report_dates = _report_dates(summary, steps)
    if len(report_dates) < 2:
        raise ValueError(f"В {case} меньше двух отчётных шагов — считать нечего.")

    known = set(summary.keys())
    names = list(wells) if wells is not None else _wells_in_summary(summary)

    # Выгрузка без единого скважинного вектора — это не «скважины ничего не
    # дали», а расчёт, к которому нашу SUMMARY-секцию не подключили: файл
    # SUMMARY в поставке организаторов приходит пустым. Читается такой файл
    # без ошибок, и до 20.08 он давал полную таблицу нулей, ноль замечаний
    # санити-чека и ЧДД 0,000 млн руб. как обычный результат. Ноль здесь опаснее
    # отрицательного числа: скважина с нулевым дебитом не входит в действующий
    # фонд, поэтому и OPEX ей не начисляется, и число выглядит безобидно.
    # Тот же отказ закрывает вторую дорогу к нулям — имена скважин, не совпавшие
    # с именами в выгрузке.
    if not any(f"{vector}:{well}" in known
               for well in names for vector in SECTION_VECTORS):
        raise ValueError(
            f"В выгрузке {case} нет ни одного вектора по скважинам "
            f"({', '.join(SECTION_VECTORS)}): SUMMARY-секция к расчёту не "
            "подключена либо имена скважин не совпадают с именами выгрузки. "
            "Секцию пишет aios.summary.render_summary_section — файл SUMMARY "
            "в поставке организаторов приходит пустым."
        )

    for well in names:
        present = [v for v in REQUIRED_WELL_VECTORS if f"{v}:{well}" in known]
        missing = [v for v in REQUIRED_WELL_VECTORS if f"{v}:{well}" not in known]
        if not present:
            # Скважина названа снаружи, но в выгрузке её нет вовсе. Прежде это
            # проходило молча: все её векторы читались нулями, и скважина
            # выглядела «неработающей» — то есть отсутствие данных было
            # неотличимо от простоя. Отказ здесь того же рода, что и ниже: разница
            # только в том, что там пропала часть величин, а тут — все (Р-179).
            raise ValueError(
                f"В выгрузке {case} по скважине {well} нет ни одного вектора: "
                "она либо не попала в SUMMARY-секцию, либо названа иначе. Нули "
                "вместо её величин выдали бы отсутствие данных за простой."
            )
        if present and missing:
            raise ValueError(
                f"В выгрузке {case} по скважине {well} есть часть векторов "
                f"({', '.join(present)}), но не хватает {', '.join(missing)}: "
                "недостающие величины иначе подставятся нулями и выдадут себя "
                "за реальную физику."
            )

    rows: list[dict[str, Any]] = []
    for well in names:
        # плотность берётся по PVT-региону скважины, если он определён (aios/pvt.py);
        # иначе — общая. Регионы физически разные, усреднять их по фонду нельзя.
        own = (well_fluids or {}).get(well, fluids)
        oil_to_t = own.oil_density / 1000.0
        water_to_t = own.water_density / 1000.0

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


def case_problems(rows: list[dict[str, Any]],
                  max_watercut: float | None = None) -> list[str]:
    """Нарушения **условий кейса**, которые видны только по выгрузке.

    Отдельно от `sanity_problems` намеренно. Тот — физический страж границы
    «симулятор → экономика»: отрицательные приращения, дебит выше 500 м³/сут,
    одновременная добыча и закачка. Его непустой список означает «выгрузка
    нефизична, считать по ней ЧДД нельзя», и `run_cycle --confirm` на этом
    останавливается.

    Обводнённость выше предела кейса — не нефизичность. Выгрузка верна, нарушено
    условие задачи, и вывод из этого другой: менять сценарий, а не отказываться
    от числа. Смешение двух смыслов закрывало бы путь получения заявляемого ЧДД
    ровно тогда, когда кейс назовёт предел (ревизия 02.09).
    """
    problems: list[str] = []
    if max_watercut is None:
        return problems
    for row in rows:
        if row["WLPT_Diff"] <= 0:
            continue
        watercut = 1.0 - row["WOMT_Diff"] / row["WLPT_Diff"]
        if watercut > max_watercut:
            problems.append(
                f"{row['DATA']} скв. {row['well']}: обводнённость {watercut:.4g} > "
                f"{max_watercut:g}"
            )
    return problems


def sanity_problems(rows: list[dict[str, Any]],
                    max_liquid_rate: float = 500.0,
                    min_reservoir_pressure: float | None = None,
                    max_reservoir_pressure: float | None = None) -> list[str]:
    """Санити-чеки на границе «симулятор → экономика».

    Симулятор и суррогат не падают, а тихо врут, поэтому физический диапазон
    проверяется явно, до того как числа попадут в ЧДД.

    `min_reservoir_pressure` и `max_reservoir_pressure` — пределы **пластового**
    давления (колонка `THP`, это `WBP9`): снизу у добывающих, сверху у
    нагнетательных. Роль строки определяется так же, как действующий фонд, —
    по `WLPR` и `WWIR`. None у обоих означает «предел неизвестен, не проверяем»:
    выдуманная граница остановила бы прогон на числе, которого никто не измерял.
    """
    problems: list[str] = list(_monthly_step_problems(rows))
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
        if (min_reservoir_pressure is not None
                and row["WLPR"] > 0 and row["THP"] < min_reservoir_pressure):
            problems.append(
                f"{where}: пластовое давление {row['THP']:.6g} < "
                f"{min_reservoir_pressure:g} атм у добывающей"
            )
        if (max_reservoir_pressure is not None
                and row["WWIR"] > 0 and row["THP"] > max_reservoir_pressure):
            problems.append(
                f"{where}: пластовое давление {row['THP']:.6g} > "
                f"{max_reservoir_pressure:g} атм у нагнетательной"
            )

        last = previous.get(well)
        if last is not None:
            for column in ("WLPT", "WOMT", "WWIT"):
                drop = last[column] - row[column]
                if drop <= 1e-6:
                    continue
                increment = row[f"{column}_Diff"]
                # На выгрузке симулятора `*_Diff` — это и есть разность соседних
                # накопленных (`monthly_rows`), поэтому отрицательный прирост и
                # уменьшение накопленного — один факт, уже названный выше. Второй
                # строки он не заслуживает: на опорном расписании такие пары дают
                # четверть всех нарушений.
                # Снять проверку целиком нельзя: прогноз суррогата собирается не
                # разностями, там ряд накопленных едет и при неотрицательном
                # приросте, и на большую величину, чем прирост объясняет.
                if increment < 0 and abs(drop + increment) <= 1e-6 * max(1.0, drop):
                    continue
                problems.append(
                    f"{where}: накопленное {column} уменьшилось на {drop:.6g} "
                    f"при приросте {increment:.6g}"
                )
        previous[well] = {column: row[column] for column in ("WLPT", "WOMT", "WWIT")}

    return problems


def _monthly_step_problems(rows: list[dict[str, Any]]) -> list[str]:
    """Проверить, что календарь выгрузки — сплошные месяцы.

    Одна строка = скважина × **месяц**: на этом стоит и наш скрининг
    (`fund_m_per_well_month`), и сам эталонный расчётчик — он считает
    скважино-месяцы строками, а не сутками. Замер 23.08 (Б28): тот же годовой
    объём, разложенный на кварталы, даёт OPEX фонда 0,333 вместо 1,0 млн руб. в
    год и ЧДД на 1,7 % выше, а на годовом шаге — 0,083 млн и +2,3 %. Ни ошибки,
    ни предупреждения при этом не возникает нигде: деньги просто становятся
    другими.

    В валидационном кейсе (ТЗ п. 4.3) шаг отчётности задают организаторы. Если
    он не месячный, выгрузку нужно привести к месяцам **до** экономики, а не
    считать как есть; эта проверка не даёт пропустить такой кейс молча.
    Ограничение сюда же: `aios.balance.days_of` переводит суточные лимиты в
    объём по числу присутствующих месяцев и на квартальном шаге занижает его
    втрое (246 суток вместо 730).
    """
    stamps = sorted({str(row["DATA"])[:7] for row in rows if row.get("DATA")})
    if len(stamps) < 2:
        return []

    gaps: list[str] = []
    for previous, following in zip(stamps, stamps[1:], strict=False):
        year, month = int(previous[:4]), int(previous[5:7])
        month += 1
        if month == 13:
            year, month = year + 1, 1
        if f"{year}-{month:02d}" != following:
            gaps.append(f"{previous} → {following}")

    if not gaps:
        return []
    shown = ", ".join(gaps[:5])
    more = f" и ещё {len(gaps) - 5}" if len(gaps) > 5 else ""
    return [
        f"шаг выгрузки не месячный (разрывов календаря {len(gaps)}: {shown}{more}); "
        "экономика считает строку за скважино-месяц, поэтому такую выгрузку надо "
        "привести к месяцам до расчёта ЧДД"
    ]


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
