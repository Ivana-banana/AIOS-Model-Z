"""Сверка прогноза пластового давления с полным расчётом ГГДМ.

    python scripts/pressure_check.py runs/matched-0995 --model artifacts/surrogate.joblib

Зачем. Р-128 поставил предел по пластовому давлению (`WBP9`, колонка `THP`) и
проверку `summary.sanity_problems` на каждой строке выгрузки. На выгрузке полного
расчёта такая проверка осмысленна. В цикле МАС она попадает на **прогноз
суррогата** — а прогноз давления с ГГДМ не сверялся ни разу, и гейт стоял бы на
числе, которому нет доверия. Этот скрипт даёт число.

Меряется не «хороший ли R²», а то, что решает гейт: **совпадает ли решение**
«строка нарушает предел / не нарушает», принятое по прогнозу, с решением,
принятым по факту. Поэтому кроме ошибки печатается таблица по нескольким
кандидатам предела: ложные тревоги (прогноз кричит, факт спокоен) и — что
дороже — пропуски (факт нарушает, прогноз молчит).

Сравнение идёт по парам «скважина × месяц» одного и того же расписания: прогноз
считается на том же `wells_schedule.inc`, который посчитан симулятором, поэтому
разница — это ошибка модели, а не разница сценариев.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout, missing_deck_refusal  # noqa: E402
from aios.model import SurrogateLoadError, for_cycle  # noqa: E402
from aios.web.app import read_rows_csv  # noqa: E402

#: Кандидаты предела, на которых показывается цена ошибки прогноза. Числа взяты
#: не с потолка: 300 атм стоит в `WCONINJE` дека как забойное ограничение
#: нагнетателей, а 290 и 280 — это «предел чуть строже», на котором заявка уже
#: близка к границе (замер 01.09: 17 строк заявки выше 290 атм при нуле у базы).
INJECTOR_LIMITS = (280.0, 290.0, 300.0)

#: То же снизу для добывающих: 50 атм — забойный предел из `WCONPROD`, ниже него
#: пластовое опускаться не должно тем более.
PRODUCER_LIMITS = (40.0, 50.0, 60.0)


def pairs(fact: list[dict], predicted: list[dict]) -> list[tuple[str, str, float, float, float, float]]:
    """Совпавшие по «скважина × месяц» строки: факт и прогноз рядом.

    Возвращает кортежи `(дата, скважина, THP факта, THP прогноза, WLPR, WWIR)`.
    Роль строки берётся **по факту**: в контракте расчётчика другого признака
    добывающей и нагнетательной скважины нет, а прогнозный режим — это то самое,
    что мы и проверяем.
    """
    predicted_by_key = {(row["DATA"], row["well"]): row for row in predicted}
    matched = []
    for row in fact:
        key = (row["DATA"], row["well"])
        other = predicted_by_key.get(key)
        if other is None:
            continue
        matched.append((row["DATA"], row["well"], float(row["THP"]), float(other["THP"]),
                        float(row["WLPR"]), float(row["WWIR"])))
    return matched


def gate_table(matched: list[tuple[str, str, float, float, float, float]],
               limit: float, *, injectors: bool) -> dict[str, int]:
    """Что решил бы гейт по прогнозу против того, что решает факт.

    `injectors=True` — предел сверху (нагнетательные), иначе снизу
    (добывающие). Считаются только строки соответствующей роли и только те, где
    скважина работает: остановленная не несёт давления вовсе.
    """
    counts = {"строк": 0, "нарушает по факту": 0, "нарушает по прогнозу": 0,
              "ложных тревог": 0, "пропусков": 0}
    for _date, _well, fact_thp, predicted_thp, liquid, injection in matched:
        working = injection > 0 if injectors else liquid > 0
        if not working:
            continue
        counts["строк"] += 1
        fact_bad = fact_thp > limit if injectors else fact_thp < limit
        predicted_bad = predicted_thp > limit if injectors else predicted_thp < limit
        counts["нарушает по факту"] += int(fact_bad)
        counts["нарушает по прогнозу"] += int(predicted_bad)
        counts["ложных тревог"] += int(predicted_bad and not fact_bad)
        counts["пропусков"] += int(fact_bad and not predicted_bad)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="каталог прогона: rows.csv и wells_schedule.inc")
    parser.add_argument("--model", required=True, help="путь к обученному суррогату")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    run_dir = Path(arguments.run_dir)
    rows_path = run_dir / "rows.csv"
    schedule_path = run_dir / "wells_schedule.inc"
    for path in (rows_path, schedule_path):
        if not path.is_file():
            print(f"нет файла {path}", file=sys.stderr)
            return 1

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(arguments.config)
    fact = [row for row in read_rows_csv(rows_path)
            if row["DATA"] >= config.case.control_start.isoformat()]
    try:
        deck_schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        print(missing_deck_refusal(config.case.model_dir, error), file=sys.stderr)
        return 1
    deck_steps = [step for step in deck_schedule.steps
                  if step.date >= config.case.control_start]
    candidate = [step for step in parse_schedule(schedule_path).steps
                 if step.date >= config.case.control_start]
    if not candidate:
        print(f"в {schedule_path} нет шагов с {config.case.control_start}", file=sys.stderr)
        return 1

    try:
        selected = for_cycle(neural_path=Path(arguments.model), history=fact,
                             reference_steps=deck_steps, fluids=config.fluids)
    except SurrogateLoadError as error:
        print(error, file=sys.stderr)
        return 1

    predicted = selected.model.predict(candidate)
    matched = pairs(fact, predicted)
    if not matched:
        print("ни одна строка прогноза не совпала с фактом по «скважина × месяц»",
              file=sys.stderr)
        return 1

    print(f"прогон:    {run_dir}")
    print(f"суррогат:  {selected.name} ({arguments.model})")
    print(f"пар:       {len(matched)} из {len(fact)} строк факта")
    print()

    errors = [predicted_thp - fact_thp for _d, _w, fact_thp, predicted_thp, _l, _i in matched]
    absolute = sorted(abs(value) for value in errors)
    print("Ошибка прогноза пластового давления, атм (прогноз минус факт):")
    print(f"  средняя по модулю   {statistics.fmean(absolute):8.2f}")
    print(f"  медиана смещения    {statistics.median(errors):8.2f}")
    print(f"  p95 по модулю       {absolute[int(0.95 * (len(absolute) - 1))]:8.2f}")
    print(f"  максимум по модулю  {absolute[-1]:8.2f}")
    print()

    for injectors, limits, title in ((True, INJECTOR_LIMITS, "нагнетательные, предел сверху"),
                                     (False, PRODUCER_LIMITS, "добывающие, предел снизу")):
        print(f"Что решил бы гейт — {title}:")
        print("  предел   строк   по факту   по прогнозу   ложных тревог   пропусков")
        for limit in limits:
            counts = gate_table(matched, limit, injectors=injectors)
            print(f"  {limit:6.0f}  {counts['строк']:6d}  {counts['нарушает по факту']:9d}  "
                  f"{counts['нарушает по прогнозу']:12d}  {counts['ложных тревог']:14d}  "
                  f"{counts['пропусков']:10d}")
        print()

    print("Пропуск дороже ложной тревоги: ложная тревога отправляет кандидата на")
    print("полный расчёт, пропуск отправляет его организаторам.")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать кириллицы
    # роняет скрипт уже после замера, на выдаче результата (п. 3.5 ТЗ).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
