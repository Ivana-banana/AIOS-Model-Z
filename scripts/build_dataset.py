"""Собрать датасет «управление → отклик» из посчитанных сценариев DoE.

Сценарий берётся, только если у него есть отметка `.doe-done` — наличия `.UNSMRY`
недостаточно. 18.08 сборщик подхватил сценарий, который в этот момент считался
заново, и положил в датасет 7107 строк вместо 38110: обрезок истории, молча, без
единой жалобы. Суррогат, обученный на такой выборке, врёт именно там, где данных
не хватило.

    python scripts/build_dataset.py --run-root runs/doe --out data/dataset.csv

Одна строка = скважина × месяц одного сценария: то же, что отдаёт симулятор в
контракте 14 колонок расчётчика, плюс столбцы управления добычей и закачкой
этой скважины в этом сценарии. Именно на этом учится
суррогат: вход — управление и состояние, выход — отклик объекта.

Множители берутся из манифеста `doe.json`, который пишет `build_doe.py`. Без
манифеста связать отклик с управлением нечем: имя каталога не несёт значений.

Сценарии, которые не досчитались, пропускаются с явным сообщением — молча
недособранный датасет хуже, чем его отсутствие.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import (  # noqa: E402
    economics,
    pvt,  # noqa: E402
    summary,
)
from aios.deck import ScheduleStep, parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402

CONTROL_COLUMNS = [
    "scenario", "multiplier", "production_multiplier", "injection_multiplier", "is_shut"
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", default="runs/doe")
    parser.add_argument("--out", default="data/dataset.csv")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    root = Path(arguments.run_root)
    manifest_path = root / "doe.json"
    if not manifest_path.is_file():
        print(f"нет манифеста {manifest_path}: пересоберите пакет через scripts/build_doe.py",
              file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    control_start = _control_start(manifest, config.case.control_start)
    controls = {
        scenario["name"]: (
            _periods(scenario),
            frozenset(scenario.get("converted_wells", ())),
        )
        for scenario in manifest["scenarios"]
    }
    reference = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    # плотности по PVT-регионам скважин; пометка печатается всегда — молча
    # посчитать чужой пласт по нашей плотности нельзя (ТЗ п. 4.3)
    deck_fluids = pvt.fluids_by_well(config.case.model_dir)
    print(f"плотности: {deck_fluids.note}")

    target = Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    lengths: dict[str, int] = {}
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTROL_COLUMNS + economics.REQUIRED_COLUMNS)
        writer.writeheader()

        for name in sorted(controls):
            case = root / name / config.case.case_name
            if not case.with_suffix(".UNSMRY").is_file():
                print(f"{name}: не досчитан, пропускаю")
                skipped += 1
                continue
            # наличия .UNSMRY недостаточно: у оборванного прогона он тоже есть, но
            # несёт кусок истории. Отметку пишет run_doe.py после успешного расчёта
            if not (root / name / ".doe-done").is_file():
                print(f"{name}: нет отметки о завершении — прогон оборван, пропускаю")
                skipped += 1
                continue

            try:
                rows = summary.monthly_rows(case, fluids=config.fluids,
                                            well_fluids=deck_fluids.by_well)
            except ValueError as error:
                # негодная выгрузка — это пропуск сценария с названной причиной,
                # а не оборванная сборка датасета на сороковом прогоне
                print(f"{name}: {error}")
                skipped += 1
                continue
            lengths[name] = len(rows)
            roles = control_roles(reference.steps, {str(row["DATA"]) for row in rows})
            problems = summary.sanity_problems(rows)
            if problems:
                print(f"{name}: санити-чеки — нарушений {len(problems)}, первое: {problems[0]}")

            periods, converted = controls[name]
            for row in rows:
                producer_multipliers, injector_multipliers = _at(periods, str(row["DATA"]))
                role = roles.get((str(row["DATA"]), row["well"]))
                converted_well = row["well"] in converted
                if str(row["DATA"])[:10] < control_start:
                    # История до начала управления у всех сценариев пакета одна и
                    # та же: сценарий меняет расписание только с control_start.
                    # Приписать ей множители — значит учить суррогат тому, что
                    # управление на отклик не влияет, на 39 % строк выборки
                    # (`docs/experiments.md`, ревизия 19.08). Пустые значения
                    # `_pivot_controls` подставит нейтральной единицей.
                    role, converted_well = None, False
                # Роль берётся из опорного расписания, и для переведённой под
                # закачку скважины она говорит «добывающая» — сценарий-то её и
                # меняет. Поэтому у переведённых пишутся обе величины: иначе
                # множитель закачки отбрасывается, колонка `inj:<скважина>` не
                # появляется, и весь пакет ВНС даёт датасет, неотличимый от
                # обычного (`aios/neural.py`, `_pivot_controls`).
                producer = (
                    producer_multipliers.get(row["well"])
                    if role == "production" or converted_well else None
                )
                injector = (
                    injector_multipliers.get(row["well"])
                    if role == "injection" or converted_well else None
                )
                # Старая одиночная колонка описывает «что с этой скважиной
                # сделали». У переведённой это закачка, а не ноль по добыче:
                # иначе `is_shut` объявил бы работающую нагнетательную скважину
                # остановленной.
                if converted_well and injector is not None:
                    multiplier = injector
                else:
                    multiplier = producer if producer is not None else injector
                writer.writerow({
                    "scenario": name,
                    "multiplier": "" if multiplier is None else multiplier,
                    "production_multiplier": "" if producer is None else producer,
                    "injection_multiplier": "" if injector is None else injector,
                    "is_shut": "" if multiplier is None else int(multiplier == 0.0),
                    **row,
                })
            written += 1
            print(f"{name}: {len(rows)} строк")

    size_mb = target.stat().st_size / 1e6
    print()
    print(f"датасет:   {target} ({size_mb:.1f} МБ)")
    print(f"сценариев: {written} собрано, {skipped} пропущено")

    # защита в глубину на случай датасета, собранного до появления отметок:
    # у сценариев одного пакета горизонт один, значит и число строк одно
    if len(set(lengths.values())) > 1:
        expected = max(set(lengths.values()), key=list(lengths.values()).count)
        odd = {name: count for name, count in lengths.items() if count != expected}
        print("\nВНИМАНИЕ: у сценариев разное число строк при одинаковом горизонте.")
        print(f"  ожидалось {expected}, отличаются: " +
              ", ".join(f"{name}={count}" for name, count in sorted(odd.items())))
        print("  это признак оборванного расчёта; такие сценарии надо пересчитать, "
              "иначе суррогат учится на обрезке истории")
    if skipped:
        print("пропущенные сценарии придётся досчитать: суррогат, обученный на дырявой "
              "выборке, врёт именно в дырах")
    return 0 if written else 1


#: Периоды управления сценария: `(с какой даты, множители добычи, множители закачки)`,
#: по возрастанию даты. У пакета со сценарным управлением период ровно один.
Periods = list[tuple[str, dict[str, float], dict[str, float]]]


def _periods(scenario: dict[str, Any]) -> Periods:
    """Разобрать управление сценария, поддерживая оба вида манифеста.

    Старый вид — одни множители на весь горизонт (`multipliers`,
    `injector_multipliers`). Новый — список `periods`, где режим меняется внутри
    сценария по датам. Смысл нового: измерено 19.08, что множитель в DoE
    постоянен по месяцам, поэтому пакет из 12 прогонов даёт суррогату **12**
    независимых точек управления, а не 457 320 строк
    (`docs/experiments.md`). Периоды умножают число точек без новых прогонов.

    Оба вида читаются одним кодом: у старого просто один период «с начала
    времён», поэтому уже посчитанные пакеты продолжают собираться как были.
    """
    listed = scenario.get("periods")
    if not listed:
        return [("", dict(scenario.get("multipliers", {}) or {}),
                 dict(scenario.get("injector_multipliers", {}) or {}))]
    periods = [
        (str(entry["from"]), dict(entry.get("multipliers", {}) or {}),
         dict(entry.get("injector_multipliers", {}) or {}))
        for entry in listed
    ]
    periods.sort(key=lambda item: item[0])
    return periods


def _control_start(manifest: dict[str, Any], fallback: date) -> str:
    """Дата начала управления пакета: из манифеста, иначе из конфига.

    Манифест главнее: он описывает, как пакет был сгенерирован на самом деле, а
    конфиг с тех пор мог измениться.
    """
    listed = manifest.get("control_start")
    return str(listed)[:10] if listed else fallback.isoformat()


def _at(periods: Periods, data: str) -> tuple[dict[str, float], dict[str, float]]:
    """Множители, действующие в этот месяц: последний период с датой ≤ месяца."""
    current = periods[0]
    for period in periods:
        if period[0] <= data:
            current = period
        else:
            break
    return current[1], current[2]


def control_roles(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...], dates: set[str]
) -> dict[tuple[str, str], str]:
    """Активная роль скважины по исходному расписанию, включая нулевой отклик."""
    ordered_steps = sorted(steps, key=lambda step: step.date)
    producer_status: dict[str, str] = {}
    injector_status: dict[str, str] = {}
    last_role: dict[str, str] = {}
    result: dict[tuple[str, str], str] = {}
    index = 0

    for value in sorted(dates):
        current = date.fromisoformat(value[:10])
        while index < len(ordered_steps) and ordered_steps[index].date <= current:
            step = ordered_steps[index]
            for control in step.producers:
                producer_status[control.well] = control.status
                if control.status == "OPEN":
                    last_role[control.well] = "production"
            for control in step.injectors:
                injector_status[control.well] = control.status
                if control.status == "OPEN":
                    last_role[control.well] = "injection"
            index += 1

        for well in producer_status.keys() | injector_status.keys():
            if producer_status.get(well) == "OPEN" and injector_status.get(well) != "OPEN":
                role = "production"
            elif injector_status.get(well) == "OPEN" and producer_status.get(well) != "OPEN":
                role = "injection"
            else:
                role = last_role.get(well)
            if role is not None:
                result[(value, well)] = role
    return result


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
