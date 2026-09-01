"""Репетиция приёмки валидационного кейса: условия → ключи → срезка → проверка.

    python scripts/rehearse_case.py --config config-rehearsal.toml --rows runs/base/rows.csv

Протокол первых суток (`docs/validation-case-checklist.md`) написан, но до 01.09
ни разу не проходился с секундомером. Этот скрипт проходит его машинную часть:
берёт условия из конфигурации, реальное расписание из дека, реальный отклик из
выгрузки — и печатает по каждому условию **фактическое значение, предел, период и
решение планировщика**, до срезки и после.

Что он проверяет по существу: что объявленное ограничение действительно
применяется. Так 30.08 нашлась мёртвая ручка `max_events_per_year` — она
разбиралась конфигурацией и не читалась ни одной веткой планировщика (Р-113).
Проверка процесса такого не находит: её проходит и планировщик, который ничего не
сделал. Поэтому код возврата 1 здесь означает «после срезки условие всё ещё
нарушено», а не «расписание плохое».

Чего он не делает: не читает текст кейса и не переводит формулировку в ключ — это
работа человека, и она названа в чек-листе отдельно. Не запускает симулятор: отклик
берётся из готовой выгрузки, то есть проверяется реакция контура на условия, а не
поведение пласта.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from aios import balance, casecheck, planner, summary  # noqa: E402
from aios import config as config_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402
from aios.economics import load_norms  # noqa: E402
from aios.margin import Norms  # noqa: E402

#: Ключи, которые проверяются не `casecheck`, а своим механизмом. Список нужен,
#: чтобы вывод не создавал иллюзию полного покрытия: условие может быть объявлено
#: и работать, но здесь о нём будет сказано словами, а не числом.
CHECKED_ELSEWHERE = {
    "min_producer_bhp_atm": "уставки давления правит планировщик (_fit_pressures)",
    "max_injector_bhp_atm": "уставки давления правит планировщик (_fit_pressures)",
    "max_events_per_year": "квота мероприятий считается по расписанию (_fit_events_to_annual_quota)",
    "controllable_wells": "возврат к опорному режиму делает цикл (aios.loop), планировщик его не видит",
    "enforce_water_balance": "баланс воды проверяется отдельно, ниже",
    "water_balance_rule": "баланс воды проверяется отдельно, ниже",
    "water_tolerance": "баланс воды проверяется отдельно, ниже",
    "max_external_water_m3_per_day": "баланс воды проверяется отдельно, ниже",
    "water_tolerance_min": "нижняя граница коридора баланса воды — проверяется отдельно, ниже",
    "min_reservoir_pressure_atm": "пластовое давление — отклик пласта, проверяется по выгрузке",
    "max_reservoir_pressure_atm": "пластовое давление — отклик пласта, проверяется по выгрузке",
}


#: Чем в решениях планировщика называется срезка каждого ключа. Нужно, чтобы
#: отличить «условие выполнено своим механизмом» от «выполнено попутно чужим».
_MARKS = {
    "max_field_liquid_m3_per_day": ("жидкость", "среднесуточный отбор за год"),
    "max_field_injection_m3_per_day": ("закачка", "среднесуточная закачка за год"),
    "max_well_liquid_m3_per_day": ("потолок скважины", "жидкость"),
    "max_well_injection_m3_per_day": ("потолок скважины", "закачка"),
    "max_field_liquid_m3_per_year": ("годовой объём отбора жидкости",),
    "max_field_injection_m3_per_year": ("годовой объём закачки",),
    "max_field_oil_t_per_day": ("нефть",),
    "max_field_water_t_per_day": ("попутная вода",),
    "max_active_wells": ("предел числа действующих",),
    "forced_shut_wells": ("обязательная остановка",),
    "forced_shut_windows": ("обязательная остановка в окне",),
    "well_caps_m3_per_day": ("потолок скважины",),
    "max_events_per_year": ("квота мероприятий",),
}


def declared(limits: planner.Limits) -> list[str]:
    """Ключи, реально объявленные в конфигурации: не None, не пусто, не False."""
    names = []
    for field, value in vars(limits).items():
        if value is None or value == {} or value is False or value == ():
            continue
        if not limits.enforce_water_balance and field in (
                "water_tolerance", "water_balance_rule"):
            # правило и допуск существуют всегда, но условием становятся только
            # вместе с включённым балансом — иначе вывод обещает проверку,
            # которой не будет
            continue
        names.append(field)
    return names


def report(title: str, checks: list[casecheck.LimitCheck]) -> None:
    print(f"--- {title} ---")
    if not checks:
        print("  проверяемых числом условий нет")
    for check in checks:
        print(f"  {check}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml", help="конфигурация с условиями кейса")
    parser.add_argument("--rows", required=True, help="выгрузка отклика (rows.csv)")
    parser.add_argument("--schedule", default=None,
                        help="файл расписания; по умолчанию SCHEDULE из дека конфигурации")
    args = parser.parse_args(argv)

    started = time.monotonic()
    config = config_module.load(Path(args.config))
    limits = config.limits
    assumptions, _ = load_norms(config.economics.norms_file)
    norms = Norms.from_assumptions(assumptions)

    schedule_file = Path(args.schedule) if args.schedule \
        else layout(config.case.model_dir).one("SCHEDULE")
    steps = [step for step in parse_schedule(schedule_file).steps
             if step.date >= config.case.control_start]
    forecast = pd.read_csv(args.rows).to_dict("records")
    read_at = time.monotonic()

    print(f"конфигурация: {config.source}")
    print(f"расписание:   {schedule_file} — {len(steps)} шагов с {config.case.control_start}")
    print(f"отклик:       {args.rows} — {len(forecast)} строк")
    print(f"чтение:       {read_at - started:.1f} с")
    print()

    names = declared(limits)
    print(f"--- объявлено условий: {len(names)} ---")
    for name in names:
        note = CHECKED_ELSEWHERE.get(name)
        print(f"  {name}" + (f" — {note}" if note else ""))
    print()

    before = casecheck.check(steps, forecast, limits)
    report("до планировщика", before)

    fixed, decisions = planner.enforce(steps, forecast, norms, limits, config.fluids)
    enforced_at = time.monotonic()
    after = casecheck.check(fixed, forecast, limits)
    report("после планировщика", after)

    print(f"--- решения планировщика: {len(decisions)} ---")
    for decision in decisions[:10]:
        print(f"  {decision}")
    if len(decisions) > 10:
        print(f"  … ещё {len(decisions) - 10}")
    print()

    if limits.enforce_water_balance:
        # правило берётся из конфигурации, а не подставляется буквальным прочтением:
        # первая редакция скрипта звала `problems` всегда и печатала 29 нарушений
        # на сценарии, который выбранное правило проходит целиком
        if limits.water_balance_rule == "compensation":
            water = balance.compensation_problems(forecast, config.fluids,
                                                  limits.water_tolerance)
        elif limits.water_balance_rule == "external_water":
            water = balance.external_water_problems(forecast, config.fluids,
                                                    limits.max_external_water_m3_per_day)
        else:
            water = balance.problems(forecast, config.fluids, limits.water_tolerance)
        print(f"--- баланс воды ({limits.water_balance_rule}): "
              f"{'нарушений нет' if not water else len(water)} ---")
        for line in water[:5]:
            print(f"  {line}")
        print()

    if (limits.min_reservoir_pressure_atm is not None
            or limits.max_reservoir_pressure_atm is not None):
        # Пластовое давление расписанием не задать: это отклик пласта. Проверяется
        # по выгрузке — тем же кодом, который стоит на границе «симулятор →
        # экономика», и на том же прогнозе, что и остальные условия.
        pressure = [
            line for line in summary.sanity_problems(
                forecast,
                min_reservoir_pressure=limits.min_reservoir_pressure_atm,
                max_reservoir_pressure=limits.max_reservoir_pressure_atm,
            ) if "пластовое" in line or "атм" in line
        ]
        print(f"--- пластовое давление: "
              f"{'нарушений нет' if not pressure else len(pressure)} ---")
        for line in pressure[:5]:
            print(f"  {line}")
        print()

    left = casecheck.violations(after)
    print(f"срезка:       {enforced_at - read_at:.1f} с")
    print(f"всего:        {time.monotonic() - started:.1f} с")
    print()

    # Квота мероприятий — единственное условие, которое планировщик нарушает
    # осознанно: мероприятие, порождённое жёстким ограничением, не отменяется
    # (Р-138), потому что его отмена вернула бы расписание в недопустимое
    # состояние. Такая уступка печатается отдельно и провалом не считается — но
    # только если планировщик её объявил; молчаливое превышение остаётся дефектом.
    conceded = [check for check in left if check.key == "max_events_per_year"
                and any("вопреки квоте" in decision.reason for decision in decisions)]
    left = [check for check in left if check not in conceded]

    if conceded:
        print("УСТУПКА ПО ПРАВИЛУ (не провал): квота мероприятий превышена "
              "обязательными мероприятиями —")
        for check in conceded:
            print(f"  {check}")
        for decision in decisions:
            if "вопреки квоте" in decision.reason:
                print(f"  {decision}")
        print()

    if left:
        print("НАРУШЕНИЯ ОСТАЛИСЬ ПОСЛЕ СРЕЗКИ — условие объявлено, но не применяется:")
        for check in left:
            print(f"  {check}")
        return 1

    untouched = [one.key for one in before if one.satisfied]
    if untouched:
        print("условия, которые расписание не нарушало и до срезки "
              "(проверены, но ничего не доказывают о механизме):")
        for key in untouched:
            print(f"  {key}")

    # «Нарушалось до, соблюдено после» ещё не значит, что сработал механизм именно
    # этого условия: предел мог быть выполнен чужой срезкой. Найдено критиком
    # 01.09 на `max_active_wells` — отчёт печатал «103 → 95 соблюдено», а решений
    # по числу действующих скважин не было ни одного.
    worked = {decision.resource.split(",")[0].split("[")[0].strip() for decision in decisions}
    def marks_of(key: str) -> tuple[str, ...]:
        if key.startswith("group_liquid_caps"):
            return ("жидкость группы",)
        if key.startswith("group_injection_caps"):
            return ("закачка группы",)
        return _MARKS.get(key, (key,))

    silent = [
        one.key for one in before
        if not one.satisfied and not any(mark in resource for resource in worked
                                         for mark in marks_of(one.key))
    ]
    if silent:
        print()
        print("условия, выполнившиеся без собственных решений — их механизм "
              "репетиция НЕ проверила (предел выполнен другой срезкой):")
        for key in silent:
            print(f"  {key}")

    print("репетиция пройдена: каждое условие, выражаемое расписанием, соблюдено.")
    print("Пластовое давление в этот счёт не входит: оно отклик пласта, а не уставка,")
    print("и нарушение по нему — повод менять сценарий, а не срезать расписание.")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→» или
    # части кириллицы роняет скрипт уже после счёта, на выдаче результата.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
