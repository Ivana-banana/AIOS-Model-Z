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
from aios.deckfiles import (  # noqa: E402
    layout,
    missing_deck_refusal,
    norms_refusal,  # noqa: E402
)
from aios.economics import load_norms  # noqa: E402
from aios.margin import Norms  # noqa: E402
from aios.schedule import liquid_limit  # noqa: E402

#: Ключи, которые проверяются не `casecheck`, а своим механизмом. Список нужен,
#: чтобы вывод не создавал иллюзию полного покрытия: условие может быть объявлено
#: и работать, но здесь о нём будет сказано словами, а не числом.
CHECKED_ELSEWHERE = {
    "min_producer_bhp_atm": "уставки давления правит планировщик (_fit_pressures)",
    "max_injector_bhp_atm": "уставки давления правит планировщик (_fit_pressures)",
    "max_events_per_year": "квота мероприятий считается по расписанию (_fit_events_to_quota)",
    "controllable_wells": "возврат к опорному режиму делает цикл (aios.loop), планировщик его не видит",
    "enforce_water_balance": "баланс воды проверяется отдельно, ниже",
    "water_balance_rule": "баланс воды проверяется отдельно, ниже",
    "water_tolerance": "баланс воды проверяется отдельно, ниже",
    "max_external_water_m3_per_day": "мощность водозабора — помесячный расход, ниже",
    "water_tolerance_min": "нижняя граница коридора баланса воды — проверяется отдельно, ниже",
    "min_reservoir_pressure_atm": "пластовое давление — отклик пласта, проверяется по выгрузке",
    "max_reservoir_pressure_atm": "пластовое давление — отклик пласта, проверяется по выгрузке",
    "max_watercut": "обводнённость — отклик пласта, проверяется по выгрузке",
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


def _announced_for(check: casecheck.LimitCheck, decisions) -> bool:
    """Уступка в том же периоде. Правило общее с циклом — `casecheck`."""
    return casecheck.announced_for(check, decisions)


def verdict(conceded: list[str], declared: int) -> list[str]:
    """Последние строки протокола. Уступка обязана стоять в итоге, а не только выше.

    До 05.09 итог говорил «каждое условие, выражаемое расписанием, соблюдено»
    даже тогда, когда сотней строк выше стояло «УСТУПКА ПО ПРАВИЛУ: квота
    мероприятий превышена обязательными мероприятиями» — на репетиции 05.09 это
    было 88 мероприятий против объявленной квоты 3. Провалом уступка не
    считается и код возврата не меняет (Р-216, Р-217): планировщик объявил её за
    свой период, и породило её жёсткое ограничение объекта. Но «соблюдено» про
    превышение сказать нельзя — на приёмке процитируют последнюю строку, а не
    сто первую (Р-228).
    """
    if conceded:
        lines = [
            f"репетиция пройдена С УСТУПКОЙ: условий соблюдено — "
            f"{declared - len(conceded)}, превышено обязательными мероприятиями — "
            f"{len(conceded)} ({', '.join(conceded)}).",
            "Уступка объявлена планировщиком за свой период и потому не провал,",
            "но это превышение объявленного условия, а не его соблюдение.",
        ]
    else:
        lines = ["репетиция пройдена: каждое условие, выражаемое расписанием, соблюдено."]
    return lines + [
        "Отклик пласта — пластовое давление и обводнённость — в этот счёт не входит:",
        "расписанием его не задать, и нарушение по нему повод менять сценарий,",
        "а не срезать расписание.",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml", help="конфигурация с условиями кейса")
    parser.add_argument("--rows", required=True, help="выгрузка отклика (rows.csv)")
    parser.add_argument("--schedule", default=None,
                        help="файл расписания; по умолчанию SCHEDULE из дека конфигурации")
    args = parser.parse_args(argv)

    started = time.monotonic()
    refusal = config_module.refusal(args.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(Path(args.config))
    refusal = norms_refusal(config.economics.norms_file)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    limits = config.limits
    assumptions, _ = load_norms(config.economics.norms_file)
    norms = Norms.from_assumptions(assumptions)

    try:
        schedule_file = (Path(args.schedule) if args.schedule
                         else layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        print(missing_deck_refusal(config.case.model_dir, error), file=sys.stderr)
        return 1
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
            # нижняя граница коридора передаётся тоже: без неё скрипт печатал
            # «объявлено» и не проверял её вовсе — ровно мёртвая ручка, ради
            # поиска которых репетиция и написана (ревизия 02.09, находка 5)
            water = balance.compensation_problems(
                forecast, config.fluids, limits.water_tolerance,
                floor=limits.water_tolerance_min,
            )
        elif limits.water_balance_rule == "external_water":
            water = balance.external_water_problems(forecast, config.fluids,
                                                    limits.max_external_water_m3_per_day)
        else:
            water = balance.problems(forecast, config.fluids, limits.water_tolerance,
                                     floor=limits.water_tolerance_min)
        print(f"--- баланс воды ({limits.water_balance_rule}): "
              f"{'нарушений нет' if not water else len(water)} ---")
        for line in water[:5]:
            print(f"  {line}")
        print()

        # Мощность водозабора — отдельное условие, и оно действует **при любом
        # правиле**: правило говорит, сколько закачивать по норме, источник — что
        # столько воды физически есть. До 03.09 ручка проверялась только внутри
        # ветки `external_water`, то есть при правиле `compensation` (нашем
        # рабочем) была объявлена и не проверена — та самая мёртвая ручка, ради
        # поиска которых репетиция и написана.
        #
        # Счёт помесячный, а не годовой: расход источника накопить нельзя, и
        # годовое среднее прячет пиковый месяц (`balance.external_water_by_month`).
        source = balance.external_water_capacity_problems(
            forecast, config.fluids, limits.max_external_water_m3_per_day)
        known = ("мощность источника неизвестна"
                 if limits.max_external_water_m3_per_day is None
                 else f"мощность {limits.max_external_water_m3_per_day:,.10g} м³/сут")
        print(f"--- расход водозабора по месяцам ({known}): "
              f"{'нарушений нет' if not source else len(source)} ---")
        for line in source[:5]:
            print(f"  {line}")
        print()

    # Физическая годность самой выгрузки — до всяких условий кейса и независимо
    # от того, объявил ли он давление с обводнённостью. Репетиция звала
    # `sanity_problems` только при объявленном отклике пласта и оставляла из
    # ответа одни строки со словом «пластовое» — то есть выбрасывала всё
    # остальное: отрицательные приращения, нефть больше жидкости, дебит выше
    # предела ЭЦН. Кейс с `WLPR = 900` проходил репетицию с кодом 0, хотя на нём
    # эталонный расчётчик просто отказывается считать (`if row["WLPR"] > 500`).
    # Предел берётся из таблицы ЭЦН нормативов кейса, а не из литерала 500.
    physical = [line for line in summary.sanity_problems(
        forecast, max_liquid_rate=liquid_limit(
            load_norms(config.economics.norms_file)[1]),
    ) if "пластовое" not in line]
    print(f"--- физическая годность выгрузки: "
          f"{'нарушений нет' if not physical else len(physical)} ---")
    for line in physical[:5]:
        print(f"  {line}")
    print()

    if (limits.min_reservoir_pressure_atm is not None
            or limits.max_reservoir_pressure_atm is not None
            or limits.max_watercut is not None):
        # Пластовое давление и обводнённость расписанием не задать: это отклик
        # пласта. Проверяются по выгрузке — тем же кодом, который стоит на границе
        # «симулятор → экономика», и на том же прогнозе, что и остальные условия.
        response = [
            line for line in summary.sanity_problems(
                forecast,
                min_reservoir_pressure=limits.min_reservoir_pressure_atm,
                max_reservoir_pressure=limits.max_reservoir_pressure_atm,
            ) if "пластовое" in line
        ] + summary.case_problems(forecast, max_watercut=limits.max_watercut)
        print(f"--- отклик пласта (давление, обводнённость): "
              f"{'нарушений нет' if not response else len(response)} ---")
        for line in response[:5]:
            print(f"  {line}")
        print()

    left = casecheck.violations(after)
    print(f"срезка:       {enforced_at - read_at:.1f} с")
    print(f"всего:        {time.monotonic() - started:.1f} с")
    print()

    # Квота мероприятий — единственное условие, которое планировщик нарушает
    # осознанно: мероприятие, порождённое жёстким ограничением, не отменяется
    # (Р-139), потому что его отмена вернула бы расписание в недопустимое
    # состояние. Такая уступка печатается отдельно и провалом не считается — но
    # только если планировщик её объявил; молчаливое превышение остаётся дефектом.
    # Уступка засчитывается **за тот же период**, в котором нарушение. До 05.09
    # хватало любого объявления где угодно по горизонту: уступка, объявленная в
    # 2012-01, покрывала бы молчаливое превышение квоты в 2021-01 — то есть
    # ровно то, что абзац выше обещает считать дефектом. Период у проверки лежит
    # в `where` («2021» или «2021-01»), у решения — в тексте причины.
    conceded = [check for check in left
                if check.key in ("max_events_per_year", "max_events_per_month")
                and _announced_for(check, decisions)]
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

    if physical:
        print("ВЫГРУЗКА НЕФИЗИЧНА — считать по ней ЧДД нельзя:")
        for line in physical[:10]:
            print(f"  {line}")
        return 1

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

    for line in verdict([check.key for check in conceded], len(before)):
        print(line)
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→» или
    # части кириллицы роняет скрипт уже после счёта, на выдаче результата.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
