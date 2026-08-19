"""Собрать пакет расчётных кейсов под датасет суррогата (DoE).

    python scripts/build_doe.py --samples 40 --run-root runs/doe

Латинский гиперкуб по множителям дебита каждой добывающей скважины, seed — из
`config.toml`: регламент прямо запрещает неконтролируемые случайные параметры.

Покрытие печатается до прогонов, а не после: сотня расчётов по несколько часов —
слишком дорогая цена за дыру в датасете.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import doe  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=40, help="сколько сценариев (по умолчанию 40)")
    parser.add_argument("--run-root", default="runs/doe", help="куда собирать кейсы")
    parser.add_argument("--low", type=float, default=doe.DEFAULT_RANGE[0])
    parser.add_argument("--high", type=float, default=doe.DEFAULT_RANGE[1])
    parser.add_argument("--level-low", type=float, default=None,
                        help="нижняя граница общего уровня отбора сценария")
    parser.add_argument("--level-high", type=float, default=None,
                        help="верхняя граница общего уровня отбора сценария")
    parser.add_argument(
        "--control-kind",
        choices=("producers", "injectors", "all", "conversions", "cyclic"),
        default="producers",
        help="какие режимы варьировать (по умолчанию только добывающие)",
    )
    parser.add_argument("--shut-low", type=float, default=None,
                        help="нижняя граница доли остановленных скважин в периоде")
    parser.add_argument("--shut-high", type=float, default=None,
                        # %% — argparse разворачивает help как формат-строку
                        help="верхняя граница доли остановленных: без этой оси на "
                             "остановку приходится около 4 %% точек на скважину")
    parser.add_argument("--periods", type=int, default=1,
                        help="на сколько периодов делить горизонт: режим меняется на их "
                             "границах, и один прогон даёт столько точек управления, "
                             "сколько в нём периодов (по умолчанию 1)")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="только покрытие, без сборки кейсов на диск")
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    control_steps = [step for step in schedule.steps if step.date >= config.case.control_start]
    if not control_steps:
        print(f"в расписании нет шагов с {config.case.control_start}", file=sys.stderr)
        return 1

    levels = None
    if (arguments.level_low is None) != (arguments.level_high is None):
        print("--level-low и --level-high задаются вместе", file=sys.stderr)
        return 1
    if arguments.level_low is not None:
        levels = (arguments.level_low, arguments.level_high)

    shut_share = None
    if (arguments.shut_low is None) != (arguments.shut_high is None):
        print("--shut-low и --shut-high задаются вместе", file=sys.stderr)
        return 1
    if arguments.shut_low is not None:
        shut_share = (arguments.shut_low, arguments.shut_high)

    scenarios = doe.design(
        control_steps,
        samples=arguments.samples,
        seed=config.run.seed,
        bounds=(arguments.low, arguments.high),
        levels=levels,
        control_kind=arguments.control_kind,
        periods=arguments.periods,
        shut_share=shut_share,
    )
    report = doe.coverage(scenarios)

    print(f"seed:            {config.run.seed} (из {config.source})")
    # для переводов диапазон свой: множитель здесь — доля от бывшего дебита
    # скважины, и ноль в нём смысла не имеет. Печатаем то, что применено
    own_range = {"conversions": doe.CONVERSION_RANGE, "cyclic": doe.CYCLIC_AMPLITUDE}
    effective = own_range.get(arguments.control_kind, (arguments.low, arguments.high)) if (
        (arguments.low, arguments.high) == doe.DEFAULT_RANGE
    ) else (arguments.low, arguments.high)
    print(f"диапазон:        {effective[0]} — {effective[1]}")
    print(f"управление:      {arguments.control_kind}")
    print(f"сценариев:       {report['scenarios']}")
    print(f"скважин:         {report['wells']}")
    if arguments.control_kind == "cyclic":
        # доля остановок здесь не тревожный признак, а сам режим: каждая группа
        # стоит половину времени, поэтому около 0,5 — это норма, а не провал
        halves = sorted({s.cyclic_half_period for s in scenarios})
        print(f"полупериод:      от {halves[0]} до {halves[-1]} мес.")
        print(f"доля простоя:    {report['min_shut_share']:.3f} — {report['max_shut_share']:.3f} "
              "(половина — это и есть противофаза)")
        if report["max_shut_share"] > 0.75:
            print("\nвнимание: группы простаивают больше трёх четвертей времени — "
                  "это уже не пульсация, а остановка закачки")
    elif arguments.control_kind == "conversions":
        # размах множителей здесь ничего не значит: скважина либо переведена,
        # либо её в сценарии нет вовсе. Смотреть надо на охват фонда переводами
        converted = sorted({well for s in scenarios for well in s.converted_wells})
        per_scenario = [len(s.converted_wells) for s in scenarios]
        print(f"переводов всего: {report['converted_total']}")
        print(f"в сценарии:      от {min(per_scenario)} до {max(per_scenario)}")
        print(f"разных скважин:  {len(converted)}")
        if len(converted) < 2:
            print("\nвнимание: переводится слишком мало разных скважин — "
                  "отклик соседей суррогату выучить не на чем")
    else:
        print(f"размах, мин:     {report['min_span']:.3f}")
        print(f"размах, средний: {report['mean_span']:.3f}")
        print(f"доля остановок:  {report['min_shut_share']:.3f} — {report['max_shut_share']:.3f}")

        if report["min_span"] < 0.5 * (arguments.high - arguments.low):
            print("\nвнимание: по какой-то скважине выборка покрыла меньше половины диапазона — "
                  "суррогат будет экстраполировать именно там")

    # Периоды — это и есть число точек управления с одного прогона, поэтому в
    # dry-run они печатаются поимённо: провал покрытия в отдельном периоде общий
    # размах по пакету не показывает
    if report.get("periods"):
        print(f"периодов:        {len(report['periods'])} "
              f"(столько точек управления даёт один прогон)")
        for entry in report["periods"]:
            print(f"  с {entry['from']}: размах {entry['min_span']:.3f} (мин) / "
                  f"{entry['mean_span']:.3f} (средний), "
                  f"остановок {entry['min_shut_share']:.3f}—{entry['max_shut_share']:.3f}")

    if arguments.dry_run:
        return 0

    written = doe.write_cases(
        scenarios,
        arguments.run_root,
        config.case.model_dir,
        control_start=config.case.control_start,
    )

    # Манифест: без него отклик не связать с управлением. Восстанавливать
    # множители повторным запуском генератора можно, но тогда датасет зависит от
    # того, что никто не поменял seed, диапазон и число точек, — а это ровно тот
    # молчаливый сдвиг, который потом не найти.
    manifest = Path(arguments.run_root) / "doe.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": config.run.seed,
                "samples": arguments.samples,
                "bounds": [arguments.low, arguments.high],
                "control_kind": arguments.control_kind,
                "periods": arguments.periods,
                "shut_share": list(shut_share) if shut_share else None,
                "control_start": config.case.control_start.isoformat(),
                "coverage": report,
                "scenarios": [
                    {
                        "name": s.name,
                        # Режим по периодам: сборщик датасета берёт множители
                        # последнего периода с датой ≤ месяца строки. Старые поля
                        # рядом — по ним собираются уже посчитанные пакеты
                        "periods": [
                            {
                                "from": period.start.isoformat(),
                                "multipliers": period.multipliers,
                                "injector_multipliers": period.injector_multipliers,
                            }
                            for period in s.periods
                        ],
                        "multipliers": s.multipliers,
                        "injector_multipliers": s.injector_multipliers,
                        "shut_wells": list(s.shut_wells),
                        "shut_injectors": list(s.shut_injectors),
                        # Без этого списка сборщик датасета не отличит перевод
                        # под закачку от остановки: роль скважины он берёт из
                        # опорного расписания, а там она добывающая.
                        "converted_wells": list(s.converted_wells),
                    }
                    for s in scenarios
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nманифест:       {manifest}")
    print(f"собрано кейсов: {len(written)} в {arguments.run_root}")
    print("прогон пакетом:")
    print(f"  python scripts/run_doe.py {arguments.run_root} {config.run.processes} -- "
          "--parsing-strictness=low")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
