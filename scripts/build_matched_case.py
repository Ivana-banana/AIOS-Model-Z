"""Кейс, в котором закачку просят по достигнутому, а не равномерным множителем.

    python scripts/build_matched_case.py runs/matched --rows runs/inj-1.6/rows.csv \
        --production 1.5 --injection 1.6 --slack 1.05

Зачем это отдельный сценарий. В заявляемом прогоне (отбор ×1.5, закачка ×1.6)
**18 нагнетательных из 41 живут на пределе 300 атм**, и их доля в закачке —
38,8 %. У пятнадцати из них факт не дотягивает до просьбы: до пласта доходит от
0,11 до 0,84 назначенного. Для этих скважин множитель ×1.6 существует только в
файле — расход им задаёт давление, а не уставка.

Отсюда правило этого кейса: **просить столько, сколько прогон показал
достижимым**. Множитель скважины — не общий ×1.6, а `×1.6 · доля дошедшего · запас`,
и он никогда не выше общего. Доля берётся из выгрузки прогона, который шёл ровно
с тем расписанием, к которому она относится (`aios.deliverable.delivered_ratios`).

Что это меняет и что нет:

* **просьба к системе ППД падает** — довод 3 решения Р-52 усиливается числом, а
  не рассуждением;
* **доля закачки, которую определяет предел давления, а не наше расписание,
  падает** — довод 1 того же решения: тем больше числа принадлежит нам;
* **сколько воды дойдёт до пласта — вопрос к полному расчёту, а не к этому
  скрипту.** Уставка, опущенная до достигнутого, в части месяцев станет
  связывающей, и закачка там уменьшится. Насколько — показывает прогон.

Запас (`--slack`) — единственная ручка: 1.0 просит ровно достигнутое, 1.3
оставляет скважине треть сверху. Чем больше запас, тем ближе кейс к равномерному.

Настройка по выгрузке **нашего** симулятора — это подгонка под OPM, и её надо
называть. Она асимметрична и потому допустима: уставка ниже пропускной
способности даёт один и тот же расход в любом симуляторе, а разойтись они могут
только там, где расход задаёт призабойная зона, — то есть ровно там, откуда этот
кейс уходит.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import deliverable, optimise  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import find_data_file, layout  # noqa: E402
from aios.run import build_case  # noqa: E402
from aios.schedule import assert_valid  # noqa: E402


def multipliers(ratios: dict[str, float], wells: list[str], injection: float,
                slack: float) -> dict[str, float]:
    """Множитель по скважине: общий, ослабленный долей дошедшего до пласта.

    Скважина без замера (её нет в выгрузке или уставка была нулевой) сохраняет
    общий множитель: молча просить у неё меньше нельзя — про неё ничего не
    известно, а не известно, что она не возьмёт.
    """
    result: dict[str, float] = {}
    for well in wells:
        ratio = ratios.get(well)
        result[well] = injection if ratio is None else min(injection,
                                                           injection * ratio * slack)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="куда собрать кейс")
    parser.add_argument("--rows", required=True,
                        help="выгрузка прогона с равномерным множителем --injection")
    parser.add_argument("--production", type=float, default=1.5)
    parser.add_argument("--injection", type=float, default=1.6,
                        help="общий множитель, с которым считался прогон --rows")
    parser.add_argument("--slack", type=float, default=1.05,
                        help="во сколько раз просить больше достигнутого")
    parser.add_argument("--match-producers", action="store_true",
                        help="то же правило для добывающих: их предел WCONPROD снизу, "
                             "и на пределе уставка дебита тоже не работает")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    if min(arguments.production, arguments.injection, arguments.slack) < 0:
        print("множители не бывают отрицательными", file=sys.stderr)
        return 1

    config = config_module.load(arguments.config)
    model_dir = config.case.model_dir
    try:
        find_data_file(model_dir)
        schedule_file = layout(model_dir).one("SCHEDULE")
    except ValueError as error:
        print(f"Дек не читается: {error}", file=sys.stderr)
        return 1

    schedule = parse_schedule(schedule_file)
    steps = [step for step in schedule.steps if step.date >= config.case.control_start]
    if not steps:
        print(f"в деке нет шагов с {config.case.control_start}", file=sys.stderr)
        return 1

    rows = pd.read_csv(arguments.rows)
    reference = optimise.scale_injection(steps, arguments.injection)
    ratios = deliverable.delivered_ratios(
        rows, reference, since=config.case.control_start.isoformat())
    if not ratios:
        print(f"в выгрузке {arguments.rows} нет закачки с "
              f"{config.case.control_start} — не по чему считать долю", file=sys.stderr)
        return 1

    injectors = sorted({control.well for step in steps for control in step.injectors})
    chosen = multipliers(ratios, injectors, arguments.injection, arguments.slack)

    if arguments.match_producers:
        producers = sorted({control.well for step in steps for control in step.producers})
        produced = deliverable.delivered_ratios(
            rows, optimise.scale(steps, arguments.production),
            since=config.case.control_start.isoformat(), kind="producer")
        chosen_producers = multipliers(produced, producers, arguments.production,
                                       arguments.slack)
        scaled = steps
        for well, multiplier in chosen_producers.items():
            scaled = optimise.scale(scaled, multiplier, {well})
    else:
        produced, chosen_producers = {}, {}
        scaled = optimise.scale(steps, arguments.production)

    for well, multiplier in chosen.items():
        scaled = optimise.scale_injection(scaled, multiplier, {well})
    assert_valid(scaled)

    case = build_case(arguments.run_dir, model_dir, steps=scaled,
                      control_start=config.case.control_start,
                      wells=tuple(schedule.wells), case_name=config.case.case_name)

    asked_uniform = _asked(reference)
    asked_here = _asked(scaled)
    reduced = [well for well, value in chosen.items() if value < arguments.injection - 1e-9]

    print(f"дек:          {model_dir}")
    print(f"выгрузка:     {arguments.rows}")
    print(f"наших шагов:  {len(scaled)}")
    print(f"множители:    отбор ×{arguments.production}, закачка ×{arguments.injection} "
          f"с поправкой на достигнутое, запас ×{arguments.slack}")
    print(f"нагнетательных: {len(injectors)}; замер есть у {len(ratios)}; "
          f"просьба снижена у {len(reduced)}")
    if arguments.match_producers:
        cut = sum(1 for well, value in chosen_producers.items()
                  if value < arguments.production - 1e-9)
        print(f"добывающих:     {len(chosen_producers)}; замер есть у {len(produced)}; "
              f"просьба снижена у {cut}")
    print(f"просьба по закачке: {asked_here:.0f} против {asked_uniform:.0f} м³/сут·мес "
          f"равномерного ×{arguments.injection} "
          f"({asked_here / asked_uniform * 100:.1f} %)")
    print(f"кейс:         {case.data_file}")
    print()
    print(f"{'скв':>6} {'дошло':>7} {'множитель':>10}")
    for well in sorted(reduced, key=lambda name: chosen[name]):
        print(f"{well:>6} {ratios[well]:>7.2f} {chosen[well]:>10.2f}")
    print()
    print("дальше:")
    print(f"  scripts/run_flow.sh {case.data_file} {config.run.processes}")
    print(f"  python scripts/export_rows.py {arguments.run_dir}")
    return 0


def _asked(steps) -> float:
    """Сумма назначенных уставок закачки по всем шагам, м³/сут·мес."""
    return sum(control.rate or 0.0 for step in steps for control in step.injectors)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
