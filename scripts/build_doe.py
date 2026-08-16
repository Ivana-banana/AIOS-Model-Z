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
        "--control-kind", choices=("producers", "injectors", "all"), default="producers",
        help="какие режимы варьировать (по умолчанию только добывающие)",
    )
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="только покрытие, без сборки кейсов на диск")
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    schedule = parse_schedule(config.case.model_dir / "Model_Z_sch.inc")
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

    scenarios = doe.design(
        control_steps,
        samples=arguments.samples,
        seed=config.run.seed,
        bounds=(arguments.low, arguments.high),
        levels=levels,
        control_kind=arguments.control_kind,
    )
    report = doe.coverage(scenarios)

    print(f"seed:            {config.run.seed} (из {config.source})")
    print(f"диапазон:        {arguments.low} — {arguments.high}")
    print(f"управление:      {arguments.control_kind}")
    print(f"сценариев:       {report['scenarios']}")
    print(f"скважин:         {report['wells']}")
    print(f"размах, мин:     {report['min_span']:.3f}")
    print(f"размах, средний: {report['mean_span']:.3f}")
    print(f"доля остановок:  {report['min_shut_share']:.3f} — {report['max_shut_share']:.3f}")

    if report["min_span"] < 0.5 * (arguments.high - arguments.low):
        print("\nвнимание: по какой-то скважине выборка покрыла меньше половины диапазона — "
              "суррогат будет экстраполировать именно там")

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
                "control_start": config.case.control_start.isoformat(),
                "coverage": report,
                "scenarios": [
                    {
                        "name": s.name,
                        "multipliers": s.multipliers,
                        "injector_multipliers": s.injector_multipliers,
                        "shut_wells": list(s.shut_wells),
                        "shut_injectors": list(s.shut_injectors),
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
    print(f"  for d in {arguments.run_root}/doe-*; do scripts/run_flow.sh "
          f"\"$d/{config.case.case_name}.DATA\" {config.run.processes}; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
