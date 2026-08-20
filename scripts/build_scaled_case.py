"""Собрать кейс с равномерным масштабированием отбора и закачки.

    python scripts/build_scaled_case.py runs/inject20 --production 1.5 --injection 2.0

Зачем отдельная команда. Лучший подтверждённый результат проекта — форсаж ×1.5
с закачкой ×2.0, ЧДД 13 642.4 млн руб. (+15,1 % к истории, 16.08). Его прогон
считался в среде, которая исчезла, и **команды, которой он собирался, в
репозитории не осталось**: сценарий строился разовым кодом. Восстановить число
без неё нельзя, а заявлять ЧДД, которого не можешь пересчитать, запрещено прямо.

Масштабирование равномерное и намеренно тупое: это не решение агентов, а точка
отсчёта, с которой они обязаны сравниваться (принцип 3 `CLAUDE.md`). Ограничение
методики в 500 м³/сут соблюдается внутри `optimise.scale`, а ноль означает
остановку, а не открытую скважину с нулевым дебитом.

История до `control_start` не трогается: масштабируются только наши шаги.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import optimise  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import find_data_file, layout  # noqa: E402
from aios.run import build_case  # noqa: E402
from aios.schedule import assert_valid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="куда собрать кейс")
    parser.add_argument("--production", type=float, default=1.0,
                        help="множитель дебита добывающих скважин")
    parser.add_argument("--injection", type=float, default=1.0,
                        help="множитель приёмистости нагнетательных")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    if arguments.production < 0 or arguments.injection < 0:
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

    scaled = optimise.scale(steps, arguments.production)
    scaled = optimise.scale_injection(scaled, arguments.injection)
    # валидатор до записи: невалидное расписание не должно доходить до файла,
    # который считает симулятор (`Р-04`)
    assert_valid(scaled)

    case = build_case(arguments.run_dir, model_dir, steps=scaled,
                      control_start=config.case.control_start,
                      wells=tuple(schedule.wells), case_name=config.case.case_name)

    print(f"дек:          {model_dir}")
    print(f"шагов истории до {config.case.control_start}: "
          f"{len(schedule.steps) - len(steps)}")
    print(f"наших шагов:  {len(scaled)}")
    print(f"множители:    отбор ×{arguments.production}, закачка ×{arguments.injection}")
    print(f"кейс:         {case.data_file}")
    print()
    print("дальше:")
    print(f"  scripts/run_flow.sh {case.data_file} {config.run.processes}")
    print(f"  python scripts/export_rows.py {arguments.run_dir}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
