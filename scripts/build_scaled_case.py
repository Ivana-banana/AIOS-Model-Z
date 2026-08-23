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
import json
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

    try:
        case = build_case(arguments.run_dir, model_dir, steps=scaled,
                          control_start=config.case.control_start,
                          wells=tuple(schedule.wells), case_name=config.case.case_name)
    except ValueError as error:
        # Дек, ведущий режимы через WCONHIST/WCONINJH, мы не считаем — и это
        # законный исход. Но узнавать о нём надо строкой с ценой, а не
        # трассировкой на середине сборки (Б29).
        print(f"кейс не собран: {error}", file=sys.stderr)
        return 1

    print(f"дек:          {model_dir}")
    print(f"шагов истории до {config.case.control_start}: "
          f"{len(schedule.steps) - len(steps)}")
    print(f"наших шагов:  {len(scaled)}")
    print(f"множители:    отбор ×{arguments.production}, закачка ×{arguments.injection}")
    print(f"кейс:         {case.data_file}")

    report = optimise.scaling_report(steps, scaled, production=arguments.production,
                                     injection=arguments.injection)
    report_path = Path(arguments.run_dir) / "scaling.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"production": arguments.production, "injection": arguments.injection,
                    **report.as_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"множитель применён буквально к {report.literal} записям из "
          f"{report.records} ({100.0 * report.share:.1f} %)")
    print(f"отчёт:        {report_path}")
    for line in _honest_multiplier_notes(report):
        print(line)
    print()
    print("дальше:")
    print(f"  scripts/run_flow.sh {case.data_file} {config.run.processes}")
    print(f"  python scripts/export_rows.py {arguments.run_dir}")
    return 0


def _honest_multiplier_notes(report: optimise.ScalingReport) -> list[str]:
    """Где заявленный множитель не был применён буквально.

    Масштабирование делает с записью одну из четырёх вещей, и три из них меняют
    смысл ярлыка «×1.5».

    **Предел дебита жидкости.** Методика запрещает больший дебит, поэтому
    `optimise.scale` обрезает результат. На Модели Z это не срабатывает ни разу
    (максимальный исторический дебит 110 м³/сут, замер 21.08), но заявляемое
    число считается на **чужом** деке валидационного кейса.

    **Не уставочный режим.** У скважины под `BHP`, `THP`, `GRUP`, `ORAT`, `WRAT`
    или `RESV` поле расхода — верхний предел, а не уставка: умножив его, мы
    ослабляем ограничение, но режима не форсируем.

    **Записи без поля расхода** (`ORAT`/`WRAT`/`RESV` с пустой колонкой LRAT,
    `RESV` у нагнетательной) остаются как есть: умножать нечего.

    Печатается, а не исправляется: сдаваемый файл обязан остаться ровно тем, на
    котором посчитано заявляемое число. Проверяемой величиной служит
    `share` — она же уходит в `scaling.json` и в предполётную проверку.
    """
    notes: list[str] = []
    if report.capped:
        notes.append(
            f"внимание:     {report.capped} записей упёрлись в предел дебита жидкости — "
            "у них множитель меньше заявленного"
        )
    if report.off_mode:
        listed = ", ".join(f"{mode} ×{count}" for mode, count in sorted(report.off_mode.items()))
        notes.append(
            f"внимание:     {sum(report.off_mode.values())} записей не под уставочным "
            f"режимом ({listed}) — для них поле расхода это предел, а не уставка, "
            "и множитель режима не форсирует"
        )
    if report.without_rate:
        listed = ", ".join(f"{mode} ×{count}"
                           for mode, count in sorted(report.without_rate.items()))
        notes.append(
            f"внимание:     {sum(report.without_rate.values())} записей вовсе без поля "
            f"расхода ({listed}) — множитель к ним неприменим, записи оставлены как есть"
        )
    if not notes:
        notes.append("ни одна запись не потеряла заявленный множитель: "
                     "предел не сработал, все режимы уставочные")
    return notes


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
