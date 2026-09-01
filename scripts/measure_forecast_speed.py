"""Замерить, во сколько раз суррогат быстрее гидродинамического расчёта.

    python scripts/measure_forecast_speed.py --model artifacts/surrogate.joblib \\
        --rows runs/base/rows.csv

Требование ТЗ п. 3.2.2 сформулировано как «ускоренное прогнозирование поведения
объекта разработки», и до 01.09 у нас на него не было ни одного числа: про
тривиальный суррогат в документах сказано «мгновенно против 11–13 минут ГГДМ»,
про обученный — ничего.

Меряется то же, что делает цикл: прогноз отклика на **полном** горизонте
управления и **полном** фонде — вход `predict(steps)`, выход в контракте 14
колонок. Сравнение идёт с временем полного прогона ГГДМ, которое задаётся
параметром: свои 11 минут мы подставляем по умолчанию, но число это машины и
конфигурации, а не свойство симулятора, поэтому оно печатается вместе с
источником.

Чего этот замер не значит: **ускорение имеет смысл только вместе с точностью.**
Суррогат переоценивает отклик кандидатов в 3,4–4,4 раза (`docs/experiments.md`,
24.08), и скорость без этой оговорки — подача, а не результат.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from aios import config as config_module  # noqa: E402
from aios import model as model_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402

#: Время полного прогона Модели Z на этой машине: 11 минут на четырёх процессах
#: MPI (`docs/report.md`, раздел 8). Параметр, а не константа: на другом железе и
#: другом числе рангов оно другое.
DEFAULT_GGDM_SECONDS = 660.0


def _ru(value: float) -> str:
    """Число в русском виде: разделитель тысяч — пробел, а не запятая."""
    return f"{value:,.0f}".replace(",", " ")


def measure(predict, steps, repeats: int) -> tuple[float, int]:
    """Медиана времени прогноза и число строк отклика."""
    times: list[float] = []
    rows: list[dict] = []
    for _ in range(repeats):
        started = time.perf_counter()
        rows = predict(steps)
        times.append(time.perf_counter() - started)
    return statistics.median(times), len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None,
                        help="обученный артефакт; без него меряется только бейзлайн")
    parser.add_argument("--rows", required=True,
                        help="выгрузка истории — из неё строится тривиальный суррогат")
    parser.add_argument("--config", default=str(config_module.DEFAULT_PATH))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ggdm-seconds", type=float, default=DEFAULT_GGDM_SECONDS,
                        help="время полного прогона ГГДМ, с которым сравниваем")
    args = parser.parse_args(argv)

    config = config_module.load(Path(args.config))
    schedule_file = layout(config.case.model_dir).one("SCHEDULE")
    steps = [step for step in parse_schedule(schedule_file).steps
             if step.date >= config.case.control_start]
    history = pd.read_csv(args.rows).to_dict("records")

    wells = len({control.well for step in steps for control in step.producers}
                | {control.well for step in steps for control in step.injectors})
    print(f"горизонт:  {len(steps)} шагов с {config.case.control_start}, скважин {wells}")
    print(f"история:   {args.rows} — {len(history)} строк")
    print(f"повторов:  {args.repeats}; сравнение с ГГДМ: {args.ggdm_seconds:.0f} с")
    print()

    baseline = model_module.for_cycle(neural_path=None, history=history,
                                      reference_steps=steps, fluids=config.fluids)
    seconds, rows = measure(baseline.model.predict, steps, args.repeats)
    print(f"{baseline.name}: {seconds:.2f} с на {rows} строк отклика — "
          f"быстрее ГГДМ в {_ru(args.ggdm_seconds / seconds)} раз")

    if args.model:
        trained = model_module.for_cycle(neural_path=Path(args.model), history=history,
                                         reference_steps=steps, fluids=config.fluids)
        seconds, rows = measure(trained.model.predict, steps, args.repeats)
        print(f"{trained.name}: {seconds:.2f} с на {rows} строк отклика — "
              f"быстрее ГГДМ в {_ru(args.ggdm_seconds / seconds)} раз")

    print()
    print("Оговорка, без которой число нельзя называть: ускорение имеет смысл")
    print("только вместе с точностью. Суррогат переоценивает отклик кандидатов")
    print("в 3,4–4,4 раза, поэтому заявляемое ЧДД получено расписанием, а не")
    print("прогнозом, и каждый кандидат подтверждается полным расчётом.")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «≈» роняет
    # скрипт уже после замера.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
