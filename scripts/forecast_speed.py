"""Во сколько раз прогноз суррогата быстрее полного расчёта ГГДМ.

    python scripts/forecast_speed.py --model artifacts/surrogate.joblib \
        --ggdm-seconds 1123 --ggdm-source "runs/speed-0901, flow 2026.04, 4 процесса"

П. 3.2.2 ТЗ требует «ускоренного прогнозирования поведения объекта разработки»,
то есть скорость здесь — предмет требования, а не приятный побочный эффект. До
01.09 у нас не было ни одного замера: про тривиальный бейзлайн говорилось
«мгновенно», про обученный суррогат — ничего.

Меряется то же, что считает симулятор: **весь фонд на всём горизонте
управления**, от `[case].control_start` до конца расписания дека. Разбирается
время на три части, потому что смысл у них разный:

* загрузка артефакта — платится один раз за сессию;
* сам прогноз — платится за каждый вариант расписания, и именно он умножается
  на число вариантов в переборе;
* полный расчёт ГГДМ — число берётся не отсюда, а из прогона симулятора на
  **той же машине** (`--ggdm-seconds`), потому что померить его этим скриптом
  нельзя: на Windows симулятора нет вовсе.

Скрипт ничего не оптимизирует и не считает ЧДД. Он отвечает на один вопрос
жюри — «во сколько раз быстрее» — и обязан отвечать с оговоркой про точность:
ускорение имеет смысл только вместе с ней, а суррогат переоценивает отклик в
3,4–4,4 раза (`docs/experiments.md`, 24.08).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402
from aios.model import SurrogateLoadError, for_cycle  # noqa: E402
from aios.web.app import read_rows_csv  # noqa: E402


def measure(callable_, repeats: int) -> list[float]:
    """Время каждого повтора в секундах, включая прогревочный (он первый).

    Первый вызов считается вместе с остальными и печатается отдельно: у
    обученного суррогата он дороже — прогреваются BLAS и кэши sklearn, — и
    прятать его в медиану значит показывать не то время, которое платит
    пользователь за единственный прогноз.
    """
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        times.append(time.perf_counter() - started)
    return times


def report(name: str, times: list[float], rows: int, ggdm_seconds: float | None) -> dict:
    """Одна строка замера: сколько заняло и во сколько раз быстрее ГГДМ."""
    median = statistics.median(times)
    entry = {
        "модель": name,
        "строк прогноза": rows,
        "первый вызов, с": round(times[0], 3),
        "медиана, с": round(median, 3),
        "минимум, с": round(min(times), 3),
        "повторов": len(times),
    }
    if ggdm_seconds and median > 0:
        entry["быстрее ГГДМ, раз"] = round(ggdm_seconds / median, 1)
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument("--model", default=None,
                        help="путь к обученному суррогату; без него меряется только бейзлайн")
    parser.add_argument("--rows", default=None,
                        help="выгрузка для обучения бейзлайна (по умолчанию rows.csv прогона из конфига)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="сколько раз повторить прогноз (по умолчанию 5)")
    parser.add_argument("--ggdm-seconds", type=float, default=None,
                        help="время полного расчёта ГГДМ на этой же машине, секунд")
    parser.add_argument("--ggdm-source", default=None,
                        help="чем и где получено время ГГДМ — уходит в отчёт как подпись числа")
    parser.add_argument("--json", default=None, help="куда положить замер машиночитаемо")
    arguments = parser.parse_args(argv)

    if arguments.repeats < 1:
        print("повторов должно быть хотя бы один", file=sys.stderr)
        return 1

    config = config_module.load(arguments.config)
    rows_path = Path(arguments.rows) if arguments.rows else config.case.run_dir / "rows.csv"
    if not rows_path.is_file():
        print(f"нет выгрузки {rows_path}: бейзлайну не на чем учиться", file=sys.stderr)
        return 1

    history = read_rows_csv(rows_path)
    schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    steps = [step for step in schedule.steps if step.date >= config.case.control_start]
    if not steps:
        print(f"в расписании дека нет шагов с {config.case.control_start}", file=sys.stderr)
        return 1

    wells = sorted({control.well for step in steps
                    for control in step.producers + step.injectors})
    print(f"горизонт:  {steps[0].date} → {steps[-1].date}, шагов {len(steps)}")
    print(f"фонд:      {len(wells)} скважин с режимом")
    print(f"история:   {rows_path} ({len(history)} строк)")
    if arguments.ggdm_seconds:
        print(f"ГГДМ:      {arguments.ggdm_seconds:.0f} с"
              + (f" — {arguments.ggdm_source}" if arguments.ggdm_source else ""))
    print()

    measurements = []

    baseline = for_cycle(neural_path=None, history=history,
                         reference_steps=steps, fluids=config.fluids)
    times = measure(lambda: baseline.model.predict(steps), arguments.repeats)
    measurements.append(report(baseline.name, times, len(baseline.model.predict(steps)),
                               arguments.ggdm_seconds))

    if arguments.model:
        model_path = Path(arguments.model)
        started = time.perf_counter()
        try:
            trained = for_cycle(neural_path=model_path, history=history,
                                reference_steps=steps, fluids=config.fluids)
        except SurrogateLoadError as error:
            print(error, file=sys.stderr)
            return 1
        load_seconds = time.perf_counter() - started
        times = measure(lambda: trained.model.predict(steps), arguments.repeats)
        entry = report(trained.name, times, len(trained.model.predict(steps)),
                       arguments.ggdm_seconds)
        entry["загрузка артефакта, с"] = round(load_seconds, 3)
        entry["артефакт"] = str(model_path)
        measurements.append(entry)

    width = max(len(str(entry["модель"])) for entry in measurements)
    for entry in measurements:
        speedup = entry.get("быстрее ГГДМ, раз")
        print(f"{str(entry['модель']):<{width}}  медиана {entry['медиана, с']:>8.3f} с  "
              f"первый {entry['первый вызов, с']:>8.3f} с  строк {entry['строк прогноза']}"
              + (f"  быстрее ГГДМ в {speedup:.1f} раз" if speedup else ""))

    print()
    print("Скорость без точности ничего не значит: суррогат переоценивает отклик")
    print("в 3,4–4,4 раза (docs/experiments.md, 24.08), поэтому найденный на нём")
    print("оптимум подтверждается полным расчётом, а не принимается по прогнозу.")

    if arguments.json:
        payload = {
            "горизонт": [str(steps[0].date), str(steps[-1].date)],
            "шагов": len(steps),
            "скважин": len(wells),
            "ггдм_секунд": arguments.ggdm_seconds,
            "ггдм_источник": arguments.ggdm_source,
            "замеры": measurements,
        }
        Path(arguments.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"замер записан: {arguments.json}")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→» или
    # части кириллицы роняет скрипт уже после замера, на выдаче результата.
    # Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
