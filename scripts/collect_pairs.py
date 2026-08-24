"""Пары «прогноз суррогата — полный расчёт» для конформного запаса.

    python scripts/collect_pairs.py --model artifacts/surrogate.joblib \
        --runs runs/base runs/matched-0995 runs/srg-narrow \
        --out docs/data/margin-pairs.csv

Зачем. Односторонний конформный запас (`aios.trust.OptimismMargin`, Р-72) требует
пар «что обещал суррогат и что дал эталонный расчётчик». Прогонов у нас накоплено
много, но пары из них никто не собирал: прогноз считался в момент цикла и
терялся вместе с выводом команды. Здесь прогноз считается заново — по тому самому
расписанию, которое ушло в расчёт, — а факт берётся из выгрузки того же прогона.

**Чего этот сбор не делает, и это важнее самих чисел.** Конформная гарантия
держится на обмениваемости: калибровочная выборка обязана быть из того же
распределения, что и то, к чему запас применяют. Применяем мы его к **кандидатам
оптимизатора**, а большинство наших прогонов — ручные равномерные сценарии.
Поэтому колонка `family` пишется в CSV рядом с числами, а не выводится: запас,
посчитанный по ручным сценариям, защищает от ошибки на ручных сценариях.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, pvt, summary, trust  # noqa: E402  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402
from aios.model import SurrogateLoadError, for_cycle  # noqa: E402
from aios.neural import NeuralSurrogate  # noqa: E402


def actual_npv(run_dir: Path, start_date: str, norms_file: Path) -> float | None:
    """Факт: эталонный расчётчик по выгрузке прогона. Своей формулы здесь нет."""
    rows_file = run_dir / "rows.csv"
    if not rows_file.is_file():
        return None
    rows = pd.read_csv(rows_file).to_dict("records")
    return economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                 start_date=start_date, norms_file=norms_file,
                                 name=run_dir.name).chdd_m


def predicted_npv(model, run_dir: Path, control_start, start_date: str,
                  norms_file: Path) -> float | None:
    """Прогноз: суррогат по тому же расписанию, что ушло в расчёт."""
    schedule_file = run_dir / "wells_schedule.inc"
    if not schedule_file.is_file():
        return None
    steps = [step for step in parse_schedule(schedule_file).steps
             if step.date >= control_start]
    if not steps:
        return None
    rows = model.predict(steps)
    return economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                 start_date=start_date, norms_file=norms_file,
                                 name=f"{run_dir.name}-прогноз").chdd_m


def distance_of(model, region, run_dir: Path, control_start) -> float | None:
    """Расстояние расписания до области доверия — тем же кодом, что и шлюз.

    Считается здесь, а не отдельной командой, потому что вопрос «растёт ли ошибка
    с расстоянием» проверяется только парой «остаток и расстояние» на одной
    строке. На трёх точках связь заметила машина Б (Б34); здесь их 23.

    Область лежит в артефакте (`NeuralBundle.trust_region`), а не в самой модели —
    поэтому передаётся отдельно, как это делает и `scripts/trust_gate.py`.
    """
    if region is None or not isinstance(model, NeuralSurrogate):
        return None
    schedule_file = run_dir / "wells_schedule.inc"
    if not schedule_file.is_file():
        return None
    steps = [step for step in parse_schedule(schedule_file).steps
             if step.date >= control_start]
    if not steps:
        return None
    return float(region.distance(model.control_profile(steps)).value)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="артефакт суррогата")
    parser.add_argument("--runs", nargs="+", required=True, help="каталоги прогонов")
    parser.add_argument("--family", nargs="*", default=None,
                        help="к какому семейству относится каждый прогон "
                             "(manual / search); по умолчанию manual")
    parser.add_argument("--base", default=None,
                        help="каталог базового прогона: на его выгрузке суррогат "
                             "знает состояние фонда (по умолчанию из конфига)")
    parser.add_argument("--out", default="docs/data/margin-pairs.csv",
                        help="куда писать пары; каталог docs/data/ ведётся в git, "
                             "в отличие от data/ — файл крошечный и он часть довода")
    parser.add_argument("--alpha", type=float, default=trust.DEFAULT_ALPHA)
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    # суррогату нужны те же два входа, что и в цикле: состояние фонда из выгрузки
    # базового прогона и опорное расписание дека. Без них он предскажет что
    # угодно — проверено: на пустых входах все сценарии дают одно число
    base_dir = Path(arguments.base) if arguments.base else config.case.run_dir
    deck_fluids = pvt.fluids_by_well(config.case.model_dir)
    history = summary.monthly_rows(base_dir / config.case.case_name,
                                   fluids=config.fluids,
                                   well_fluids=deck_fluids.by_well)
    if not history:
        print(f"нет выгрузки базового прогона в {base_dir}", file=sys.stderr)
        return 1
    deck_schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    reference_steps = [step for step in deck_schedule.steps
                       if step.date >= config.case.control_start]
    try:
        selected = for_cycle(neural_path=Path(arguments.model), history=history,
                             reference_steps=reference_steps, fluids=config.fluids)
    except (SurrogateLoadError, ValueError) as error:
        print(f"суррогат не загрузился: {error}", file=sys.stderr)
        return 1
    print(f"суррогат: {selected.name}")
    region = getattr(joblib.load(arguments.model), "trust_region", None)
    if region is None:
        print("области доверия в артефакте нет — расстояния не будет")

    families = arguments.family or []
    start = config.economics.start_date.isoformat()
    rows_out: list[tuple[str, float, float, str, float | None]] = []
    for index, name in enumerate(arguments.runs):
        run_dir = Path(name)
        family = families[index] if index < len(families) else "manual"
        actual = actual_npv(run_dir, start, config.economics.norms_file)
        predicted = predicted_npv(selected.model, run_dir, config.case.control_start,
                                  start, config.economics.norms_file)
        if actual is None or predicted is None:
            print(f"{run_dir.name:24} пропущен: нет выгрузки или расписания")
            continue
        distance = distance_of(selected.model, region, run_dir, config.case.control_start)
        rows_out.append((run_dir.name, predicted, actual, family, distance))
        shown = f"{distance:6.1f}" if distance is not None else "     —"
        print(f"{run_dir.name:24} прогноз {predicted:10.1f}  факт {actual:10.1f}  "
              f"остаток {predicted - actual:+9.1f}  расстояние {shown}  [{family}]")

    if not rows_out:
        print("ни одной пары не собрано", file=sys.stderr)
        return 1

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for name, predicted, actual, family, distance in rows_out:
            writer.writerow([f"{predicted:.6f}", f"{actual:.6f}", name, family,
                             "" if distance is None else f"{distance:.4f}"])
    print()
    print(f"пар собрано: {len(rows_out)} → {out}")

    for family in sorted({row[3] for row in rows_out}) + ["все"]:
        chosen = [(row[1], row[2]) for row in rows_out
                  if family == "все" or row[3] == family]
        margin = trust.OptimismMargin.fit(chosen, alpha=arguments.alpha)
        verdict = (f"{margin.value:+.1f} млн руб." if margin.calibrated
                   else f"не калиброван — {margin.reason}")
        print(f"  {family:8} пар {len(chosen):2}: {verdict}")

    ranked = [row for row in rows_out if row[4] is not None]
    if len(ranked) >= 3:
        ranked.sort(key=lambda row: row[4])
        print()
        print("остаток против расстояния до области доверия (по возрастанию расстояния):")
        for name, predicted, actual, family, distance in ranked:
            print(f"  {distance:6.2f}  {predicted - actual:+9.1f}  "
                  f"{100.0 * (predicted - actual) / actual:+6.2f} %  {name} [{family}]")
        outside = [row for row in ranked if row[4] > ranked[0][4]]
        rising = sum(1 for one, two in zip(ranked, ranked[1:], strict=False)
                     if (two[1] - two[2]) >= (one[1] - one[2]))
        print(f"  пар, где остаток не убывает с расстоянием: {rising} из {len(ranked) - 1}"
              f" (проверено на {len(outside) + 1} точках)")
    return 0



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
