"""Насколько суррогат предсказывает скважину, модели которой у него нет.

    python scripts/holdout_wells.py --model artifacts/surrogate.joblib \
        --data data/dataset-all-near.csv data/dataset-all-wide.csv \
               data/dataset-injection.csv data/dataset-periods.csv

**Зачем.** Трекер на КТ-2 (31.08) спросила про масштабируемость и сама назвала
слабое место: «обучаете MLP — переносим на другое месторождение, всё сломается».
Архитектурную часть ответа мы даём (`docs/scalability.md`), а вот числа не было
ни одного: разбиение при обучении идёт **только по сценариям**
(`split_scenarios`), то есть каждая скважина присутствует и в обучении, и в
отложенной части. Про скважину, которой суррогат не видел, мы не знаем ничего.

**Что здесь считается.** На одних и тех же отложенных строках сравниваются три
предсказателя. Уровень `level(скважина, месяц)` у всех трёх один и тот же — он
берётся из истории самой скважины и не обучается, поэтому доступен и для новой
скважины тоже. Различаются они только откликом на управление:

* **своя модель** — то, что есть сегодня для известной скважины. Верхняя планка;
* **фонд** — медиана откликов моделей **других** скважин на входах этой. Это
  заменитель общей модели фонда: если общая модель возможна, её качество будет
  где-то здесь. Годится именно потому, что при `features="own"` вход модели —
  семь столбцов, не зависящих ни от имени скважины, ни от их числа;
* **уровень** — отклик ноль, то есть «взять историю скважины и не пытаться
  предсказать реакцию на управление». Нижняя планка, и она честная: именно это
  остаётся, если модели нет вовсе.

**Как читать результат.** Если «фонд» близок к «своей» — общая модель возможна, и
новая скважина перестаёт выпадать из прогноза. Если «фонд» близок к «уровню» —
отклик скважино-специфичен, и общая модель не даст ничего сверх истории. Если
«фонд» хуже «уровня» — перенос вреден, и это надо сказать вслух, а не прятать.

**Чего замер не делает.** Он не обучает общую модель — он оценивает её потолок
сверху заменителем. Настоящая общая модель училась бы на объединённых строках и
могла бы оказаться лучше медианы чужих откликов; хуже — вряд ли. Поэтому вывод
«общая модель не нужна» из этого замера следует, а вывод «общая модель даст ровно
столько» — нет.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aios.neural import (  # noqa: E402
    TARGET_COLUMNS,
    _centre_features,
    _clipped_months,
    _control_matrix,
    _features,
    _raw_space,
    _response,
    _select_controls,
)
from scripts.train_surrogate import load_datasets  # noqa: E402


def roles_of(control_wells: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """Роли каждой скважины по колонкам управления: добыча, закачка или обе."""
    roles: dict[str, set[str]] = {}
    for key in control_wells:
        role, _, well = key.partition(":")
        if well:
            roles.setdefault(well, set()).add(role)
    return {well: frozenset(value) for well, value in roles.items()}


def evaluate(bundle, data: pd.DataFrame, wells: list[str], donors: int,
             seed: int, control_start: pd.Timestamp | None,
             role_matched: bool = True) -> list[dict]:
    """Три предсказателя на отложенных строках каждой названной скважины.

    `role_matched` — брать донорами только скважины той же роли. Первый прогон
    02.09 делался без этого и дал по закачке `Skill` ровно ноль: фонд — 71
    добывающая против 41 нагнетательной, случайные доноры на две трети
    добывающие, а модель добывающей отвечает по закачке нулём. Медиана такой
    смеси — тоже ноль, то есть «общая модель фонда» по закачке молча
    вырождалась в уровень. Это дефект замера, а не свойство переноса.
    """
    controls = _control_matrix(data)
    control_wells = tuple(controls.columns)
    start_month = bundle.start_month
    month_span = bundle.month_span
    centre = bundle.control_center
    targets = bundle.report.targets
    test_scenarios = set(bundle.report.test_scenarios)
    roles = roles_of(control_wells)
    rng = np.random.default_rng(seed)

    results = []
    for well in wells:
        rows = data[(data["well"] == well) & (data["scenario"].isin(test_scenarios))]
        if control_start is not None:
            rows = rows[rows["DATA"] >= control_start]
        if rows.empty:
            continue

        wide = _features(rows, controls, start_month, month_span)
        base = _centre_features(rows, centre, start_month, month_span)
        x = _select_controls(wide, control_wells, well, "own")
        x_base = _select_controls(base, control_wells, well, "own")
        months = _clipped_months(rows["DATA"], start_month, month_span)
        level = bundle.levels[well][months]
        actual = rows.loc[:, TARGET_COLUMNS].astype(float).to_numpy()

        # чужие модели: выборка доноров фиксирована seed'ом, чтобы замер повторялся
        others = [name for name in bundle.models if name != well]
        if role_matched:
            mine = roles.get(well, frozenset())
            same = [name for name in others if roles.get(name, frozenset()) & mine]
            if same:
                others = same
        chosen = (others if donors >= len(others)
                  else [others[i] for i in rng.choice(len(others), donors, replace=False)])
        responses = np.stack([
            _response(bundle.models[name], x, x_base) for name in chosen
        ])

        predictions = {
            "своя": _raw_space(level + _response(bundle.models[well], x, x_base), targets),
            "фонд": _raw_space(level + np.median(responses, axis=0), targets),
            "уровень": _raw_space(level, targets),
        }
        results.append({
            "well": well,
            "rows": int(len(rows)),
            "donors": len(chosen),
            "mae": {
                name: {
                    target: float(np.abs(predicted[:, index] - actual[:, index]).mean())
                    for index, target in enumerate(TARGET_COLUMNS)
                }
                for name, predicted in predictions.items()
            },
        })
    return results


def summarise(results: list[dict], target: str) -> dict[str, float]:
    """Средняя ошибка по скважинам и `Skill` относительно уровня.

    `Skill = 1 − MAE / MAE(уровень)`: ноль означает «не лучше истории», единица —
    идеальный отклик, отрицательное значение — «хуже, чем не предсказывать вовсе».
    Та же форма, что у `Skill_MAE` в пороге качества (`aios/holdout.py`), чтобы
    число читалось привычно.
    """
    level = float(np.mean([one["mae"]["уровень"][target] for one in results]))
    summary = {"уровень": level}
    for name in ("своя", "фонд"):
        mae = float(np.mean([one["mae"][name][target] for one in results]))
        summary[name] = mae
        summary[f"skill:{name}"] = 0.0 if level == 0 else 1.0 - mae / level
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/surrogate.joblib")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--wells", type=int, default=12,
                        help="сколько скважин отложить (0 — все)")
    parser.add_argument("--donors", type=int, default=24,
                        help="сколько чужих моделей усредняется в «фонд»")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--control-start", default="2007-01-01")
    parser.add_argument("--any-role", action="store_true",
                        help="брать донорами любые скважины, а не только той же роли")
    parser.add_argument("--out", default="artifacts/holdout-wells.json")
    arguments = parser.parse_args()

    bundle = joblib.load(arguments.model)
    try:
        data = load_datasets(arguments.data)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    data["scenario"] = data["scenario"].astype(str)
    data["well"] = data["well"].astype(str)
    data["DATA"] = pd.to_datetime(data["DATA"], errors="raise")
    control_start = (None if not arguments.control_start
                     else pd.Timestamp(arguments.control_start))

    known = sorted(bundle.models)
    rng = np.random.default_rng(arguments.seed)
    wells = (known if arguments.wells <= 0 or arguments.wells >= len(known)
             else [known[i] for i in
                   sorted(rng.choice(len(known), arguments.wells, replace=False))])

    print(f"моделей в артефакте: {len(known)}; откладываем: {len(wells)}; "
          f"доноров на скважину: {arguments.donors}")
    results = evaluate(bundle, data, wells, arguments.donors,
                       arguments.seed, control_start,
                       role_matched=not arguments.any_role)
    if not results:
        print("отложенных строк не нашлось — проверьте сценарии и --control-start")
        return 1

    rows = sum(one["rows"] for one in results)
    print(f"скважин посчитано: {len(results)}, строк: {rows}\n")
    print(f"{'цель':<12} {'уровень':>12} {'своя':>12} {'фонд':>12} "
          f"{'Skill своя':>12} {'Skill фонд':>12}")
    summary = {}
    for target in TARGET_COLUMNS:
        one = summarise(results, target)
        summary[target] = one
        print(f"{target:<12} {one['уровень']:>12.2f} {one['своя']:>12.2f} "
              f"{one['фонд']:>12.2f} {one['skill:своя']:>12.4f} "
              f"{one['skill:фонд']:>12.4f}")

    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.out).write_text(json.dumps({
        "seed": arguments.seed,
        "модель": arguments.model,
        "датасеты": sorted(arguments.data),
        "скважин": len(results),
        "строк": rows,
        "доноров": arguments.donors,
        "доноры по роли": not arguments.any_role,
        "сводка": summary,
        "по скважинам": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзамер записан: {arguments.out}")
    return 0


if __name__ == "__main__":
    # русская Windows по умолчанию отдаёт cp1251, а таблица замера печатается
    # уже после того, как всё посчитано: падать здесь — терять работу целиком
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
