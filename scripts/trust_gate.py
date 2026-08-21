"""Собрать доверенную область по датасетам DoE и замерить, что она ловит.

Замер, который эта команда обязана дать, задан разбором 4-7: **новый шлюз должен
отбраковывать кандидатов, которых покоординатный min–max пропускает**. Поэтому
скрипт не только строит область, но и строит нарочно сдвинутые режимы — законные
по каждой координате в отдельности, но совместно нетипичные, — и печатает, сколько
из них проходит одну проверку и сколько другую.

    python scripts/trust_gate.py --dataset data/dataset-all-wide.csv \
        --dataset data/dataset-periods.csv --out artifacts/trust-gate.json

Пары для конформной калибровки берутся отдельным файлом: строки
`прогноз_суррогата,значение_полного_расчёта` в млн руб. Полных расчётов у нас
единицы, поэтому обычный ответ — «не калибровано», и это правильный ответ, а не
недоработка.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd

from aios import trust


def load_profiles(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Сценарные представления из датасетов DoE.

    Единица — сценарий пакета, а не строка. Имя сценария дополняется именем файла:
    пакеты нумеруют сценарии независимо, и без префикса `s01` из двух разных
    пакетов слились бы в один — то есть выборка выглядела бы вдвое гуще, чем есть.
    """
    profiles: dict[str, dict[str, float]] = {}
    for path in paths:
        frame = pd.read_csv(path, usecols=["scenario", "DATA", "well", "multiplier",
                                           "production_multiplier", "injection_multiplier"])
        frame["DATA"] = pd.to_datetime(frame["DATA"])
        for name, block in frame.groupby("scenario"):
            months: dict = {}
            for stamp, month in block.groupby("DATA"):
                controls: dict[str, float] = {}
                for row in month.itertuples():
                    production = row.production_multiplier
                    injection = row.injection_multiplier
                    if pd.isna(production) and pd.isna(injection):
                        production = row.multiplier
                    if not pd.isna(production):
                        controls[f"prod:{row.well}"] = float(production)
                    if not pd.isna(injection):
                        controls[f"inj:{row.well}"] = float(injection)
                months[stamp.date()] = controls
            profiles[f"{path.stem}:{name}"] = trust.profile(months)
    return profiles


def shifted(profiles: dict[str, dict[str, float]], *, seed: int,
            count: int, share: float) -> list[dict[str, float]]:
    """Законные по каждой координате, но совместно нетипичные режимы.

    Рецепт из разбора 4-7, §8, этап 4: **не выходя за min–max**, согласованно
    подтянуть большую часть управлений к одному краю. Прямоугольник такое
    пропускает по построению — каждая координата внутри своего отрезка, — а
    сценарное расстояние обязано это увидеть.
    """
    rng = random.Random(seed)
    # объединение по всем пакетам: управление, которого нет в одном пакете, в нём
    # просто не двигалось, и его нейтральная единица — такое же наблюдение
    every = sorted({key for one in profiles.values() for key in one})
    levels = [key for key in every if key.endswith("|уровень")]
    bounds = {
        key: (min(one.get(key, 1.0) for one in profiles.values()),
              max(one.get(key, 1.0) for one in profiles.values()))
        for key in levels
    }
    baseline = list(profiles.values())

    result = []
    for index in range(count):
        source = {key: baseline[index % len(baseline)].get(key, 1.0) for key in every}
        chosen = rng.sample(levels, k=max(1, int(len(levels) * share)))
        for key in chosen:
            low, high = bounds[key]
            source[key] = high if index % 2 == 0 else low
        result.append(source)
    return result


def coordinatewise(one: dict[str, float],
                   profiles: dict[str, dict[str, float]]) -> bool:
    """Прошёл бы кандидат покоординатную проверку min–max — та самая база сравнения."""
    for key, value in one.items():
        values = [other.get(key, 1.0) for other in profiles.values()]
        if value < min(values) - 1e-9 or value > max(values) + 1e-9:
            return False
    return True


def load_pairs(path: Path | None) -> list[tuple[float, float]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [(float(row[0]), float(row[1])) for row in csv.reader(handle) if row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="CSV «прогноз, полный расчёт» в млн руб. для калибровки")
    parser.add_argument("--out", type=Path, default=Path("artifacts/trust-gate.json"))
    parser.add_argument("--k", type=int, default=trust.DEFAULT_K)
    parser.add_argument("--alpha", type=float, default=trust.DEFAULT_ALPHA)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--shifted", type=int, default=200)
    parser.add_argument("--share", type=float, default=0.6,
                        help="какую долю управлений сдвигать к краю")
    args = parser.parse_args()

    profiles = load_profiles(args.dataset)
    region = trust.TrustRegion.fit(profiles, k=args.k, quantile=args.quantile)
    margin = trust.OptimismMargin.fit(load_pairs(args.pairs), alpha=args.alpha)

    print(f"сценариев {len(profiles)}, признаков {len(region.features)}, "
          f"замороженных {len(region.frozen)}")
    print(f"порог t = {region.threshold:.3f} "
          f"(квантиль {args.quantile:.2f} leave-one-out расстояний)")
    print(f"конформный запас: {margin.value if margin.calibrated else 'не калиброван'}"
          f"{'' if margin.calibrated else ' — ' + margin.reason}")

    candidates = shifted(profiles, seed=args.seed, count=args.shifted, share=args.share)
    passed_box = sum(coordinatewise(one, profiles) for one in candidates)
    passed_gate = sum(not region.distance(one).outside for one in candidates)
    caught = sum(coordinatewise(one, profiles) and region.distance(one).outside
                 for one in candidates)

    print()
    print(f"сдвинутых кандидатов {len(candidates)}")
    print(f"  прошли покоординатный min–max: {passed_box}")
    print(f"  прошли сценарное расстояние:   {passed_gate}")
    print(f"  пропущены прямоугольником и отброшены новым шлюзом: {caught}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = trust.contract(region, margin,
                             dataset_hash=";".join(sorted(path.name for path in args.dataset)))
    written["shift_audit"] = {
        "candidates": len(candidates),
        "passed_coordinatewise": passed_box,
        "passed_scenario_distance": passed_gate,
        "caught_by_the_second_gate_only": caught,
        "share_shifted": args.share,
        "seed": args.seed,
    }
    args.out.write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nартефакт доверия: {args.out}")


if __name__ == "__main__":
    main()
