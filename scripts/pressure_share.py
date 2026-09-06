"""Какая доля закачки идёт на пределе давления — то есть считается не нами.

    python scripts/pressure_share.py --rows runs/inj-1.6/rows.csv

Число из решения Р-52: скважина, упёршаяся в предел `WCONINJE`, берёт столько,
сколько пропускает призабойная зона, и считает это модель околоскважинного
пространства — место, где численные схемы OPM и тНавигатора расходятся сильнее
всего. Чем больше такой закачки, тем меньше заявляемого числа принадлежит нашему
расписанию.

Таблица Р-52 (15,9 % на истории, 44,0 % при ×1.6, 58,1 % при ×2.0) была снята
разовым разбором, и команды под неё в репозитории не осталось. Здесь она есть, и
она эти числа воспроизводит.

Считается по обе стороны фонда, и это не симметрия ради симметрии: у
нагнетательной предел `WCONINJE` **верхний** (выше не продавить), у добывающей
`WCONPROD` — **нижний** (ниже не опустить депрессию). Р-52 мерил только закачку;
добывающая сторона до 23.08 не мерялась вовсе.

Допуск `--tolerance` — на сколько атмосфер давление может не дотягивать до
предела и всё ещё считаться «на пределе». Единица воспроизводит числа Р-52;
`aios.deliverable` для своей задачи берёт 0,1 атм и даёт 15,5 % и 43,6 %.
Обе величины верны, отличается граница, и её надо называть.
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
from aios import deliverable  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout, missing_deck_refusal  # noqa: E402

#: Чем меряется режим и чем объём, по типу скважины.
RATE = {"injector": ("WWIR", "WWIT_Diff"), "producer": ("WLPR", "WLPT_Diff")}


def share(rows: pd.DataFrame, limits: dict[str, float], since: str,
          tolerance: float, kind: str = "injector") -> dict[str, float]:
    """Доли объёма и месяцев на пределе, медиана забойного давления.

    Пределы приходят из дека по каждой скважине: общего «300 атм» здесь нет —
    на чужом деке они разные, а у части скважин их не будет вовсе. Сторона
    предела зависит от типа: нагнетательная упирается сверху, добывающая снизу.
    """
    rate, volume_column = RATE[kind]
    frame = rows.loc[:, ["DATA", "well", "BHP", rate, volume_column]].copy()
    frame["well"] = frame["well"].astype(str)
    frame = frame[(frame["DATA"].astype(str) >= since) & (frame[rate] > 0)]
    frame["limit"] = frame["well"].map(limits)
    frame = frame.dropna(subset=["limit"])
    if frame.empty:
        return {"wells": 0, "months": 0, "volume_share": 0.0,
                "month_share": 0.0, "median_bhp": 0.0, "volume": 0.0}
    at_limit = (frame["BHP"] >= frame["limit"] - tolerance) if kind == "injector"         else (frame["BHP"] <= frame["limit"] + tolerance)
    volume = float(frame[volume_column].sum())
    return {
        "wells": int(frame["well"].nunique()),
        "months": int(len(frame)),
        "volume_share": float(frame.loc[at_limit, volume_column].sum() / volume * 100)
        if volume else 0.0,
        "month_share": float(at_limit.mean() * 100),
        "median_bhp": float(frame["BHP"].median()),
        "volume": volume,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, nargs="+",
                        help="одна или несколько выгрузок rows.csv")
    parser.add_argument("--since", default="",
                        help="с какой даты считать; по умолчанию — control_start конфига")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="допуск до предела, атм (1.0 воспроизводит числа Р-52)")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    since = arguments.since or config.case.control_start.isoformat()
    try:
        schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        print(missing_deck_refusal(config.case.model_dir, error), file=sys.stderr)
        return 1
    assessed = deliverable.bhp_limits(schedule)
    by_kind = {
        kind: {str(well): limit for well, (own, limit) in assessed.items() if own == kind}
        for kind in RATE
    }
    if not any(by_kind.values()):
        print("в деке нет скважин с пределом забойного давления", file=sys.stderr)
        return 1

    print(f"окно: с {since}, допуск {arguments.tolerance} атм")
    for kind, title in (("injector", "нагнетательные (предел WCONINJE, сверху)"),
                        ("producer", "добывающие (предел WCONPROD, снизу)")):
        limits = by_kind[kind]
        if not limits:
            continue
        print()
        print(f"{title}: предел {sorted(set(limits.values()))} атм")
        print(f"{'выгрузка':<28} {'скв':>4} {'объём,тыс.':>12} "
              f"{'на пределе,%':>13} {'месяцев,%':>10} {'медиана BHP':>12}")
        for path in arguments.rows:
            result = share(pd.read_csv(path), limits, since, arguments.tolerance, kind)
            print(f"{Path(path).parent.name:<28} {result['wells']:>4} "
                  f"{result['volume'] / 1000:>12.0f} {result['volume_share']:>13.1f} "
                  f"{result['month_share']:>10.1f} {result['median_bhp']:>12.1f}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
