"""Достижим ли форсаж отбора — по помесячной выгрузке прогона.

    python scripts/forcing_check.py --rows runs/base/rows.csv --min-bhp 50 --since 2007-01-01

Отвечает не «да/нет», а тремя категориями: «предел», «есть запас» и «нельзя
сказать». Третья на Модели Z самая большая, и это результат, а не отговорка:
история почти не содержит опытов по изменению режима, поэтому подтвердить ею
достижимость форсажа нельзя.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import forcing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("runs/base/rows.csv"))
    parser.add_argument(
        "--min-bhp", type=float, required=True,
        help="нижняя уставка забойного давления из WCONPROD, атм",
    )
    parser.add_argument(
        "--since", default="",
        help="считать с этой даты (ГГГГ-ММ-ДД); раньше объект жил другой жизнью",
    )
    parser.add_argument(
        "--watercut", type=float, default=None,
        help="обводнённость объекта, доля — тогда печатается порог поглощения",
    )
    parser.add_argument("--ratio", type=float, default=None,
                        help="заявляемый множитель отбора для того же порога")
    args = parser.parse_args()

    # Отказ словами, а не FileNotFoundError: выгрузку называет человек (Р-62).
    if not args.rows.is_file():
        print(f"нет выгрузки {args.rows}: проверять форсаж не на чем. "
              "Ожидается rows.csv из scripts/export_rows.py", file=sys.stderr)
        return 1
    with args.rows.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if not args.since or str(row["DATA"]) >= args.since]

    verdicts = forcing.diagnose(rows, min_bhp_atm=args.min_bhp)
    print(forcing.summary(verdicts))

    for verdict in ("предел", "есть запас"):
        named = [one for one in verdicts if one.verdict == verdict]
        if named:
            print(f"\n«{verdict}» ({len(named)}):")
            print("\n".join(f"  {one}" for one in named[:15]))

    if args.watercut is not None and args.ratio is not None:
        absorbing = forcing.absorbing_watercut(args.watercut, args.ratio)
        print(f"\nпри обводнённости {args.watercut:.0%} и форсаже ×{args.ratio} весь прирост "
              f"жидкости станет водой уже при {absorbing:.2%} — "
              f"рост всего на {(absorbing - args.watercut) * 100:.2f} п.п.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
