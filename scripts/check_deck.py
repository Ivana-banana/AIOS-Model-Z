"""Входной контроль чужого дека перед генерацией SCHEDULE.

    python scripts/check_deck.py vendor/Model_Z_OPM/Model_Z.data

Код возврата 1, если есть хотя бы один отказ, — чтобы шаг можно было поставить в
сквозной прогон и в CI, а не читать глазами. Заметки на код возврата не влияют:
они не мешают сборке, но их надо видеть.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aios import deckcheck


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path, help="корневой .DATA проверяемого дека")
    args = parser.parse_args()

    if not args.deck.exists():
        print(f"дека нет: {args.deck}")
        return 2

    print(deckcheck.report(args.deck))
    return 1 if deckcheck.check(args.deck) else 0


if __name__ == "__main__":
    sys.exit(main())
