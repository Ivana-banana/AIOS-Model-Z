"""Заморозить внешний тест: сценарии, которых модель не увидит никогда.

    python scripts/freeze_holdout.py --dataset data/dataset-all-wide.csv \
        data/dataset-periods.csv --count 8 --seed 20260822

Выполняется **один раз**. Повторный запуск поверх существующего файла — отказ:
замороженный тест, который можно пересобрать, не заморожен. Если он устарел
(набор сценариев вырос), файл удаляется руками, и это видимое решение с записью
в `docs/experiments.md`, а не побочный эффект команды.

Зачем вообще. Мы делали десятки выборов — представление целей, набор признаков,
архитектура, четыре порога допуска — и все по одной и той же отложенной части.
После этого она измеряет не обобщение, а то, насколько хорошо мы под неё
подстроились. Разбор 4-6 требует отдельного набора, не участвующего в выборе
ничего.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import holdout as holdout_module  # noqa: E402
from scripts.train_surrogate import load_datasets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=8,
                        help="сколько сценариев заморозить")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=holdout_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    frame = load_datasets(arguments.dataset)
    scenarios = sorted(frame["scenario"].astype(str).unique())
    try:
        frozen = holdout_module.freeze(
            scenarios, seed=arguments.seed, count=arguments.count,
            note=f"из {len(scenarios)} сценариев пакетов: "
                 + ", ".join(Path(name).name for name in arguments.dataset),
        )
        holdout_module.save(frozen, arguments.out)
    except (ValueError, FileExistsError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"сценариев всего:  {len(scenarios)}")
    print(f"заморожено:       {len(frozen.scenarios)}")
    print(f"осталось на обучение и отбор: {len(scenarios) - len(frozen.scenarios)}")
    print(f"файл:             {arguments.out}")
    print("\nэти сценарии считаются один раз — когда все решения приняты")
    return 0


if __name__ == "__main__":
    sys.exit(main())
