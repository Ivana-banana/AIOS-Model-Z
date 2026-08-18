"""Собрать расчётный кейс из `config.toml`.

Базовый кейс — история организаторов как есть, добавлена только наша
SUMMARY-секция: `Model_Z_summary.inc` в поставке пустой, и без своих векторов
симулятор не выгрузит ничего.

    python scripts/build_case.py                 # runs/base из config.toml
    python scripts/build_case.py runs/base-2     # другой каталог прогона

Дек при этом не редактируется: собранный `.DATA` ссылается на включаемые файлы
поставки по абсолютным путям (`aios/run.py`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import find_data_file, layout  # noqa: E402
from aios.run import build_case  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="куда собрать кейс (по умолчанию [case].run_dir из конфигурации)")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH,
                        help="файл конфигурации (по умолчанию config.toml в корне)")
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    run_dir = Path(arguments.run_dir) if arguments.run_dir else config.case.run_dir
    model_dir = config.case.model_dir

    # состав дека выясняется из самого дека: на кейсе организаторов (ТЗ п. 4.3)
    # файлы называются иначе, а зашитое имя отказало бы на первом же шаге
    try:
        find_data_file(model_dir)
        schedule_file = layout(model_dir).one("SCHEDULE")
    except ValueError as error:
        print(f"Дек не читается: {error}\n"
              f"Ожидается каталог поставки с корневым .DATA: {model_dir}\n"
              f"Как распаковать поставку — README.md, раздел «Установка».",
              file=sys.stderr)
        return 1

    schedule = parse_schedule(schedule_file)
    case = build_case(run_dir, model_dir, case_name=config.case.case_name)

    print(f"конфигурация:  {config.source}")
    print(f"дек:           {model_dir}")
    print(f"скважин:       {len(schedule.wells)}")
    print(f"шагов SCHEDULE:{len(schedule.steps):>4}")
    print(f"кейс:          {case.data_file}")
    print(f"SUMMARY:       {case.summary_file}")
    print()
    print("дальше:")
    print(f"  scripts/run_flow.sh {case.data_file} {config.run.processes}")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
