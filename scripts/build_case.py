"""Собрать расчётный кейс из `config.toml`.

Базовый кейс — история организаторов как есть, добавлена только наша
SUMMARY-секция: в поставке Модели Z SUMMARY-файл пустой, и без своих векторов
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
from aios.deckfiles import layout as deck_layout  # noqa: E402
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

    try:
        deck = deck_layout(model_dir)
    except ValueError as error:
        print(f"Дек не найден или неполон: {error}\n"
              f"Распаковать поставку: unzip -q materials/Model_Z_final_OPM.zip "
              f'-d vendor -x "__MACOSX/*"', file=sys.stderr)
        return 1

    schedule = parse_schedule(deck.one("SCHEDULE"))
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
