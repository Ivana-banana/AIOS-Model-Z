#!/usr/bin/env python3
"""Пересчитать заявленное ЧДД точкой входа организаторов и сверить с документами.

Зачем отдельный скрипт, когда есть `check_package.py`. Тот проверяет, что файл
расписания на месте, что его `md5` совпадает с заявленным и что в
`docs/submission.md` вообще есть число. Чего он не проверяет — **что это число
получено из этой выгрузки**: в каталоге прогона нет машинного следа расчёта, и
заявленная тройка вписана в документы руками. Сегодня она совпадает (проверено
03.09), но гарантии на день сдачи нет, а организаторы делают **один** расчёт и
сравнивают его результат с заявленным.

Что делает этот скрипт:

1. запускает `vendor/CHDD_PYTHON/РАСЧЕТ_ЧДД.py` — **ту самую точку входа**, а не
   нашу обвязку, — на `rows.csv` прогона, с годом дисконтирования из конфига;
2. сверяет полученное число с тем, что написано в `docs/submission.md`, с
   допуском в половину последнего знака документа (см. ниже);
3. считает `md5` расписания и выгрузки, `sha256` нормативов, забирает версию
   вендора и пишет всё это в `runs/<прогон>/npv.json` — машинный след, которого
   не было;
4. возвращает 1 при любом расхождении, чтобы вызов из `check_package.py`
   становился отказом допуска, а не заметкой.

Точность сверки. Число в документах округлено до одного знака (5 579,5), поэтому
допуск сравнения — половина последнего знака: `0,05` млн руб. Совпадение до
`1e-9` требуется от другой пары — нашей обвязки и прямого запуска эталона, и
она сверяется именно так.

Запуск:

    python scripts/verify_declared.py --run runs/matched-0995
    python scripts/verify_declared.py --run runs/base --claimed 5300.520
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics  # noqa: E402

#: Заявленное число в `docs/submission.md`: строка таблицы «**ЧДД** | **5 579,5 млн руб.**».
#: Пробелы внутри числа — неразрывные и обычные, разделитель дробной части — запятая.
DECLARED = re.compile(r"\*\*ЧДД\*\*\s*\|\s*\*\*([\d   ,.]+)\s*млн руб")

#: Допуск сверки с документом: число там округлено до одного знака.
ROUNDING_TOLERANCE_M = 0.05


def parse_declared(text: str) -> float | None:
    """Заявленное число из `docs/submission.md` в млн руб."""
    match = DECLARED.search(text)
    if match is None:
        return None
    digits = (match.group(1)
              .replace(" ", "").replace(" ", "").replace(" ", "")
              .replace(",", "."))
    try:
        return float(digits)
    except ValueError:
        return None


def digest(path: Path, algorithm: str = "md5") -> str:
    """Контрольная сумма файла — по частям: выгрузка бывает в десятки мегабайт."""
    accumulator = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            accumulator.update(block)
    return accumulator.hexdigest()


def recompute(rows_file: Path, start_year: int, norms_file: Path,
              work_dir: Path) -> tuple[float, list[str]]:
    """Посчитать ЧДД точкой входа организаторов. Возвращает число и команду."""
    json_file = work_dir / "npv-vendor.json"
    command = [
        sys.executable,
        str(economics.VENDOR_DIR / "РАСЧЕТ_ЧДД.py"),
        "--input", str(rows_file),
        "--norms", str(norms_file),
        "--start-year", str(start_year),
        "--output", str(work_dir / "npv-vendor.xlsx"),
        "--json", str(json_file),
    ]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"эталонный расчётчик отказал (код {completed.returncode}):\n{completed.stderr}"
        )
    raw = json.loads(json_file.read_text(encoding="utf-8"))
    return float(raw["summary"]["totalChddM"]), command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=Path("runs/matched-0995"),
                        help="каталог прогона с rows.csv и wells_schedule.inc")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--claimed", type=float,
                        help="заявленное число; по умолчанию берётся из docs/submission.md")
    parser.add_argument("--submission", type=Path, default=Path("docs/submission.md"))
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    run_dir = arguments.run if arguments.run.is_absolute() else ROOT / arguments.run
    rows_file = run_dir / "rows.csv"
    if not rows_file.is_file():
        print(f"нет выгрузки {rows_file}: пересчитывать нечего", file=sys.stderr)
        return 1

    config = config_module.load(arguments.config)
    start_year = config.economics.start_date.year
    norms_file = config.economics.norms_file

    claimed = arguments.claimed
    source = "аргумент --claimed"
    if claimed is None:
        submission = (arguments.submission if arguments.submission.is_absolute()
                      else ROOT / arguments.submission)
        claimed = parse_declared(submission.read_text(encoding="utf-8")) \
            if submission.is_file() else None
        source = str(arguments.submission)
    if claimed is None:
        print(f"заявленное число не найдено в {source}", file=sys.stderr)
        return 1

    recomputed, command = recompute(rows_file, start_year, norms_file, run_dir)
    difference = abs(recomputed - claimed)
    agreed = difference <= ROUNDING_TOLERANCE_M

    schedule_file = run_dir / "wells_schedule.inc"
    record = {
        "run": run_dir.name,
        "chddM": recomputed,
        "claimedM": claimed,
        "claimedFrom": source,
        "differenceM": difference,
        "agreed": agreed,
        "startYear": start_year,
        "vendorVersion": economics.VENDOR_VERSION,
        "normsSha256": digest(norms_file, "sha256"),
        "rowsMd5": digest(rows_file),
        "scheduleMd5": digest(schedule_file) if schedule_file.is_file() else None,
        "command": " ".join(command),
    }
    (run_dir / "npv.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"прогон:      {run_dir}")
    print(f"выгрузка:    rows.csv, md5 {record['rowsMd5']}")
    if record["scheduleMd5"]:
        print(f"расписание:  wells_schedule.inc, md5 {record['scheduleMd5']}")
    print(f"расчётчик:   {economics.VENDOR_DIR.name}, версия {economics.VENDOR_VERSION}")
    print(f"дисконт от:  {start_year}")
    print(f"пересчёт:    {recomputed:,.3f} млн руб.")
    print(f"заявлено:    {claimed:,.3f} млн руб. ({source})")
    print(f"расхождение: {difference:,.6f} млн руб.")
    print(f"след:        {(run_dir / 'npv.json')}")
    print()
    if not agreed:
        print("РАСХОЖДЕНИЕ: заявленное число не воспроизводится этой выгрузкой. "
              "Отправлять пакет нельзя — сначала разберите, откуда взялось число.",
              file=sys.stderr)
        return 1
    print("заявленное число воспроизведено эталонным расчётчиком из этой выгрузки")
    return 0


if __name__ == "__main__":
    # печать не должна зависеть от локали консоли (п. 3.5 ТЗ): под cp1251
    # результат упал бы уже после того, как посчитан
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
