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
from aios.deckfiles import norms_refusal  # noqa: E402

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


def declared_run(text: str) -> str | None:
    """Каталог прогона, который документ называет заявленным.

    Нужен ровно затем, чтобы отличить «число не воспроизводится» от «сверяется
    не тот прогон»: по паспорту дня отправки `verify_declared` гоняют по обоим
    прогонам подряд, и на базовом он до 09.09 кричал «отправлять пакет нельзя»
    — притом что базовый и не обязан давать заявленное число.
    """
    match = re.search(r"`(runs/[\w.-]+(?:/[\w.-]+)*)/wells_schedule\.inc`", text)
    if match is None:
        return None
    path = match.group(1)
    return path[: -len("/submission")] if path.endswith("/submission") else path


def portable(path: Path) -> str:
    """Путь так, как его напишет любая машина: от корня репозитория, через `/`.

    След `npv.json` отслеживается git и попадает в сдаваемый пакет, поэтому в нём
    не место ни букве диска, ни домашнему каталогу того, кто запускал последним.
    До 05.09 запись хранила `sys.executable` и абсолютные пути: файл менялся при
    каждом пересчёте на другой машине (домашний каталог Linux против пути с
    буквой диска в Windows), а
    записанная команда не запускалась ни у кого, кроме автора записи, — то есть
    поле «команда» в машинном следе заявки было невоспроизводимым (Р-227).

    Путь вне репозитория остаётся абсолютным: подменять его относительным значило
    бы записать команду, которая молча возьмёт другой файл.
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def recompute(rows_file: Path, start_year: int, norms_file: Path,
              work_dir: Path) -> tuple[float, list[str]]:
    """Посчитать ЧДД точкой входа организаторов. Возвращает число и команду.

    Возвращается команда в переносимом виде — та, которую жюри сможет повторить
    из корня репозитория. Запускается она абсолютными путями: подпроцессу не
    важно, откуда его позвали, а записи важно, чтобы её можно было выполнить.
    """
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
    recorded = [
        "python",
        portable(economics.VENDOR_DIR / "РАСЧЕТ_ЧДД.py"),
        "--input", portable(rows_file),
        "--norms", portable(norms_file),
        "--start-year", str(start_year),
        "--output", portable(work_dir / "npv-vendor.xlsx"),
        "--json", portable(json_file),
    ]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"эталонный расчётчик отказал (код {completed.returncode}):\n{completed.stderr}"
        )
    raw = json.loads(json_file.read_text(encoding="utf-8"))
    return float(raw["summary"]["totalChddM"]), recorded


def charge_initial_pump(norms_file: Path) -> bool | None:
    """Значение флага `chargeInitialPump` из нормативов — или None, если его нет.

    Флаг решает, платим ли мы за ЭЦН, уже стоящий в скважине на начало расчёта.
    На заявке это 92,9 млн руб. разницы (`BLOCKED.md`, п. 0.1), и до 04.09 след
    её не фиксировал: два расчёта одного расписания с разными значениями флага
    давали одинаково выглядящий машинный след (С9).
    """
    assumptions, _ = economics.load_norms(norms_file)
    value = assumptions.get("chargeInitialPump")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("да", "true", "yes", "1")
    return None


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

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(arguments.config)
    refusal = norms_refusal(config.economics.norms_file)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    start_year = config.economics.start_date.year
    norms_file = config.economics.norms_file

    claimed = arguments.claimed
    source = "аргумент --claimed"
    wrong_run: str | None = None
    if claimed is None:
        submission = (arguments.submission if arguments.submission.is_absolute()
                      else ROOT / arguments.submission)
        text = submission.read_text(encoding="utf-8") if submission.is_file() else ""
        claimed = parse_declared(text) if text else None
        source = portable(submission)
        declared = declared_run(text)
        if declared is not None and portable(run_dir) != declared:
            wrong_run = declared
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
        if wrong_run:
            print(f"НЕ ТОТ ПРОГОН: заявлен {wrong_run}, а сверяется "
                  f"{portable(run_dir)}. Это не расхождение заявки: у базового "
                  "и любого другого сценария своё число. Чтобы сверить его, "
                  "назовите его ключом --claimed.", file=sys.stderr)
            return 1
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
