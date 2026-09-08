#!/usr/bin/env python3
"""Собрать то, что уходит трекеру: расписание за период управления и его ЧДД.

    python scripts/build_submission.py --run runs/matched-0995

Письмо организаторов от 08.09.2026 изменило **форму** сдачи, а не содержание:

> и файл управляющих последовательностей, и значение ЧДД — только за период,
> когда месторождением управляет AIOS, без исторической части.

До этого письма мы собирали `wells_schedule.inc`, подменяющий секцию SCHEDULE
целиком: история дека дословно, дальше наши шаги. Историю организаторы теперь
подставляют сами — второй её экземпляр в нашем файле означал бы повторение тех
же дат, а шаг, не превосходящий текущего времени, симулятор считает ошибкой.

Что делает эта команда:

1. режет собранное расписание прогона по `[case].control_start` и берёт **хвост**
   (`aios.run.control_tail`) — ту же обрезку, которой собирается кейс, только с
   другой стороны;
2. проверяет разрез: склейка истории и хвоста обязана дать исходный файл строка
   в строку. Это единственная гарантия, что при переносе формы ничего не
   потеряно и не удвоено — вскрытия новых скважин лежат между `DATES`, и молча
   потерять их значит получить нули за весь горизонт;
3. прогоняет по хвосту предполётную проверку `check_submission.check` — тот же
   разбор, каким файл проверяется перед отправкой;
4. считает ЧДД **точкой входа организаторов** по выгрузке прогона, с началом
   расчёта на `control_start`, и оставляет машинный след рядом с файлом.

Своей формулы ЧДД здесь нет и быть не может: единственный источник числа —
`vendor/CHDD_PYTHON/РАСЧЕТ_ЧДД.py` (`CLAUDE.md`). Окно расчёта и база
дисконтирования у эталона — **один и тот же** параметр `--start-year`: строки
раньше него в экономику не попадают, а дисконт считается от него же. Поэтому
«ЧДД за период управления» выражается ровно одним значением этого параметра, и
подбирать окно отдельно нам не нужно и нельзя.

Код возврата 1 — при любом отказе: пустой хвост, потеря строк на разрезе,
нарушение предполётной проверки, отказ эталонного расчётчика.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_submission import check  # noqa: E402

from aios import config as config_module  # noqa: E402
from aios import deck as deck_module  # noqa: E402
from aios import economics  # noqa: E402
from aios import run as run_module  # noqa: E402
from aios import schedule as schedule_module  # noqa: E402
from aios.deckfiles import (  # noqa: E402
    layout,
    missing_deck_refusal,
    norms_refusal,  # noqa: E402
)


def meaningful_lines(text: str) -> list[str]:
    """Строки, несущие содержание. Пустые на стыке разреза значения не имеют."""
    return [line for line in text.splitlines() if line.strip()]


def split_off_history(schedule_text: str, control_start: date) -> str:
    """Хвост расписания за период управления, с проверкой самого разреза.

    Отказ, а не пустая строка, в двух случаях: управляемых шагов нет вовсе
    (сдавать нечего) и разрез потерял или удвоил содержательные строки. Второе
    невозможно по построению `control_tail`, и именно поэтому проверяется здесь:
    цена ошибки — единственный расчёт организаторов.
    """
    tail = run_module.control_tail(schedule_text, control_start)
    if not tail.strip():
        raise ValueError(
            f"в расписании нет ни одного шага начиная с {control_start}: "
            "сдавать за период управления нечего. Проверьте [case].control_start "
            "и горизонт прогона"
        )
    prefix = run_module.history_prefix(schedule_text, control_start)
    if meaningful_lines(prefix) + meaningful_lines(tail) != meaningful_lines(schedule_text):
        raise ValueError(
            "разрез по дате начала управления потерял или удвоил строки: "
            f"история {len(meaningful_lines(prefix))} + хвост {len(meaningful_lines(tail))} "
            f"против исходных {len(meaningful_lines(schedule_text))}"
        )
    return tail


def digest(path: Path, algorithm: str = "md5") -> str:
    """Контрольная сумма файла по частям: выгрузка бывает в десятки мегабайт."""
    accumulator = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            accumulator.update(block)
    return accumulator.hexdigest()


def portable(path: Path) -> str:
    """Путь от корня репозитория: машинный след уходит в пакет и в git."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def reference_chdd(rows_file: Path, start_year: int, norms_file: Path,
                   work_dir: Path) -> tuple[float, list[str]]:
    """ЧДД точкой входа организаторов. Возвращает число и воспроизводимую команду."""
    json_file = work_dir / "npv-vendor.json"
    arguments = [
        "--input", rows_file, "--norms", norms_file,
        "--start-year", start_year,
        "--output", work_dir / "npv-vendor.xlsx", "--json", json_file,
    ]
    entry_point = economics.VENDOR_DIR / "РАСЧЕТ_ЧДД.py"
    command = [sys.executable, str(entry_point)] + [str(item) for item in arguments]
    recorded = ["python", portable(entry_point)] + [
        portable(item) if isinstance(item, Path) else str(item) for item in arguments
    ]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"эталонный расчётчик отказал (код {completed.returncode}):\n{completed.stderr}"
        )
    raw = json.loads(json_file.read_text(encoding="utf-8"))
    return float(raw["summary"]["totalChddM"]), recorded


#: Заявленное число в `docs/submission.md`: строка «**ЧДД** | **13 133,8 млн руб.**».
#: Разделитель дробной части — запятая, пробелы внутри числа бывают неразрывными.
DECLARED = re.compile(r"\*\*ЧДД\*\*\s*\|\s*\*\*([\d   ,.]+)\s*млн руб")

#: Допуск сверки с документом: число там округлено до одного знака.
ROUNDING_TOLERANCE_M = 0.05


def parse_declared(text: str) -> float | None:
    """Заявленное число из `docs/submission.md` в млн руб."""
    match = DECLARED.search(text)
    if match is None:
        return None
    digits = (match.group(1)
              .replace(" ", "").replace(" ", "").replace(" ", "")
              .replace(",", "."))
    try:
        return float(digits)
    except ValueError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=Path("runs/matched-0995"),
                        help="каталог прогона с wells_schedule.inc и rows.csv")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--out", type=Path, default=None,
                        help="куда положить пакет; по умолчанию <прогон>/submission")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="каталог дека для предполётной проверки; по умолчанию из конфига")
    parser.add_argument("--claimed", type=float, default=None,
                        help="заявленное число; по умолчанию из docs/submission.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_dir = arguments.run if arguments.run.is_absolute() else ROOT / arguments.run
    schedule_file = run_dir / "wells_schedule.inc"
    if not schedule_file.is_file():
        print(f"нет расписания {schedule_file}: собирать нечего", file=sys.stderr)
        return 1

    out_dir = arguments.out or (run_dir / "submission")
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    target = out_dir / "wells_schedule.inc"
    if (out_dir.resolve() == run_dir.resolve()
            or target.resolve() == schedule_file.resolve()
            or (target.exists() and target.samefile(schedule_file))):
        print("ОТКАЗ: выходной файл совпадает с расписанием исходного прогона; "
              "укажите отдельный каталог --out", file=sys.stderr)
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
    control_start = config.case.control_start
    out_dir.mkdir(parents=True, exist_ok=True)

    source = schedule_file.read_text(encoding="utf-8", errors="replace")
    try:
        tail = split_off_history(source, control_start)
    except ValueError as failure:
        print(f"отказ: {failure}", file=sys.stderr)
        return 1

    # Invalidate the previous confirmation before replacing any package input.
    # Every later refusal (or interruption) must leave an unconfirmed package.
    (out_dir / "npv.json").write_text(
        json.dumps({"agreed": False, "status": "unconfirmed"}) + "\n",
        encoding="utf-8",
    )
    target.write_text(tail, encoding="utf-8", newline="\n")

    parsed = deck_module.parse_schedule_text(tail)
    print(f"прогон:      {portable(run_dir)}")
    print(f"файл:        {portable(target)}, md5 {digest(target)}")
    print(f"период:      {parsed.steps[0].date} — {parsed.steps[-1].date}, "
          f"шагов {len(parsed.steps)}")
    print(f"история:     осталась за организаторами (до {control_start})")

    # Предполётная проверка требует дека: только он знает состав фонда и горизонт.
    # Дека может не быть — он не ведётся в git и распаковывается отдельным шагом
    # инструкции. Тогда команда отказывает, а не собирает пакет «на слово»: файл
    # уходит в единственный расчёт организаторов, и непроверенный он стоит там
    # всего результата. Сам файл к этому моменту уже записан и его можно
    # посмотреть — но пакетом он не объявляется и числа не получает.
    model_dir = arguments.model_dir or config.case.model_dir
    try:
        deck_text = layout(model_dir).one("SCHEDULE").read_text(
            encoding="utf-8", errors="replace")
    except ValueError as error:
        print(missing_deck_refusal(model_dir, error), file=sys.stderr)
        print("предполётная проверка не выполнена: пакет не собран", file=sys.stderr)
        return 1

    _, pumps = economics.load_norms(config.economics.norms_file)
    failures, notes = check(
        tail, deck_text, control_start,
        schedule_module.liquid_limit(pumps),
        # предел давления нагнетания приходит из кейса, а 300 атм — умолчание
        # Модели Z: на валидационном кейсе (ТЗ п. 4.3) число будет другим
        (schedule_module.MAX_INJECTION_BHP_ATM
         if config.limits.max_injector_bhp_atm is None
         else config.limits.max_injector_bhp_atm),
    )

    for note in notes:
        print(f"  замечание: {note}")
    for failure in failures:
        print(f"  ОТКАЗ: {failure}", file=sys.stderr)
    if failures:
        print("файл к отправке не пригоден", file=sys.stderr)
        return 1

    rows_file = run_dir / "rows.csv"
    if not rows_file.is_file():
        print(f"нет выгрузки {rows_file}: ЧДД за период управления не посчитан",
              file=sys.stderr)
        print("файл собран, числа нет", file=sys.stderr)
        return 1

    if config.economics.start_date != control_start:
        print(
            f"ОТКАЗ: окно ЧДД ({config.economics.start_date}) не совпадает с началом "
            f"управления ({control_start}). Письмо организаторов от 08.09.2026 требует "
            "число за период управления; при таком конфиге сдаваемый файл и "
            "сдаваемое число описывали бы разные периоды",
            file=sys.stderr,
        )
        return 1

    try:
        chdd_m, command = reference_chdd(
            rows_file, control_start.year, config.economics.norms_file, out_dir)
    except RuntimeError as failure:
        print(f"отказ: {failure}", file=sys.stderr)
        return 1

    # Сверка с документом. Без неё пакет был бы «числом из прогона», а заявка —
    # числом из текста, и совпадение этих двух держалось бы на внимательности.
    # Организаторы делают один расчёт и сравнивают его с заявленным (`CLAUDE.md`).
    submission_doc = ROOT / "docs" / "submission.md"
    claimed = (parse_declared(submission_doc.read_text(encoding="utf-8"))
               if submission_doc.is_file() else None)
    if arguments.claimed is not None:
        claimed = arguments.claimed
    difference = None if claimed is None else abs(chdd_m - claimed)
    agreed = difference is not None and difference <= ROUNDING_TOLERANCE_M

    record = {
        "run": run_dir.name,
        "period": f"{parsed.steps[0].date}—{parsed.steps[-1].date}",
        "controlStart": control_start.isoformat(),
        "chddM": chdd_m,
        "claimedM": claimed,
        "claimedFrom": ("аргумент --claimed" if arguments.claimed is not None
                        else portable(submission_doc)),
        "differenceM": difference,
        "agreed": agreed,
        "startYear": control_start.year,
        "vendorVersion": economics.VENDOR_VERSION,
        "normsSha256": digest(config.economics.norms_file, "sha256"),
        "rowsMd5": digest(rows_file),
        "scheduleMd5": digest(target),
        "sourceScheduleMd5": digest(schedule_file),
        "command": " ".join(command),
    }
    (out_dir / "npv.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"расчётчик:   {economics.VENDOR_DIR.name}, версия {economics.VENDOR_VERSION}")
    print(f"окно ЧДД:    с {control_start.year} года, оно же база дисконтирования")
    print(f"ЧДД:         {chdd_m:,.3f} млн руб.")
    print(f"след:        {portable(out_dir / 'npv.json')}")

    if claimed is None:
        print(f"заявленное число не найдено в {portable(submission_doc)}", file=sys.stderr)
        return 1
    print(f"в документе: {claimed:,.3f} млн руб., расхождение {difference:.3f}")
    if not agreed:
        print(f"ОТКАЗ: пересчёт {chdd_m:,.3f} против заявленного {claimed:,.3f} — "
              f"расхождение {difference:.3f} больше допуска {ROUNDING_TOLERANCE_M}",
              file=sys.stderr)
        return 1

    print("пакет за период управления собран")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
