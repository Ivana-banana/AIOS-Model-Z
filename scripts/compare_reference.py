"""Сверка нашего расчёта базового сценария с эталоном организаторов.

    python scripts/compare_reference.py
    python scripts/compare_reference.py --rows runs/base/rows.csv --start-year 1991
    python scripts/compare_reference.py --start-year 2007 --brief

Зачем. КТ-2 требует «расчёт выполнен на базовом сценарии, полученный ЧДД сверен с
результатом организаторов». 30.08 организаторы опубликовали годовую раскладку по
Модели Z, посчитанную **в OPM Flow** — том же симуляторе, которым считаем мы.
Значит, сверять можно не только итог, но и каждое слагаемое: добычу, состав OPEX,
события, налоги и коэффициент дисконтирования по отдельности.

Число ЧДД здесь не вычисляется: наша сторона считается эталонным расчётчиком
организаторов через `aios.economics`, эталонная — читается из книги.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import economics, reference  # noqa: E402

#: Книга эталона ищется в каталоге, а не по имени: в валидационном кейсе
#: организаторы пришлют файл с другим названием, и зашитое имя отказало бы
#: ровно там, где его уже не поправить.
REFERENCE_DIR = ROOT / "docs" / "reference"
DEFAULT_ROWS = ROOT / "runs" / "base" / "rows.csv"


def default_reference() -> Path | None:
    """Единственная книга в `docs/reference/`; при неоднозначности — `None`."""

    books = sorted(REFERENCE_DIR.glob("*.xlsx"))
    return books[0] if len(books) == 1 else None


#: Что показывать в кратком режиме: итог, физика и то, из чего собран FCF.
BRIEF_LABELS = (
    "Добыча нефти",
    "Добыча жидкости",
    "Закачка",
    "Выручка от основной деятельности",
    "Полный OPEX",
    "FCF",
    "Коэффициент дисконтирования",
    "ЧДД",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=default_reference(),
                        help="книга организаторов (xlsx)")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS,
                        help="наша выгрузка в 14 колонок контракта")
    parser.add_argument("--start-year", type=int, default=None,
                        help="год базы дисконтирования; по умолчанию — как в книге")
    parser.add_argument("--brief", action="store_true",
                        help="только итоги и ключевые строки, без годовой таблицы")
    parser.add_argument("--charge-initial-pump", action="store_true",
                        help="начислять ЭЦН при первом пуске скважины (как в эталоне)")
    return parser


def norms_with_initial_pump() -> Path:
    """Копия нормативов с `chargeInitialPump = ДА`.

    Вендоренный файл не трогаем (`CLAUDE.md`): книга копируется во временный
    каталог, и одна ячейка переключается там. Ячейка ищется по имени параметра,
    а не по адресу, — иначе правка молча уедет при обновлении архива.
    """

    target = Path(tempfile.gettempdir()) / "aios-norms-initial-pump.xlsx"
    shutil.copy(economics.NORMS_FILE, target)
    book = openpyxl.load_workbook(target)
    for sheet in book.worksheets:
        for row in sheet.iter_rows():
            if row and row[0].value == "chargeInitialPump":
                row[2].value = "ДА"
                book.save(target)
                return target
    raise ValueError(
        f"в {economics.NORMS_FILE.name} не найден параметр chargeInitialPump"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.reference is None:
        print(f"в {REFERENCE_DIR} не одна книга эталона — укажите --reference",
              file=sys.stderr)
        return 2
    if not arguments.reference.is_file():
        print(f"нет книги эталона: {arguments.reference}", file=sys.stderr)
        return 2
    if not arguments.rows.is_file():
        print(f"нет выгрузки: {arguments.rows}", file=sys.stderr)
        return 2

    table = reference.load(arguments.reference)
    start_year = arguments.start_year or table.start_year
    norms = norms_with_initial_pump() if arguments.charge_initial_pump else economics.NORMS_FILE
    result = economics.compute_npv_from_file(
        arguments.rows, start_date=f"{start_year}-01-01", norms_file=norms
    )

    print(f"эталон:  {arguments.reference.name} (лист {table.sheet})")
    print(f"наш:     {arguments.rows}")
    print(f"нормативы: {norms.name}"
          f"{' (chargeInitialPump=ДА)' if arguments.charge_initial_pump else ''}")
    print(f"база дисконтирования: эталон {table.start_year}, наш расчёт {start_year}")
    if start_year != table.start_year:
        print("  ВНИМАНИЕ: базы разные — итоги несопоставимы напрямую.")
    print()

    ours = table.cumulative_chdd_m
    delta = result.chdd_m - ours
    print(f"ЧДД эталона:  {ours:12.3f} млн руб.")
    print(f"ЧДД наш:      {result.chdd_m:12.3f} млн руб.")
    print(f"расхождение:  {delta:+12.3f} млн руб. ({100 * delta / abs(ours):+.2f} %)")
    print()

    labels = BRIEF_LABELS if arguments.brief else tuple(reference.ROW_TO_ANNUAL)
    diffs = reference.compare(table, result.annual, labels=labels)

    print("итоги по показателям (сумма за все годы нашей раскладки):")
    print(f"{'показатель':38s} {'эталон':>14s} {'наш':>14s} {'Δ':>12s} {'Δ%':>8s}")
    for label in labels:
        group = [d for d in diffs if d.label == label]
        if not group:
            continue
        if label == "Коэффициент дисконтирования":
            worst = max(group, key=lambda d: abs(d.delta))
            print(f"{label:38s} {'—':>14s} {'—':>14s} "
                  f"{worst.delta:+12.2e} {'':>8s}  (худший год {worst.year})")
            continue
        ref_total = sum(d.reference for d in group)
        our_total = sum(d.ours for d in group)
        gap = our_total - ref_total
        share = f"{100 * gap / abs(ref_total):+7.2f}%" if abs(ref_total) > 1e-12 else "     — "
        print(f"{label:38s} {ref_total:14.3f} {our_total:14.3f} {gap:+12.3f} {share:>8s}")

    if not arguments.brief:
        print()
        print("ЧДД по годам:")
        print(f"{'год':>5s} {'эталон':>12s} {'наш':>12s} {'Δ':>10s} {'Δ%':>9s}")
        for diff in (d for d in diffs if d.label == "ЧДД"):
            relative = diff.relative
            share = f"{100 * relative:+8.2f}%" if relative is not None else "       — "
            print(f"{diff.year:5d} {diff.reference:12.2f} {diff.ours:12.2f} "
                  f"{diff.delta:+10.2f} {share:>9s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
