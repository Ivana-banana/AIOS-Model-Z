"""Сверка нашего расчёта базового сценария с эталонами организаторов.

    python scripts/compare_reference.py --charge-initial-pump
    python scripts/compare_reference.py --reference "docs/reference/... .csv"
    python scripts/compare_reference.py --start-year 2007 --brief

Зачем. КТ-2 требует «расчёт выполнен на базовом сценарии, полученный ЧДД сверен с
результатом организаторов». 30.08 организаторы опубликовали годовую раскладку по
Модели Z **в двух симуляторах**, и у каждой своя роль:

- **OPM Flow** — тот же симулятор, которым считаем мы: построчная сверка контура,
  где расхождение по физике означало бы ошибку у нас;
- **тНавигатор** — симулятор, которым будут проверять сдачу: разница между двумя
  файлами организаторов и есть разрыв, который мы до сих пор не могли измерить.

Без `--reference` сверка идёт **со всеми** раскладками в `docs/reference/` подряд,
чтобы обе картины были на одном экране.

Число ЧДД здесь не вычисляется: наша сторона считается эталонным расчётчиком
организаторов через `aios.economics`, эталонная — читается из раскладки.
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

#: Раскладки эталона ищутся в каталоге, а не по имени: в валидационном кейсе
#: организаторы пришлют файлы с другими названиями, и зашитое имя отказало бы
#: ровно там, где его уже не поправить.
REFERENCE_DIR = ROOT / "docs" / "reference"
REFERENCE_SUFFIXES = (".xlsx", ".csv")
DEFAULT_ROWS = ROOT / "runs" / "base" / "rows.csv"


def discover_references() -> list[Path]:
    """Все раскладки эталона в `docs/reference/`, по имени файла."""

    return sorted(
        path
        for path in REFERENCE_DIR.glob("*")
        if path.suffix.lower() in REFERENCE_SUFFIXES
    )


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
    parser.add_argument("--reference", type=Path, default=None,
                        help="одна раскладка организаторов; по умолчанию — все из "
                             "docs/reference/")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS,
                        help="наша выгрузка в 14 колонок контракта")
    parser.add_argument("--start-year", type=int, default=None,
                        help="год базы дисконтирования; по умолчанию — как в книге")
    parser.add_argument("--brief", action="store_true",
                        help="только итоги и ключевые строки, без годовой таблицы")
    parser.add_argument("--charge-initial-pump", action="store_true",
                        help="начислять ЭЦН при первом пуске скважины (как в эталоне)")
    parser.add_argument("--explain", action="store_true",
                        help="разложить расхождение ЧДД на слагаемые и показать, "
                             "объём это или распределение по годам")
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


def report(
    table: reference.ReferenceTable,
    result: economics.NpvResult,
    *,
    labels: tuple[str, ...],
    detailed: bool,
    explain: bool = False,
) -> float:
    """Напечатать сверку с одной раскладкой. Возвращает ЧДД эталона."""

    reference_chdd_m = table.cumulative_chdd_m
    delta = result.chdd_m - reference_chdd_m
    print(f"--- {table.path.name} ({table.sheet}) ---")
    print(f"ЧДД эталона:  {reference_chdd_m:12.3f} млн руб.")
    print(f"ЧДД наш:      {result.chdd_m:12.3f} млн руб.")
    print(f"расхождение:  {delta:+12.3f} млн руб. "
          f"({100 * delta / abs(reference_chdd_m):+.2f} %)")
    print()

    diffs = reference.compare(table, result.annual, labels=labels)
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

    if explain:
        print()
        print("откуда расхождение (дисконтированные вклады слагаемых FCF):")
        parts = reference.decompose(table, result.annual)
        for label, value in sorted(parts.items(), key=lambda item: -abs(item[1])):
            print(f"  {label:38s} {value:+10.2f}")
        residual = delta - sum(parts.values())
        print(f"  {'сумма слагаемых':38s} {sum(parts.values()):+10.2f} "
              f"(невязка {residual:+.3f})")

        factors = table.series("Коэффициент дисконтирования")
        annual_by_year = {int(item["year"]): item for item in result.annual}
        oil = table.series("Добыча нефти")
        volume = sum(annual_by_year[y]["oilKt"] - oil[y]
                     for y in table.years if y in annual_by_year)
        weighted = sum((annual_by_year[y]["oilKt"] - oil[y]) * factors[y]
                       for y in table.years if y in annual_by_year)
        print()
        print("нефть: объём против распределения по годам")
        print(f"  расхождение по объёму:            {volume:+8.2f} тыс. т")
        print(f"  оно же, взвешенное дисконтом:     {weighted:+8.2f} тыс. т")
        print("  вторая величина и есть деньги: тонна раннего года весит больше")

    if detailed:
        print()
        print("ЧДД по годам:")
        print(f"{'год':>5s} {'эталон':>12s} {'наш':>12s} {'Δ':>10s} {'Δ%':>9s}")
        for diff in (d for d in diffs if d.label == "ЧДД"):
            relative = diff.relative
            share = f"{100 * relative:+8.2f}%" if relative is not None else "       — "
            print(f"{diff.year:5d} {diff.reference:12.2f} {diff.ours:12.2f} "
                  f"{diff.delta:+10.2f} {share:>9s}")
    print()
    return reference_chdd_m


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    references = [arguments.reference] if arguments.reference else discover_references()
    if not references:
        print(f"в {REFERENCE_DIR} нет ни одной раскладки эталона", file=sys.stderr)
        return 2
    for path in references:
        if not path.is_file():
            print(f"нет раскладки эталона: {path}", file=sys.stderr)
            return 2
    if not arguments.rows.is_file():
        print(f"нет выгрузки: {arguments.rows}", file=sys.stderr)
        return 2

    tables = [reference.load(path) for path in references]
    bases = {table.start_year for table in tables}
    if arguments.start_year is None and len(bases) > 1:
        print(f"раскладки дисконтируют от разных лет {sorted(bases)} — "
              "укажите --start-year явно", file=sys.stderr)
        return 2
    start_year = arguments.start_year or tables[0].start_year

    norms = norms_with_initial_pump() if arguments.charge_initial_pump else economics.NORMS_FILE
    result = economics.compute_npv_from_file(
        arguments.rows, start_date=f"{start_year}-01-01", norms_file=norms
    )

    print(f"наш:       {arguments.rows}")
    print(f"нормативы: {norms.name}"
          f"{' (chargeInitialPump=ДА)' if arguments.charge_initial_pump else ''}")
    print(f"база дисконтирования: {start_year}")
    for table in tables:
        if table.start_year != start_year:
            print(f"  ВНИМАНИЕ: {table.path.name} дисконтирует от "
                  f"{table.start_year} — итоги несопоставимы напрямую.")
    print()

    labels = BRIEF_LABELS if arguments.brief else tuple(reference.ROW_TO_ANNUAL)
    totals = {table.path.name: report(table, result, labels=labels,
                                      detailed=not arguments.brief,
                                      explain=arguments.explain)
              for table in tables}

    if len(totals) > 1:
        print("=== разрыв между эталонами организаторов ===")
        print("Это разница их собственных расчётов одного и того же сценария в")
        print("разных симуляторах — то, что наш стенд измерить не может.")
        best = min(totals.items(), key=lambda item: abs(item[1] - result.chdd_m))
        for name, value in totals.items():
            gap = result.chdd_m - value
            mark = "  ← мы ближе всего сюда" if name == best[0] else ""
            print(f"  {name:45s} {value:10.2f}   наш Δ {gap:+8.2f} "
                  f"({100 * gap / abs(value):+.2f} %){mark}")
        spread = max(totals.values()) - min(totals.values())
        low = min(totals.values())
        print(f"  разброс между эталонами: {spread:.2f} млн руб. "
              f"({100 * spread / abs(low):.2f} %)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
