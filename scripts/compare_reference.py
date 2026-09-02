"""Сверка нашего расчёта базового сценария с эталонами организаторов.

    python scripts/compare_reference.py --charge-initial-pump
    python scripts/compare_reference.py --reference "docs/reference/... .csv"
    python scripts/compare_reference.py --brief

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
from typing import Any

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
                        help="год начала экономического расчёта; обязан совпадать с эталоном")
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


def coverage_problem(table: reference.ReferenceTable,
                     annual: list[dict[str, Any]]) -> str | None:
    """Вернуть причину, по которой горизонт нашего расчёта не сравним с эталоном.

    Первые пустые годы эталона допустимы: в Модели Z выгрузка начинается лишь
    в 1994, хотя дисконтирование организаторы ведут с 1991. Но пропуск года с
    деньгами (или лишний год с нашей стороны) делает итоговую разность ЧДД
    бессмысленной: `reference.compare` построчно его пропустит, а итог книги —
    нет. Пересчитывать эталон своей формулой нельзя.
    """

    reference_years = set(table.years)
    our_years = {int(item["year"]) for item in annual}
    extra = sorted(our_years - reference_years)
    # Коэффициент дисконтирования сам по себе ненулевой до начала выгрузки и
    # не является денежным потоком. Остальные сопоставляемые строки и
    # накопленный ЧДД должны быть нулевыми в пропущенном году.
    checked_rows = (*reference.ROW_TO_ANNUAL, "Накопленный ЧДД")
    missing_nonzero = [
        year for year in sorted(reference_years - our_years)
        if any(abs(table.rows.get(label, {}).get(year, 0.0)) > 1e-12
               for label in checked_rows
               if label != "Коэффициент дисконтирования")
    ]
    parts = []
    if extra:
        parts.append("в нашей выгрузке есть годы вне эталона: " + ", ".join(map(str, extra)))
    if missing_nonzero:
        parts.append("в эталоне есть ненулевые годы, которых нет в нашей выгрузке: "
                     + ", ".join(map(str, missing_nonzero)))
    return "; ".join(parts) if parts else None


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
        # ЧДД эталона хранится отдельно: годовые строки пояснения не должны
        # менять числовой итог, который возвращается для сравнения эталонов.
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
    if len(bases) != 1:
        print(f"раскладки рассчитаны от разных лет {sorted(bases)} — "
              "сверять их вместе нельзя, выберите одну через --reference", file=sys.stderr)
        return 2
    reference_start_year = next(iter(bases))
    if arguments.start_year is not None and arguments.start_year != reference_start_year:
        print(
            f"эталон рассчитан с {reference_start_year}, а запрошен {arguments.start_year}: "
            "для другой даты нужны раскладка или исходные rows организаторов; "
            "эталон сами не пересчитываем",
            file=sys.stderr,
        )
        return 2
    start_year = reference_start_year

    norms = norms_with_initial_pump() if arguments.charge_initial_pump else economics.NORMS_FILE
    result = economics.compute_npv_from_file(
        arguments.rows, start_date=f"{start_year}-01-01", norms_file=norms
    )
    for table in tables:
        problem = coverage_problem(table, result.annual)
        if problem is not None:
            print(f"{table.path.name}: {problem}; сверка остановлена", file=sys.stderr)
            return 2

    print(f"наш:       {arguments.rows}")
    print(f"нормативы: {norms.name}"
          f"{' (chargeInitialPump=ДА)' if arguments.charge_initial_pump else ''}")
    print(f"начало экономического расчёта: {start_year}")
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
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «Δ»
    # или кириллицы роняет скрипт уже после расчёта, на выдаче результата.
    # Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
