"""Разрыв «наш симулятор ↔ тНавигатор» в окне, за которое заявляется число.

    python scripts/simulator_gap.py
    python scripts/simulator_gap.py --since 1991
    python scripts/simulator_gap.py --rows runs/base/rows.csv

Зачем отдельная команда. `compare_reference.py` отвечает на вопрос «сошёлся ли
наш прогон с эталоном» и считает накопленное с начала книги — с 1991. А
заявляется число за **период управления**: у эталонного расчётчика окно и база
дисконтирования это один параметр `--start-year`, и с 08.09 он равен 2007
(Р-249). Разрыв двух симуляторов, посчитанный с 1991, к заявляемому числу
отношения не имеет: 1991–2006 в него не входят вовсе, а именно там разрыв и
велик — на разбуривании он доходит до 2,7 % по нефти.

Что считается. Обе раскладки организаторов посчитаны на **одном и том же**
историческом управлении, поэтому разница между ними — это разница численных
схем, а не разных сценариев. Годовые ряды передисконтируются к началу окна их
же коэффициентом дисконтирования, то есть арифметика остаётся их.

Ключ `--rows` добавляет вторую половину: координаты режима по годам и долю
объёма, который решает **призабойная зона**, а не наша уставка (скважина на
пределе `WCONPROD`/`WCONINJE` берёт столько, сколько пропускает пласт). Это и
есть та координата, вдоль которой разрыв обязан расти, — и по ней же видно,
насколько история вообще позволяет судить о форсированном режиме.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import reference  # noqa: E402

REFERENCE_DIR = ROOT / "docs" / "reference"
REFERENCE_SUFFIXES = (".xlsx", ".csv")

#: Показатели, по которым сравниваются две схемы. Закачка здесь не для полноты:
#: она задаётся расписанием напрямую и обязана совпасть в ноль — расхождение по
#: ней означало бы разные сценарии, а не разные схемы.
VOLUMES = (("Добыча нефти", "нефть"), ("Добыча жидкости", "жидкость"),
           ("Закачка", "закачка"))


def books() -> list[reference.ReferenceTable]:
    """Все раскладки организаторов из `docs/reference/`, отсортированные по имени."""
    found = sorted(path for path in REFERENCE_DIR.glob("*")
                   if path.suffix.lower() in REFERENCE_SUFFIXES)
    return [reference.load(path) for path in found]


def npv_in_window(table: reference.ReferenceTable, since: int) -> float:
    """ЧДД за окно с `since`, продисконтированный к `since` их же коэффициентом.

    Годовой «ЧДД» в книге — уже дисконтированный к началу книги. Деление на
    коэффициент стартового года переносит базу, ничего не пересчитывая: своей
    формулы ЧДД в проекте нет и быть не может (`CLAUDE.md`).
    """
    factors = table.series("Коэффициент дисконтирования")
    if since not in factors:
        raise ValueError(f"в книге {table.sheet} нет года {since}")
    yearly = table.series("ЧДД")
    return sum(yearly[year] for year in table.years if year >= since) / factors[since]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denominator = (sum((x - mx) ** 2 for x in xs)
                   * sum((y - my) ** 2 for y in ys)) ** 0.5
    return numerator / denominator if denominator else float("nan")


def relative(new: float, old: float) -> float:
    return (new - old) / old * 100 if old else float("nan")


def exposure_by_year(rows_file: Path, config_path: Path,
                     tolerance: float) -> dict[int, tuple[float, float]] | str:
    """Доля объёма, который решает призабойная зона, по годам: (добыча, закачка).

    Скважина, упёршаяся в предел, берёт столько, сколько пропускает пласт, и
    считает это модель околоскважинного пространства — место, где схемы OPM и
    тНавигатора расходятся сильнее всего (Р-52).
    """
    import pandas as pd

    from aios import deliverable
    from aios.deck import parse_schedule
    from aios.deckfiles import layout, missing_deck_refusal

    sys.path.insert(0, str(ROOT / "scripts"))
    from pressure_share import share

    config = config_module.load(config_path)
    try:
        schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        # Пределы давления по скважинам живут в деке, а дек в git не ведётся:
        # промах мимо него — обычный первый запуск на чистом клоне, и отвечать
        # на него трассировкой нельзя (Р-248).
        return missing_deck_refusal(config.case.model_dir, error)
    limits = deliverable.bhp_limits(schedule)
    producers = {well: value for well, (kind, value) in limits.items() if kind == "producer"}
    injectors = {well: value for well, (kind, value) in limits.items() if kind == "injector"}

    rows = pd.read_csv(rows_file)
    rows["год"] = rows["DATA"].astype(str).str[:4].astype(int)
    result: dict[int, tuple[float, float]] = {}
    for year in sorted(rows["год"].unique()):
        part = rows[rows["год"] == year]
        start = f"{year}-01-01"
        result[int(year)] = (
            share(part, producers, start, tolerance, kind="producer")["volume_share"],
            share(part, injectors, start, tolerance, kind="injector")["volume_share"],
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", type=int, default=None,
                        help="год начала окна; по умолчанию — [economics].start_date конфига")
    parser.add_argument("--config", type=Path, default=Path(config_module.DEFAULT_PATH))
    parser.add_argument("--rows", type=Path, default=None,
                        help="выгрузка прогона: добавит координаты режима и экспозицию")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="допуск до предела давления, атм")
    arguments = parser.parse_args(argv)

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(arguments.config)
    since = arguments.since or config.economics.start_date.year

    tables = books()
    if len(tables) < 2:
        print(f"в {REFERENCE_DIR} меньше двух раскладок: сравнивать нечего. "
              "Разрыв схем меряется разницей ДВУХ расчётов организаторов на одном "
              "и том же управлении.", file=sys.stderr)
        return 1

    print(f"окно расчёта: с {since} (у эталонного расчётчика это же и база дисконта)")
    print()
    print(f"{'раскладка организаторов':44s} {'ЧДД в окне, млн руб.':>22}")
    for table in tables:
        print(f"{table.path.name[:44]:44s} {npv_in_window(table, since):>22,.1f}")

    first, last = tables[0], tables[-1]
    gap = npv_in_window(last, since) - npv_in_window(first, since)
    print()
    print(f"РАЗРЫВ СХЕМ: {gap:+,.1f} млн руб. "
          f"({relative(npv_in_window(last, since), npv_in_window(first, since)):+.2f} %)")
    print(f"  между «{first.path.name}» и «{last.path.name}»")

    years = [year for year in first.years if year >= since and year in last.years]
    print()
    print(f"{'год':>5}" + "".join(f"{'Δ' + name + ',%':>12}" for _, name in VOLUMES))
    series: dict[str, list[float]] = {name: [] for _, name in VOLUMES}
    for year in years:
        line = f"{year:>5}"
        for label, name in VOLUMES:
            old, new = first.series(label)[year], last.series(label)[year]
            value = relative(new, old)
            series[name].append(value)
            line += f"{value:>12.2f}" if value == value else f"{'—':>12}"
        print(line)

    print()
    for _, name in VOLUMES:
        values = [one for one in series[name] if one == one]
        if not values:
            continue
        print(f"{name:10s} медиана {statistics.median(values):+6.2f} %, "
              f"размах {min(values):+.2f} … {max(values):+.2f} %, "
              f"|макс| {max(abs(one) for one in values):.2f} %")

    if arguments.rows is not None:
        if not arguments.rows.is_file():
            print(f"\nнет выгрузки {arguments.rows}: экспозицию считать не из чего",
                  file=sys.stderr)
            return 1
        exposure = exposure_by_year(arguments.rows, arguments.config, arguments.tolerance)
        if isinstance(exposure, str):
            print(exposure, file=sys.stderr)
            return 1
        shared = [year for year in years if year in exposure]
        print()
        print(f"режим по выгрузке {arguments.rows} — доля объёма, которую решает "
              "призабойная зона")
        print(f"{'год':>5} {'добыча на пределе,%':>21} {'закачка на пределе,%':>22}")
        for year in shared:
            producers, injectors = exposure[year]
            print(f"{year:>5} {producers:>21.1f} {injectors:>22.1f}")
        producers = [exposure[year][0] for year in shared]
        print()
        print(f"доля добычи «на пределе» прошла {min(producers):.1f} → {max(producers):.1f} % "
              f"(размах {max(producers) - min(producers):.1f} п.п.)")
        oil = [series["нефть"][years.index(year)] for year in shared]
        print(f"корреляция расхождения по нефти с этой долей: "
              f"{pearson(producers, oil):+.3f}")
        if max(producers) - min(producers) < 15.0:
            print("  размах доли мал — история НЕ содержит вариации, по которой "
                  "можно было бы судить о форсированном режиме")

    print()
    print("ЧЕГО ЭТОТ ЗАМЕР НЕ ДОКАЗЫВАЕТ: он снят на ИСТОРИЧЕСКОМ управлении. "
          "Наш сценарий в тНавигаторе не считал никто, и перенести разрыв на него "
          "нечем — второго симулятора у нас нет.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
