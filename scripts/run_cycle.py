"""Прогнать цикл управления поверх результата ГГДМ и собрать кейс под проверку.

Два шага, между ними — полный расчёт на симуляторе:

    python scripts/run_cycle.py --from runs/base --out runs/policy
    docker run --rm -v "$PWD:$PWD" -w "$PWD" aios \
        scripts/run_flow.sh runs/policy/RUN.DATA 4 --parsing-strictness=low
    python scripts/run_cycle.py --confirm runs/policy --from runs/base

Первый шаг обучает суррогат на выгрузке базового прогона, проходит цикл
«состояние → диагноз → решения → расписание → прогноз ЧДД» и пишет кейс с
получившимся расписанием. Число, которое он печатает, — **прогноз суррогата**.

Третий шаг заменяет прогноз результатом полного расчёта: только после него
прирост можно называть результатом (`aios.loop.confirm`, `CycleResult.confirmed`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, loop, summary  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.margin import Norms  # noqa: E402
from aios.run import build_case  # noqa: E402
from aios.surrogate import DeclineSurrogate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", default=None,
                        help="каталог прогона, на котором учится суррогат (по умолчанию из конфига)")
    parser.add_argument("--out", default="runs/policy", help="куда собрать кейс с решениями цикла")
    parser.add_argument("--confirm", default=None,
                        help="каталог посчитанного кейса: заменить прогноз полным расчётом")
    parser.add_argument("--no-tune-rates", action="store_true",
                        help="только остановка убыточных, без подбора режимов")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    source = Path(arguments.source) if arguments.source else config.case.run_dir
    case_name = config.case.case_name

    history = summary.monthly_rows(source / case_name, fluids=config.fluids)
    if not history:
        print(f"нет выгрузки в {source}: сначала посчитайте базовый прогон", file=sys.stderr)
        return 1

    assumptions, _pumps = economics.load_norms(config.economics.norms_file)
    norms = Norms.from_assumptions(assumptions)
    schedule = parse_schedule(config.case.model_dir / "Model_Z_sch.inc")
    base_steps = [step for step in schedule.steps if step.date >= config.case.control_start]

    surrogate = DeclineSurrogate.fit(history, fluids=config.fluids)
    result = loop.run_cycle(
        surrogate,
        base_steps,
        norms,
        control_start=config.case.control_start,
        start_date=config.economics.start_date,
        lookback_months=config.policy.lookback_months,
        known_wells=frozenset(schedule.wells),
        tune_rates=not arguments.no_tune_rates,
        limits=config.limits,
        fluids=config.fluids,
    )

    if arguments.confirm:
        confirmed_dir = Path(arguments.confirm)
        rows = summary.monthly_rows(confirmed_dir / case_name, fluids=config.fluids)
        if not rows:
            print(f"нет выгрузки в {confirmed_dir}", file=sys.stderr)
            return 1

        problems = summary.sanity_problems(rows)
        if problems:
            print(f"санити-чеки расчёта: нарушений {len(problems)}")
            for line in problems[:10]:
                print(f"  {line}")

        # ЧДД базового сценария берётся из полного расчёта, а не из прогноза:
        # сравнивать подтверждённое число с прогнозным нельзя, разойдётся сам
        # предмет сравнения.
        base_npv = economics.compute_npv(
            history, headers=economics.REQUIRED_COLUMNS,
            start_date=config.economics.start_date.isoformat(),
        )
        confirmed = loop.confirm(result, rows, config.economics.start_date)

        print("--- подтверждение полным расчётом ---")
        print(f"ЧДД базового сценария (ГГДМ):  {base_npv.chdd_m:12.1f} млн руб.")
        print(f"ЧДД сценария цикла (ГГДМ):     {confirmed.chdd_m:12.1f} млн руб.")
        print(f"прирост:                       {confirmed.chdd_m - base_npv.chdd_m:+12.1f} млн руб. "
              f"({100.0 * (confirmed.chdd_m - base_npv.chdd_m) / abs(base_npv.chdd_m):+.2f} %)")
        print()
        print(f"прогноз суррогата обещал:      {result.gain_m:+12.1f} млн руб.")
        print(f"ошибка прогноза:               "
              f"{result.gain_m - (confirmed.chdd_m - base_npv.chdd_m):+12.1f} млн руб.")
        print()
        print(f"нефть базовая / цикла: {base_npv.oil_kt:.1f} / {confirmed_oil(rows):.1f} тыс. т")
        print(f"подтверждено: {confirmed.confirmed}")
        return 0

    print(result.summary())
    print()

    if result.problems:
        print("расписание нарушает ограничения — кейс не собираю", file=sys.stderr)
        return 1

    case = build_case(
        arguments.out,
        config.case.model_dir,
        steps=list(result.steps),
        control_start=config.case.control_start,
        wells=schedule.wells,
        case_name=case_name,
    )
    print(f"кейс собран: {case.data_file}")
    print(f"расписание:  {case.schedule_file}")
    print()
    print("дальше — полный расчёт, без него прирост остаётся прогнозом:")
    # путь печатается относительным, когда кейс внутри репозитория, и абсолютным,
    # когда снаружи: команда должна копироваться как есть в обоих случаях
    data_path = case.data_file.resolve()
    shown = data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path
    print('  docker run --rm -v "$PWD:$PWD" -w "$PWD" aios \\')
    print(f"      scripts/run_flow.sh {shown} "
          f"{config.run.processes} --parsing-strictness=low")
    print(f"  python scripts/run_cycle.py --confirm {arguments.out} --from {source}")
    return 0


def confirmed_oil(rows: list[dict[str, object]]) -> float:
    """Накопленная нефть выгрузки, тыс. т — для сравнения сценариев глазами."""
    return sum(float(row["WOMT_Diff"]) for row in rows) / 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
