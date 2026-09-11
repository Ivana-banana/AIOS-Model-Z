"""Как растёт время контура, когда фонд крупнее Модели Z.

    python scripts/benchmark_scale.py --rows runs/matched-0995/rows.csv --factors 1,2,4,8

Трек называется «масштабируемая мультиагентная система», и до 31.08 слово это в
проекте ничем не подтверждалось: в `docs/tz-compliance.md` строка про
масштабирование стояла «частично» со ссылками на конфиг и отсутствие зашитых
имён. Это про переносимость постановки, а не про то, выдержит ли контур фонд
вчетверо крупнее. Здесь — число вместо обещания.

**Фонд раздувается репликацией, и это осознанное упрощение.** Копия скважины
получает суффикс `_k1`, `_k2` и так далее, история и режимы копируются как есть.
Новой физики такой фонд не создаёт — межскважинных связей между копиями нет, —
и потому замер отвечает ровно на один вопрос: **как растёт стоимость счёта от
размера фонда**. Он не отвечает, останется ли решение осмысленным на настоящем
объекте вчетверо крупнее; это разные вопросы, и путать их нельзя.

Проверка, что репликация не врёт, встроена в сам замер: ЧДД раздутого фонда
обязан быть кратен исходному с той же кратностью. Разошлось — печатается отказ,
потому что нелинейный ЧДД означает ошибку копирования, а не находку.

Замеряются три величины порознь, и это важнее суммы: у них разные причины расти.
Экономика — вендоренный расчётчик организаторов, его мы не переписываем и на его
время влиять не можем. Суррогат — наш прогноз. Цикл — координация, то самое
место, где «агент на скважину» обычно перестаёт масштабироваться.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, loop  # noqa: E402
from aios.deck import ScheduleStep, parse_schedule  # noqa: E402
from aios.deckfiles import layout, norms_refusal  # noqa: E402
from aios.margin import Norms  # noqa: E402
from aios.surrogate import DeclineSurrogate  # noqa: E402

#: Насколько ЧДД раздутого фонда может отойти от кратного, прежде чем замер
#: объявляется недействительным. Копии независимы и дисконтируются одинаково,
#: поэтому расхождение здесь берётся только из ошибки репликации.
NPV_TOLERANCE = 1e-6


def suffixed(name: str, copy: int) -> str:
    """Имя скважины в копии фонда. Нулевая копия — исходное имя."""
    return name if copy == 0 else f"{name}_k{copy}"


def replicate_rows(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    """Размножить выгрузку в `factor` раз, переименовав скважины копий."""
    return [dict(row, well=suffixed(str(row["well"]), copy))
            for copy in range(factor) for row in rows]


def replicate_steps(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                    factor: int) -> list[ScheduleStep]:
    """Размножить расписание: у каждого шага те же даты, фонд в `factor` раз шире."""
    return [
        ScheduleStep(
            date=step.date,
            producers=tuple(replace(control, well=suffixed(control.well, copy))
                            for copy in range(factor) for control in step.producers),
            injectors=tuple(replace(control, well=suffixed(control.well, copy))
                            for copy in range(factor) for control in step.injectors),
        )
        for step in steps
    ]


def _numeric(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Привести колонки контракта к числам: из CSV всё приходит строками."""
    return [
        {key: (value if key in ("DATA", "well") else float(value))
         for key, value in row.items()}
        for row in rows
    ]


def _warmup(rows: list[dict[str, Any]], steps: list[ScheduleStep],
            norms: Norms, config: Any) -> None:
    """Пройти контур один раз вхолостую, чтобы разовые издержки не легли на замер.

    Берётся первая скважина фонда: прогреть надо пути кода, а не потратить время.
    """
    well = str(rows[0]["well"])
    history = [row for row in rows if str(row["well"]) == well]
    warm = replicate_steps(steps, 1)
    economics.compute_npv(history, start_date=config.economics.start_date,
                          norms_file=config.economics.norms_file)
    surrogate = DeclineSurrogate.fit(history, fluids=config.fluids)
    loop.run_cycle(
        surrogate, warm, norms,
        control_start=config.case.control_start,
        start_date=config.economics.start_date,
        lookback_months=config.policy.lookback_months,
        limits=config.limits,
        fluids=config.fluids,
        norms_file=config.economics.norms_file,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default="runs/matched-0995/rows.csv",
                        help="выгрузка в контракте расчётчика, из которой берётся фонд")
    parser.add_argument("--factors", default="1,2,4,8",
                        help="во сколько раз раздувать фонд, через запятую")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    # Конфигурация проверяется ПЕРВОЙ: без неё не выполнить ничего, а её путь
    # печатают руками. Обратный порядок означал, что на чистом клоне (где нет
    # ни одной выгрузки) команда отвечала «нет выгрузки» на битый --config —
    # и обход точек входа краснел на клоне, хотя на машине разработки был
    # зелёным. Ровно эта же ошибка была у обоих прежних сторожей класса.
    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1

    factors = [int(value) for value in arguments.factors.split(",")]
    if any(factor < 1 for factor in factors):
        print("кратность не бывает меньше единицы", file=sys.stderr)
        return 1

    source = Path(arguments.rows)
    if not source.is_file():
        print(f"нет выгрузки {source}: сначала посчитайте прогон", file=sys.stderr)
        return 1

    config = config_module.load(arguments.config)
    refusal = norms_refusal(config.economics.norms_file)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    with source.open(encoding="utf-8") as handle:
        rows = _numeric(list(csv.DictReader(handle)))

    try:
        schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        # дек в git не ведётся: промах мимо него на чистом клоне — обычный
        # первый запуск, а не поломка. Трассировка читалась бы как второе
        # (Р-248)
        print(f"Дек не читается: {error}", file=sys.stderr)
        print(f"Ожидается каталог поставки с корневым .DATA: "
              f"{config.case.model_dir}", file=sys.stderr)
        print("Как распаковать поставку — README.md, раздел «Установка».",
              file=sys.stderr)
        return 1
    base_steps = [step for step in schedule.steps
                  if step.date >= config.case.control_start]
    assumptions, _pumps = economics.load_norms(config.economics.norms_file)
    norms = Norms.from_assumptions(assumptions)

    wells = len({row["well"] for row in rows})
    print(f"исходный фонд: {wells} скважин, {len(rows)} строк выгрузки, "
          f"{len(base_steps)} шагов под управлением")
    print()
    print(f"{'фонд':>6} {'строк':>9} {'экономика':>10} {'суррогат':>10} "
          f"{'цикл':>9} {'действий':>9} {'урезаний':>9} {'ЧДД, млн':>14}")

    # Прогревочный проход, результат которого выбрасывается. Без него первая
    # измеренная точка завышена почти вдвое (замер 31.08: 2,84 с против 1,55 с
    # на повторе того же ×1) — ленивые импорты, кэши pandas и прогрев страниц
    # памяти платятся один раз. Первой точкой обычно идёт ×1, то есть шум садится
    # ровно на знаменатель всех отношений и делает рост похожим на нелинейный.
    _warmup(rows, base_steps, norms, config)

    reference: float | None = None
    for factor in factors:
        history = replicate_rows(rows, factor)
        steps = replicate_steps(base_steps, factor)

        started = time.perf_counter()
        npv = economics.compute_npv(history, start_date=config.economics.start_date,
                                    norms_file=config.economics.norms_file)
        economics_seconds = time.perf_counter() - started

        started = time.perf_counter()
        surrogate = DeclineSurrogate.fit(history, fluids=config.fluids)
        surrogate.predict(steps)
        surrogate_seconds = time.perf_counter() - started

        known = frozenset(
            {control.well for step in steps for control in step.producers}
            | {control.well for step in steps for control in step.injectors}
        )
        started = time.perf_counter()
        result = loop.run_cycle(
            surrogate, steps, norms,
            control_start=config.case.control_start,
            start_date=config.economics.start_date,
            lookback_months=config.policy.lookback_months,
            known_wells=known,
            limits=config.limits,
            fluids=config.fluids,
            norms_file=config.economics.norms_file,
        )
        cycle_seconds = time.perf_counter() - started

        if reference is None:
            reference = npv.chdd_m
        elif abs(npv.chdd_m - reference * factor) > NPV_TOLERANCE * abs(reference * factor):
            print(f"\nЧДД при ×{factor} — {npv.chdd_m:,.1f}, а кратный дал бы "
                  f"{reference * factor:,.1f}: репликация фонда неверна, "
                  "замер недействителен", file=sys.stderr)
            return 1

        print(f"×{factor:<5} {len(history):>9} {economics_seconds:>9.2f}с "
              f"{surrogate_seconds:>9.2f}с {cycle_seconds:>8.2f}с "
              f"{len(result.actions):>9} {len(result.allocations):>9} "
              f"{npv.chdd_m:>14,.1f}")

    print()
    print("Полный расчёт той же истории на ГГДМ — 11–13 минут (docs/experiments.md).")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
