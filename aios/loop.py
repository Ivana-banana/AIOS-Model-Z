"""Цикл управления: от состояния фонда к расписанию и его оценке.

Это тот самый контур МАС из постановки — оценка состояния → диагностика →
управляющее воздействие → проверка, — собранный так, чтобы работать на любом
источнике прогноза: на тривиальном суррогате сегодня, на нейросети завтра, на
полном расчёте ГГДМ при финальной проверке.

Цикл ничего не решает сам. Он связывает уже проверенные части: суррогат
(`aios.surrogate`), диагностику (`aios.margin`), политику (`aios.policy`),
генерацию расписания (`aios.schedule`) и деньги (`aios.economics`). Любое число,
которое он выдаёт, посчитано эталонным расчётчиком организаторов.

Честная оговорка, которая обязана попасть и в отчёт: **прирост, полученный на
суррогате, — это прогноз, а не результат.** Существующим он становится только
после подтверждения полным расчётом. `CycleResult.confirmed` про это и говорит.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from aios import economics, optimise
from aios.deck import ScheduleStep
from aios.margin import Norms
from aios.policy import Action, apply, explain, plan_shutdowns
from aios.schedule import ScheduleProblem, validate
from aios.surrogate import Surrogate


@dataclass(frozen=True)
class CycleResult:
    """Итог одного прохода цикла."""

    base_chdd_m: float
    """ЧДД исходного сценария по тому же источнику прогноза — точка отсчёта."""

    chdd_m: float
    """ЧДД после применения действий."""

    actions: tuple[Action, ...]

    rate_choices: tuple[optimise.RateChoice, ...]
    """Подбор режимов по скважинам. Пусто, если подбор был выключен."""

    steps: tuple[ScheduleStep, ...]
    warnings: tuple[str, ...]
    """Выход за обучающий диапазон суррогата и прочее, что делает прогноз шатким."""

    problems: tuple[ScheduleProblem, ...]
    """Нарушения ограничений в получившемся расписании. Непустой список — брак."""

    confirmed: bool = False
    """Подтверждён ли прирост полным расчётом ГГДМ. Пока False — это прогноз."""

    rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def gain_m(self) -> float:
        return self.chdd_m - self.base_chdd_m

    @property
    def is_improvement(self) -> bool:
        return self.gain_m > 0 and not self.problems

    def summary(self) -> str:
        """Человеко-читаемый отчёт — в лог, в UI и в презентацию."""
        status = "подтверждено полным расчётом" if self.confirmed else "ПРОГНОЗ, не подтверждён"
        lines = [
            f"ЧДД базового сценария: {self.base_chdd_m:,.1f} млн руб.",
            f"ЧДД после решений:     {self.chdd_m:,.1f} млн руб.",
            f"Прирост:               {self.gain_m:+,.1f} млн руб. ({status})",
            "",
        ]
        if self.rate_choices:
            lines += [optimise.explain(list(self.rate_choices)), ""]
        lines.append(explain(list(self.actions)))
        if self.warnings:
            lines += ["", f"Предупреждения ({len(self.warnings)}):"]
            lines += [f"  ! {warning}" for warning in self.warnings[:10]]
        if self.problems:
            lines += ["", f"НАРУШЕНИЯ ОГРАНИЧЕНИЙ ({len(self.problems)}) — расписание непригодно:"]
            lines += [f"  ! {problem}" for problem in self.problems[:10]]
        return "\n".join(lines)


def run_cycle(surrogate: Surrogate,
              base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
              norms: Norms,
              *,
              control_start: date,
              start_date: date,
              lookback_months: int = 6,
              known_wells: frozenset[str] | None = None,
              tune_rates: bool = True) -> CycleResult:
    """Пройти цикл один раз: спрогнозировать, продиагностировать, решить, оценить.

    Порядок именно такой и не иначе: диагноз ставится по прогнозу базового
    сценария, то есть по тому, что произойдёт, если ничего не делать. Решать по
    факту прошлого нельзя — управляем-то будущим.

    Два разных решения, и разница между ними существенна для отчёта:

    * **подбор режимов** (`aios.optimise`) смотрит на весь горизонт сразу. Это
      законно: мы формируем расписание на весь период вперёд, и «будущее» здесь —
      прогноз суррогата, обученного только на истории, а не подсмотренный факт;
    * **остановка убыточных** (`aios.policy`) причинная: решение на дату `t`
      принимается по данным до `t`. Она ловит скважины, которые становятся
      убыточными по ходу горизонта даже на лучшем из режимов.
    """
    base_rows = surrogate.predict(base_steps)
    base = _npv(base_rows, start_date)

    choices: list[optimise.RateChoice] = []
    tuned = list(base_steps)
    if tune_rates:
        choices = optimise.choose_rates(surrogate, base_steps, norms, start_date=start_date)
        tuned = optimise.apply_choices(base_steps, choices)

    actions = plan_shutdowns(
        surrogate.predict(tuned), norms, control_start=control_start, lookback=lookback_months
    )
    steps = apply(tuned, actions)

    rows = surrogate.predict(steps)
    result = _npv(rows, start_date)

    warnings = list(surrogate.in_training_range(steps))
    problems = validate(steps, known_wells)

    return CycleResult(
        base_chdd_m=base.chdd_m,
        chdd_m=result.chdd_m,
        actions=tuple(actions),
        rate_choices=tuple(choices),
        steps=tuple(steps),
        warnings=tuple(warnings),
        problems=tuple(problems),
        rows=rows,
    )


def confirm(result: CycleResult, rows: list[dict[str, Any]], start_date: date) -> CycleResult:
    """Заменить прогнозные числа результатом полного расчёта.

    `rows` — выгрузка ГГДМ по тому же расписанию (`aios.summary.monthly_rows`).
    После этого `confirmed` становится True, и только тогда прирост можно
    называть результатом, а ЧДД — заявлять организаторам.
    """
    confirmed = _npv(rows, start_date)
    return CycleResult(
        base_chdd_m=result.base_chdd_m,
        chdd_m=confirmed.chdd_m,
        actions=result.actions,
        rate_choices=result.rate_choices,
        steps=result.steps,
        warnings=result.warnings,
        problems=result.problems,
        confirmed=True,
        rows=rows,
    )


def _npv(rows: list[dict[str, Any]], start_date: date) -> economics.NpvResult:
    return economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date=start_date.isoformat()
    )
