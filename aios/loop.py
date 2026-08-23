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
from pathlib import Path
from typing import Any

from aios import economics, optimise, planner
from aios.deck import ScheduleStep
from aios.margin import Norms
from aios.policy import Action, apply, explain, plan_shutdowns
from aios.schedule import ScheduleProblem, liquid_limit, validate
from aios.summary import Fluids
from aios.surrogate import Surrogate


@dataclass(frozen=True)
class CycleResult:
    """Итог одного прохода цикла."""

    base_chdd_m: float
    """ЧДД исходного сценария по тому же источнику прогноза — точка отсчёта."""

    chdd_m: float
    """ЧДД после применения действий."""

    actions: tuple[Action, ...]

    rate_choices: tuple[optimise.RateChoice | optimise.JointRateChoice, ...]
    """Подбор режимов по скважинам. Пусто, если подбор был выключен."""

    allocations: tuple[planner.Allocation, ...]
    """Что урезал планировщик, согласуя предложения с ограничениями объекта."""

    steps: tuple[ScheduleStep, ...]
    warnings: tuple[str, ...]
    """Выход за обучающий диапазон суррогата и прочее, что делает прогноз шатким."""

    problems: tuple[ScheduleProblem, ...]
    """Нарушения ограничений в получившемся расписании. Непустой список — брак."""

    excluded_rows: int = 0
    """Сколько строк прогноза расчётчик выбросил как отрицательные приращения.

    Прогнозный ЧДД считается по строкам суррогата, а строку с отрицательным
    приростом методика исключает целиком. Падение выходит почти
    пропорциональным: 5 % негодных строк дают −5,3 % ЧДД, 20 % — −20,0 %
    (измерено 20.08). Само исключение верно, а вот молчать о нём нельзя —
    иначе прогноз, посчитанный на обрезанном наборе, неотличим от честного.

    На странице настоящего прогона этот счётчик показывается с самого начала;
    здесь он появился позже, хотя нужнее именно тут: у суррогата нефизичные
    строки бывают тысячами (замер А4 — 22 563 на сценарии с ВНС).
    """

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
        if self.excluded_rows:
            lines[-1:] = [
                f"ВНИМАНИЕ: исключено строк прогноза {self.excluded_rows} — "
                "расчётчик отбросил их как отрицательные приращения, "
                "то есть число посчитано на обрезанном наборе",
                "",
            ]
        if self.rate_choices:
            lines += [optimise.explain(list(self.rate_choices)), ""]
        lines.append(explain(list(self.actions)))
        if self.allocations:
            lines += ["", planner.explain(list(self.allocations))]
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
              tune_rates: bool = True,
              joint_rate_search: bool = False,
              limits: planner.Limits | None = None,
              fluids: Fluids | None = None,
              multipliers: tuple[float, ...] | None = None,
              norms_file: Path = economics.NORMS_FILE) -> CycleResult:
    """Пройти цикл один раз: спрогнозировать, продиагностировать, решить, оценить.

    Порядок именно такой и не иначе: диагноз ставится по прогнозу базового
    сценария, то есть по тому, что произойдёт, если ничего не делать. Решать по
    факту прошлого нельзя — управляем-то будущим.

    Два разных решения, и разница между ними существенна для отчёта:

    * **подбор режимов** (`aios.optimise`) смотрит на весь горизонт сразу. Это
      законно: мы формируем расписание на весь период вперёд, и «будущее» здесь —
      прогноз суррогата, обученного только на истории, а не подсмотренный факт;
      для модели с межскважинными связями `joint_rate_search` обязан быть True,
      чтобы каждый кандидат оценивался по ЧДД всего фонда;
    * **остановка убыточных** (`aios.policy`) причинная: решение на дату `t`
      принимается по данным до `t`. Она ловит скважины, которые становятся
      убыточными по ходу горизонта даже на лучшем из режимов.
    """
    base_rows = surrogate.predict(base_steps)
    base = _npv(base_rows, start_date, norms_file)

    choices: list[optimise.RateChoice] | list[optimise.JointRateChoice] = []
    tuned = list(base_steps)
    if tune_rates:
        if joint_rate_search:
            choices = optimise.choose_joint_rates(
                surrogate, base_steps, start_date=start_date,
                limits=limits, norms=norms, fluids=fluids,
                # набор множителей задаётся снаружи: узкий набор держит поиск
                # ближе к обучающим сценариям, широкий уводит из области
                # доверия — и то и другое надо уметь померить (Б31)
                **({} if multipliers is None else {"multipliers": multipliers}),
                # второй контур шлюза доверия (`Р-72`) — область лежит в самом
                # артефакте суррогата, а не собирается здесь заново: иначе ничто
                # не мешает применить область одной модели к прогнозу другой.
                # `None` у артефактов до 22.08 — поиск работает как раньше
                region=getattr(getattr(surrogate, "bundle", None), "trust_region", None),
                norms_file=norms_file,
            )
        else:
            choices = optimise.choose_rates(
                surrogate, base_steps, norms, start_date=start_date,
                **({} if multipliers is None else {"multipliers": multipliers}),
            )
        tuned = optimise.apply_choices(base_steps, choices)

    actions = plan_shutdowns(
        surrogate.predict(tuned), norms, control_start=control_start, lookback=lookback_months
    )
    steps = apply(tuned, actions)

    # последним слово за планировщиком: скважинные агенты предлагают, глядя каждый
    # на себя, а вода и мощность по жидкости принадлежат объекту целиком
    allocations: list[planner.Allocation] = []
    if limits is not None:
        steps, allocations = planner.enforce(
            steps, surrogate.predict(steps), norms, limits, fluids
        )

    rows = surrogate.predict(steps)
    result = _npv(rows, start_date, norms_file)

    warnings = list(surrogate.in_training_range(steps))
    # предел дебита жидкости — из таблицы ЭЦН тех нормативов, по которым считается
    # ЧДД этого прогона, а не из литерала 500 от Модели Z (Б28)
    problems = validate(steps, known_wells,
                        max_liquid_rate=liquid_limit(economics.load_norms(norms_file)[1]))

    return CycleResult(
        base_chdd_m=base.chdd_m,
        chdd_m=result.chdd_m,
        actions=tuple(actions),
        rate_choices=tuple(choices),
        allocations=tuple(allocations),
        steps=tuple(steps),
        warnings=tuple(warnings),
        problems=tuple(problems),
        excluded_rows=result.excluded_negative_rows,
        rows=rows,
    )


def confirm(result: CycleResult, rows: list[dict[str, Any]], start_date: date,
            *, base_rows: list[dict[str, Any]],
            norms_file: Path = economics.NORMS_FILE) -> CycleResult:
    """Заменить прогнозные числа результатом полного расчёта.

    `rows` — выгрузка ГГДМ по расписанию цикла, `base_rows` — выгрузка ГГДМ по
    базовому сценарию (обе из `aios.summary.monthly_rows`). После этого
    `confirmed` становится True, и только тогда прирост можно называть
    результатом, а ЧДД — заявлять организаторам.

    **Базовое число пересчитывается тоже, и это не формальность.** Раньше
    заменялся только `chdd_m`, а `base_chdd_m` оставался прогнозом суррогата —
    то есть `gain_m` становился разностью двух разных источников и при этом
    подписывался «подтверждено полным расчётом». Замер 21.08: прогноз обещал
    прирост −100,0, после подтверждения печаталось −999,2. `run_cycle.py` эту
    ловушку обходил вручную, считая базовый ЧДД сам, — то есть защита держалась
    на том, что единственный вызывающий про неё помнит.

    Аргумент обязателен по той же причине, по какой обязателен `start_date` в
    `economics.compute_npv`: забыть его можно ровно один раз.
    """
    confirmed = _npv(rows, start_date, norms_file)
    return CycleResult(
        base_chdd_m=_npv(base_rows, start_date, norms_file).chdd_m,
        chdd_m=confirmed.chdd_m,
        actions=result.actions,
        rate_choices=result.rate_choices,
        allocations=result.allocations,
        steps=result.steps,
        warnings=result.warnings,
        problems=result.problems,
        confirmed=True,
        rows=rows,
    )


def _npv(rows: list[dict[str, Any]], start_date: date,
         norms_file: Path = economics.NORMS_FILE) -> economics.NpvResult:
    return economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date=start_date.isoformat(),
        norms_file=norms_file,
    )
