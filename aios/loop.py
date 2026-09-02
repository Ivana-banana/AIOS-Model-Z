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

from aios import balance, economics, optimise, planner, trust
from aios.deck import ScheduleStep
from aios.margin import Norms
from aios.policy import Action, apply, explain, plan_conversions, plan_shutdowns
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
    raw_chdd_m: float | None = None
    safe_chdd_m: float | None = None
    margin: trust.OptimismMargin | None = None

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
        if self.margin is None:
            lines += ["Конформный запас: не считался — прогноз с запасом недоступен.", ""]
        elif not self.margin.calibrated:
            lines += [f"Конформный запас: НЕ КАЛИБРОВАН — {self.margin.reason}", ""]
        else:
            lines += [
                f"Конформный запас: {self.margin.value:+,.1f} млн руб. ({self.margin.samples} пар).",
                f"Прогноз после запаса: {self.safe_chdd_m:,.1f} млн руб. "
                f"(сырой {self.raw_chdd_m:,.1f} млн руб.)",
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
              start_multipliers: dict[tuple[str, str], float] | None = None,
              margin: trust.OptimismMargin | None = None,
              search_seed: int | None = None,
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
    # Точка отсчёта — то расписание, с которого цикл стартует. Если задан старт
    # из готового сценария, сравнивать надо с ним, а не с опорным: иначе
    # «прирост» смешает эффект самого сценария с тем, что добавил цикл (Б34).
    origin = (base_steps if start_multipliers is None
              else optimise.apply_start(base_steps, start_multipliers))
    base_rows = surrogate.predict(origin)
    base = _npv(base_rows, start_date, norms_file)

    choices: list[optimise.RateChoice] | list[optimise.JointRateChoice] = []
    tuned = list(origin)
    if tune_rates:
        if joint_rate_search:
            choices = optimise.choose_joint_rates(
                surrogate, base_steps, start_date=start_date,
                limits=limits, norms=norms, fluids=fluids,
                # набор множителей задаётся снаружи: узкий набор держит поиск
                # ближе к обучающим сценариям, широкий уводит из области
                # доверия — и то и другое надо уметь померить (Б31)
                **({} if multipliers is None else {"multipliers": multipliers}),
                # старт поиска из уже найденного сценария: опорное расписание при
                # этом не подменяется — модель продолжает получать вход в тех же
                # координатах, в которых училась
                **({} if start_multipliers is None else {"start": start_multipliers}),
                # второй контур шлюза доверия (`Р-72`) — область лежит в самом
                # артефакте суррогата, а не собирается здесь заново: иначе ничто
                # не мешает применить область одной модели к прогнозу другой.
                # `None` у артефактов до 22.08 — поиск работает как раньше
                region=getattr(getattr(surrogate, "bundle", None), "trust_region", None),
                margin=margin,
                seed=search_seed,
                norms_file=norms_file,
            )
        else:
            choices = optimise.choose_rates(
                surrogate, base_steps, norms, start_date=start_date,
                **({} if multipliers is None else {"multipliers": multipliers}),
            )
        tuned = optimise.apply_choices(base_steps, choices)

    forecast = surrogate.predict(tuned)
    actions = plan_shutdowns(
        forecast, norms, control_start=control_start, lookback=lookback_months
    )
    # Перевод под закачку — второе решение фонда, и оно предлагается только там,
    # где у объекта есть запас под собственным потолком компенсации: закачка
    # добавляется, а она у нас ограничена (Р-61). Без запаса предложение было бы
    # обещанием, которое планировщик тут же срежет.
    converted = plan_conversions(
        forecast, norms, control_start=control_start, lookback=lookback_months,
        headroom=_water_headroom(forecast, limits, fluids),
    )
    # остановка и перевод по одной скважине — одно решение, а не два: переводим
    shut_wells = {action.well for action in converted}
    actions = [action for action in actions if action.well not in shut_wells] + converted
    steps = apply(tuned, actions)

    # последним слово за планировщиком: скважинные агенты предлагают, глядя каждый
    # на себя, а вода и мощность по жидкости принадлежат объекту целиком
    allocations: list[planner.Allocation] = []
    if limits is not None:
        steps, allocations = planner.enforce(
            steps, surrogate.predict(steps), norms, limits, fluids
        )

    # Скважины, которыми управлять не разрешено, возвращаются к опорному режиму.
    # Это единственная из трёх форм ограничения по фонду, которую нельзя сделать
    # в планировщике: чтобы вернуть скважину к исходному режиму, нужно само
    # опорное расписание, а планировщик получает уже изменённые шаги (Б40).
    if limits is not None and limits.controllable_wells is not None:
        steps = _restore_uncontrolled(origin, steps, limits.controllable_wells)

    rows = surrogate.predict(steps)
    result = _npv(rows, start_date, norms_file)
    region = getattr(getattr(surrogate, "bundle", None), "trust_region", None)
    distance = optimise._distance_of(surrogate, steps, region)
    safe_chdd = trust.safe_value(result.chdd_m, distance, margin)

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
        raw_chdd_m=result.chdd_m,
        safe_chdd_m=safe_chdd,
        margin=margin,
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


def _restore_uncontrolled(origin: list[ScheduleStep] | tuple[ScheduleStep, ...],
                          steps: list[ScheduleStep],
                          allowed: frozenset[str]) -> list[ScheduleStep]:
    """Вернуть неразрешённым скважинам режим опорного расписания.

    «Разрешено управлять этими» означает, что остальные остаются как были, а не
    «меняются осторожно»: кейс может прямо запретить трогать часть фонда, и
    тогда наше расписание обязано совпасть с исходным на этих скважинах
    побайтово, а не приблизительно.
    """
    by_date = {step.date: step for step in origin}
    result: list[ScheduleStep] = []
    for step in steps:
        reference = by_date.get(step.date)
        if reference is None:
            result.append(step)
            continue
        original_producers = {control.well: control for control in reference.producers}
        original_injectors = {control.well: control for control in reference.injectors}
        producers = tuple(
            control if control.well in allowed
            else original_producers.get(control.well, control)
            for control in step.producers
        )
        injectors = tuple(
            control if control.well in allowed
            else original_injectors.get(control.well, control)
            for control in step.injectors
        )
        result.append(ScheduleStep(step.date, producers, injectors))
    return result


def _water_headroom(rows: list[dict[str, Any]],
                    limits: planner.Limits | None,
                    fluids: Fluids | None) -> bool:
    """Есть ли у объекта запас под собственным потолком компенсации.

    Перевод под закачку добавляет воду в пласт, а закачка ограничена нашим же
    правилом (Р-61): без запаса предлагать перевод — значит обещать то, что
    планировщик срежет на следующем шаге. Без ограничений запас считается
    неограниченным: правило выключено, и мешать переводу нечему.
    """
    if limits is None or not limits.enforce_water_balance:
        return True
    years = balance.by_year(rows, fluids)
    if not years:
        return True
    return max(year.compensation for year in years) < limits.water_tolerance


def _npv(rows: list[dict[str, Any]], start_date: date,
         norms_file: Path = economics.NORMS_FILE) -> economics.NpvResult:
    return economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date=start_date.isoformat(),
        norms_file=norms_file,
    )
