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

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from aios import balance, casecheck, economics, optimise, planner, trust
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

    case_problems: tuple[str, ...] = ()
    """Условия кейса, которые расписание нарушает после всех срезок.

    Отдельно от `problems`: там нарушения самого расписания (дебит выше предела
    ЭЦН, незнакомая скважина) с привязкой к шагу и скважине, здесь — условия
    объекта (`aios.casecheck`, баланс воды, мощность водозабора), у которых адрес
    другой: год, месяц или группа. Непустой список так же означает брак — см.
    `_case_problems`.
    """

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
        return self.gain_m > 0 and not self.problems and not self.case_problems

    def summary(self) -> str:
        """Человеко-читаемый отчёт — в лог, в UI и в презентацию."""
        status = "подтверждено полным расчётом" if self.confirmed else "ПРОГНОЗ, не подтверждён"
        lines = [
            f"ЧДД базового сценария: {self.base_chdd_m:,.1f} млн руб.",
            f"ЧДД после решений:     {self.chdd_m:,.1f} млн руб.",
            f"Прирост:               {self.gain_m:+,.1f} млн руб. ({status})",
            "",
        ]
        if self.confirmed:
            lines += ["Конформный запас не применяется: число получено полным расчётом, "
                      "поправлять на оптимизм суррогата здесь нечего.", ""]
        elif self.margin is None:
            lines += ["Конформный запас: не считался — прогноз с запасом недоступен.", ""]
        elif not self.margin.calibrated:
            lines += [f"Конформный запас: НЕ КАЛИБРОВАН — {self.margin.reason}", ""]
        else:
            lines += [
                f"Конформный запас: {self.margin.value:+,.1f} млн руб. ({self.margin.samples} пар).",
                f"Прогноз после запаса: {self.safe_chdd_m:,.1f} млн руб. "
                f"(сырой {self.raw_chdd_m:,.1f} млн руб.)",
            ]
            # Кроме запаса из величины вычитается штраф за удаление от обучающих
            # сценариев, и вне области он бывает на два порядка больше самого
            # запаса: замер 05.09 дал «после запаса −214 989,8» при сыром 2 647,2
            # и запасе 927,8. Подпись, называющая только запас, превращала сигнал
            # «безнадёжно вне области» в бессмыслицу на глазах у читателя.
            penalty = self.raw_chdd_m - self.margin.value - self.safe_chdd_m
            if penalty > 1e-9:
                lines.append(
                    f"  из них штраф за удаление от обучающих сценариев: "
                    f"-{penalty:,.1f} млн руб. — это не оценка денег, а признак "
                    f"«режим вне доверенной области»"
                )
            lines.append("")
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
        if self.case_problems:
            lines += ["", f"УСЛОВИЯ КЕЙСА НАРУШЕНЫ ({len(self.case_problems)}) — "
                          "кандидат непригоден:"]
            lines += [f"  ! {problem}" for problem in self.case_problems[:10]]
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
              norms_file: Path = economics.NORMS_FILE,
              history_rows: list[dict[str, Any]] | None = None,
              progress: Callable[[str], None] | None = None) -> CycleResult:
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
    def step(note: str) -> None:
        # Цикл на обученном суррогате идёт десятки минут и до 05.09 не подавал
        # признаков жизни между «конформный запас» и концом: отличить работу от
        # зависания было нечем. Печатает по-прежнему вызывающий, а не модуль.
        if progress is not None:
            progress(note)

    origin = (base_steps if start_multipliers is None
              else optimise.apply_start(base_steps, start_multipliers))
    step("прогноз базы")
    base_rows = surrogate.predict(origin)
    base = _npv(base_rows, start_date, norms_file)

    choices: list[optimise.RateChoice] | list[optimise.JointRateChoice] = []
    tuned = list(origin)
    if tune_rates:
        step("подбор режимов: совместный поиск по ЧДД фонда")
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
            frame = base_steps
        else:
            # Одиночный перебор ведётся от **стартового сценария**, а не от дека.
            # У `choose_rates` нет параметра `start`, каким пользуется совместный
            # перебор, поэтому карта из `--base-schedule` здесь просто терялась:
            # поиск шёл по деку, а `base_chdd_m` считался по `origin` со стартом —
            # «прирост» сравнивал две разные точки отсчёта (2.1 #8).
            choices = optimise.choose_rates(
                surrogate, origin, norms, start_date=start_date,
                **({} if multipliers is None else {"multipliers": multipliers}),
            )
            frame = list(origin)
        # Пустой ответ поиска означает «ничего не менять», а не «вернуться к
        # деку»: `apply_choices(base_steps, [])` отдаёт опорное расписание и
        # молча теряет карту старта из `--base-schedule`. Тогда прирост считался
        # бы от одного сценария, а расписание собиралось из другого (Р-190).
        # Множители осмысленны только в той системе координат, в которой их
        # искали, поэтому применяются к ней же.
        tuned = optimise.apply_choices(frame, choices) if choices else list(origin)

    forecast = surrogate.predict(tuned)
    step("политика остановок")
    actions = plan_shutdowns(
        forecast, norms, control_start=control_start, lookback=lookback_months
    )
    # Перевод под закачку — второе решение фонда, и оно предлагается только там,
    # где у объекта есть запас под собственным потолком компенсации: закачка
    # добавляется, а она у нас ограничена (Р-61). Без запаса предложение было бы
    # обещанием, которое планировщик тут же срежет.
    converted = plan_conversions(
        forecast, norms, control_start=control_start, lookback=lookback_months,
        headroom=_water_headroom(forecast, limits, fluids, history_rows),
    )
    # остановка и перевод по одной скважине — одно решение, а не два: переводим
    shut_wells = {action.well for action in converted}
    actions = [action for action in actions if action.well not in shut_wells] + converted
    steps = apply(tuned, actions)

    # последним слово за планировщиком: скважинные агенты предлагают, глядя каждый
    # на себя, а вода и мощность по жидкости принадлежат объекту целиком
    allocations: list[planner.Allocation] = []
    if limits is not None:
        # Коридор баланса воды судится **тем же окном**, что и подтверждение —
        # по всей выгрузке, а не с начала управления (решение владельца 04.09,
        # Р-204). Историю до горизонта прогноз не содержит, поэтому накопленное
        # за неё переносится числом.
        carried = _carried_water(
            history_rows,
            min((step.date.year for step in steps), default=None),
            limits, fluids,
        )
        step("приведение к ограничениям объекта")
        steps, allocations = _enforce_to_fixed_point(
            surrogate, steps, norms, limits, fluids, carried
        )

    # Скважины, которыми управлять не разрешено, возвращаются к опорному режиму.
    # Это единственная из трёх форм ограничения по фонду, которую нельзя сделать
    # в планировщике: чтобы вернуть скважину к исходному режиму, нужно само
    # опорное расписание, а планировщик получает уже изменённые шаги (Б40).
    if limits is not None and limits.controllable_wells is not None:
        steps = _restore_uncontrolled(origin, steps, limits.controllable_wells)

    step("прогноз получившегося сценария")
    rows = surrogate.predict(steps)
    result = _npv(rows, start_date, norms_file)
    region = getattr(getattr(surrogate, "bundle", None), "trust_region", None)
    distance = optimise._distance_of(surrogate, steps, region)
    safe_chdd = trust.safe_value(result.chdd_m, distance, margin)

    warnings = list(surrogate.in_training_range(steps))
    warnings.extend(_baseline_outside_the_corridor(base_rows, limits, fluids,
                                                   history_rows))
    # Причины, по которым кандидат обязан пойти на полный расчёт, до 03.09
    # печатал только `scripts/trust_gate.py`, то есть отдельная команда, которую
    # надо было вспомнить запустить. В самом цикле шлюз молчал (ревизия 2.2 #1).
    #
    # Спрашивается только у обученной модели: у тривиального бейзлайна доверенной
    # области нет и быть не может, и «область не собрана» про него — не
    # предупреждение, а описание другой вещи. Заявку он не обосновывает.
    if getattr(surrogate, "bundle", None) is not None:
        # В начало, а не в хвост: `summary()` печатает первые десять
        # предупреждений из семидесяти трёх, и причина, по которой поиск ничего
        # не выбрал, оказывалась невидимой в отчёте цикла (Р-190).
        warnings = trust.routed_to_full_run(distance, margin, region=region) + warnings
    # предел дебита жидкости — из таблицы ЭЦН тех нормативов, по которым считается
    # ЧДД этого прогона, а не из литерала 500 от Модели Z (Б28)
    problems = validate(
        steps, known_wells,
        max_liquid_rate=liquid_limit(economics.load_norms(norms_file)[1]),
        # предел давления нагнетания — свойство кейса, а не Модели Z: у неё во всей
        # истории 300 атм, и это умолчание, а не физический закон (ревизия 2.3 #5)
        **({} if limits is None or limits.max_injector_bhp_atm is None
           else {"max_injection_bhp": limits.max_injector_bhp_atm}),
    )
    # Fail-closed: планировщик прошёл — это ещё не «условия кейса соблюдены».
    # Считается по тому же прогнозу, по которому посчитан ЧДД кандидата.
    case_left = _case_problems(steps, rows, limits, fluids, allocations)

    return CycleResult(
        base_chdd_m=base.chdd_m,
        chdd_m=result.chdd_m,
        actions=tuple(actions),
        rate_choices=tuple(choices),
        allocations=tuple(allocations),
        steps=tuple(steps),
        warnings=tuple(warnings),
        problems=tuple(problems),
        case_problems=tuple(case_left),
        excluded_rows=result.excluded_negative_rows,
        rows=rows,
        raw_chdd_m=result.chdd_m,
        safe_chdd_m=safe_chdd,
        margin=margin,
    )


def confirm(result: CycleResult, rows: list[dict[str, Any]], start_date: date,
            *, base_rows: list[dict[str, Any]],
            limits: planner.Limits | None = None,
            fluids: Fluids | None = None,
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

    **Условия кейса пересчитываются на подтверждённой выгрузке, а не переносятся
    от прогноза.** Прогноз суррогата и отклик ГГДМ — разные числа, и баланс воды
    у них разный: перенос прогнозных `case_problems` давал подтверждённый
    результат с пустым списком нарушений при реальной компенсации 324 % против
    предела 110 % (найдено критиком 03.09). `limits` не передан — значит условий
    не объявлено, и список пуст, ровно как в `run_cycle`.
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
        case_problems=tuple(_case_problems(result.steps, rows, limits, fluids,
                                           list(result.allocations))),
        excluded_rows=confirmed.excluded_negative_rows,
        confirmed=True,
        rows=rows,
        margin=result.margin,
    )


def _restore_uncontrolled(origin: list[ScheduleStep] | tuple[ScheduleStep, ...],
                          steps: list[ScheduleStep],
                          allowed: frozenset[str]) -> list[ScheduleStep]:
    """Вернуть неразрешённым скважинам режим опорного расписания.

    «Разрешено управлять этими» означает, что остальные остаются как были, а не
    «меняются осторожно»: кейс может прямо запретить трогать часть фонда, и
    тогда наше расписание обязано совпасть с исходным на этих скважинах
    побайтово, а не приблизительно.

    **Опора разворачивается в полные шаги, и без этого форма не работала.** В
    деке запись действует до следующей записи по той же скважине, поэтому в
    опорном расписании скважина упомянута не на каждом шаге: у Модели Z это
    84…103 записи при 103 скважинах. Пока восстановление искало запись по дате в
    дельтовом расписании, на шагах без упоминания оно не находило ничего и
    оставляло режим планировщика — то есть запрет действовал на тех месяцах, где
    скважина случайно оказалась переписанной, и не действовал на остальных
    (ревизия 03.09, 2.1 #1). Разворачивать здесь дешевле, чем требовать
    развёрнутую опору от всех вызывающих: та же причина, по которой развёрнут
    вход планировщика (`planner.expand_deltas`).
    """
    by_date = {step.date: step for step in planner.expand_deltas(origin)}
    result: list[ScheduleStep] = []
    for step in steps:
        reference = by_date.get(step.date)
        if reference is None:
            result.append(step)
            continue
        # Шаг собирается заново, а не правится по записям: управляемые скважины
        # берутся из кандидата, все остальные — из опоры целиком, вместе с ролью.
        # Проходом по записям кандидата (как было до 03.09) нельзя отменить
        # **появление** роли: перевод запрещённой скважины под закачку добавлял
        # запись `WCONINJE`, которой в опоре нет, и восстановление её не видело —
        # либо получалось противоречие «открыта и на добычу, и на закачку», либо,
        # если в опоре скважина стояла закрытой, перевод проходил молча.
        producers = tuple(
            [control for control in step.producers if control.well in allowed]
            + [control for control in reference.producers if control.well not in allowed]
        )
        injectors = tuple(
            [control for control in step.injectors if control.well in allowed]
            + [control for control in reference.injectors if control.well not in allowed]
        )
        result.append(ScheduleStep(step.date, producers, injectors))
    return result


#: Сколько раз цикл готов пересчитать прогноз и повторить приведение к
#: ограничениям. Три — не подобранное число, а потолок: на замерах хватало двух
#: проходов, третий оставлен запасом. Больше значило бы прятать расхождение,
#: которое надо не сглаживать, а называть.
MAX_ENFORCE_PASSES = 3


def _carried_water(history_rows: list[dict[str, Any]] | None,
                   first_year: int | None,
                   limits: planner.Limits | None,
                   fluids: Fluids | None) -> tuple[float, float] | None:
    """Отобранное и закачанное **до** первого года горизонта, м³.

    Владелец решил 04.09: коридор судится тем же окном, что и подтверждение, —
    по всей выгрузке. Прогноз истории не содержит (суррогат считает управляемую
    часть), поэтому накопленное за неё переносится одним числом.

    `None` означает «истории нет» — тогда счёт начинается с нуля, как было до
    решения. Так работает всякий вызов без выгрузки: тесты, разбор, `web`.
    """
    if not history_rows or limits is None or first_year is None:
        return None
    withdrawn = 0.0
    injected = 0.0
    for year in balance.by_year(history_rows, fluids):
        if year.year >= first_year:
            continue
        withdrawn += (year.water_m3 if limits.water_balance_rule == "produced_water"
                      else year.liquid_m3)
        injected += year.injected_m3
    if withdrawn <= 0 and injected <= 0:
        return None
    return (withdrawn, injected)


def _enforce_to_fixed_point(surrogate: Surrogate,
                            steps: list[ScheduleStep],
                            norms: Norms,
                            limits: planner.Limits,
                            fluids: Fluids | None,
                            carried: tuple[float, float] | None = None,
                            ) -> tuple[list[ScheduleStep], list[planner.Allocation]]:
    """Приводить расписание к ограничениям, пока оно не перестанет меняться.

    Один проход планировщика этого не даёт, и причина не в реализации, а в
    устройстве задачи. База коридора баланса воды — **отобранный объём**, то есть
    отклик на расписание; планировщик получает прогноз, посчитанный для
    расписания **до** срезок, и после того, как он срежет отбор, тот же коридор
    считался бы уже по другой базе. Замер ревизии 03.09: на репетиции полный
    цикл давал 16 нарушений коридора и компенсацию 133–141 % — то есть
    расписание, только что приведённое к ограничениям, их нарушало (Р-189).

    Поэтому прогноз спрашивается заново, и приведение повторяется. Суррогат для
    того и нужен: спросить его ещё раз стоит миллисекунды, а полный расчёт —
    четверть часа.

    Итерация останавливается, когда расписание совпало с предыдущим, либо на
    `MAX_ENFORCE_PASSES`. Недостижение неподвижной точки не глушится: расписание
    остаётся последним посчитанным, а нарушения назовёт `_case_problems` — они
    для того и считаются по свежему прогнозу.
    """
    decisions: list[planner.Allocation] = []
    current = list(steps)
    for _ in range(MAX_ENFORCE_PASSES):
        fitted, made = planner.enforce(
            current, surrogate.predict(current), norms, limits, fluids,
            carried_water=carried,
        )
        # Решения копятся по всем проходам, а не берутся с последнего: последний
        # проход ничего не режет (потому и последний), и брать его список значило
        # бы отчитаться пустотой о расписании, которое на самом деле урезано.
        decisions.extend(made)
        if fitted == current:
            break
        current = fitted
    return current, decisions


def _case_problems(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   rows: list[dict[str, Any]],
                   limits: planner.Limits | None,
                   fluids: Fluids | None,
                   allocations: list[planner.Allocation]) -> list[str]:
    """Условия кейса, которые получившееся расписание всё-таки нарушает.

    Планировщик приводит расписание к ограничениям объекта, но пройти его и
    остаться допустимым — разные утверждения, и между ними лежат три вещи:
    возврат неуправляемых скважин к опоре идёт **после** планировщика и может
    сломать его срезку; коридор баланса воды планировщик подбирает одним
    множителем, а проверяется он по годам; мощность водозабора — предел расхода,
    и она считается по месяцам. Поэтому итог пересчитывается тем же кодом, что
    и в репетиции приёмки (`scripts/rehearse_case.py`), а непустой список
    делает `is_improvement` ложным: кандидат, нарушающий условие кейса,
    кандидатом не является (А2 ревизии 03.09).

    Возвращаются строки, а не `ScheduleProblem`: у объектного условия нет одной
    скважины и одной даты — «2024-10 по объекту» и «скв. 66, 2021-12» это разные
    адреса, и выдумывать им общий формат значит терять место нарушения.

    **Уступка по квоте мероприятий провалом не считается**, но только если
    планировщик её объявил: мероприятие, порождённое жёстким ограничением, не
    отменяется (Р-139, Р-150), а молчаливое превышение остаётся дефектом. То же
    правило и в тех же словах — в репетиции приёмки.

    Отклик пласта (пластовое давление, обводнённость) сюда не входит:
    расписанием его не задать, и нарушение по нему — повод менять сценарий, а не
    признак негодного расписания. Он идёт в предупреждения.
    """
    if limits is None:
        return []
    conceded = ("max_events_per_year", "max_events_per_month")
    declared = any("вопреки квоте" in one.reason for one in allocations)
    left = [str(check) for check in casecheck.violations(casecheck.check(steps, rows, limits))
            if not (declared and check.key in conceded)]

    if limits.enforce_water_balance:
        if limits.water_balance_rule == "compensation":
            left += balance.compensation_problems(
                rows, fluids, limits.water_tolerance, floor=limits.water_tolerance_min)
        elif limits.water_balance_rule == "external_water":
            left += balance.external_water_problems(
                rows, fluids, limits.max_external_water_m3_per_day)
        else:
            left += balance.problems(rows, fluids, limits.water_tolerance,
                                     floor=limits.water_tolerance_min)

    # мощность источника действует при любом правиле; при неизвестной мощности
    # функция возвращает справочную строку про пиковый месяц — она не нарушение
    if limits.max_external_water_m3_per_day is not None:
        left += balance.external_water_capacity_problems(
            rows, fluids, limits.max_external_water_m3_per_day)
    return left


def _baseline_outside_the_corridor(base_rows: list[dict[str, Any]],
                                   limits: planner.Limits | None,
                                   fluids: Fluids | None,
                                   history_rows: list[dict[str, Any]] | None = None,
                                   ) -> list[str]:
    """Сказать вслух, если **бездействие** уже нарушает наш же потолок.

    Один и тот же потолок 1,3 живёт в трёх разных окнах (С1): в конфигурации он
    обоснован историей **с 1994** (там накопленная компенсация 1,239), цикл
    считает по прогнозу **с 2007** (там сама история даёт 1,401 в 2011 и 1,291 на
    конец), а подтверждение судит по выгрузке снова с 1994. Замер 04.09 на
    `runs/base/rows.csv` — числа именно эти.

    Следствие тяжелее, чем расхождение в третьем знаке: в окне цикла нарушает
    потолок **само бездействие**, поэтому fail-closed отвергает любого кандидата,
    и оператор читает «условия кейса нарушены», не зная, что ничего не делать —
    тоже нарушение. Выбор окна — вопрос к организаторам и решение владельца
    (`BLOCKED.md`), и здесь он не принимается. Здесь он перестаёт быть незаметным.
    """
    if limits is None or not limits.enforce_water_balance:
        return []
    # Тем же окном, что судит подтверждение: накопленное за историю переносится
    # (решение владельца 04.09, Р-204). Иначе предупреждение срабатывало бы на
    # опорном расписании там, где нарушения нет, — окно и создавало нарушение.
    first_year = min((int(str(row["DATA"])[:4]) for row in base_rows), default=None)
    carried = _carried_water(history_rows, first_year, limits, fluids)
    worst = balance.worst_cumulative_compensation(
        base_rows, fluids, limits.water_balance_rule, carried=carried)
    if worst is None or worst[1] <= limits.water_tolerance:
        return []
    year, ratio = worst
    return [
        f"опорное расписание само выходит за потолок компенсации: {ratio:.3f} "
        f"в {year} при потолке {limits.water_tolerance:.2f} — в окне, которым "
        f"считает цикл. Значит, непригодным будет объявлен любой кандидат, включая "
        f"бездействие; окно и потолок надо согласовать до того, как верить отказу"
    ]


def _water_headroom(rows: list[dict[str, Any]],
                    limits: planner.Limits | None,
                    fluids: Fluids | None,
                    history_rows: list[dict[str, Any]] | None = None) -> bool:
    """Есть ли у объекта запас под собственным потолком компенсации.

    Перевод под закачку добавляет воду в пласт, а закачка ограничена нашим же
    правилом (Р-61): без запаса предлагать перевод — значит обещать то, что
    планировщик срежет на следующем шаге. Без ограничений запас считается
    неограниченным: правило выключено, и мешать переводу нечему.

    **Считается тем же счётом, что и аудит, и по тому же правилу (С4).** До 04.09
    здесь стоял годовой максимум `year.compensation` — отношение закачки к
    отобранному **внутри одного года**, — тогда как срезает планировщик и судит
    `balance.compensation_problems` по **накопленному** отношению на конец года.
    Две разные величины на одном пороге расходятся в обе стороны: запас
    объявлялся там, где его нет, и наоборот. Отсюда 56 переводов под закачку на
    репетиции (280 млн руб. OPEX), которые коридор затем срезал: новая закачка
    вытесняла существующую.

    И правило бралось всегда как `compensation`, даже когда кейс объявил
    `produced_water`: там знаменатель — добытая вода, а не отобранная жидкость,
    и число меньше в разы.
    """
    if limits is None or not limits.enforce_water_balance:
        return True
    years = balance.by_year(rows, fluids)
    if not years:
        return True

    # То же окно, что у коридора и у подтверждения: накопленное до горизонта
    # переносится, а не начинается с нуля (решение владельца 04.09, Р-204).
    first_year = min((int(str(row["DATA"])[:4]) for row in rows), default=None)
    carried = _carried_water(history_rows, first_year, limits, fluids)
    withdrawn, injected = carried if carried is not None else (0.0, 0.0)
    worst = 0.0
    for year in years:
        withdrawn += (year.water_m3 if limits.water_balance_rule == "produced_water"
                      else year.liquid_m3)
        injected += year.injected_m3
        if withdrawn > 0:
            worst = max(worst, injected / withdrawn)
    return worst < limits.water_tolerance


def _npv(rows: list[dict[str, Any]], start_date: date,
         norms_file: Path = economics.NORMS_FILE) -> economics.NpvResult:
    return economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date=start_date.isoformat(),
        norms_file=norms_file,
    )
