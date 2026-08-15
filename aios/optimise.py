"""Подбор режимов работы скважин поверх суррогатной модели.

Остановить убыточную скважину — самое дешёвое улучшение ЧДД, но и самое грубое:
между «работает как есть» и «выключена» лежит весь диапазон режимов. Здесь
перебираются варианты дебита и выбирается лучший **по деньгам**, а не по
инженерному чутью.

## Почему перебор так дёшев

Тривиальный суррогат (`aios.surrogate.DeclineSurrogate`) не знает межскважинного
взаимодействия: обводнённость каждой скважины зависит только от её собственного
накопленного отбора. Из этого следует, что вклад скважины в результат **не
зависит от того, что делают остальные**, и оптимум по фонду распадается на
независимые оптимумы по скважинам. Поэтому вместо `скважины × варианты` прогонов
достаточно `варианты`: прогоняем весь фонд на каждом множителе дебита и по
каждой скважине берём лучший.

**Это свойство исчезнет вместе с тривиальным суррогатом.** Как только появится
нейросеть, обученная на прогонах ГГДМ, закачка соседа начнёт менять отклик
скважины, и раскладывать задачу на независимые куски будет нельзя — понадобится
согласование через планировщика. Здесь это записано явно, чтобы при переходе не
унаследовать молча неверное допущение.

## Чем меряем

Дисконтированной маржой скважины: сумма месячных марж (`aios.margin`), каждая со
своим годовым коэффициентом дисконтирования. Это ровно та величина, которая
складывается в дисконтированную EBITDA методики — закреплено тестом. Полный ЧДД
считается один раз, на собранном плане, и только эталонным расчётчиком.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from aios.deck import ProducerControl, ScheduleStep
from aios.margin import Norms, month_margin
from aios.schedule import MAX_LIQUID_RATE_M3_PER_DAY
from aios.surrogate import Surrogate

#: Множители к дебиту базового сценария. Ноль — остановка, это тоже режим.
#:
#: Вверх по умолчанию не идём, и это не осторожность, а следствие честности
#: бейзлайна. Тривиальный суррогат не знает пластового давления: обводнённость у
#: него зависит только от накопленного отбора, поэтому удвоить дебит — значит
#: получить ту же нефть раньше, а дисконтирование за это доплачивает. Форсаж в
#: такой модели выгоден **всегда** и **всегда** необоснован: в жизни он упирается
#: в давление и резко обводняет скважину. Ставить в расписание решение, которого
#: модель не может подтвердить, — это ровно тот случай, когда суррогат ошибается
#: в нашу пользу, а платим за это мы.
DEFAULT_MULTIPLIERS = (0.0, 0.5, 0.75, 1.0)

#: Форсирование — только осознанно и только когда появится суррогат, обученный
#: на прогонах ГГДМ: `choose_rates(..., multipliers=DEFAULT_MULTIPLIERS + FORCING)`.
FORCING_MULTIPLIERS = (1.25, 1.5)


@dataclass(frozen=True)
class RateChoice:
    """Выбранный режим скважины и то, чем он лучше исходного."""

    well: str
    multiplier: float
    baseline_margin_m: float
    chosen_margin_m: float

    @property
    def gain_m(self) -> float:
        return self.chosen_margin_m - self.baseline_margin_m

    @property
    def changed(self) -> bool:
        return self.multiplier != 1.0

    def __str__(self) -> str:
        if not self.changed:
            return f"скв. {self.well}: режим без изменений"
        what = "остановка" if self.multiplier == 0 else f"дебит ×{self.multiplier:g}"
        return (
            f"скв. {self.well}: {what}, дисконтированная маржа "
            f"{self.baseline_margin_m:+.2f} → {self.chosen_margin_m:+.2f} млн руб."
        )


def discounted_margin_by_well(rows: list[dict[str, Any]],
                              norms: Norms,
                              start_year: int) -> dict[str, float]:
    """Дисконтированная маржа по скважинам, млн руб.

    Коэффициент дисконтирования — годовой, как в методике: `1/(1+WACC)^(y−y₀)`.
    Годы раньше стартового не дисконтируются в обратную сторону: они просто не
    относятся к расчётному горизонту, и их вклад берётся с коэффициентом 1.
    """
    totals: dict[str, float] = {}
    for row in rows:
        year = int(str(row["DATA"])[:4])
        factor = 1.0 / (1.0 + norms.wacc_rate) ** max(year - start_year, 0)
        totals[row["well"]] = totals.get(row["well"], 0.0) + month_margin(row, norms) * factor
    return totals


def choose_rates(surrogate: Surrogate,
                 base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                 norms: Norms,
                 *,
                 start_date: date,
                 multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS) -> list[RateChoice]:
    """Перебрать множители дебита и выбрать по каждой скважине лучший.

    Возвращает решение по каждой добывающей скважине, включая «оставить как
    есть» — так видно и то, что система рассмотрела и отвергла.
    """
    if 1.0 not in multipliers:
        raise ValueError("в наборе множителей обязан быть 1.0 — режим «как есть» это база")

    scores: dict[float, dict[str, float]] = {}
    for multiplier in multipliers:
        rows = surrogate.predict(scale(base_steps, multiplier))
        scores[multiplier] = discounted_margin_by_well(rows, norms, start_date.year)

    wells = sorted(
        {control.well for step in base_steps for control in step.producers},
        key=_well_key,
    )

    choices: list[RateChoice] = []
    for well in wells:
        baseline = scores[1.0].get(well, 0.0)
        best_multiplier, best_margin = 1.0, baseline
        for multiplier in multipliers:
            margin = scores[multiplier].get(well, 0.0)
            if margin > best_margin + 1e-9:
                best_multiplier, best_margin = multiplier, margin
        choices.append(
            RateChoice(
                well=well,
                multiplier=best_multiplier,
                baseline_margin_m=baseline,
                chosen_margin_m=best_margin,
            )
        )
    return choices


def scale(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
          multiplier: float,
          wells: set[str] | None = None) -> list[ScheduleStep]:
    """Умножить дебит добывающих скважин на множитель.

    Ноль означает остановку, а не открытую скважину с нулевым дебитом: последнее
    методика считает некорректной записью, да и физического смысла в нём нет.
    Верхняя граница 500 м³/сут соблюдается здесь же — расписание, которое её
    нарушает, не должно даже собираться.
    """
    result: list[ScheduleStep] = []
    for step in steps:
        producers = []
        for control in step.producers:
            if wells is not None and control.well not in wells:
                producers.append(control)
                continue
            producers.append(_scaled(control, multiplier))
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))
    return result


def apply_choices(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                  choices: list[RateChoice]) -> list[ScheduleStep]:
    """Собрать расписание из выбранных по скважинам режимов."""
    by_well = {choice.well: choice.multiplier for choice in choices}
    result: list[ScheduleStep] = []
    for step in steps:
        producers = tuple(
            _scaled(control, by_well.get(control.well, 1.0)) for control in step.producers
        )
        result.append(ScheduleStep(step.date, producers, step.injectors))
    return result


def explain(choices: list[RateChoice]) -> str:
    """Отчёт о подборе — в лог, UI и презентацию."""
    changed = [choice for choice in choices if choice.changed]
    if not changed:
        return f"Режимы не менялись: рассмотрено {len(choices)} скважин, лучше исходного нет."

    gain = sum(choice.gain_m for choice in changed)
    stopped = sum(1 for choice in changed if choice.multiplier == 0)
    lines = [
        f"Изменено режимов: {len(changed)} из {len(choices)} "
        f"(из них остановок {stopped}); прирост дисконтированной маржи {gain:+.1f} млн руб.",
        "",
    ]
    lines.extend(f"  {choice}" for choice in sorted(changed, key=lambda item: -item.gain_m))
    return "\n".join(lines)


def _scaled(control: ProducerControl, multiplier: float) -> ProducerControl:
    if multiplier == 1.0 or control.status != "OPEN":
        return control
    if multiplier == 0.0:
        return ProducerControl(control.well, "SHUT", control.mode, None, control.bhp)
    rate = (control.liquid_rate or 0.0) * multiplier
    if rate <= 0:
        return ProducerControl(control.well, "SHUT", control.mode, None, control.bhp)
    return ProducerControl(
        control.well,
        control.status,
        control.mode,
        min(rate, MAX_LIQUID_RATE_M3_PER_DAY),
        control.bhp,
    )


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
