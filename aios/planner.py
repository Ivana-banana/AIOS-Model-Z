"""Планировщик: согласование решений по скважинам через общие ограничения.

Скважинные агенты предлагают режимы, глядя каждый на себя. Планировщик смотрит
на объект целиком и режет предложения там, где они не помещаются в общий ресурс.
Это и есть та координация, без которой «агент на скважину» не работает: у фонда
есть величины, которые принадлежат всем сразу.

Что здесь координируется по-настоящему:

* **вода на закачку.** Дополнительной воды на объекте нет — закачиваем ту, что
  подняли (разбор организаторов, `docs/method/qa-organisers.md`). Значит, объём
  закачки ограничен добычей воды, и это ограничение уровня объекта, а не скважины;
* **мощность по жидкости.** Система сбора и подготовки имеет предел; в
  валидационном кейсе ограничения по жидкости будут заданы явно. Когда предел
  есть, вопрос «кого резать» — экономический, и решается он одинаково для всех:
  первой режется скважина, которая приносит меньше всех денег на кубометр
  дефицитного ресурса.
* **мощность ППД.** Когда известен общий предел КНС и водоводов, закачка каждого
  шага пропорционально укладывается в него. Ноль в конфиге означает, что число
  организаторами пока не задано, а не что закачка запрещена.

Чего планировщик **не** делает и почему. Он не решает, где закачка полезнее: для
этого надо знать, чья закачка кому поднимает давление, а тривиальный суррогат
межскважинного взаимодействия не видит. Поэтому дефицит воды он раскладывает
пропорционально — честное «никого не выделяем» вместо выдуманного приоритета.
Адресное распределение закачки появится вместе с нейросетевым суррогатом.

Один проход, а не итерации до сходимости: срезав дебиты, мы меняем и добычу воды,
то есть ограничение по закачке сдвигается. Пересчитывать до неподвижной точки
можно, но на прогнозе, который сам приблизителен, это точность, которой нет.
Итог всегда проверяется по факту (`aios.balance.problems`) на полном расчёте.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms, month_margin
from aios.summary import Fluids


@dataclass(frozen=True)
class Limits:
    """Ограничения уровня объекта. Приходят из конфигурации валидационного кейса."""

    max_field_liquid_m3_per_day: float | None = None
    """Предел системы сбора по жидкости. None — предела нет."""

    max_field_injection_m3_per_day: float | None = None
    """Предел поверхностной системы ППД по закачке. None — число неизвестно."""

    enforce_water_balance: bool = True
    """Ограничивать закачку добытой водой."""

    water_tolerance: float = 1.0
    """Допуск к балансу: 1.0 — закачиваем ровно столько, сколько подняли."""


@dataclass(frozen=True)
class Allocation:
    """Одно решение планировщика: кого и почему урезали."""

    well: str
    resource: str
    before: float
    after: float
    reason: str

    @property
    def cut(self) -> float:
        return self.before - self.after

    def __str__(self) -> str:
        return (
            f"скв. {self.well}: {self.resource} {self.before:.4g} → {self.after:.4g} м³/сут — "
            f"{self.reason}"
        )


def enforce(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
            forecast: list[dict[str, Any]],
            norms: Norms,
            limits: Limits,
            fluids: Fluids | None = None) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Привести расписание в соответствие с ограничениями объекта.

    `forecast` — прогноз отклика на это же расписание (суррогат или расчёт).
    Возвращает исправленное расписание и список решений с обоснованием.
    """
    fluids = fluids or Fluids()
    result = list(steps)
    decisions: list[Allocation] = []

    if limits.enforce_water_balance:
        result, water = _fit_injection_to_available_water(result, forecast, fluids,
                                                          limits.water_tolerance)
        decisions.extend(water)

    if limits.max_field_liquid_m3_per_day is not None:
        result, liquid = _fit_liquid_to_capacity(result, forecast, norms,
                                                 limits.max_field_liquid_m3_per_day)
        decisions.extend(liquid)

    if limits.max_field_injection_m3_per_day is not None:
        result, injection = _fit_injection_to_capacity(
            result, limits.max_field_injection_m3_per_day
        )
        decisions.extend(injection)

    return result, decisions


def _fit_injection_to_capacity(
    steps: list[ScheduleStep], capacity: float
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Пропорционально уложить каждый шаг в мощность КНС/водоводов."""
    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []
    for step in steps:
        total = sum(
            control.rate or 0.0
            for control in step.injectors
            if control.status == "OPEN"
        )
        if total <= capacity or total <= 0:
            result.append(step)
            continue

        factor = capacity / total
        injectors = []
        for control in step.injectors:
            if control.status != "OPEN" or control.rate is None:
                injectors.append(control)
                continue
            reduced = control.rate * factor
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(
                    Allocation(
                        well=control.well,
                        resource="мощность ППД",
                        before=control.rate,
                        after=reduced,
                        reason=(
                            f"закачка по объекту {total:,.0f} > предела "
                            f"{capacity:,.0f} м³/сут"
                        ),
                    )
                )
            injectors.append(
                InjectorControl(
                    control.well, control.fluid, control.status, control.mode,
                    reduced, control.bhp,
                )
            )
        result.append(ScheduleStep(step.date, step.producers, tuple(injectors)))
    return result, decisions


def explain(decisions: list[Allocation]) -> str:
    """Отчёт планировщика — в лог, UI и на защиту."""
    if not decisions:
        return "Планировщик не вмешивался: предложения агентов помещаются в ограничения объекта."

    by_resource: dict[str, list[Allocation]] = {}
    for decision in decisions:
        by_resource.setdefault(decision.resource, []).append(decision)

    lines = []
    for resource, items in by_resource.items():
        total = sum(item.cut for item in items)
        lines.append(f"{resource}: урезано {len(items)} скважин, суммарно {total:,.0f} м³/сут")
        lines.extend(f"  {item}" for item in sorted(items, key=lambda x: -x.cut)[:10])
        lines.append("")
    return "\n".join(lines).rstrip()


def _fit_injection_to_available_water(steps: list[ScheduleStep],
                                      forecast: list[dict[str, Any]],
                                      fluids: Fluids,
                                      tolerance: float) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ужать закачку до объёма добытой воды.

    Пропорционально по всем нагнетательным: чья закачка полезнее, тривиальный
    суррогат сказать не может, и выдумывать приоритет здесь было бы хуже, чем
    признать, что мы его не знаем.
    """
    produced = 0.0
    planned = 0.0
    for row in forecast:
        oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]
        if liquid_t > 0 and 0 <= oil_t <= liquid_t:
            produced += (liquid_t - oil_t) * 1000.0 / fluids.water_density
        planned += max(row["WWIT_Diff"], 0.0)

    allowed = produced * tolerance
    if planned <= allowed or planned <= 0:
        return steps, []

    factor = allowed / planned
    decisions: list[Allocation] = []
    seen: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        injectors = []
        for control in step.injectors:
            if control.status != "OPEN" or control.rate is None:
                injectors.append(control)
                continue
            reduced = control.rate * factor
            if control.well not in seen:
                seen.add(control.well)
                decisions.append(
                    Allocation(
                        well=control.well,
                        resource="закачка",
                        before=control.rate,
                        after=reduced,
                        reason=(
                            f"закачка {planned:,.0f} м³ превышает добытую воду "
                            f"{produced:,.0f} м³; источника сверх неё нет"
                        ),
                    )
                )
            injectors.append(
                InjectorControl(control.well, control.fluid, control.status, control.mode,
                                reduced, control.bhp)
            )
        result.append(ScheduleStep(step.date, step.producers, tuple(injectors)))

    return result, decisions


def _fit_liquid_to_capacity(steps: list[ScheduleStep],
                            forecast: list[dict[str, Any]],
                            norms: Norms,
                            capacity: float) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ужать отбор жидкости до пропускной способности по объекту.

    Режем по возрастанию удельной маржи: первой уходит скважина, которая приносит
    меньше всех денег на кубометр поднятой жидкости. Это единственный порядок,
    который можно объяснить и защитить — остальные сводятся к «нам так показалось».
    """
    value = _margin_per_liquid(forecast, norms)
    decisions: list[Allocation] = []
    result: list[ScheduleStep] = []
    reported: set[str] = set()

    for step in steps:
        working = [
            control for control in step.producers
            if control.status == "OPEN" and control.liquid_rate
        ]
        total = sum(control.liquid_rate or 0.0 for control in working)
        if total <= capacity:
            result.append(step)
            continue

        # худшие по деньгам на кубометр — первыми под нож
        order = sorted(working, key=lambda control: value.get(control.well, 0.0))
        excess = total - capacity
        cuts: dict[str, float] = {}
        for control in order:
            if excess <= 0:
                break
            rate = control.liquid_rate or 0.0
            cut = min(rate, excess)
            cuts[control.well] = rate - cut
            excess -= cut

        producers = []
        for control in step.producers:
            if control.well not in cuts:
                producers.append(control)
                continue
            new_rate = cuts[control.well]
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(
                    Allocation(
                        well=control.well,
                        resource="жидкость",
                        before=control.liquid_rate or 0.0,
                        after=new_rate,
                        reason=(
                            f"отбор по объекту {total:,.0f} > предела {capacity:,.0f} м³/сут; "
                            f"маржа {value.get(control.well, 0.0):.4f} млн руб./м³ — "
                            f"наименьшая в фонде"
                        ),
                    )
                )
            producers.append(
                ProducerControl(control.well, "SHUT" if new_rate <= 0 else control.status,
                                control.mode, new_rate or None, control.bhp)
            )
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))

    return result, decisions


def _margin_per_liquid(forecast: list[dict[str, Any]], norms: Norms) -> dict[str, float]:
    """Сколько скважина приносит на кубометр поднятой жидкости, млн руб./м³."""
    margin: dict[str, float] = {}
    liquid: dict[str, float] = {}
    for row in forecast:
        well = row["well"]
        margin[well] = margin.get(well, 0.0) + month_margin(row, norms)
        liquid[well] = liquid.get(well, 0.0) + max(row["WLPT_Diff"], 0.0)
    return {
        well: (margin[well] / volume if volume > 0 else margin[well])
        for well, volume in liquid.items()
    }
