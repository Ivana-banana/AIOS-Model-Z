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

from aios import balance
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms, counted_by_methodology, month_margin
from aios.summary import Fluids


@dataclass(frozen=True)
class Limits:
    """Ограничения уровня объекта. Приходят из конфигурации валидационного кейса."""

    max_field_liquid_m3_per_day: float | None = None
    """Предел системы сбора по жидкости. None — предела нет."""

    max_field_injection_m3_per_day: float | None = None
    """Предел поверхностной системы ППД по закачке. None — число неизвестно."""

    enforce_water_balance: bool = True
    """Ограничивать закачку балансом воды вообще."""

    max_external_water_m3_per_day: float | None = None
    """Мощность стороннего водоисточника, м³/сут. None — число неизвестно.

    Нужна только правилу `external_water`. Значение по умолчанию для конкретного
    объекта выводится из его же истории — `aios.balance.historic_external_water`
    даёт максимум, который источник заведомо выдавал.
    """

    water_balance_rule: str = "produced_water"
    """Чем ограничена закачка: `produced_water`, `compensation` или `external_water`.

    `produced_water` — буквальное прочтение ответа организаторов «дополнительной
    воды нет»: накопленная закачка не выше накопленной добытой воды. На истории
    Модели Z оно даёт 2,32, то есть **отвергает саму историю месторождения**.

    `compensation` — компенсация отбора закачкой: закачка возмещает отобранный
    объём в пластовых условиях (нефть вместе с водой). На той же истории — 1,291,
    обычная величина для заводнения с поддержанием давления. Отраслевой нормы
    вида «закачка ≤ добытая вода» не существует: нормируются источник, качество и
    максимальный годовой объём (`docs/manus/vrr-water-balance.md`).

    `external_water` — закачка ограничена попутной водой **плюс лимитом
    стороннего источника**. Это единственная из трёх формулировок, которая
    ограничивает то, чем закачка ограничена физически. На Модели Z сторонняя вода
    даёт 60,8 % всей закачки за историю и 682 м³/сут в 2024 — то есть
    `produced_water` описывает не этот объект (`docs/manus/round4.md`, Р-71).
    """

    water_tolerance: float = 1.0
    """Во сколько раз закачка может превышать базу правила.

    Для `produced_water` 1.0 означает «закачиваем ровно столько, сколько подняли»;
    для `compensation` это потолок компенсации — 1.3 держит нас не выше
    исторической компенсации объекта (1,291).
    """


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

    if limits.water_balance_rule not in ("produced_water", "compensation", "external_water"):
        raise ValueError(
            f"неизвестное water_balance_rule: {limits.water_balance_rule!r}; "
            "допустимы 'produced_water', 'compensation' и 'external_water'"
        )

    if limits.enforce_water_balance:
        if limits.water_balance_rule == "compensation":
            result, water = _fit_injection_to_compensation(result, forecast, fluids,
                                                           limits.water_tolerance)
        elif limits.water_balance_rule == "external_water":
            result, water = _fit_injection_to_external_water(
                result, forecast, fluids, limits.max_external_water_m3_per_day
            )
        else:
            result, water = _fit_injection_to_available_water(result, forecast, fluids,
                                                              limits.water_tolerance)
        decisions.extend(water)

    if limits.max_field_liquid_m3_per_day is not None:
        result, liquid = _fit_liquid_to_capacity(result, forecast, norms,
                                                 limits.max_field_liquid_m3_per_day, fluids)
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
    """Ужать закачку до объёма добытой воды."""
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

    return _scale_injection(
        steps, allowed / planned,
        f"закачка {planned:,.0f} м³ превышает добытую воду "
        f"{produced:,.0f} м³; источника сверх неё нет",
    )


def _fit_injection_to_external_water(
    steps: list[ScheduleStep],
    forecast: list[dict[str, Any]],
    fluids: Fluids,
    limit_m3_per_day: float | None,
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ужать закачку до попутной воды плюс лимит стороннего источника.

    Физически закачка ограничена не отношением объёмов, а тем, откуда берётся
    вода: своя попутная плюс то, что даёт водозабор. На Модели Z второе слагаемое
    не поправка, а больше половины — 60,8 % закачки за историю (Р-71).

    `limit_m3_per_day = None` означает «мощность источника неизвестна». Тогда
    **ничего не режем**: выдуманный предел изменит заявляемый ЧДД на число,
    которого никто не измерял. Потребность в сторонней воде при этом остаётся
    видимой — её печатает `aios.balance.external_water_problems`, и она обязана
    попасть в отчёт как оговорка к сценарию.
    """
    if limit_m3_per_day is None:
        return steps, []

    produced = 0.0
    planned = 0.0
    stamps: set[str] = set()
    for row in forecast:
        oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]
        if liquid_t > 0 and 0 <= oil_t <= liquid_t:
            produced += (liquid_t - oil_t) * 1000.0 / fluids.water_density
        planned += max(row["WWIT_Diff"], 0.0)
        stamp = str(row.get("DATA", ""))[:7]
        if len(stamp) == 7:
            stamps.add(stamp)

    if not stamps:
        # Тихо подставить 365 нельзя: на девятнадцатилетнем горизонте это занизило
        # бы допустимую закачку в двадцать раз и срезало бы её без всякой причины.
        # Лимит источника — суточный, и без календаря он не переводится в объём.
        raise ValueError(
            "правилу external_water нужна колонка DATA в прогнозе: лимит источника "
            "задан в м³/сут, и без календаря его не с чем сравнивать"
        )

    days = balance.days_of(stamps)
    allowed = produced + limit_m3_per_day * days
    if planned <= allowed or planned <= 0:
        return steps, []

    need = (planned - produced) / days
    return _scale_injection(
        steps, allowed / planned,
        f"закачке нужно {need:,.0f} м³/сут воды со стороны при мощности источника "
        f"{limit_m3_per_day:,.0f} м³/сут",
    )


def _fit_injection_to_compensation(steps: list[ScheduleStep],
                                   forecast: list[dict[str, Any]],
                                   fluids: Fluids,
                                   ceiling: float) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ужать закачку до компенсации отобранного объёма в пластовых условиях.

    База — нефть вместе с водой, а не одна вода: закачка при заводнении возмещает
    отобранную пустотность, иначе поддерживать пластовое давление было бы нечем.
    Объёмные коэффициенты Модели Z измерены по деку (`Bв = 1,0000`,
    `Bн` = 0,9862…1,0078), то есть поверхностные объёмы здесь практически равны
    пластовым; на другом кейсе это нужно будет пересчитать (Р-61).

    Ужимаем пропорционально по всем нагнетательным — по той же причине, что и в
    `_fit_injection_to_available_water`: чья закачка полезнее, тривиальный
    суррогат сказать не может.
    """
    withdrawn = 0.0
    planned = 0.0
    for row in forecast:
        oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]
        if liquid_t > 0 and 0 <= oil_t <= liquid_t:
            withdrawn += oil_t * 1000.0 / fluids.oil_density
            withdrawn += (liquid_t - oil_t) * 1000.0 / fluids.water_density
        planned += max(row["WWIT_Diff"], 0.0)

    allowed = withdrawn * ceiling
    if planned <= allowed or planned <= 0:
        return steps, []

    return _scale_injection(
        steps, allowed / planned,
        f"закачка {planned:,.0f} м³ даёт компенсацию {planned / withdrawn:.2f} "
        f"от отобранных {withdrawn:,.0f} м³ при потолке {ceiling:.2f}",
    )


def _scale_injection(steps: list[ScheduleStep], factor: float,
                     reason: str) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Пропорционально ужать закачку всех открытых нагнетательных."""
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
                    Allocation(well=control.well, resource="закачка",
                               before=control.rate, after=reduced, reason=reason)
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
                            capacity: float,
                            fluids: Fluids) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ужать отбор жидкости до пропускной способности по объекту.

    Режем по возрастанию удельной маржи: первой уходит скважина, которая приносит
    меньше всех денег на кубометр поднятой жидкости. Это единственный порядок,
    который можно объяснить и защитить — остальные сводятся к «нам так показалось».
    """
    value = _margin_per_liquid(forecast, norms, fluids)
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
        # место в очереди нужно для обоснования: режется несколько скважин, и
        # «наименьшая в фонде» верно только про первую. Ложное обоснование хуже
        # отсутствующего — его нельзя проверить на глаз
        place = {control.well: index + 1 for index, control in enumerate(order)}
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
                            # значащими цифрами, а не четырьмя знаками после
                            # запятой: удельная маржа — тысячные доли миллиона, и
                            # при `.4f` две разные скважины печатались одинаково
                            # (`-0.0001` у обеих, замер 20.08). Единица остаётся
                            # той же, в которой считается ранжирование
                            f"маржа {value.get(control.well, 0.0):.4g} млн руб./м³ — "
                            + ("наименьшая в фонде"
                               if place[control.well] == 1 else
                               f"{place[control.well]}-я снизу из {len(working)} "
                               "работающих; режем по возрастанию, пока отбор не "
                               "уложится в предел")
                        ),
                    )
                )
            producers.append(
                ProducerControl(control.well, "SHUT" if new_rate <= 0 else control.status,
                                control.mode, new_rate or None, control.bhp)
            )
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))

    return result, decisions


def _margin_per_liquid(forecast: list[dict[str, Any]], norms: Norms,
                       fluids: Fluids | None = None) -> dict[str, float]:
    """Сколько скважина приносит на кубометр поднятой жидкости, млн руб./м³.

    Знаменатель — именно объём. В контракте расчётчика `WLPT_Diff` это **тонны**,
    а дефицитный ресурс, предел системы сбора и сама уставка `WCONPROD` заданы в
    **м³/сут**: делить на тонны значит ранжировать фонд по чужой единице.
    Плотность жидкости меняется с обводнённостью от 913 до 1172 кг/м³, так что
    расхождение доходит до 28 % — на порядок среза влияет слабо, потому что маржу
    определяет нефть, но число уходит в объяснение решения человеку.

    Перевод массы в объём — той же конвенцией, что в `aios.balance.by_year`.
    """
    fluids = fluids or Fluids()
    margin: dict[str, float] = {}
    liquid: dict[str, float] = {}
    for row in forecast:
        # Строку, исключённую методикой, ранжирование не считает вовсе. Раньше
        # знаменатель отрицательные приращения обрезал (`max(..., 0.0)`), а
        # числитель считал строку как есть — и первой под нож уходила не худшая
        # по деньгам, а та, у кого выгрузка откатила накопленное.
        if not counted_by_methodology(row):
            continue
        well = row["well"]
        oil_t = max(row["WOMT_Diff"], 0.0)
        liquid_t = max(row["WLPT_Diff"], 0.0)
        volume = (oil_t * 1000.0 / fluids.oil_density
                  + max(liquid_t - oil_t, 0.0) * 1000.0 / fluids.water_density)
        margin[well] = margin.get(well, 0.0) + month_margin(row, norms)
        liquid[well] = liquid.get(well, 0.0) + volume
    return {
        well: (margin[well] / volume if volume > 0 else margin[well])
        for well, volume in liquid.items()
    }
