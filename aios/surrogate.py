"""Суррогатные модели: интерфейс и тривиальный бейзлайн.

Суррогат отвечает на один вопрос: *что даст фонд, если управлять им вот так* —
и отвечает за миллисекунды вместо часов полного расчёта. Выход сразу в контракте
эталонного расчётчика (14 колонок), чтобы прогноз можно было оценить теми же
деньгами, какими нас будут оценивать.

Здесь два предмета:

* `Surrogate` — протокол. Нейросеть встанет на это же место.
* `DeclineSurrogate` — тупой бейзлайн: обводнённость растёт по накопленному отбору
  жидкости, нефть — остаток. Никакого обучения, только подгонка прямой на скважину.

Зачем бейзлайн нужен раньше нейросети. Во-первых, он замыкает цикл МАС до того,
как посчитан хоть один сценарий на ГГДМ, — вся работа над агентами и UI перестаёт
ждать расчётов. Во-вторых, он задаёт планку: нейросеть, не побившая подгонку
прямой, не существует, и это тоже результат для отчёта.

Чего бейзлайн **не** умеет и не притворяется, что умеет: межскважинного
взаимодействия. Он не знает, что закачка соседа поднимет давление, и что
остановка скважины перераспределит потоки. Ровно это и должна принести нейросеть,
обученная на прогонах ГГДМ, — и разница между ними будет измеримой.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.summary import Fluids

#: Дальше этой обводнённости не экстраполируем: 1.0 означает нефти ровно ноль,
#: а прямая, продолженная за единицу, даёт отрицательную добычу.
MAX_WATERCUT = 0.9995

#: Минимум месяцев работы, по которым имеет смысл что-то подгонять.
MIN_HISTORY_MONTHS = 3

#: Во сколько раз дебит может превысить исторический максимум скважины, прежде чем
#: прогноз считается экстраполяцией. Единица означает: **любой** режим выше того,
#: на котором скважина реально работала, — уже экстраполяция.
#:
#: Порог именно такой не из осторожности, а по механике бейзлайна. Он не знает
#: пластового давления: обводнённость у него зависит только от накопленного отбора,
#: поэтому поднять дебит вдвое — значит просто получить ту же нефть раньше, а
#: дисконтирование за это платит. Форсаж в нём выгоден всегда, и всегда неверно:
#: в жизни он упирается в давление и резко обводняет скважину, о чём организаторы
#: говорили прямо. Пока нет суррогата, обученного на прогонах ГГДМ, решение
#: «форсировать» обязано выходить наверх с предупреждением, а не тихо в расписание.
EXTRAPOLATION_MARGIN = 1.0

#: Насколько закачка может отойти от исторической, прежде чем прогноз объявляется
#: недостоверным. Порог маленький намеренно: бейзлайн отклик на закачку не
#: моделирует вовсе — добыча в нём считается по собственному дебиту скважины, —
#: поэтому изменение закачки не меняет прогноз ни на тонну. Молчать об этом
#: нельзя: именно так 16.08 планировщик урезал закачку в 4,7 раза под обещание
#: +646,5 млн руб., а полный расчёт дал −9295,4.
INJECTION_CHANGE_TOLERANCE = 0.05


class Surrogate(Protocol):
    """Что обязана уметь любая суррогатная модель проекта."""

    def predict(self, steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[dict[str, Any]]:
        """Прогноз отклика фонда на управление, строки «скважина × месяц»."""

    def in_training_range(self, steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[str]:
        """Чем предложенное управление выходит за то, на чём модель обучалась.

        Пустой список — прогноз внутри диапазона. Непустой — предупреждения,
        которые обязаны дойти до решения, а не остаться в логе: оптимизатор
        целенаправленно тянет управление туда, где суррогат ошибается в нашу
        пользу.
        """


@dataclass(frozen=True)
class NumericRange:
    """Замкнутый диапазон величины, фактически встречавшийся в обучении."""

    minimum: float
    maximum: float

    def contains(self, value: float, *, tolerance: float = 1e-9) -> bool:
        return self.minimum - tolerance <= value <= self.maximum + tolerance


@dataclass(frozen=True)
class TrainingEnvelope:
    """Область применимости суррогата по скважинам и объекту целиком."""

    producer_rates: dict[str, NumericRange]
    injector_rates: dict[str, NumericRange]
    field_liquid_rate: NumericRange
    field_injection_rate: NumericRange

    @classmethod
    def fit(cls, rows: list[dict[str, Any]]) -> TrainingEnvelope:
        """Собрать границы из строк `scenario x well x month` датасета DoE."""
        if not rows:
            raise ValueError("обучающий датасет пуст")

        producers: dict[str, list[float]] = {}
        injectors: dict[str, list[float]] = {}
        field: dict[tuple[str, str], tuple[float, float]] = {}
        for row in rows:
            scenario = str(row.get("scenario", ""))
            month = str(row["DATA"])
            well = str(row["well"])
            liquid = max(float(row["WLPR"]), 0.0)
            injection = max(float(row["WWIR"]), 0.0)
            if liquid > 0:
                producers.setdefault(well, []).append(liquid)
            if injection > 0:
                injectors.setdefault(well, []).append(injection)
            total_liquid, total_injection = field.get((scenario, month), (0.0, 0.0))
            field[(scenario, month)] = (total_liquid + liquid, total_injection + injection)

        return cls(
            producer_rates={well: _numeric_range(values) for well, values in producers.items()},
            injector_rates={well: _numeric_range(values) for well, values in injectors.items()},
            field_liquid_rate=_numeric_range([value[0] for value in field.values()]),
            field_injection_rate=_numeric_range([value[1] for value in field.values()]),
        )

    def warnings(self,
                 steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[str]:
        """Объяснить, чем расписание выходит за область обучающих прогонов."""
        warnings: list[str] = []
        for month_start, controls in expand(steps).items():
            total_liquid = 0.0
            total_injection = 0.0
            for well, control in controls.items():
                if control.status != "OPEN":
                    continue
                if isinstance(control, ProducerControl):
                    rate = max(control.liquid_rate or 0.0, 0.0)
                    total_liquid += rate
                    warnings.extend(
                        _range_warning(month_start, well, "дебит", rate,
                                       self.producer_rates.get(well))
                    )
                else:
                    rate = max(control.rate or 0.0, 0.0)
                    total_injection += rate
                    warnings.extend(
                        _range_warning(month_start, well, "закачка", rate,
                                       self.injector_rates.get(well))
                    )

            warnings.extend(_field_range_warning(
                month_start, "суммарный дебит", total_liquid, self.field_liquid_rate
            ))
            warnings.extend(_field_range_warning(
                month_start, "суммарная закачка", total_injection, self.field_injection_rate
            ))
        return warnings


@dataclass(frozen=True)
class WellDecline:
    """Подгонка по одной скважине: обводнённость как функция накопленного отбора."""

    well: str
    watercut_at_start: float
    """Обводнённость (объёмная) на конец истории, доля."""

    watercut_slope: float
    """Прирост обводнённости на 1000 м³ накопленного отбора жидкости."""

    cumulative_liquid_m3: float
    max_liquid_rate_seen: float
    months_fitted: int


@dataclass(frozen=True)
class DeclineSurrogate:
    """Тупой бейзлайн: планируемый отбор жидкости минус растущая обводнённость."""

    wells: dict[str, WellDecline]
    fluids: Fluids

    injection_range: dict[str, NumericRange] = field(default_factory=dict)
    """Исторический диапазон приёмистости по нагнетательным, м³/сут.

    Именно диапазон, а не максимум: закачка по скважине за историю гуляет, и
    сравнение с одним лишь максимумом объявляло бы изменением любой спокойный
    месяц. Отклик на закачку бейзлайн не моделирует, поэтому единственное, что он
    может честно сделать, — заметить выход за пределы виденного и сказать об этом.
    """

    @classmethod
    def fit(cls, history: list[dict[str, Any]], fluids: Fluids | None = None) -> DeclineSurrogate:
        """Подогнать по историческим строкам в контракте расчётчика."""
        fluids = fluids or Fluids()
        by_well: dict[str, list[dict[str, Any]]] = {}
        for row in history:
            by_well.setdefault(row["well"], []).append(row)

        fitted: dict[str, WellDecline] = {}
        injection_range: dict[str, NumericRange] = {}
        for well, rows in by_well.items():
            rows.sort(key=lambda item: item["DATA"])
            decline = _fit_well(well, rows, fluids)
            if decline is not None:
                fitted[well] = decline
            injected = [float(row["WWIR"]) for row in rows if float(row["WWIR"]) > 0]
            if injected:
                injection_range[well] = _numeric_range(injected)
        return cls(wells=fitted, fluids=fluids, injection_range=injection_range)

    def predict(self, steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[dict[str, Any]]:
        oil_to_t = self.fluids.oil_density / 1000.0
        water_to_t = self.fluids.water_density / 1000.0

        state = {well: decline.cumulative_liquid_m3 for well, decline in self.wells.items()}
        cumulative_oil_t: dict[str, float] = {}
        cumulative_liquid_t: dict[str, float] = {}
        cumulative_injection: dict[str, float] = {}

        rows: list[dict[str, Any]] = []
        for month_start, controls in expand(steps).items():
            days = _days_in_month(month_start)
            for well, control in sorted(controls.items(), key=lambda item: _well_key(item[0])):
                if isinstance(control, InjectorControl):
                    rate = control.rate or 0.0 if control.status == "OPEN" else 0.0
                    volume = rate * days
                    cumulative_injection[well] = cumulative_injection.get(well, 0.0) + volume
                    rows.append(
                        _row(month_start, well, liquid_rate=0.0, oil_rate_t=0.0,
                             injection_rate=rate, cumulative_liquid_t=cumulative_liquid_t.get(well, 0.0),
                             cumulative_oil_t=cumulative_oil_t.get(well, 0.0),
                             cumulative_injection=cumulative_injection[well],
                             liquid_t=0.0, oil_t=0.0, injection_m3=volume)
                    )
                    continue

                decline = self.wells.get(well)
                rate = control.liquid_rate or 0.0 if control.status == "OPEN" else 0.0
                if decline is None or rate <= 0:
                    rows.append(
                        _row(month_start, well, liquid_rate=0.0, oil_rate_t=0.0, injection_rate=0.0,
                             cumulative_liquid_t=cumulative_liquid_t.get(well, 0.0),
                             cumulative_oil_t=cumulative_oil_t.get(well, 0.0),
                             cumulative_injection=cumulative_injection.get(well, 0.0),
                             liquid_t=0.0, oil_t=0.0, injection_m3=0.0)
                    )
                    continue

                liquid_m3 = rate * days
                produced = state.get(well, decline.cumulative_liquid_m3)
                watercut = _watercut_at(decline, produced + liquid_m3 / 2.0)  # середина месяца
                state[well] = produced + liquid_m3

                oil_m3 = liquid_m3 * (1.0 - watercut)
                water_m3 = liquid_m3 - oil_m3
                oil_t = oil_m3 * oil_to_t
                liquid_t = oil_t + water_m3 * water_to_t

                cumulative_oil_t[well] = cumulative_oil_t.get(well, 0.0) + oil_t
                cumulative_liquid_t[well] = cumulative_liquid_t.get(well, 0.0) + liquid_t

                rows.append(
                    _row(month_start, well, liquid_rate=rate, oil_rate_t=oil_t / days,
                         injection_rate=0.0,
                         cumulative_liquid_t=cumulative_liquid_t[well],
                         cumulative_oil_t=cumulative_oil_t[well],
                         cumulative_injection=cumulative_injection.get(well, 0.0),
                         liquid_t=liquid_t, oil_t=oil_t, injection_m3=0.0)
                )

        rows.sort(key=lambda row: (row["DATA"], _well_key(row["well"])))
        return rows

    def in_training_range(self,
                          steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[str]:
        """Предупреждения о режимах, которых скважина в истории не видела.

        Подгонка прямой честна только рядом с данными, на которых подогнана.
        Форсированный отбор вдвое выше исторического — не прогноз, а фантазия,
        и решение обязано об этом знать.
        """
        warnings: list[str] = []
        for month_start, controls in expand(steps).items():
            for well, control in controls.items():
                if isinstance(control, InjectorControl):
                    warning = self._injection_warning(month_start, well, control)
                    if warning:
                        warnings.append(warning)
                    continue
                if control.status != "OPEN":
                    continue
                decline = self.wells.get(well)
                rate = control.liquid_rate or 0.0
                if decline is None:
                    if rate > 0:
                        warnings.append(
                            f"{month_start.isoformat()} скв. {well}: истории работы нет, "
                            f"прогноз не на чем строить"
                        )
                    continue
                if rate > decline.max_liquid_rate_seen * EXTRAPOLATION_MARGIN:
                    warnings.append(
                        f"{month_start.isoformat()} скв. {well}: дебит {rate:g} м3/сут выше "
                        f"исторического максимума {decline.max_liquid_rate_seen:g} м3/сут — "
                        f"форсаж, прогноз недостоверен"
                    )
        return warnings

    def _injection_warning(self, month_start: date, well: str,
                           control: InjectorControl) -> str | None:
        """Закачку изменили — а бейзлайн этого изменения не видит.

        Отклик на закачку в модель не входит: добыча считается по собственному
        дебиту скважины и кривой обводнённости. Значит, любое отклонение от
        исторической приёмистости прогноз не меняет — и именно поэтому о нём
        обязано быть сказано вслух, а не выясняться полным расчётом задним числом.
        """
        rate = (control.rate or 0.0) if control.status == "OPEN" else 0.0
        seen = self.injection_range.get(well)

        if seen is None:
            if rate <= 0:
                return None
            return (f"{month_start.isoformat()} скв. {well}: закачка {rate:g} м3/сут, "
                    f"а в истории закачки не было — отклик на неё не моделируется")

        slack = max(seen.maximum, 1.0) * INJECTION_CHANGE_TOLERANCE
        if seen.minimum - slack <= rate <= seen.maximum + slack:
            return None
        return (f"{month_start.isoformat()} скв. {well}: закачка {rate:g} м3/сут вне "
                f"исторических {seen.minimum:g}…{seen.maximum:g} — бейзлайн отклик на "
                f"закачку не моделирует, прогноз это изменение не учитывает")


def expand(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
           ) -> dict[date, dict[str, ProducerControl | InjectorControl]]:
    """Развернуть расписание в действующие режимы на каждом шаге.

    В Eclipse режим действует до следующего явного изменения: скважину, не
    перечисленную на шаге, не останавливают — ей оставляют прежний режим. Все,
    кто читает расписание (суррогат, валидатор, UI), обязаны понимать это
    одинаково, поэтому раскрытие живёт в одном месте.
    """
    effective: dict[str, ProducerControl | InjectorControl] = {}
    timeline: dict[date, dict[str, ProducerControl | InjectorControl]] = {}

    for step in steps:
        for control in step.producers:
            effective[control.well] = control
        for control in step.injectors:
            effective[control.well] = control
        timeline[step.date] = dict(effective)
    return timeline


def _fit_well(well: str, rows: list[dict[str, Any]], fluids: Fluids) -> WellDecline | None:
    """Подогнать прямую «обводнённость от накопленного отбора»."""
    points: list[tuple[float, float]] = []
    cumulative = 0.0
    max_rate = 0.0

    for row in rows:
        oil_t = row["WOMT_Diff"]
        liquid_t = row["WLPT_Diff"]
        if liquid_t <= 0 or oil_t < 0 or oil_t > liquid_t:
            continue  # простой или мусорная строка — в подгонку не берём

        oil_m3 = oil_t * 1000.0 / fluids.oil_density
        water_m3 = (liquid_t - oil_t) * 1000.0 / fluids.water_density
        liquid_m3 = oil_m3 + water_m3
        if liquid_m3 <= 0:
            continue

        cumulative += liquid_m3
        max_rate = max(max_rate, row["WLPR"])
        points.append((cumulative, min(water_m3 / liquid_m3, MAX_WATERCUT)))

    if len(points) < MIN_HISTORY_MONTHS:
        return None

    slope = _slope(points)
    return WellDecline(
        well=well,
        watercut_at_start=points[-1][1],
        watercut_slope=slope,
        cumulative_liquid_m3=cumulative,
        max_liquid_rate_seen=max_rate,
        months_fitted=len(points),
    )


def _slope(points: list[tuple[float, float]]) -> float:
    """Наклон прямой методом наименьших квадратов; при вырождении — ноль."""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    if variance <= 0:
        return 0.0
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return covariance / variance


def _numeric_range(values: list[float]) -> NumericRange:
    if not values:
        return NumericRange(0.0, 0.0)
    return NumericRange(min(values), max(values))


def _range_warning(month_start: date, well: str, label: str, value: float,
                   bounds: NumericRange | None) -> list[str]:
    if value <= 0:
        return []
    prefix = f"{month_start.isoformat()} скв. {well}: {label} {value:g} м3/сут"
    if bounds is None:
        return [f"{prefix}, такой режим отсутствует в обучающих прогонах"]
    if not bounds.contains(value):
        return [
            f"{prefix} вне обучающего диапазона "
            f"{bounds.minimum:g}..{bounds.maximum:g} м3/сут"
        ]
    return []


def _field_range_warning(month_start: date, label: str, value: float,
                         bounds: NumericRange) -> list[str]:
    if bounds.contains(value):
        return []
    return [
        f"{month_start.isoformat()}: {label} {value:g} м3/сут вне обучающего "
        f"диапазона {bounds.minimum:g}..{bounds.maximum:g} м3/сут"
    ]


def _watercut_at(decline: WellDecline, cumulative_liquid_m3: float) -> float:
    """Обводнённость при заданном накопленном отборе, зажатая в физический диапазон."""
    delta = cumulative_liquid_m3 - decline.cumulative_liquid_m3
    value = decline.watercut_at_start + decline.watercut_slope * delta
    return min(max(value, 0.0), MAX_WATERCUT)


def _row(month_start: date, well: str, *, liquid_rate: float, oil_rate_t: float,
         injection_rate: float, cumulative_liquid_t: float, cumulative_oil_t: float,
         cumulative_injection: float, liquid_t: float, oil_t: float,
         injection_m3: float) -> dict[str, Any]:
    """Строка в контракте эталонного расчётчика. Порядок колонок обязателен."""
    return {
        "DATA": month_start.isoformat(),
        "well": well,
        "WLPT": cumulative_liquid_t,
        "WLPR": liquid_rate,
        "WOMT": cumulative_oil_t,
        "WOMR": oil_rate_t,
        "WWIR": injection_rate,
        "WWIT": cumulative_injection,
        "THP": 0.0,   # давления бейзлайн не прогнозирует; на ЧДД они не влияют
        "BHP": 0.0,
        "WEFF": 1.0,
        "WLPT_Diff": liquid_t,
        "WOMT_Diff": oil_t,
        "WWIT_Diff": injection_m3,
    }


def _days_in_month(value: date) -> int:
    return calendar.monthrange(value.year, value.month)[1]


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
