"""Сколько закачки скважина реально примет: предел отдельной нагнетательной.

Мощность ППД лимитом не является — это снято расчётом (Р-59). Настоящее
ограничение живёт на скважине: при равномерном масштабировании закачки максимум
по отдельной скважине уходит с 245 к ~490 м³/сут, а принять его надо при том же
предельном давлении. Справка по геомеханике и заводнению
(`docs/manus/injectivity-limit.md`) даёт критерий:

    J ≥ q_цель / (P_предел − P_пласт)

то есть скважина примет цель, если её коэффициент приёмистости `J` хватает,
чтобы при допустимом забойном давлении продавить нужный расход.

**Предел у нас не выдуманный, а записан в деке.** `WCONINJE` каждой
нагнетательной содержит ограничение по забойному давлению (для Модели Z — 300
атм). Это меняет постановку из справки: переводить устьевое давление в забойное
через TVD, плотность и трение не нужно — ограничение уже задано там, где живёт
критерий приёмистости.

Как считается `J`. Справка требует брать **квазистационарные рабочие точки**
одного режима, а не облако всей истории. Проверено на наших данных, почему это
не придирка: регрессия «BHP по расходу» на всех 224 месяцах даёт отрицательный
наклон у 27 скважин из 41 — за 18 лет меняется пластовое давление, и оно
затирает связь «больше качаем — выше давление». Поэтому наклон оценивается в
скользящем окне 24 месяца, а по скважине берётся медиана наклонов тех окон, где
связь читается. Свободный член той же прямой — оценка пластового давления на
датуме скважины; отдельного замера `P_r` нам не нужно.

Чего этот расчёт **не** доказывает. Давления здесь — из скважинной модели OPM
Flow, а не с промысла: мы диагностируем то, во что верит симулятор, и именно это
правильно для ограничения наших сценариев. Давления гидроразрыва в модели нет
вовсе — симулятор примет любое забойное давление ниже заданного предела, поэтому
вопрос «не уйдёт ли вода в трещину» нашим стендом не проверяется в принципе.
Промысловые методы (SRT, falloff, ПГИ, диагностика Холла с замеренным `P_r`)
остаются единственным способом это утверждать.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW_MONTHS = 24
"""Окно квазистационарности. Внутри него пластовое давление считаем неизменным."""

MIN_WINDOWS = 5
"""Сколько окон должно прочитаться, чтобы оценке `J` вообще верить."""

MIN_CORRELATION = 0.70
"""Порог |ρ| внутри окна: ниже — это не матричная ветвь, а шум режима."""

AT_LIMIT_ATM = 0.1
"""На сколько атмосфер давление может не дотягивать до предела и всё ещё «на пределе»."""


@dataclass(frozen=True)
class Injectivity:
    """Что известно о пределе одной нагнетательной скважины."""

    well: str
    limit: float
    """Предел забойного давления из `WCONINJE`, атм."""

    months: int
    months_at_limit: int
    """Сколько месяцев скважина упиралась в предел: там расход задаёт давление, а не уставка."""

    index: float | None
    """Коэффициент приёмистости, м³/(сут·атм). None — матричная ветвь не читается."""

    reservoir: float | None
    """Оценка пластового давления на датуме скважины, атм."""

    peak: float
    """Максимальный исторический расход, м³/сут."""

    @property
    def deliverable(self) -> float | None:
        """Расход при давлении на пределе, м³/сут. None — оценить нечем."""
        if self.index is None or self.reservoir is None:
            return None
        return max(0.0, self.index * (self.limit - self.reservoir))

    @property
    def limited(self) -> bool:
        """Скважина живёт на пределе давления: прибавка уставки ей ничего не даст."""
        return self.months > 0 and self.months_at_limit / self.months >= 0.5


def bhp_limits(schedule) -> dict[str, float]:
    """Предел забойного давления по скважинам из `WCONINJE` дека."""
    limits: dict[str, float] = {}
    for step in schedule.steps:
        for control in step.injectors:
            if control.bhp is not None:
                limits[control.well] = control.bhp
    return limits


def assess(rows: pd.DataFrame, limits: dict[str, float], since: str,
           window: int = WINDOW_MONTHS) -> dict[str, Injectivity]:
    """Оценка предела по каждой нагнетательной из помесячной выгрузки."""
    injection = rows[(rows["WWIR"] > 0) & (rows["DATA"] >= since)]
    assessment: dict[str, Injectivity] = {}
    for well, history in injection.groupby(injection["well"].astype(str)):
        limit = limits.get(well)
        if limit is None:
            continue
        history = history.sort_values("DATA")
        at_limit = history["BHP"] >= limit - AT_LIMIT_ATM
        index, reservoir = _matrix_branch(history[~at_limit], window)
        assessment[well] = Injectivity(
            well=well, limit=limit, months=len(history),
            months_at_limit=int(at_limit.sum()), index=index, reservoir=reservoir,
            peak=float(history["WWIR"].max()),
        )
    return assessment


def problems(assessment: dict[str, Injectivity], targets: dict[str, float]) -> list[str]:
    """Что в назначенной закачке скважины не примут. Пустой список — возражений нет.

    Две разные причины отказа, и путать их нельзя. Скважина **на пределе
    давления** не примет ничего сверх текущего: её расход уже определяется
    давлением, а не уставкой, и прибавка в `WCONINJE` останется на бумаге.
    Скважине **с читаемой матричной ветвью** можно назвать число: `J·(предел −
    пластовое)`. Скважина, у которой ветвь не читается, не запрещается и не
    разрешается — про неё сказано, что мы не знаем, и это тоже результат.
    """
    found: list[str] = []
    for well, target in sorted(targets.items(), key=_well_order):
        state = assessment.get(well)
        if state is None or target <= state.peak:
            continue
        if state.limited:
            found.append(f"скважина {well}: {state.months_at_limit} из {state.months} "
                         f"месяцев на пределе {state.limit:.0f} атм — прибавка до "
                         f"{target:.0f} м³/сут останется в файле, а не в пласте")
            continue
        deliverable = state.deliverable
        if deliverable is None:
            found.append(f"скважина {well}: цель {target:.0f} м³/сут выше исторического "
                         f"максимума {state.peak:.0f}, а матричная ветвь не читается — "
                         "принять или отвергнуть цель нечем")
        elif target > deliverable:
            found.append(f"скважина {well}: цель {target:.0f} м³/сут выше предела "
                         f"{deliverable:.0f} при давлении {state.limit:.0f} атм "
                         f"(J={state.index:.2f} м³/(сут·атм), пластовое ≈"
                         f"{state.reservoir:.0f} атм)")
    return found


def _matrix_branch(history: pd.DataFrame, window: int) -> tuple[float | None, float | None]:
    """Медианные наклон и свободный член прямой `BHP = P_пласт + q/J` по окнам."""
    rate = history["WWIR"].to_numpy()
    pressure = history["BHP"].to_numpy()
    slopes, intercepts = [], []
    for start in range(len(rate) - window + 1):
        piece, response = rate[start:start + window], pressure[start:start + window]
        if piece.std() <= 0 or response.std() <= 0:
            continue
        slope, intercept = np.polyfit(piece, response, 1)
        if slope > 0 and np.corrcoef(piece, response)[0, 1] >= MIN_CORRELATION:
            slopes.append(slope)
            intercepts.append(intercept)
    if len(slopes) < MIN_WINDOWS:
        return None, None
    return float(1.0 / np.median(slopes)), float(np.median(intercepts))


def _well_order(item: tuple[str, float]) -> tuple[int, str]:
    name = item[0]
    return (int(name), "") if name.isdigit() else (10**9, name)
