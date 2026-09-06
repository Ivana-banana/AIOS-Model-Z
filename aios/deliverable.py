"""Сколько скважина реально отдаст и примет: предел по забойному давлению.

Мощность ППД лимитом не является — это снято расчётом (Р-59). Настоящее
ограничение живёт на скважине и записано прямо в деке:

* у нагнетательной `WCONINJE` задаёт **верхний** предел забойного давления
  (Модель Z: 300 атм). Выше него закачку не продавить;
* у добывающей `WCONPROD` задаёт **нижний** предел (Модель Z: 50 атм). Ниже него
  депрессию не увеличить.

Справка по достижимости роста приёмистости (`docs/manus/injectivity-limit.md`)
даёт критерий в общем виде: скважина возьмёт цель, если коэффициента
приёмистости (продуктивности) хватает, чтобы продавить нужный расход при
допустимом давлении:

    q_возможный = J · |P_предел − P_пласт|

Скважина, которая **уже** живёт на своём пределе, управляется давлением, а не
уставкой: прибавка в `WCONPROD`/`WCONINJE` останется в файле и не дойдёт до
пласта. На Модели Z таких большинство, и это меряется, а не предполагается
(Р-65, Р-66).

Как считается `J`. Справка требует брать **квазистационарные рабочие точки**
одного режима, а не облако всей истории. Проверено, почему это не придирка:
регрессия «BHP по расходу» на всех 224 месяцах даёт наклон неверного знака у 27
нагнетательных из 41 — за 18 лет меняется пластовое давление и затирает связь
«больше качаем — выше давление». Поэтому наклон оценивается в скользящем окне 24
месяца, а по скважине берётся медиана наклонов тех окон, где связь читается.
Свободный член той же прямой — оценка пластового давления на датуме скважины;
отдельного замера `P_r` не нужно.

Чего этот расчёт **не** доказывает. Давления здесь — из скважинной модели OPM
Flow, а не с промысла: мы диагностируем то, во что верит симулятор, и именно это
правильно для ограничения наших сценариев. Давления гидроразрыва в модели нет
вовсе — симулятор примет любое забойное давление ниже заданного предела, поэтому
вопрос «не уйдёт ли вода в трещину» нашим стендом не проверяется в принципе.
Промысловые методы (SRT, falloff, ПГИ, диагностика Холла с замеренным `P_r`)
остаются единственным способом это утверждать.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

WINDOW_MONTHS = 24
"""Окно квазистационарности. Внутри него пластовое давление считаем неизменным."""

MIN_WINDOWS = 5
"""Сколько окон должно прочитаться, чтобы оценке вообще верить."""

MIN_CORRELATION = 0.70
"""Порог |ρ| внутри окна: ниже — это не рабочая ветвь, а шум режима."""

AT_LIMIT_ATM = 0.1
"""На сколько атмосфер давление может не дотягивать до предела и всё ещё «на пределе»."""

RATE = {"injector": "WWIR", "producer": "WLPR"}
"""Чем меряется режим: закачка или отбор жидкости, м³/сут."""

SIDE = {"injector": 1.0, "producer": -1.0}
"""С какой стороны предел. У нагнетательной давление растёт с расходом, у добывающей падает."""


@dataclass(frozen=True)
class Well:
    """Что известно о пределе одной скважины."""

    well: str
    kind: str
    """`injector` или `producer`."""

    limit: float
    """Предел забойного давления из `WCONINJE`/`WCONPROD`, атм."""

    months: int
    months_at_limit: int
    """Сколько месяцев скважина стояла на пределе: там расход задаёт давление, а не уставка."""

    index: float | None
    """Коэффициент приёмистости или продуктивности, м³/(сут·атм). None — ветвь не читается."""

    reservoir: float | None
    """Оценка пластового давления на датуме скважины, атм."""

    peak: float
    """Максимальный исторический расход, м³/сут."""

    @property
    def deliverable(self) -> float | None:
        """Расход при давлении на пределе, м³/сут. None — оценить нечем."""
        if self.index is None or self.reservoir is None:
            return None
        return max(0.0, self.index * SIDE[self.kind] * (self.limit - self.reservoir))

    @property
    def limited(self) -> bool:
        """Скважина живёт на пределе давления: прибавка уставки ей ничего не даст."""
        return self.months > 0 and self.months_at_limit / self.months >= 0.5


def bhp_limits(schedule) -> dict[str, tuple[str, float]]:
    """Предел забойного давления по скважинам: `скважина → (тип, атм)`.

    Берётся последнее назначение по ходу расписания: если организаторы меняли
    предел, действует тот, с которым скважина работает в конце истории.
    """
    limits: dict[str, tuple[str, float]] = {}
    for step in schedule.steps:
        for control in step.producers:
            if control.bhp is not None:
                limits[control.well] = ("producer", control.bhp)
        for control in step.injectors:
            if control.bhp is not None:
                limits[control.well] = ("injector", control.bhp)
    return limits


def assess(rows: pd.DataFrame, limits: dict[str, tuple[str, float]], since: str,
           window: int = WINDOW_MONTHS) -> dict[str, Well]:
    """Оценка предела по каждой скважине из помесячной выгрузки."""
    window_rows = rows[rows["DATA"] >= since]
    names = window_rows["well"].astype(str)
    assessment: dict[str, Well] = {}
    for well, history in window_rows.groupby(names):
        if well not in limits:
            continue
        kind, limit = limits[well]
        history = history[history[RATE[kind]] > 0].sort_values("DATA")
        if history.empty:
            continue
        at_limit = _at_limit(history["BHP"], kind, limit)
        index, reservoir = _working_branch(history[~at_limit], kind, window)
        assessment[well] = Well(
            well=well, kind=kind, limit=limit, months=len(history),
            months_at_limit=int(at_limit.sum()), index=index, reservoir=reservoir,
            peak=float(history[RATE[kind]].max()),
        )
    return assessment


def problems(assessment: dict[str, Well], targets: dict[str, float]) -> list[str]:
    """Что в назначенных режимах скважины не выполнят. Пустой список — возражений нет.

    Три разных исхода, и путать их нельзя. Скважина **на пределе давления** не
    даст ничего сверх текущего: её расход уже определяется давлением. Скважине
    **с читаемой рабочей ветвью** можно назвать число: `J·|предел − пластовое|`.
    Скважина, у которой ветвь не читается, не запрещается и не разрешается — про
    неё сказано, что мы не знаем, и это тоже результат.
    """
    found: list[str] = []
    what = {"injector": "прибавка", "producer": "форсаж"}
    for well, target in sorted(targets.items(), key=_well_order):
        state = assessment.get(well)
        if state is None or target <= state.peak:
            continue
        if state.limited:
            found.append(f"скважина {well}: {state.months_at_limit} из {state.months} "
                         f"месяцев на пределе {state.limit:.0f} атм — "
                         f"{what[state.kind]} до {target:.0f} м³/сут останется в файле, "
                         "а не в пласте")
            continue
        deliverable = state.deliverable
        if deliverable is None:
            found.append(f"скважина {well}: цель {target:.0f} м³/сут выше исторического "
                         f"максимума {state.peak:.0f}, а рабочая ветвь не читается — "
                         "принять или отвергнуть цель нечем")
        elif target > deliverable:
            found.append(f"скважина {well}: цель {target:.0f} м³/сут выше предела "
                         f"{deliverable:.0f} при давлении {state.limit:.0f} атм "
                         f"(J={state.index:.2f} м³/(сут·атм), пластовое ≈"
                         f"{state.reservoir:.0f} атм)")
    return found


def _at_limit(pressure: pd.Series, kind: str, limit: float) -> pd.Series:
    if kind == "injector":
        return pressure >= limit - AT_LIMIT_ATM
    return pressure <= limit + AT_LIMIT_ATM


def _working_branch(history: pd.DataFrame, kind: str,
                    window: int) -> tuple[float | None, float | None]:
    """Медианные наклон и свободный член прямой `BHP = P_пласт ± q/J` по окнам."""
    side = SIDE[kind]
    rate = history[RATE[kind]].to_numpy()
    pressure = history["BHP"].to_numpy()
    slopes, intercepts = [], []
    for start in range(len(rate) - window + 1):
        piece, response = rate[start:start + window], pressure[start:start + window]
        # Три подмены в этих двух строках мутационный прогон помечает
        # эквивалентными, и все три вердикта верны (Р-243).
        # `or` против `and`: вырожденное по любой оси окно всё равно отсеется
        # ниже — `np.corrcoef` при нулевой дисперсии даёт `nan`, а сравнение с
        # `nan` ложно. `side * slope` против `side / slope`: знак произведения и
        # знак частного совпадают при любом ненулевом наклоне — проверено
        # перебором, — а нулевой наклон даёт ту же `nan`-корреляцию.
        # `>=` против `>` у порога: попасть коэффициентом корреляции ровно в
        # 0,70 нельзя, он считается из тех же float.
        if piece.std() <= 0 or response.std() <= 0:
            continue
        slope, intercept = np.polyfit(piece, response, 1)
        if side * slope > 0 and side * np.corrcoef(piece, response)[0, 1] >= MIN_CORRELATION:
            slopes.append(slope)
            intercepts.append(intercept)
    if len(slopes) < MIN_WINDOWS:
        return None, None
    return float(side / np.median(slopes)), float(np.median(intercepts))


def _well_order(item: tuple[str, float]) -> tuple[int, str]:
    name = item[0]
    return (int(name), "") if name.isdigit() else (10**9, name)


def planned_targets(steps: Sequence[Any], kind: str = "injector") -> pd.DataFrame:
    """Уставки расписания по шагам: колонки `well`, `DATA`, `target`.

    Отдельной функцией, потому что уставка действует **с** даты шага и до
    следующей: сопоставлять её с выгрузкой по точному совпадению дат можно
    только там, где шаги и выгрузка помесячные. На чужом деке это неверно.
    """
    records = [
        {"well": str(control.well), "DATA": pd.Timestamp(step.date), "target": target}
        for step in steps
        for control in (step.injectors if kind == "injector" else step.producers)
        for target in ((control.rate if kind == "injector" else control.liquid_rate),)
        if target is not None
    ]
    frame = pd.DataFrame.from_records(records, columns=["well", "DATA", "target"])
    # единица времени приводится явно: `merge_asof` отказывается сливать
    # `datetime64[s]` с `datetime64[us]`, а даты шагов приходят из `date`
    frame["DATA"] = frame["DATA"].astype("datetime64[ns]")
    return frame


def delivered_ratios(rows: pd.DataFrame, steps: Sequence[Any], since: str,
                     kind: str = "injector") -> dict[str, float]:
    """Какая доля запрошенного дошла до пласта — по каждой скважине.

    Считается по прогону, который шёл **с этим же расписанием**: отношение суммы
    фактического расхода к сумме уставок за то же окно. Единица означает, что
    скважина берёт (отдаёт) всё, что просят; 0,2 — что четыре пятых просьбы
    остались в файле, потому что расход задаёт предел давления, а не уставка.

    Это не оценка приёмистости или продуктивности: `J` здесь не участвует, и по
    этой величине нельзя сказать, сколько скважина возьмёт при **другом**
    пластовом давлении. Она отвечает ровно на один вопрос — сколько из уже
    посчитанной просьбы дошло до пласта.
    """
    targets = planned_targets(steps, kind)
    if targets.empty:
        return {}
    rate = RATE[kind]
    frame = rows.loc[:, ["DATA", "well", rate]].copy()
    frame["well"] = frame["well"].astype(str)
    frame["DATA"] = pd.to_datetime(frame["DATA"]).astype("datetime64[ns]")
    frame = frame[frame["DATA"] >= pd.Timestamp(since)]
    if frame.empty:
        return {}
    merged = pd.merge_asof(
        frame.sort_values("DATA"),
        targets.sort_values("DATA"),
        on="DATA",
        by="well",
    )
    merged = merged[merged["target"] > 0]
    if merged.empty:
        return {}
    total = merged.groupby("well")[[rate, "target"]].sum()
    return {
        str(well): float(row[rate] / row["target"])
        for well, row in total.iterrows()
        if row["target"] > 0
    }
