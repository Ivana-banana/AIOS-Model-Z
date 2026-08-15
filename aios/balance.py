"""Баланс воды и диагностика заводнения.

Организаторы на разборе 12.08 сказали прямо: **дополнительной воды нет**. Вся
вода, которую добыли, — её и закачиваем, единый норматив закачки, выбирать между
сточной, пресной и товарной не приходится (`docs/method/qa-organisers.md`).

Отсюда балансовое ограничение, которого нет ни в методике, ни в презентациях:
закачка не берётся из воздуха и ограничена добытой водой. Сценарий, который
закачивает больше, чем поднял, физически не обеспечен — и это ровно тот случай,
когда модель посчитает «отлично», а решение будет нереализуемым.

Здесь же — водонефтяной фактор. Организаторы подсветили его отдельно: это не
отчётный показатель, а **признак для агента**. «Максимум жидкости — значит
максимум нефти? Не всегда. И когда вы это агенту объясните, гораздо легче будет
управлять системой.»
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aios.summary import Fluids


@dataclass(frozen=True)
class YearBalance:
    """Баланс по объекту за календарный год, поверхностные условия."""

    year: int
    oil_m3: float
    water_m3: float
    injected_m3: float

    @property
    def liquid_m3(self) -> float:
        return self.oil_m3 + self.water_m3

    @property
    def watercut(self) -> float:
        """Обводнённость по объёму, доля."""
        return self.water_m3 / self.liquid_m3 if self.liquid_m3 > 0 else 0.0

    @property
    def water_oil_ratio(self) -> float:
        """ВНФ: сколько кубов воды подняли на куб нефти."""
        return self.water_m3 / self.oil_m3 if self.oil_m3 > 0 else 0.0

    @property
    def compensation(self) -> float:
        """Отношение закачки к добытой жидкости.

        Это упрощение: методически компенсация считается в пластовых условиях
        (`Qотб,прив ≈ Qн·Bн + Qв·Bв`), а объёмные коэффициенты нам из выгрузки
        недоступны. Показатель годится для сравнения сценариев между собой и для
        диагностики тренда, но не как абсолютная величина — так и подаётся.
        """
        return self.injected_m3 / self.liquid_m3 if self.liquid_m3 > 0 else 0.0

    @property
    def water_reuse(self) -> float:
        """Какую долю закачки покрывает собственная добытая вода."""
        return self.injected_m3 / self.water_m3 if self.water_m3 > 0 else 0.0


def by_year(rows: list[dict[str, Any]], fluids: Fluids | None = None) -> list[YearBalance]:
    """Свести баланс по годам. Массы из выгрузки переводятся обратно в объёмы."""
    fluids = fluids or Fluids()
    totals: dict[int, list[float]] = {}

    for row in rows:
        oil_t = row["WOMT_Diff"]
        liquid_t = row["WLPT_Diff"]
        if oil_t < 0 or liquid_t < 0:
            continue  # такие строки методика исключает из экономики целиком
        year = int(str(row["DATA"])[:4])
        bucket = totals.setdefault(year, [0.0, 0.0, 0.0])
        bucket[0] += oil_t * 1000.0 / fluids.oil_density
        bucket[1] += max(liquid_t - oil_t, 0.0) * 1000.0 / fluids.water_density
        bucket[2] += max(row["WWIT_Diff"], 0.0)

    return [
        YearBalance(year=year, oil_m3=values[0], water_m3=values[1], injected_m3=values[2])
        for year, values in sorted(totals.items())
    ]


def problems(rows: list[dict[str, Any]], fluids: Fluids | None = None,
             tolerance: float = 1.05) -> list[str]:
    """Нарушения баланса воды.

    Проверка накопительная, а не помесячная: воду можно копить в системе
    подготовки и закачивать неравномерно, поэтому отдельный месяц с превышением
    ничего не значит. Значение имеет накопленный итог — за ним источника уже нет.

    `tolerance` — допуск на неучтённые потери и на упрощение с плотностями.
    """
    over: list[str] = []
    produced = 0.0
    injected = 0.0

    for year in by_year(rows, fluids):
        produced += year.water_m3
        injected += year.injected_m3
        if produced <= 0:
            if injected > 0:
                over.append(
                    f"{year.year}: закачано {injected:,.0f} м³ при нулевой добыче воды — "
                    f"источника закачки нет"
                )
            continue
        if injected > produced * tolerance:
            over.append(
                f"{year.year}: накопленная закачка {injected:,.0f} м³ превышает добытую воду "
                f"{produced:,.0f} м³ на {(injected / produced - 1) * 100:.0f}% — "
                f"дополнительной воды на объекте нет"
            )
    return over


def well_water_oil_ratio(rows: list[dict[str, Any]],
                         fluids: Fluids | None = None) -> dict[str, float]:
    """ВНФ по каждой скважине за весь горизонт — признак для агента и диагностика.

    Скважина с ВНФ 50 поднимает пятьдесят кубов воды на куб нефти: платим за
    подъём жидкости и за закачку этой же воды обратно, а выручка идёт только с
    нефти. Именно на таких скважинах и живёт двойной штраф за воду.
    """
    fluids = fluids or Fluids()
    oil: dict[str, float] = {}
    water: dict[str, float] = {}

    for row in rows:
        oil_t = row["WOMT_Diff"]
        liquid_t = row["WLPT_Diff"]
        if oil_t < 0 or liquid_t < 0:
            continue
        well = row["well"]
        oil[well] = oil.get(well, 0.0) + oil_t * 1000.0 / fluids.oil_density
        water[well] = water.get(well, 0.0) + max(liquid_t - oil_t, 0.0) * 1000.0 / fluids.water_density

    return {
        well: (water[well] / volume if volume > 0 else float("inf"))
        for well, volume in oil.items()
        if water.get(well, 0.0) > 0 or volume > 0
    }
