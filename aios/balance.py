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

from aios.margin import counted_by_methodology
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

        Методически компенсация считается в пластовых условиях
        (`Qотб,прив ≈ Qн·Bн + Qв·Bв`), а здесь объёмные коэффициенты не
        применяются. Насколько это упрощает — измерено, а не предположено:
        `aios.pvt.parse_volume_factors` берёт их из дека, и для Модели Z
        `Bв = 1,0000`, `Bн` = 0,9862…1,0078. То есть на этом объекте показатель
        абсолютен с точностью до 2,2 %, а не только сравним между сценариями.

        На чужом деке (ТЗ п. 4.3) коэффициенты надо перемерить: там нефть может
        быть газонасыщенной, и тогда `Bн` уйдёт заметно выше единицы. До
        перемера показатель годится для сравнения сценариев, но не как
        абсолютная величина.
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
        # Правило одно на все три приращения: строка с отрицательным WLPT_Diff,
        # WOMT_Diff **или** WWIT_Diff исключается целиком. До 21.08 здесь
        # проверялись два из трёх, а закачка клампилась через max(..., 0.0) —
        # перекос односторонний: добытая вода такой строки шла в баланс, а
        # закачка обнулялась, и ограничение «закачиваем только то, что подняли»
        # выглядело выполненным щедрее, чем есть.
        if not counted_by_methodology(row):
            continue
        oil_t = row["WOMT_Diff"]
        liquid_t = row["WLPT_Diff"]
        year = int(str(row["DATA"])[:4])
        bucket = totals.setdefault(year, [0.0, 0.0, 0.0])
        bucket[0] += oil_t * 1000.0 / fluids.oil_density
        # нефти больше жидкости быть не может; прогноз суррогата это нарушает,
        # и отрицательной воды из-за него в балансе появиться не должно
        bucket[1] += max(liquid_t - oil_t, 0.0) * 1000.0 / fluids.water_density
        bucket[2] += row["WWIT_Diff"]

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


def compensation_problems(rows: list[dict[str, Any]],
                          fluids: Fluids | None = None,
                          ceiling: float = 1.5,
                          oil_fvf: float = 1.0,
                          water_fvf: float = 1.0) -> list[str]:
    """Нарушения компенсации отбора закачкой — вторая, физическая формулировка.

    Отличие от `problems` принципиальное, и от него зависят деньги.

    `problems` сравнивает накопленную закачку с накопленной **добытой водой**. Это
    буквальное прочтение фразы организаторов «дополнительной воды нет». Но при
    заводнении закачка возмещает не поднятую воду, а **отобранный объём в
    пластовых условиях** — нефть вместе с водой: `Qотб,пласт ≈ Qн·Bн + Qв·Bв`.
    Иначе поддерживать пластовое давление было бы нечем.

    Разница не теоретическая. На историческом сценарии Модели Z первая проверка
    даёт превышение в 2,32 раза, то есть **отвергает саму историю месторождения**;
    вторая — компенсацию 129 %, обычную величину для заводнения с поддержанием
    давления (`docs/experiments.md`).

    `oil_fvf` и `water_fvf` — объёмные коэффициенты. Для Модели Z они измерены по
    деку (`aios.pvt.parse_volume_factors`): `Bв = 1,0000`, `Bн` в диапазоне
    0,9862…1,0078. То есть для этого объекта поверхностные объёмы практически
    равны пластовым, и умолчание 1,0 вносит не более 2,2 % — но на другом кейсе
    это будет не так, поэтому коэффициенты параметры, а не константы.

    `ceiling` — предел компенсации. 1,5 означает «закачивать в полтора раза больше
    отобранного нельзя»; величина взята с запасом и обсуждаема, в отличие от самой
    формулировки.

    Какую из двух проверок применять на сдаче — открытый вопрос к организаторам
    (`BLOCKED.md`, п. 3). Обе доступны намеренно: подменять одну другой молча
    значит изменить ЧДД без решения человека.
    """
    over: list[str] = []
    withdrawn = 0.0
    injected = 0.0

    for year in by_year(rows, fluids):
        withdrawn += year.oil_m3 * oil_fvf + year.water_m3 * water_fvf
        injected += year.injected_m3 * water_fvf
        if withdrawn <= 0:
            if injected > 0:
                over.append(
                    f"{year.year}: закачано {injected:,.0f} м³ при нулевом отборе — "
                    f"возмещать нечего"
                )
            continue
        if injected > withdrawn * ceiling:
            over.append(
                f"{year.year}: накопленная закачка {injected:,.0f} м³ возмещает отобранный "
                f"объём {withdrawn:,.0f} м³ на {injected / withdrawn * 100:.0f}% — "
                f"выше предела {ceiling * 100:.0f}%"
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
