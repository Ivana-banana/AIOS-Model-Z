"""Баланс воды и диагностика заводнения.

Организаторы на разборе 12.08 сказали прямо: **дополнительной воды нет**. Вся
вода, которую добыли, — её и закачиваем, единый норматив закачки, выбирать между
сточной, пресной и товарной не приходится (`docs/method/qa-organisers.md`).

Отсюда балансовое ограничение, которого нет ни в методике, ни в презентациях:
закачка не берётся из воздуха. Сценарий, который закачивает воду ниоткуда,
физически не обеспечен — и это ровно тот случай, когда модель посчитает
«отлично», а решение будет нереализуемым.

**Но буквальное «закачка ≤ добытая вода» историю Модели Z отвергает, и измерение
это доказало** (22.08, `runs/base/rows.csv`): за 1991–2025 закачано 13 157 610 м³
при добытой воде 5 164 127 м³, то есть **60,8 % всей закачки объекта — вода со
стороны**; в 2024 это 682,4 м³/сут при закачке 1 940 м³/сут. Независимая
проверка пришла к 684 м³/сут (`docs/manus/round4.md`, ответ 4-2).

Поэтому здесь три формулировки, а не одна, и подменять их друг другом молча
нельзя — они ограничивают разное:

* `problems` — «дополнительной воды нет вовсе», лимит источника ноль. На Модели Z
  ложно по факту, оставлено для кейса, где источника действительно нет;
* `compensation_problems` — закачка возмещает **отобранную пустотность**. Это про
  поддержание пластового давления, а не про то, откуда взялась вода;
* `external_water_problems` — закачка ≤ попутная вода + лимит **стороннего**
  источника. Физически это и есть ограничение по источнику, и мощность источника
  нам неизвестна: она параметр, а не константа.

Здесь же — водонефтяной фактор. Организаторы подсветили его отдельно: это не
отчётный показатель, а **признак для агента**. «Максимум жидкости — значит
максимум нефти? Не всегда. И когда вы это агенту объясните, гораздо легче будет
управлять системой.»
"""

from __future__ import annotations

from calendar import monthrange
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
    days: float = 365.0
    """Сколько суток года попало в выгрузку. Нужно, чтобы перевести объёмы в м³/сут."""

    @property
    def liquid_m3(self) -> float:
        return self.oil_m3 + self.water_m3

    @property
    def external_water_m3(self) -> float:
        """Закачка, не покрытая собственной попутной водой, — вода со стороны.

        Отрицательное значение означает обратное: часть попутной воды не пошла в
        закачку и утилизируется иначе.
        """
        return self.injected_m3 - self.water_m3

    @property
    def external_water_m3_per_day(self) -> float:
        return self.external_water_m3 / self.days if self.days > 0 else 0.0

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
    months: dict[int, set[str]] = {}

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
        stamp = str(row["DATA"])
        year = int(stamp[:4])
        months.setdefault(year, set()).add(stamp[:7])
        bucket = totals.setdefault(year, [0.0, 0.0, 0.0])
        bucket[0] += oil_t * 1000.0 / fluids.oil_density
        # нефти больше жидкости быть не может; прогноз суррогата это нарушает,
        # и отрицательной воды из-за него в балансе появиться не должно
        bucket[1] += max(liquid_t - oil_t, 0.0) * 1000.0 / fluids.water_density
        bucket[2] += row["WWIT_Diff"]

    return [
        YearBalance(year=year, oil_m3=values[0], water_m3=values[1], injected_m3=values[2],
                    days=days_of(months.get(year, set())))
        for year, values in sorted(totals.items())
    ]


def days_of(stamps: set[str]) -> float:
    """Сколько суток покрывают месяцы `YYYY-MM` — неполный год не считаем полным."""
    total = 0.0
    for stamp in stamps:
        year, month = int(stamp[:4]), int(stamp[5:7])
        total += monthrange(year, month)[1]
    return total or 365.0


def problems(rows: list[dict[str, Any]], fluids: Fluids | None = None,
             tolerance: float = 1.05, floor: float | None = None) -> list[str]:
    """Нарушения баланса воды.

    Проверка накопительная, а не помесячная: воду можно копить в системе
    подготовки и закачивать неравномерно, поэтому отдельный месяц с превышением
    ничего не значит. Значение имеет накопленный итог — за ним источника уже нет.

    `tolerance` — допуск на неучтённые потери и на упрощение с плотностями.

    `floor` — нижняя граница того же отношения, когда кейс задаёт коридор, а не
    один потолок. None — снизу не проверяем.
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
        # тот же допуск и по той же причине, что в `compensation_problems`:
        # срезка ровно до предела не должна объявляться его нарушением (Р-188)
        if injected > produced * tolerance * (1 + 1e-9):
            over.append(
                f"{year.year}: накопленная закачка {injected:,.0f} м³ превышает добытую воду "
                f"{produced:,.0f} м³ на {(injected / produced - 1) * 100:.0f}% — "
                f"дополнительной воды на объекте нет"
            )
        elif floor is not None and injected < produced * floor * (1 - 1e-9):
            over.append(
                f"{year.year}: накопленная закачка {injected:,.0f} м³ составляет "
                f"{injected / produced * 100:.0f}% добытой воды {produced:,.0f} м³ — "
                f"ниже границы {floor * 100:.0f}%"
            )
    return over


def compensation_problems(rows: list[dict[str, Any]],
                          fluids: Fluids | None = None,
                          ceiling: float = 1.5,
                          oil_fvf: float = 1.0,
                          water_fvf: float = 1.0,
                          floor: float | None = None) -> list[str]:
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

    `floor` — нижняя граница той же величины. None — снизу не проверяем, и это
    умолчание: до КТ-2 коридор был односторонним. Трекер сказала, что в тестовом
    кейсе он будет задан жёстко с двух сторон, «примерно 0.85–1.15 условно»
    (`docs/kt2-feedback-2026-08-31.md`). Нижняя граница физически не менее важна
    верхней: недокомпенсация роняет пластовое давление, и потерянную энергию
    пласта не вернуть последующей закачкой.

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
        # Допуск на счёт с плавающей точкой — тот же, что у `casecheck.LimitCheck`
        # и по той же причине: планировщик режет закачку **ровно** до потолка, и
        # без допуска его собственная срезка тут же объявляется нарушением в
        # последнем знаке. Замер 03.09 на прогоне цикла: «на 130% — выше предела
        # 130%». С включённой проверкой fail-closed (Р-165) это означало бы, что
        # ни один приведённый к коридору кандидат не считается пригодным (Р-188).
        if injected > withdrawn * ceiling * (1 + 1e-9):
            over.append(
                f"{year.year}: накопленная закачка {injected:,.0f} м³ возмещает отобранный "
                f"объём {withdrawn:,.0f} м³ на {injected / withdrawn * 100:.0f}% — "
                f"выше предела {ceiling * 100:.0f}%"
            )
        elif floor is not None and injected < withdrawn * floor * (1 - 1e-9):
            over.append(
                f"{year.year}: накопленная закачка {injected:,.0f} м³ возмещает отобранный "
                f"объём {withdrawn:,.0f} м³ на {injected / withdrawn * 100:.0f}% — "
                f"ниже границы {floor * 100:.0f}%"
            )
    return over


def historic_external_water(rows: list[dict[str, Any]],
                            fluids: Fluids | None = None) -> float:
    """Максимальный годовой приток сторонней воды в истории объекта, м³/сут.

    Это единственное число про мощность водоисточника, которое у нас есть
    основание назвать: столько источник **заведомо выдавал**, потому что выдавал.
    Мощности его мы не знаем — ни паспорта КНС, ни производительности водозабора
    в поставке нет (`docs/manus/round4.md`, разбор ответа 4-2).

    Поэтому число годится как предел по умолчанию и не годится как утверждение
    «больше нельзя». Сценарий, которому нужно больше, не запрещается, а
    помечается: подтверждать его мощность — не наша задача и не задача нашего
    стенда.
    """
    years = by_year(rows, fluids)
    return max((year.external_water_m3_per_day for year in years), default=0.0)


def external_water_problems(rows: list[dict[str, Any]],
                            fluids: Fluids | None = None,
                            limit_m3_per_day: float | None = None) -> list[str]:
    """Годы, в которых закачке не хватает попутной воды сверх лимита источника.

    Третья формулировка баланса, и физически она честнее двух прежних.

    `problems` сравнивает закачку с добытой водой (лимит источника = 0) и на
    истории Модели Z отвергает саму историю: 60,8 % всей закачки объекта — вода
    со стороны. `compensation_problems` сравнивает закачку с отобранной
    пустотностью — это про поддержание пластового давления, а не про то, откуда
    берётся вода. Здесь проверяется ровно источник:

        закачка ≤ попутная вода + лимит сторонней воды

    `limit_m3_per_day = None` означает «предел неизвестен» — тогда проверка
    ничего не запрещает, но по-прежнему считает потребность, и её печатают.
    Подставлять сюда правдоподобное число вместо неизвестного нельзя: заявленный
    ЧДД тогда будет опираться на выдуманную мощность водозабора.
    """
    found: list[str] = []
    for year in by_year(rows, fluids):
        need = year.external_water_m3_per_day
        if need <= 0:
            continue
        if limit_m3_per_day is None:
            found.append(
                f"{year.year}: закачке нужно {need:,.0f} м³/сут воды со стороны — "
                f"мощность источника неизвестна, режим не подтверждён"
            )
        elif need > limit_m3_per_day:
            found.append(
                f"{year.year}: закачке нужно {need:,.0f} м³/сут воды со стороны при "
                f"пределе {limit_m3_per_day:,.0f} м³/сут — превышение "
                f"{(need / limit_m3_per_day - 1) * 100:.0f}%"
            )
    return found


def worst_cumulative_compensation(rows: list[dict[str, Any]],
                                  fluids: Fluids | None = None,
                                  rule: str = "compensation") -> tuple[int, float] | None:
    """Худший год по накопленной компенсации: `(год, отношение)`.

    Тем же счётом, каким её судит `compensation_problems`: отношение накопленной
    закачки к накопленному отбору **на конец каждого года**. Нужен, чтобы
    сравнивать одно и то же в разных окнах — и видеть, что окно решает.

    `rule` выбирает знаменатель: `produced_water` — добытая вода, иначе вся
    отобранная жидкость. Пустая выгрузка даёт None: сравнивать нечего.
    """
    withdrawn = 0.0
    injected = 0.0
    worst: tuple[int, float] | None = None
    for year in by_year(rows, fluids):
        withdrawn += year.water_m3 if rule == "produced_water" else year.liquid_m3
        injected += year.injected_m3
        if withdrawn <= 0:
            continue
        ratio = injected / withdrawn
        if worst is None or ratio > worst[1]:
            worst = (year.year, ratio)
    return worst


def external_water_by_month(rows: list[dict[str, Any]],
                            fluids: Fluids | None = None) -> list[tuple[str, float]]:
    """Потребность в сторонней воде по месяцам: `[("2019-07", 812.4), …]`, м³/сут.

    Годовая величина (`by_year`) — среднее по году, и она скрывает пик: месяц с
    тройным расходом водозабора усредняется соседями и в годовом числе не виден
    вовсе.

    Почему это не противоречит накопительной логике `problems`. Там речь про
    **объём** воды: её можно копить в системе подготовки и закачивать
    неравномерно, поэтому отдельный месяц с превышением ничего не значит. Здесь
    речь про **расход источника**, а расход накопить нельзя: насосная станция не
    отдаёт за июль больше, чем может за июль, сколько бы ни простаивала в январе.
    Две разные величины, и проверять их одним правилом — ошибка (Р-162).

    Месяцы возвращаются по возрастанию метки. Отрицательная потребность
    сохраняется как есть (закачка меньше попутной воды) — обрезать её до нуля
    значило бы прятать месяцы, в которых вода утилизируется мимо закачки.
    """
    fluids = fluids or Fluids()
    water: dict[str, float] = {}
    injected: dict[str, float] = {}

    for row in rows:
        if not counted_by_methodology(row):
            continue
        stamp = str(row["DATA"])[:7]
        water[stamp] = water.get(stamp, 0.0) + max(
            row["WLPT_Diff"] - row["WOMT_Diff"], 0.0) * 1000.0 / fluids.water_density
        injected[stamp] = injected.get(stamp, 0.0) + row["WWIT_Diff"]

    found: list[tuple[str, float]] = []
    for stamp in sorted(injected.keys() | water.keys()):
        days = monthrange(int(stamp[:4]), int(stamp[5:7]))[1]
        found.append((stamp, (injected.get(stamp, 0.0) - water.get(stamp, 0.0)) / days))
    return found


def external_water_capacity_problems(rows: list[dict[str, Any]],
                                     fluids: Fluids | None = None,
                                     limit_m3_per_day: float | None = None) -> list[str]:
    """Месяцы, в которых расход водозабора выше его мощности.

    Четвёртая формулировка баланса и единственная **мгновенная**: три прежние
    сравнивают объёмы за год или накопленным итогом, а мощность источника —
    предел расхода, и год его усредняет. Проверку назвала трекер на КТ-2
    («помесячный расход не превышает мощности»), и она ловит то, чего годовая не
    видит: расписание с ровным годовым средним и пиковым июлем.

    `limit_m3_per_day = None` — мощность неизвестна. Тогда ничего не нарушается,
    но и молчать нельзя: возвращается одна строка про месяц наибольшей
    потребности. Подставлять сюда правдоподобное число вместо неизвестного
    запрещено по той же причине, что и в `external_water_problems`: заявленный
    ЧДД опёрся бы на выдуманную мощность водозабора.
    """
    months = external_water_by_month(rows, fluids)
    if not months:
        return []

    if limit_m3_per_day is None:
        stamp, need = max(months, key=lambda item: item[1])
        if need <= 0:
            return []
        return [
            f"{stamp}: наибольший расход водозабора {need:,.0f} м³/сут — мощность "
            f"источника неизвестна, режим не подтверждён"
        ]

    found = []
    for stamp, need in months:
        if need <= limit_m3_per_day:
            continue
        # мощность ноль — «источника нет вовсе»; процент превышения над нулём не
        # определён, и печатать «inf%» вместо числа хуже, чем не печатать его.
        # Из конфигурации ноль сюда не приходит (`_non_negative(...) or None`
        # переводит его в «неизвестно»), но функция публичная и зовётся напрямую.
        excess = (f" — превышение {(need / limit_m3_per_day - 1) * 100:.0f}%"
                  if limit_m3_per_day > 0 else " — стороннего источника нет вовсе")
        found.append(
            f"{stamp}: расход водозабора {need:,.0f} м³/сут при мощности "
            f"{limit_m3_per_day:,.10g} м³/сут{excess}"
        )
    return found


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
