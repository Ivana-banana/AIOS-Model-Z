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

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Any

from aios import balance
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms, counted_by_methodology, month_margin
from aios.summary import Fluids


@dataclass(frozen=True)
class ShutWindow:
    """Окно обязательной остановки: скважина стоит с `start` по `end` включительно.

    Форма ограничения, которой у нас не было до 01.09: `forced_shut_wells`
    закрывает скважину на весь горизонт, а кейс скажет «скважину W остановить с
    01.2027 по 06.2027» (ремонт, ГТМ, ограничение подрядчика). Выбор из двух
    неверных — остановить навсегда или не останавливать вовсе — здесь и кончается.
    """

    well: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"окно остановки скважины {self.well}: конец {self.end} раньше начала "
                f"{self.start}"
            )

    def covers(self, moment: date) -> bool:
        """Попадает ли шаг расписания в окно. Границы включаются обе."""
        return self.start <= moment <= self.end


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

    Действует **при любом правиле**, а не только при `external_water`: норма
    («сколько закачивать относительно отбора») и физика («откуда взять воду») —
    разные ограничения, и кейс задаёт оба сразу. Трекер на КТ-2 назвала второе
    «докачкой дополнительного объёма воды в размере N»
    (`docs/kt2-feedback-2026-08-31.md`, запись 00:16).

    До 01.09 ключ жил внутри одного правила и в рабочей конфигурации
    (`compensation`) не работал вовсе.

    Значение по умолчанию для конкретного объекта выводится из его же истории —
    `aios.balance.historic_external_water` даёт максимум, который источник
    заведомо выдавал.
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
    """Во сколько раз закачка может превышать базу правила — **верхняя** граница.

    Для `produced_water` 1.0 означает «закачиваем ровно столько, сколько подняли»;
    для `compensation` это потолок компенсации — 1.3 держит нас не выше
    исторической компенсации объекта (1,291).
    """

    water_tolerance_min: float | None = None
    """Нижняя граница того же отношения. None — снизу закачка не ограничена.

    Вместе с `water_tolerance` задаёт **коридор**: трекер на КТ-2 сказала, что в
    тестовом кейсе он будет задан жёстко, «примерно 0.85–1.15 условно»
    (`docs/kt2-feedback-2026-08-31.md`, запись 00:00). До появления кейса ключ
    не задан ни в одном рабочем конфиге, и это осознанно: нижняя граница,
    включённая умолчанием, изменила бы заявляемое число.

    Смысл границы зависит от правила ровно так же, как у верхней: база — та же
    (отобранная пустотность у `compensation`, добытая вода у `produced_water`).
    Правило `external_water` ограничивает **источник**, а не отношение объёмов,
    и нижней границы в этих терминах не имеет: сочетание отвергается явно, а не
    молча игнорируется.

    Коридор Модели Z полезно знать заранее: историческая компенсация объекта —
    1,291, то есть при коридоре 0.85–1.15 **вся история выше верхней границы**.
    """

    #: --- ручки под валидационный кейс (Р-110) ---------------------------------
    #:
    #: Все десять появились до кейса намеренно. Организаторы сказали, что ограничения
    #: придут «по давлениям, по фонду, по закачке» (`docs/method/qa-organisers.md`,
    #: 0:44:14 и 0:55:38), а времени на кейс отведено строго: дописывать
    #: недостающую ручку под таймером дороже, чем держать неиспользованную.
    #:
    #: Умолчание каждой — «ограничения нет», и это не формальность: любая из них,
    #: включённая по умолчанию, изменила бы заявляемое число.

    min_producer_bhp_atm: float | None = None
    """Нижний предел забойного давления добывающей, атм. None — берётся из дека.

    В деке Модели Z стоит 50 атм, и 84,4 % месяцев добычи идут ровно на нём. Кейс
    «по давлениям» назовёт своё число, а подменять ради этого дек нельзя — он
    материал организаторов.
    """

    max_injector_bhp_atm: float | None = None
    """Верхний предел давления нагнетания, атм. None — берётся из дека (300)."""

    min_reservoir_pressure_atm: float | None = None
    """Нижний предел **пластового** давления у добывающей, атм. None — не проверяем.

    Не путать с `min_producer_bhp_atm`: то — забойное, наша уставка, она уезжает
    в `WCONPROD` сдаваемого файла и её планировщик правит. Пластовое давление
    (`WBP9`, приезжает в колонке `THP`) — **отклик пласта**, а не уставка:
    расписанием его не задать, оно получается расчётом. Поэтому предел здесь
    работает проверкой по факту, а не срезкой: нарушение видно в
    `aios.summary.sanity_problems` и останавливает выгрузку.

    Трекер назвала это отдельным пунктом на КТ-2 и добавила, что «многие
    забывают» (`docs/kt2-feedback-2026-08-31.md`, запись 00:35).
    """

    max_reservoir_pressure_atm: float | None = None
    """Верхний предел пластового давления у нагнетательной, атм. None — не проверяем.

    Физический смысл верхней границы — не перейти давление гидроразрыва:
    закачка выше него уводит воду в трещину и в соседа, а не вытесняет нефть.
    """

    max_well_injection_m3_per_day: float | None = None
    """Потолок приёмистости **одной** скважины. None — предела нет.

    Существующий предел — суммарный по объекту; «ограничение по закачке» может
    прийти поскважинным, и тогда суммарный его не выражает: сто скважин по
    десять кубов и десять по сто дают один и тот же объект и разные скважины.
    """

    max_well_liquid_m3_per_day: float | None = None
    """Потолок отбора одной добывающей, м³/сут. None — предела нет."""

    well_caps_m3_per_day: dict[str, float] | None = None
    """Карта «скважина → потолок», перекрывающая потолок по типу.

    Нужна, когда ограничение приходит адресным списком, а не одним числом на
    фонд: у скважины может быть свой паспорт насоса или своя договорённость.
    """

    controllable_wells: frozenset[str] | None = None
    """Скважины, которыми **разрешено** управлять. None — разрешены все.

    Остальные остаются в том режиме, который задан историей: мы не трогаем их
    вовсе, а не «трогаем осторожно».
    """

    forced_shut_wells: frozenset[str] | None = None
    """Скважины, которые обязаны быть остановлены. None — таких нет."""

    max_active_wells: int | None = None
    """Максимум действующих скважин на шаге. None — предела нет."""

    max_field_injection_m3_per_year: float | None = None
    """Годовой объём закачки по объекту, м³. None — предела нет.

    Отраслевая норма нормирует именно годовой объём, а не суточный расход
    (`docs/manus/vrr-water-balance.md`), поэтому кейс может прийти в этих
    единицах — и суточный предел его не выражает: закачка неравномерна по
    месяцам.
    """

    max_field_oil_t_per_day: float | None = None
    """Предел по нефти на объекте, т/сут. None — предела нет.

    Мощность УПН выражается по нефти, а не по жидкости, и объектным пределом по
    жидкости не заменяется: одна и та же жидкость при обводнённости 50 % и 90 %
    даёт вчетверо разную нефть. Проверяется по прогнозу — без него фазу из
    расписания не получить, там только уставки по жидкости.
    """

    max_field_water_t_per_day: float | None = None
    """Предел по попутной воде на объекте, т/сут. None — предела нет.

    Мощность УПСВ и утилизации. Вода считается как жидкость минус нефть по тому
    же прогнозу: отдельной колонки воды в контракте расчётчика нет.
    """

    well_groups: dict[str, frozenset[str]] | None = None
    """Группы скважин: куст, блок, ДНС. None — групп нет.

    Между объектом и скважиной у нас до 01.09 не было ничего, а «по кусту не
    более» — обычная форма ограничения. Скважина может входить в несколько групп:
    куст и блок заводнения это разные разрезы одного фонда.
    """

    group_liquid_caps_m3_per_day: dict[str, float] | None = None
    """Потолок отбора жидкости по группе, м³/сут. None — предела нет."""

    group_injection_caps_m3_per_day: dict[str, float] | None = None
    """Потолок закачки по группе, м³/сут. None — предела нет."""

    forced_shut_windows: tuple[ShutWindow, ...] | None = None
    """Обязательные остановки с окном дат. None — таких нет.

    Отличается от `forced_shut_wells` ровно тем, ради чего заведена: скважина
    закрывается только на шагах внутри окна, а вне его остаётся с тем режимом,
    который ей назначил цикл. Окон у одной скважины может быть несколько.
    """

    max_field_liquid_m3_per_year: float | None = None
    """Годовой объём отбора жидкости по объекту, м³. None — предела нет.

    Годовой аккумулятор до 01.09 был только у закачки, хотя условие «за год не
    более V» приходит и по отбору: квота на подготовку и утилизацию воды, лимит
    системы сбора в объёме, договор на приём жидкости. Суточный предел его не
    выражает — отбор неравномерен по месяцам, и год может выйти за норму, ни разу
    не превысив суточного потолка.
    """

    field_limits_are_annual_average: bool = False
    """Как читать суточные пределы объекта: пик каждого месяца или среднее за год.

    Все наши пределы — помесячный пик, и это опасная сторона двусмысленности
    ровно в одну сторону: **пик, прочитанный как среднее, пропускает нарушение**,
    а среднее, прочитанное как пик, всего лишь режет лишнее. Ключ включает вторую
    трактовку для `max_field_liquid_m3_per_day` и
    `max_field_injection_m3_per_day`: предел применяется к среднесуточному за
    календарный год, а не к каждому месяцу.
    """

    max_events_per_year: int | None = None
    """Максимум платных мероприятий за год: остановки, пуски, переводы. None — нет.

    Методика оценивает их поштучно (1,0 млн руб. за остановку и за пуск, 5,0 за
    ВНС), поэтому ограничить их число — естественная форма ограничения по фонду.

    Считается по расписанию, а не по выгрузке, и потому не совпадает с событиями
    эталонного расчётчика — подробности в `_fit_events_to_annual_quota`.
    """


@dataclass(frozen=True)
class Allocation:
    """Одно решение планировщика: кого и почему изменили.

    Чаще всего это урезание, но не всегда: добор до нижней границы коридора
    баланса воды даёт `after > before` и, соответственно, отрицательный `cut`.
    """

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
    forecast = _with_string_wells(forecast)
    result = list(steps)
    decisions: list[Allocation] = []

    if limits.water_balance_rule not in ("produced_water", "compensation", "external_water"):
        raise ValueError(
            f"неизвестное water_balance_rule: {limits.water_balance_rule!r}; "
            "допустимы 'produced_water', 'compensation' и 'external_water'"
        )

    if (limits.water_tolerance_min is not None
            and limits.water_balance_rule == "external_water"):
        raise ValueError(
            "water_tolerance_min несовместим с правилом external_water: оно "
            "ограничивает источник воды, а не отношение объёмов, и нижней границы "
            "в этих терминах не имеет"
        )

    if (limits.water_tolerance_min is not None
            and limits.water_tolerance_min > limits.water_tolerance):
        raise ValueError(
            f"пустой коридор баланса воды: нижняя граница {limits.water_tolerance_min:.10g} "
            f"выше верхней {limits.water_tolerance:.10g}"
        )

    if limits.enforce_water_balance:
        if limits.water_balance_rule == "external_water":
            result, water = _fit_injection_to_external_water(
                result, forecast, fluids, limits.max_external_water_m3_per_day
            )
        else:
            result, water = _fit_injection_to_water_corridor(
                result, forecast, fluids,
                rule=limits.water_balance_rule,
                ceiling=limits.water_tolerance,
                floor=limits.water_tolerance_min,
                source_m3_per_day=limits.max_external_water_m3_per_day,
            )
        decisions.extend(water)

    if limits.max_field_liquid_m3_per_day is not None:
        if limits.field_limits_are_annual_average:
            result, liquid = _fit_liquid_to_annual_volume(
                result, _annual_budget(result, limits.max_field_liquid_m3_per_day),
                reason="среднесуточный отбор за год",
            )
        else:
            result, liquid = _fit_liquid_to_capacity(result, forecast, norms,
                                                     limits.max_field_liquid_m3_per_day, fluids)
        decisions.extend(liquid)

    if limits.max_field_injection_m3_per_day is not None:
        if limits.field_limits_are_annual_average:
            result, injection = _fit_injection_to_annual_budget(
                result, _annual_budget(result, limits.max_field_injection_m3_per_day),
                reason="среднесуточная закачка за год",
            )
        else:
            result, injection = _fit_injection_to_capacity(
                result, limits.max_field_injection_m3_per_day
            )
        decisions.extend(injection)

    if limits.max_field_oil_t_per_day is not None:
        result, oil = _fit_phase_to_capacity(result, forecast,
                                             limits.max_field_oil_t_per_day, phase="нефть")
        decisions.extend(oil)

    if limits.max_field_water_t_per_day is not None:
        result, water_phase = _fit_phase_to_capacity(
            result, forecast, limits.max_field_water_t_per_day, phase="попутная вода"
        )
        decisions.extend(water_phase)

    # Групповой уровень стоит между объектом и скважиной — там же, где он стоит
    # в самой иерархии ограничений. Считать его после скважинных потолков значило
    # бы резать уже срезанное и объяснять группе то, что сделал паспорт насоса.
    result, groups = _fit_groups(result, forecast, norms, limits, fluids)
    decisions.extend(groups)

    # Ручки под валидационный кейс идут последними и в этом порядке намеренно:
    # сначала объектные пределы (выше), потом скважинные, потом состав фонда,
    # потом годовой объём. Обратный порядок дал бы срезку по объекту поверх уже
    # срезанных скважин и «двойную» экономию, которой в пласте нет.
    result, per_well = _fit_wells_to_caps(result, limits)
    decisions.extend(per_well)

    result, pressures = _fit_pressures(result, limits)
    decisions.extend(pressures)

    result, fund = _fit_fund(result, forecast, norms, limits)
    decisions.extend(fund)

    if limits.max_field_injection_m3_per_year is not None:
        result, yearly = _fit_injection_to_annual_volume(
            result, limits.max_field_injection_m3_per_year
        )
        decisions.extend(yearly)

    if limits.max_field_liquid_m3_per_year is not None:
        result, yearly_liquid = _fit_liquid_to_annual_volume(
            result,
            dict.fromkeys(_years(result), limits.max_field_liquid_m3_per_year),
            reason="годовой объём отбора жидкости",
        )
        decisions.extend(yearly_liquid)

    # Квота мероприятий идёт после всех срезок, и это обязательный порядок:
    # мероприятие может появиться от самой срезки — предел по объекту, доведённый
    # до нуля, это остановка. Считать квоту раньше значило бы считать не тот
    # график, который в итоге уедет в файл.
    if limits.max_events_per_year is not None:
        result, events = _fit_events_to_annual_quota(result, limits.max_events_per_year)
        decisions.extend(events)

    return result, decisions


def _achieved_rates(forecast: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Достигнутый дебит жидкости по прогнозу: `{"2007-01": {"скв": м³/сут}}`."""
    rates: dict[str, dict[str, float]] = {}
    for row in forecast:
        if not counted_by_methodology(row):
            continue
        stamp = str(row["DATA"])[:7]
        rates.setdefault(stamp, {})[str(row["well"])] = float(row["WLPR"] or 0.0)
    return rates


def _phase_by_month(forecast: list[dict[str, Any]], phase: str) -> dict[str, float]:
    """Тонны фазы за месяц по объекту: `{"2007-01": 12345.0}`.

    Считается по прогнозу, потому что расписание фаз не содержит вовсе: там
    уставки по жидкости и по закачке. Строки, исключённые методикой, не считаются
    — тем же правилом, что и везде в проекте.
    """
    totals: dict[str, float] = {}
    for row in forecast:
        if not counted_by_methodology(row):
            continue
        oil_t = max(row["WOMT_Diff"], 0.0)
        liquid_t = max(row["WLPT_Diff"], 0.0)
        value = oil_t if phase == "нефть" else max(liquid_t - oil_t, 0.0)
        stamp = str(row["DATA"])[:7]
        totals[stamp] = totals.get(stamp, 0.0) + value
    return totals


def _fit_phase_to_capacity(steps: list[ScheduleStep],
                           forecast: list[dict[str, Any]],
                           capacity_t_per_day: float,
                           *, phase: str) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Уложить фазу — нефть или попутную воду — в мощность подготовки.

    Расписание задаёт жидкость, а ограничена фаза, поэтому срезка идёт через
    допущение: **при неизменной обводнённости срез жидкости на k срезает фазу на
    то же k**. Допущение названо здесь, а не спрятано: обводнённость от срезки
    меняется, и проверять результат надо по факту полного расчёта, как и баланс
    воды.

    Без прогноза ограничение не проверяемо ничем — это отказ, а не молчаливый
    пропуск: объявленное ограничение, которое никак не считается, хуже
    отсутствующего.
    """
    if not forecast:
        raise ValueError(
            f"предел по фазе «{phase}» задан, но прогноза нет: из расписания фазу "
            "не получить, там только уставки по жидкости"
        )

    monthly = _phase_by_month(forecast, phase)
    reference = _achieved_rates(forecast)
    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        stamp = f"{step.date.year:04d}-{step.date.month:02d}"
        days = monthrange(step.date.year, step.date.month)[1]
        produced_t = monthly.get(stamp)
        if produced_t is None:
            result.append(step)
            continue
        rate = produced_t / days
        if rate <= capacity_t_per_day:
            result.append(step)
            continue

        factor = capacity_t_per_day / rate
        achieved = reference.get(stamp, {})
        producers = []
        for control in step.producers:
            if control.status != "OPEN" or not control.liquid_rate:
                producers.append(control)
                continue
            # Резать надо от **достигнутого** дебита, а не от уставки. Уставки
            # Модели Z выше факта в среднем в 1,47 раза: пропорциональный срез
            # уставки на k при уставке 45 и факте 30 не меняет отклика вовсе, и
            # ограничение по фазе оставалось нарушенным после «успешной» срезки
            # (найдено репетицией приёмки 01.09). Скважину без отклика в прогнозе
            # трогать не за что — она фазы и не даёт.
            achieved_rate = achieved.get(control.well)
            if achieved_rate is None or achieved_rate <= 0:
                producers.append(control)
                continue
            reduced = min(control.liquid_rate, achieved_rate * factor)
            if reduced >= control.liquid_rate:
                producers.append(control)
                continue
            if stamp not in reported:
                reported.add(stamp)
                decisions.append(Allocation(
                    well=control.well, resource=f"{phase}, {stamp}",
                    before=control.liquid_rate, after=reduced,
                    reason=(f"{phase} {rate:,.1f} > мощности {capacity_t_per_day:,.1f} т/сут; "
                            "уставка опущена до той же доли от достигнутого дебита"),
                ))
            producers.append(ProducerControl(control.well, control.status, control.mode,
                                             reduced, control.bhp))
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))
    return result, decisions


def _fit_groups(steps: list[ScheduleStep],
                forecast: list[dict[str, Any]],
                norms: Norms | None,
                limits: Limits,
                fluids: Fluids) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ограничения уровня группы: куст, блок, ДНС.

    Порядок срезки взят тот же, что на уровне объекта, и по той же причине:
    жидкость режется по возрастанию удельной маржи (первой уходит скважина,
    приносящая меньше денег на кубометр), закачка — пропорционально, потому что
    приоритета между нагнетательными у нас нет.

    Скважина может входить в несколько групп; тогда её режут все, чьи пределы она
    нарушает, а действует наименьший результат — он применяется по очереди.
    """
    groups = limits.well_groups or {}
    liquid_caps = limits.group_liquid_caps_m3_per_day or {}
    injection_caps = limits.group_injection_caps_m3_per_day or {}
    if not liquid_caps and not injection_caps:
        return steps, []

    unknown = sorted((set(liquid_caps) | set(injection_caps)) - set(groups))
    if unknown:
        raise ValueError(
            "потолок задан группам, которых нет в well_groups: " + ", ".join(unknown)
        )
    if liquid_caps and norms is None:
        raise ValueError(
            "с групповым потолком по жидкости нужны нормативы: резать надо худшие "
            "по марже, а без неё выбирать не из чего"
        )

    value = _margin_per_liquid(forecast, norms, fluids) if (liquid_caps and norms) else {}
    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        producers = list(step.producers)
        injectors = list(step.injectors)

        for group, cap in sorted(liquid_caps.items()):
            members = groups[group]
            working = [
                control for control in producers
                if control.well in members and control.status == "OPEN" and control.liquid_rate
            ]
            total = sum(control.liquid_rate or 0.0 for control in working)
            if total <= cap:
                continue
            excess = total - cap
            order = sorted(working, key=lambda control: value.get(control.well, 0.0))
            cuts: dict[str, float] = {}
            for control in order:
                if excess <= 0:
                    break
                rate = control.liquid_rate or 0.0
                cut = min(rate, excess)
                cuts[control.well] = rate - cut
                excess -= cut
            for index, control in enumerate(producers):
                if control.well not in cuts:
                    continue
                key = f"{group}:{control.well}"
                if key not in reported:
                    reported.add(key)
                    decisions.append(Allocation(
                        well=control.well, resource=f"жидкость группы «{group}»",
                        before=control.liquid_rate or 0.0, after=cuts[control.well],
                        reason=(f"группа «{group}»: {total:,.1f} > потолка {cap:,.1f} м³/сут"),
                    ))
                producers[index] = ProducerControl(control.well, control.status, control.mode,
                                                   cuts[control.well], control.bhp)

        for group, cap in sorted(injection_caps.items()):
            members = groups[group]
            working = [
                control for control in injectors
                if control.well in members and control.status == "OPEN" and control.rate
            ]
            total = sum(control.rate or 0.0 for control in working)
            if total <= cap:
                continue
            factor = cap / total
            for index, control in enumerate(injectors):
                if control.well not in members or control.status != "OPEN" or not control.rate:
                    continue
                reduced = control.rate * factor
                key = f"{group}:{control.well}"
                if key not in reported:
                    reported.add(key)
                    decisions.append(Allocation(
                        well=control.well, resource=f"закачка группы «{group}»",
                        before=control.rate, after=reduced,
                        reason=(f"группа «{group}»: {total:,.1f} > потолка {cap:,.1f} м³/сут"),
                    ))
                injectors[index] = InjectorControl(control.well, control.fluid, control.status,
                                                   control.mode, reduced, control.bhp)

        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result, decisions


def _with_string_wells(forecast: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Привести имена скважин прогноза к строкам — как в расписании.

    Найдено репетицией приёмки 01.09 и стоило бы дорого на кейсе. Прогноз,
    прочитанный из `rows.csv` через pandas, отдаёт имя скважины числом (`1`, а не
    `"1"`), а в расписании оно всегда строка. Совпадений между ними нет ни одного,
    поэтому `value.get(control.well, 0.0)` возвращал ноль **каждой** скважине — и
    ранжирование «режем худшую по деньгам на кубометр» молча вырождалось в порядок
    перечисления. Срезка при этом выглядела совершенно обычной: объяснение
    печаталось, предел соблюдался, а выбор был произвольным.

    Нормализация делается один раз на входе, а не в каждой функции: разъехаться
    заново тогда может только весь модуль сразу.
    """
    if not forecast:
        return []
    if all(isinstance(row.get("well"), str) for row in forecast):
        return list(forecast)
    return [row | {"well": str(row["well"])} for row in forecast]


def _shut_reason(well: str, forced: frozenset[str],
                 active: dict[str, ShutWindow]) -> tuple[str, str]:
    """Ресурс и причина остановки: список на весь горизонт или окно дат."""
    if well in forced:
        return ("обязательная остановка",
                "скважина названа в конфигурации как обязательно остановленная")
    window = active[well]
    return ("обязательная остановка в окне",
            f"окно обязательной остановки {window.start} … {window.end}")


def _shut_mark(well: str, forced: frozenset[str],
               active: dict[str, ShutWindow]) -> str:
    """Ключ, по которому решение печатается один раз.

    Для списка на весь горизонт это скважина: причина одна на все шаги. Для окон —
    скважина и окно: два ремонта в разные годы это два разных решения, и слить их
    в одну строку значило бы потерять второй.
    """
    if well in forced:
        return well
    window = active[well]
    return f"{well}@{window.start}"


def _fit_fund(steps: list[ScheduleStep],
              forecast: list[dict[str, Any]],
              norms: Norms | None,
              limits: Limits) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ограничения по составу фонда — три формы, все названы организаторами.

    1. **Обязательные остановки** (`forced_shut_wells`): скважина закрывается
       независимо от того, что предложил цикл. Форма самая жёсткая и самая
       вероятная: «эти скважины в ремонте, ими не управлять».
    2. **Максимум действующих** (`max_active_wells`): если открытых больше
       предела, закрываются худшие по удельной марже — тем же правилом, каким
       планировщик режет отбор при нехватке мощности. Без нормативов правило
       решать нечем, и это отказ, а не молчаливое «закроем первых попавшихся».
    3. **Разрешённые к управлению** (`controllable_wells`) применяются **не
       здесь**: чтобы вернуть скважину к историческому режиму, нужно само
       историческое расписание, а планировщик его не видит — он получает уже
       изменённые шаги. Эта форма живёт в цикле (`aios.loop`), где опорные шаги
       есть, и там же покрыта тестом.
    """
    forced = limits.forced_shut_wells or frozenset()
    windows = limits.forced_shut_windows or ()
    if not forced and not windows and limits.max_active_wells is None:
        return steps, []
    if limits.max_active_wells is not None and norms is None:
        raise ValueError(
            "с ограничением по числу действующих скважин нужны нормативы: закрывать "
            "надо худшие по марже, а без неё выбирать не из чего"
        )

    # ранжирование то же, что при нехватке мощности по жидкости: удельная маржа
    # на кубометр. Своего правила здесь заводить нельзя — две разные очереди «кого
    # резать первым» разъехались бы молча
    margins = _margin_per_liquid(forecast, norms, Fluids()) if norms else {}
    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        producers = list(step.producers)
        injectors = list(step.injectors)

        def shut(well: str, resource: str, reason: str, before: float,
                 mark: str | None = None) -> None:
            key = mark or well
            if key in reported:
                return
            reported.add(key)
            decisions.append(Allocation(well=well, resource=resource,
                                        before=before, after=0.0, reason=reason))

        # окно действует только на своих шагах: вне его скважина остаётся с тем
        # режимом, который ей назначил цикл, — в этом вся разница с forced_shut_wells
        active_windows = {
            window.well: window for window in windows if window.covers(step.date)
        }
        shut_now = forced | set(active_windows)

        if shut_now:
            for index, control in enumerate(producers):
                if control.well in shut_now and control.status == "OPEN":
                    shut(control.well, *_shut_reason(control.well, forced, active_windows),
                         control.liquid_rate or 0.0,
                         _shut_mark(control.well, forced, active_windows))
                    producers[index] = ProducerControl(control.well, "SHUT", control.mode,
                                                       None, control.bhp)
            for index, control in enumerate(injectors):
                if control.well in shut_now and control.status == "OPEN":
                    shut(control.well, *_shut_reason(control.well, forced, active_windows),
                         control.rate or 0.0,
                         _shut_mark(control.well, forced, active_windows))
                    injectors[index] = InjectorControl(control.well, control.fluid, "SHUT",
                                                       control.mode, None, control.bhp)

        limit = limits.max_active_wells
        if limit is not None:
            active = [control.well for control in producers if control.status == "OPEN"]
            active += [control.well for control in injectors if control.status == "OPEN"]
            if len(active) > limit:
                # закрываем худших по удельной марже: то же правило, что и при
                # нехватке мощности по жидкости — иначе выбор был бы произвольным
                worst = sorted(set(active), key=lambda well: margins.get(well, 0.0))
                closing = set(worst[: len(active) - limit])
                for index, control in enumerate(producers):
                    if control.well in closing and control.status == "OPEN":
                        shut(control.well, "предел числа действующих скважин",
                             f"действующих {len(active)} > предела {limit}; "
                             f"маржа {margins.get(control.well, 0.0):.4g} млн — "
                             "наименьшая в фонде",
                             control.liquid_rate or 0.0)
                        producers[index] = ProducerControl(control.well, "SHUT",
                                                           control.mode, None, control.bhp)
                for index, control in enumerate(injectors):
                    if control.well in closing and control.status == "OPEN":
                        shut(control.well, "предел числа действующих скважин",
                             f"действующих {len(active)} > предела {limit}",
                             control.rate or 0.0)
                        injectors[index] = InjectorControl(control.well, control.fluid,
                                                           "SHUT", control.mode, None,
                                                           control.bhp)

        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result, decisions


def _fit_wells_to_caps(steps: list[ScheduleStep],
                       limits: Limits) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Потолок на **одну** скважину: по типу и адресной картой.

    Суммарный предел по объекту такого ограничения не выражает: сто скважин по
    десять кубов и десять по сто дают один и тот же объект и совершенно разные
    скважины. Адресная карта перекрывает потолок по типу — у скважины может быть
    свой паспорт насоса.
    """
    caps = limits.well_caps_m3_per_day or {}
    if not caps and limits.max_well_injection_m3_per_day is None \
            and limits.max_well_liquid_m3_per_day is None:
        return steps, []

    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []

    def cap_for(well: str, default: float | None) -> float | None:
        return caps.get(well, default)

    for step in steps:
        producers = []
        for control in step.producers:
            cap = cap_for(control.well, limits.max_well_liquid_m3_per_day)
            rate = control.liquid_rate
            if (control.status != "OPEN" or cap is None or rate is None or rate <= cap):
                producers.append(control)
                continue
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(Allocation(
                    well=control.well, resource="потолок скважины по жидкости",
                    before=rate, after=cap,
                    reason=f"отбор {rate:,.10g} > потолка скважины {cap:,.10g} м³/сут",
                ))
            producers.append(ProducerControl(control.well, control.status, control.mode,
                                             cap, control.bhp))
        injectors = []
        for control in step.injectors:
            cap = cap_for(control.well, limits.max_well_injection_m3_per_day)
            rate = control.rate
            if (control.status != "OPEN" or cap is None or rate is None or rate <= cap):
                injectors.append(control)
                continue
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(Allocation(
                    well=control.well, resource="потолок скважины по закачке",
                    before=rate, after=cap,
                    reason=f"закачка {rate:,.10g} > потолка скважины {cap:,.10g} м³/сут",
                ))
            injectors.append(InjectorControl(control.well, control.fluid, control.status,
                                             control.mode, cap, control.bhp))
        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result, decisions


def _fit_pressures(steps: list[ScheduleStep],
                   limits: Limits) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Пределы давления из конфигурации, перекрывающие дек.

    В деке Модели Z стоят 50 атм снизу у добывающих и 300 сверху у
    нагнетательных. Кейс «по давлениям» назовёт свои числа, а подменять ради
    этого дек нельзя — он материал организаторов. Поэтому предел приходит
    конфигурацией, и в решении прямо сказано, что он взят оттуда: иначе на защите
    не отличить «так в деке» от «так мы задали».
    """
    if limits.min_producer_bhp_atm is None and limits.max_injector_bhp_atm is None:
        return steps, []

    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        producers = []
        for control in step.producers:
            floor = limits.min_producer_bhp_atm
            if floor is None or control.bhp is None or control.bhp >= floor:
                producers.append(control)
                continue
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(Allocation(
                    well=control.well, resource="забойное давление, предел из конфига",
                    before=control.bhp, after=floor,
                    reason=(f"забойное {control.bhp:,.10g} ниже предела "
                            f"{floor:,.10g} атм, заданного конфигурацией"),
                ))
            producers.append(ProducerControl(control.well, control.status, control.mode,
                                             control.liquid_rate, floor))
        injectors = []
        for control in step.injectors:
            ceiling = limits.max_injector_bhp_atm
            if ceiling is None or control.bhp is None or control.bhp <= ceiling:
                injectors.append(control)
                continue
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(Allocation(
                    well=control.well, resource="давление нагнетания, предел из конфига",
                    before=control.bhp, after=ceiling,
                    reason=(f"давление нагнетания {control.bhp:,.10g} выше предела "
                            f"{ceiling:,.10g} атм, заданного конфигурацией"),
                ))
            injectors.append(InjectorControl(control.well, control.fluid, control.status,
                                             control.mode, control.rate, ceiling))
        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result, decisions


def _years(steps: list[ScheduleStep]) -> list[int]:
    """Календарные годы, встречающиеся в расписании, по порядку."""
    return sorted({step.date.year for step in steps})


def _annual_budget(steps: list[ScheduleStep], rate_m3_per_day: float) -> dict[int, float]:
    """Суточная ставка, прочитанная как среднее за год → объём каждого года.

    Дни считаются только по месяцам, которые в расписании есть: горизонт
    управления начинается в середине года и в середине же кончается, а деление
    на 365 в такие годы завысило бы допустимый объём и пропустило нарушение.
    """
    days: dict[int, int] = {}
    for step in steps:
        days[step.date.year] = days.get(step.date.year, 0) + \
            monthrange(step.date.year, step.date.month)[1]
    return {year: rate_m3_per_day * count for year, count in days.items()}


def _fit_injection_to_annual_volume(
    steps: list[ScheduleStep], limit_m3: float
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Годовой объём закачки по объекту, а не суточный расход.

    Отраслевая норма нормирует именно годовой объём
    (`docs/manus/vrr-water-balance.md`), и суточный предел его не выражает:
    закачка неравномерна по месяцам, и год может выйти за норму, ни разу не
    превысив суточного потолка.
    """
    return _fit_injection_to_annual_budget(
        steps, dict.fromkeys(_years(steps), limit_m3), reason="годовой объём закачки"
    )


def _fit_injection_to_annual_budget(
    steps: list[ScheduleStep], budget_m3: dict[int, float], *, reason: str
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Уложить закачку в объём, свой у каждого года.

    Срезается каждый нарушивший год отдельно и пропорционально — приоритета между
    скважинами у нас нет по той же причине, что и в балансе воды. Пропорция
    выбрана ещё и потому, что не создаёт мероприятий: скважина остаётся открытой,
    а не закрывается, и квота мероприятий от срезки не расходуется.
    """
    by_year: dict[int, float] = {}
    for step in steps:
        for control in step.injectors:
            if control.status == "OPEN" and control.rate:
                by_year[step.date.year] = by_year.get(step.date.year, 0.0) + \
                    control.rate * monthrange(step.date.year, step.date.month)[1]

    over = {
        year: total for year, total in by_year.items()
        if year in budget_m3 and total > budget_m3[year]
    }
    if not over:
        return steps, []

    decisions: list[Allocation] = []
    reported: set[int] = set()
    result: list[ScheduleStep] = []
    for step in steps:
        factor = budget_m3[step.date.year] / over[step.date.year] \
            if step.date.year in over else 1.0
        if factor >= 1.0:
            result.append(step)
            continue
        injectors = []
        for control in step.injectors:
            if control.status != "OPEN" or not control.rate:
                injectors.append(control)
                continue
            reduced = control.rate * factor
            if step.date.year not in reported:
                reported.add(step.date.year)
                decisions.append(Allocation(
                    well=control.well, resource=f"{reason}, {step.date.year}",
                    before=control.rate, after=reduced,
                    reason=(f"за {step.date.year} закачка {over[step.date.year]:,.0f} > "
                            f"нормы {budget_m3[step.date.year]:,.0f} м³"),
                ))
            injectors.append(InjectorControl(control.well, control.fluid, control.status,
                                             control.mode, reduced, control.bhp))
        result.append(ScheduleStep(step.date, step.producers, tuple(injectors)))
    return result, decisions


def _fit_liquid_to_annual_volume(
    steps: list[ScheduleStep], budget_m3: dict[int, float], *, reason: str
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """То же для отбора жидкости: свой объём у каждого года.

    Режется пропорционально, а не по марже, и это отдельное решение. Порядок «кого
    резать первым» по деньгам осмыслен, когда дефицит мгновенный: не помещается
    этот месяц. Годовой объём — величина накопленная, и очередь по марже одного
    месяца распределяла бы годовую нехватку по случайной её части. Пропорция
    сохраняет форму профиля, не закрывает ни одной скважины и потому не расходует
    квоту мероприятий.
    """
    by_year: dict[int, float] = {}
    for step in steps:
        for control in step.producers:
            if control.status == "OPEN" and control.liquid_rate:
                by_year[step.date.year] = by_year.get(step.date.year, 0.0) + \
                    control.liquid_rate * monthrange(step.date.year, step.date.month)[1]

    over = {
        year: total for year, total in by_year.items()
        if year in budget_m3 and total > budget_m3[year]
    }
    if not over:
        return steps, []

    decisions: list[Allocation] = []
    reported: set[int] = set()
    result: list[ScheduleStep] = []
    for step in steps:
        factor = budget_m3[step.date.year] / over[step.date.year] \
            if step.date.year in over else 1.0
        if factor >= 1.0:
            result.append(step)
            continue
        producers = []
        for control in step.producers:
            if control.status != "OPEN" or not control.liquid_rate:
                producers.append(control)
                continue
            reduced = control.liquid_rate * factor
            if step.date.year not in reported:
                reported.add(step.date.year)
                decisions.append(Allocation(
                    well=control.well, resource=f"{reason}, {step.date.year}",
                    before=control.liquid_rate, after=reduced,
                    reason=(f"за {step.date.year} отбор жидкости {over[step.date.year]:,.0f} > "
                            f"нормы {budget_m3[step.date.year]:,.0f} м³"),
                ))
            producers.append(ProducerControl(control.well, control.status, control.mode,
                                             reduced, control.bhp))
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))
    return result, decisions


def _fit_events_to_annual_quota(
    steps: list[ScheduleStep], limit: int
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Квота платных мероприятий на календарный год.

    Методика оценивает мероприятия поштучно — 1,0 млн руб. за остановку и
    столько же за пуск, 5,0 млн за перевод под закачку, — поэтому «не более N
    мероприятий в год» это естественная форма ограничения по фонду, и
    организаторы назвали фонд одним из трёх видов ограничений кейса.

    **Мероприятие считается по расписанию, а не по выгрузке.** Эталонный
    расчётчик выводит события из отклика (`WLPR`, `WWIR`) и знает ещё смену ЭЦН
    и изменение режима; здесь считается то, чем управляет сдаваемый файл:
    перемена состояния скважины между соседними шагами. Смены ЭЦН среди них нет
    — насос выбирается по дебиту, а дебит это прогноз, а не наша уставка.
    Поэтому квота ограничивает **наши** решения, и число мероприятий у
    расчётчика может отличаться; сравнивать их напрямую нельзя.

    Перевод под закачку — одно мероприятие, а не два: в расписании он выглядит
    двумя записями (добыча закрывается, `WCONINJE` дописывается), и считать их
    порознь значило бы вдвое сузить квоту.

    **Квота расходуется по календарю: первые N мероприятий года проходят,
    остальные отменяются.** Ранжировать мероприятия по ценности было бы точнее,
    но ценность мероприятия — это приращение ЧДД от него, и оно известно только
    полным расчётом; удельная маржа скважины, которой мы ранжируем срезку
    отбора, ценность события не выражает. Календарный порядок детерминирован и
    объясним словами, а произвольный выбор — нет.

    Отменённое мероприятие означает, что скважина остаётся в том состоянии, в
    каком была на предыдущем шаге, — и следующий шаг сравнивается уже с
    исправленным, поэтому отмена не порождает нового мероприятия в отместку.
    У первого шага предыдущего нет, и мероприятий он не даёт: историческое
    состояние до горизонта управления планировщик не видит (та же причина, по
    которой `controllable_wells` живёт в цикле, а не здесь).
    """
    decisions: list[Allocation] = []
    result: list[ScheduleStep] = []
    spent: dict[int, int] = {}
    previous: ScheduleStep | None = None

    for step in steps:
        if previous is None:
            result.append(step)
            previous = step
            continue

        was_producing = {c.well: c for c in previous.producers if c.status == "OPEN"}
        was_injecting = {c.well: c for c in previous.injectors if c.status == "OPEN"}
        now_producing = {c.well: c for c in step.producers if c.status == "OPEN"}
        now_injecting = {c.well: c for c in step.injectors if c.status == "OPEN"}

        before_producers = {c.well: c for c in previous.producers}
        before_injectors = {c.well: c for c in previous.injectors}

        year = step.date.year
        cancelled: set[str] = set()

        # порядок обхода — как в файле: сначала добывающие, потом нагнетательные
        wells = [c.well for c in step.producers]
        wells += [c.well for c in step.injectors if c.well not in wells]

        for well in wells:
            if well not in before_producers and well not in before_injectors:
                continue  # скважина появилась в горизонте — сравнивать не с чем
            kind = _event_kind(
                (well in was_producing, well in was_injecting),
                (well in now_producing, well in now_injecting),
            )
            if kind is None:
                continue
            if spent.get(year, 0) < limit:
                spent[year] = spent.get(year, 0) + 1
                continue
            cancelled.add(well)
            asked = now_producing.get(well) or now_injecting.get(well)
            kept = before_producers.get(well) or before_injectors.get(well)
            decisions.append(Allocation(
                well=well, resource=f"квота мероприятий, {year}",
                before=_rate_of(asked), after=_rate_of(kept),
                reason=(f"{kind} отменён: квота мероприятий на {year} год "
                        f"исчерпана (предел {limit}); скважина остаётся как была"),
            ))

        if not cancelled:
            result.append(step)
            previous = step
            continue

        producers = tuple(
            before_producers[c.well] if c.well in cancelled and c.well in before_producers
            else c
            for c in step.producers
        )
        # записи закачки, которой на прошлом шаге не было, при отмене снимаются:
        # именно её появление и было мероприятием
        injectors = tuple(
            before_injectors[c.well] if c.well in cancelled and c.well in before_injectors
            else c
            for c in step.injectors
            if not (c.well in cancelled and c.well not in before_injectors)
        )
        fixed = ScheduleStep(step.date, producers, injectors)
        result.append(fixed)
        previous = fixed

    return result, decisions


def _event_kind(before: tuple[bool, bool], now: tuple[bool, bool]) -> str | None:
    """Как называется перемена состояния скважины. `(добывает, закачивает)`."""
    if before == now:
        return None
    if any(before) and not any(now):
        return "остановка"
    if not any(before) and any(now):
        return "пуск"
    if before[0] and now[1]:
        return "перевод под закачку"
    return "возврат в добычу"


def _rate_of(control: ProducerControl | InjectorControl | None) -> float:
    """Расход записи независимо от её вида; закрытая запись — ноль."""
    if control is None or control.status != "OPEN":
        return 0.0
    rate = getattr(control, "liquid_rate", None)
    if rate is None:
        rate = getattr(control, "rate", None)
    return rate or 0.0


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
                            # предел печатается без округления до целого: у
                            # организаторов он может прийти дробным, а «предела
                            # 0 м³/сут» при пределе 0,5 — объяснение, которому
                            # нельзя верить (замер 23.08, Б28)
                            f"{capacity:,.10g} м³/сут"
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
        # Знак важен: с появлением нижней границы коридора планировщик умеет не
        # только резать, но и добирать, и «урезано −120 м³/сут» читалось бы как
        # опечатка, а не как добор до нормы.
        verb = "урезано" if total >= 0 else "добрано"
        lines.append(
            f"{resource}: {verb} {len(items)} скважин, суммарно {abs(total):,.0f} м³/сут"
        )
        lines.extend(f"  {item}" for item in sorted(items, key=lambda x: -abs(x.cut))[:10])
        lines.append("")
    return "\n".join(lines).rstrip()


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
        f"{limit_m3_per_day:,.10g} м³/сут",
    )


def _fit_injection_to_water_corridor(
    steps: list[ScheduleStep],
    forecast: list[dict[str, Any]],
    fluids: Fluids,
    *,
    rule: str,
    ceiling: float,
    floor: float | None,
    source_m3_per_day: float | None,
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Привести закачку к коридору нормы и к тому, сколько воды физически есть.

    Один шаг вместо трёх, и это не стилистика. Каждая срезка здесь **умножает**
    текущие расходы, а базу берёт из прогноза — того самого, который срезки уже
    не отражают. Две срезки подряд поэтому дают произведение множителей, то есть
    «двойную экономию», которой в пласте нет; ровно об этом предупреждает
    комментарий о порядке в `enforce`. Сложить в одном шаге срезку сверху, добор
    снизу и предел источника — единственный способ не считать одно и то же
    дважды.

    Три границы, из которых собирается итог:

    * **верхняя граница нормы** — `база × ceiling`. База зависит от правила:
      отобранная пустотность (нефть вместе с водой) у `compensation`, добытая
      вода у `produced_water`;
    * **нижняя граница нормы** — `база × floor`, если коридор двусторонний;
    * **физика источника** — попутная вода плюс `source_m3_per_day × сутки`.

    Приоритет между ними один и он не обсуждается: **воды, которой нет, никакая
    норма не создаёт.** Поэтому предел источника режет и верхнюю границу, и
    добор: если водозабора не хватает на нижнюю границу коридора, расписание
    останется ниже нормы, и это правильный исход, а не ошибка. Назвать
    несоответствие — работа аудита по факту
    (`aios.balance.compensation_problems`, `aios.balance.external_water_problems`).

    Правило `external_water` сюда не попадает: оно целиком про источник и
    обслуживается `_fit_injection_to_external_water`.
    """
    base = 0.0
    produced_water = 0.0
    planned = 0.0
    stamps: set[str] = set()
    for row in forecast:
        oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]
        if liquid_t > 0 and 0 <= oil_t <= liquid_t:
            water_m3 = (liquid_t - oil_t) * 1000.0 / fluids.water_density
            produced_water += water_m3
            base += water_m3
            if rule == "compensation":
                base += oil_t * 1000.0 / fluids.oil_density
        planned += max(row["WWIT_Diff"], 0.0)
        stamp = str(row.get("DATA", ""))[:7]
        if len(stamp) == 7:
            stamps.add(stamp)

    if planned <= 0 or base <= 0:
        return steps, []

    allowed = base * ceiling
    reason = (
        f"закачка {planned:,.0f} м³ даёт компенсацию {planned / base:.2f} "
        f"от отобранных {base:,.0f} м³ при потолке {ceiling:.2f}"
        if rule == "compensation" else
        f"закачка {planned:,.0f} м³ превышает добытую воду "
        f"{base:,.0f} м³; источника сверх неё нет"
    )

    if source_m3_per_day is not None:
        if not stamps:
            # Тихо подставить 365 нельзя: на длинном горизонте это занизило бы
            # доступную воду в разы и срезало бы закачку без всякой причины.
            raise ValueError(
                "лимиту стороннего источника нужна колонка DATA в прогнозе: он "
                "задан в м³/сут, и без календаря его не с чем сравнивать"
            )
        days = balance.days_of(stamps)
        by_source = produced_water + source_m3_per_day * days
        if by_source < allowed:
            allowed = by_source
            reason = (
                f"закачке нужно {(planned - produced_water) / days:,.0f} м³/сут воды "
                f"со стороны при мощности источника {source_m3_per_day:,.10g} м³/сут"
            )

    if planned > allowed:
        return _scale_injection(steps, allowed / planned, reason)

    if floor is None:
        return steps, []

    # Добор ограничен той же доступной водой: `allowed` уже её учитывает.
    required = min(base * floor, allowed)
    if planned >= required:
        return steps, []

    return _scale_injection(
        steps, required / planned,
        f"закачка {planned:,.0f} м³ даёт {planned / base:.2f} от базы правила "
        f"{base:,.0f} м³ — ниже границы коридора {floor:.2f}",
    )


def _scale_injection(steps: list[ScheduleStep], factor: float,
                     reason: str) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Пропорционально изменить закачку всех открытых нагнетательных.

    Множитель меньше единицы режет, больше — добирает до нижней границы коридора.
    `Allocation` в обоих случаях один и тот же: он описывает изменение, а не
    только урезание.
    """
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
                            f"отбор по объекту {total:,.0f} > предела {capacity:,.10g} м³/сут; "
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
