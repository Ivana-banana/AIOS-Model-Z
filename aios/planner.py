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
        """Задевает ли окно месяц этого шага. Границы включаются обе.

        Шаг расписания — это **месяц**: режим, назначенный первым числом,
        действует до следующего шага. Значит окно короче месяца или начинающееся
        в середине («остановить с 05.02 по 20.02») выразить точно нечем, и выбор
        только между двумя огрублениями: закрыть месяц целиком или не закрывать
        вовсе. Закрываем целиком — нарушить ограничение хуже, чем потерять
        добычу, и на защите «мы остановили на месяц вместо двух недель» звучит
        иначе, чем «мы не остановили».

        До 02.09 сравнивалась только дата шага, и окно 05.02–20.02 не закрывало
        ничего, а проверка условий говорила «соблюдено» — обе стороны звали этот
        же метод (найдено критиком 01.09).
        """
        last_day = monthrange(moment.year, moment.month)[1]
        month_start = moment.replace(day=1)
        month_end = moment.replace(day=last_day)
        return self.start <= month_end and self.end >= month_start


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

    max_field_liquid_m3_total: float | None = None
    """Объём отбора жидкости за **весь горизонт**, м³. None — предела нет.

    Третья форма объёмного ограничения после суточной и годовой (п. 10 чек-листа
    приёмки): «за период разработки поднять не более V». Годовым её не выразить —
    год может укладываться в норму, а сумма лет нет.
    """

    max_field_injection_m3_total: float | None = None
    """Объём закачки за весь горизонт, м³. None — предела нет."""

    max_watercut: float | None = None
    """Предел обводнённости работающей добывающей, доля. None — предела нет.

    Как и пластовое давление, это **отклик пласта**, а не уставка: расписанием
    обводнённость не задать, поэтому предел работает проверкой по выгрузке
    (`aios.summary.sanity_problems`), а не срезкой в планировщике. Санитарный
    диапазон «обводнённость ∈ [0, 1]» — про физику и остаётся; здесь речь про
    условие кейса, п. 28 чек-листа приёмки.
    """

    min_producer_rate_m3_per_day: float | None = None
    """Минимальный дебит работающей добывающей, м³/сут. None — предела нет.

    Форма «минимальная рентабельная добыча» (п. 20 чек-листа приёмки): скважина
    либо даёт не меньше порога, либо стоит. У нас до 02.09 было только правило
    «`[policy].lookback_months` убыточных месяцев подряд — остановка», а это
    другое: оно про деньги за период, а не про режим в моменте, и кейс может
    задать именно второе.

    Скважина ниже порога **останавливается**, а не поднимается до него: поднять
    уставку значит пообещать дебит, которого пласт может не дать — ровно то, за
    что в этом проекте забракован форсаж на суррогате.
    """

    min_regime_months: int | None = None
    """Сколько месяцев режим скважины нельзя менять после изменения. None — можно.

    Вторая половина п. 25 чек-листа приёмки: «минимальная длительность режима».
    Смена уставки мероприятием не считается (методика платит за пуск, остановку и
    перевод), поэтому квота мероприятий этой формы не выражает: расписание может
    дёргать дебит каждый месяц, не потратив ни одного мероприятия.

    Правило применяется **к предложениям цикла**, а не к срезкам ограничений: если
    объектный предел требует срезать режим раньше срока, ограничение важнее — иначе
    удержание режима заставляло бы нарушать допустимость (та же развилка, что у
    квоты, Р-139). Обеспечивается это порядком: удержание стоит до всех срезок, и
    что оно вернуло, то пределы и приведут.
    """

    max_events_per_month: int | None = None
    """Максимум платных мероприятий за календарный месяц. None — предела нет.

    Годовая квота не выражает «не больше двух переключений в месяц»: год из
    двенадцати мероприятий проходит и тогда, когда все двенадцать сделаны в
    январе. Форма отдельная и в чек-листе приёмки стоит отдельной строкой
    (`docs/validation-case-checklist.md`, п. 25).
    """

    max_events_per_year: int | None = None
    """Максимум платных мероприятий за год: остановки, пуски, переводы. None — нет.

    Методика оценивает их поштучно (1,0 млн руб. за остановку и за пуск, 5,0 за
    ВНС), поэтому ограничить их число — естественная форма ограничения по фонду.

    Считается по расписанию, а не по выгрузке, и потому не совпадает с событиями
    эталонного расчётчика — подробности в `_fit_events_to_quota`.
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
    #: единица величин `before`/`after`. Почти все решения планировщика режут
    #: расход, поэтому умолчание — «м³/сут»; но пределы давления из кейса
    #: приходят в атмосферах, и до 08.09 строка решения печатала их как расход.
    #: На защите и в окне ограничений 11.09 эту строку читают глазами, и
    #: «50 → 80 м³/сут» про забойное давление — не описка, а неверная величина.
    unit: str = "м³/сут"

    @property
    def cut(self) -> float:
        return self.before - self.after

    def __str__(self) -> str:
        return (
            f"скв. {self.well}: {self.resource} {self.before:.4g} → {self.after:.4g} "
            f"{self.unit} — {self.reason}"
        )


def enforce(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
            forecast: list[dict[str, Any]],
            norms: Norms,
            limits: Limits,
            fluids: Fluids | None = None,
            carried_water: tuple[float, float] | None = None,
            ) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Привести расписание в соответствие с ограничениями объекта.

    `forecast` — прогноз отклика на это же расписание (суррогат или расчёт).
    Возвращает исправленное расписание и список решений с обоснованием.
    """
    fluids = fluids or Fluids()
    forecast = _with_string_wells(forecast)
    # расписание разворачивается в полные шаги: см. `expand_deltas` — без этого
    # все пределы объекта, группы и фонда считают не тот фонд
    result = expand_deltas(steps)
    decisions: list[Allocation] = []

    # Удержание режима идёт **до** срезок, а не после. Оно возвращает скважине
    # прежнюю уставку, и прежняя может быть выше нынешней: после всех пределов
    # такой возврат ломал объектный предел там, где он держался сам собой, без
    # единой срезки (ревизия 02.09). До срезок этого не случается — что удержание
    # вернуло, то пределы и приведут, а «ограничение важнее длительности режима»
    # остаётся верным по построению.
    if limits.min_regime_months is not None:
        result, held = _hold_regimes(result, limits.min_regime_months, protected=set())
        decisions.extend(held)

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

    result, fund = _fit_fund(result, forecast, norms, limits, fluids)
    decisions.extend(fund)

    if limits.max_field_injection_m3_per_year is not None:
        result, yearly = _fit_injection_to_annual_volume(
            result, limits.max_field_injection_m3_per_year
        )
        decisions.extend(yearly)

    if limits.max_field_liquid_m3_total is not None:
        result, total_liquid = _fit_liquid_to_annual_volume(
            result, _horizon_budget(result, limits.max_field_liquid_m3_total,
                                    producers=True),
            reason="объём отбора за горизонт",
        )
        decisions.extend(total_liquid)

    if limits.max_field_injection_m3_total is not None:
        result, total_injection = _fit_injection_to_annual_budget(
            result, _horizon_budget(result, limits.max_field_injection_m3_total,
                                    producers=False),
            reason="объём закачки за горизонт",
        )
        decisions.extend(total_injection)

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
    # Мероприятие, порождённое самой срезкой, отменять нельзя: отмена вернула бы
    # скважину в состояние, которое нарушает жёсткое ограничение. Такие события
    # квоту расходуют, но не отменяются — найдено критиком 01.09 (Р-139).
    #
    # Месячная квота идёт первой: она строже по локальности, и год, приведённый
    # ею, годовой квоте резать уже нечего. Обратный порядок оставил бы месяц с
    # двенадцатью переключениями внутри годового лимита двенадцать.
    # Минимальный рентабельный дебит проверяется **после** всех срезок: любая из
    # них может опустить скважину ниже порога, и репетиция на реальных данных это
    # показала — после «успешной» срезки оставалась скважина на 3,4 м³/сут при
    # пороге 5 (02.09).
    if limits.min_producer_rate_m3_per_day is not None:
        result, small = _shut_below_minimum(result, limits.min_producer_rate_m3_per_day)
        decisions.extend(small)

    if limits.max_events_per_month is not None or limits.max_events_per_year is not None:
        # Защищены не только мероприятия, порождённые срезками, но и те, которыми
        # держится состав фонда: отмена остановки возвращает скважину в работу, и
        # предел числа действующих ломается — а перепроверять его после квоты
        # некому (ревизия 02.09).
        result, events = _fit_events_to_quota(
            result,
            per_month=limits.max_events_per_month,
            per_year=limits.max_events_per_year,
            protected=(events_of(result) - events_of(expand_deltas(steps)))
            | _events_holding_the_fund(result, limits),
        )
        decisions.extend(events)

    # Баланс воды подбирается ПОСЛЕДНИМ, и это исправление, а не перестановка.
    #
    # До 03.09 коридор стоял первым. Каждая следующая срезка уменьшает отбор — а
    # компенсация есть отношение закачки к отбору, — то есть поднимает
    # компенсацию выше уже подобранного потолка. Замер ревизии: нетронутая
    # история при «только коридоре» даёт ноль решений, а с добавленными пределами
    # по фазам — **15 нарушений** коридора и компенсацию 1,41; полный цикл на
    # репетиции — 16 нарушений и 133–141 %. С включённой проверкой fail-closed
    # (Р-165) это означает, что ни один кандидат не проходит сертификацию, то
    # есть контур отказывается от собственных решений (Р-189).
    #
    # Здесь же кончается то, что можно сделать одним проходом: база коридора
    # берётся из **прогноза**, а прогноз посчитан для расписания ДО срезок. Чтобы
    # коридор считался по тому отбору, который расписание реально даёт, нужен
    # новый прогноз — а суррогата у планировщика нет и быть не должно. Эту часть
    # доводит до неподвижной точки цикл (`loop.run_cycle`): он умеет спросить
    # прогноз заново.
    # Мощность водозабора действует **независимо от правила нормы**, в том числе
    # когда баланс воды выключен вовсе: правило отвечает на вопрос «сколько
    # закачивать», а источник — «сколько воды физически есть». Воды, которой нет,
    # никакая норма не создаёт, и отключение правила её не создаёт тоже (2.1 #9).
    # На Модели Z ключ не задан, поэтому здесь это ничего не меняет.
    if not limits.enforce_water_balance and limits.max_external_water_m3_per_day is not None:
        result, source = _fit_injection_to_external_water(
            result, forecast, fluids, limits.max_external_water_m3_per_day
        )
        decisions.extend(source)

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
                # Коридор идёт последним (Р-189), то есть **после** адресных
                # потолков закачки, и его добор снизу мог поднять уставку выше
                # предела, который кейс объявил, — переcечь его было уже некому
                # (С5). Передаём потолки внутрь: добор ограничен ими, а не только
                # водой.
                # Накопленное до горизонта — решение владельца от 04.09: коридор
                # судится **тем же окном**, что и подтверждение, то есть по всей
                # выгрузке с 1994. Без переноса цикл считал компенсацию с 2007, а
                # там сама история даёт 1,291…1,401 при потолке 1,3 — бездействие
                # нарушало наш же потолок, и fail-closed отвергал любого
                # кандидата (С1, Р-201, Р-204).
                carried=carried_water,
                well_caps={**(limits.well_caps_m3_per_day or {})},
                default_cap=limits.max_well_injection_m3_per_day,
            )
        decisions.extend(water)

    # Квота отменяет мероприятие снятием записи со шага, поэтому на выходе
    # расписание снова частично дельтовое — а вся арифметика ограничений (и наша,
    # и в `aios.casecheck`) считает по записям шага. Разворачиваем обратно: то, что
    # verificируется, и то, что уезжает в файл, должно быть одним и тем же.
    return expand_deltas(result), decisions


def _achieved_rates(forecast: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Достигнутый дебит жидкости по прогнозу: `{"2007-01": {"скв": м³/сут}}`."""
    rates: dict[str, dict[str, float]] = {}
    for row in forecast:
        if not counted_by_methodology(row):
            continue
        stamp = str(row["DATA"])[:7]
        rates.setdefault(stamp, {})[str(row["well"])] = float(row["WLPR"] or 0.0)
    return rates


def _phase_by_well(forecast: list[dict[str, Any]],
                   phase: str) -> dict[str, dict[str, float]]:
    """Тонны фазы за месяц по скважинам: `{"2007-01": {"скв": 123.0}}`.

    Поскважинно, а не суммой по объекту: фаза месяца считается по **текущим**
    уставкам расписания, а для этого надо знать, чья именно это тонна. Строки,
    исключённые методикой, не считаются — тем же правилом, что и везде в проекте.
    """
    totals: dict[str, dict[str, float]] = {}
    for row in forecast:
        if not counted_by_methodology(row):
            continue
        oil_t = max(row["WOMT_Diff"], 0.0)
        liquid_t = max(row["WLPT_Diff"], 0.0)
        value = oil_t if phase == "нефть" else max(liquid_t - oil_t, 0.0)
        stamp = str(row["DATA"])[:7]
        well = str(row["well"])
        month = totals.setdefault(stamp, {})
        month[well] = month.get(well, 0.0) + value
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

    monthly = _phase_by_well(forecast, phase)
    reference = _achieved_rates(forecast)
    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        stamp = f"{step.date.year:04d}-{step.date.month:02d}"
        days = monthrange(step.date.year, step.date.month)[1]
        by_well = monthly.get(stamp)
        if by_well is None:
            # Месяца нет в прогнозе — фазу этого месяца посчитать нечем. Молчать
            # нельзя (докстринг обещает отказ на пустом прогнозе, и половинчатый
            # прогноз ничем не лучше), падать посреди срезки тоже нельзя: месяцы
            # за пределами горизонта прогноза законны. Поэтому месяц проходит без
            # срезки, но факт печатается решением — найдено критиком 01.09.
            if stamp not in reported:
                reported.add(stamp)
                decisions.append(Allocation(
                    well="—", resource=f"{phase}, {stamp}",
                    before=0.0, after=0.0,
                    reason=(f"месяца нет в прогнозе: предел по фазе «{phase}» за {stamp} "
                            "не проверен и не применён"),
                ))
            result.append(step)
            continue
        achieved = reference.get(stamp, {})

        # Фаза считается **по текущим уставкам**, а не по сырому прогнозу: к этому
        # месту расписание уже прошло объектные пределы, и месяц мог перестать
        # нарушать фазовый предел сам собой. До 02.09 срезка этого не видела и
        # резала повторно: на репетиции при пределе воды 1400 т/сут и фактическом
        # 1203 планировщик сделал 33 срезки и снял 471 939 м³ жидкости.
        planned = {control.well: control.liquid_rate or 0.0 for control in step.producers
                   if control.status == "OPEN"}
        produced_t = sum(
            tonnes * min(1.0, planned.get(well, 0.0) / achieved[well])
            for well, tonnes in by_well.items()
            if achieved.get(well, 0.0) > 0
        )
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
            # Масштабируется **действующий** поток, а не достигнутый дебит:
            # уставка ниже факта уже сама по себе срезала отклик, и умножать на
            # factor надо именно её. Прежняя редакция ставила `achieved × factor`
            # всем подряд, и скважина, у которой доля уже была ниже factor,
            # оставалась нетронутой — предел после «успешной» срезки оставался
            # нарушенным (ревизия 02.09, находка 1).
            reduced = min(control.liquid_rate, achieved_rate) * factor
            # Ветка защитная и **недостижима** при текущей логике: множитель
            # равен `мощность / расход` и берётся только при `расход > мощность`,
            # то есть строго меньше единицы — проверено перебором краёв, включая
            # расход, больший мощности на один ulp (05.09, Р-225). Оставлена
            # намеренно: она держит инвариант «срезка не поднимает уставку», если
            # множитель когда-нибудь начнут считать иначе. Мутационный прогон
            # помечает её эквивалентной, и это верный вердикт, а не пробел в
            # тестах — сам инвариант закреплён
            # `test_a_phase_cut_never_raises_the_setpoint`.
            if reduced >= control.liquid_rate:
                producers.append(control)
                continue
            # решение печатается каждой срезанной скважине, а не первой за месяц:
            # ключ по месяцу оставлял четыре изменённые уставки из пяти без
            # единого объяснения — а объяснимость здесь требование регламента,
            # и она же кормит отчёт репетиции (ревизия 02.09, находка 7)
            mark = f"{stamp}:{control.well}"
            if mark not in reported:
                reported.add(mark)
                decisions.append(Allocation(
                    well=control.well, resource=f"{phase}, {stamp}",
                    before=control.liquid_rate, after=reduced,
                    reason=(f"{phase} {rate:,.1f} > мощности {capacity_t_per_day:,.1f} т/сут; "
                            "уставка опущена до той же доли от достигнутого дебита"),
                ))
            producers.append(ProducerControl(
                control.well, "SHUT" if reduced <= 0 else control.status,
                control.mode, reduced or None, control.bhp,
            ))
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
                # ноль означает остановку, а не «открыта и ничего не даёт»: открытая
                # запись с нулевым дебитом делает расписание непригодным целиком
                # (`schedule.validate`), и до 02.09 групповой потолок именно её и
                # ставил — в отличие от объектного, который закрывал скважину
                rate = cuts[control.well]
                producers[index] = ProducerControl(
                    control.well, "SHUT" if rate <= 0 else control.status,
                    control.mode, rate or None, control.bhp,
                )

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
                injectors[index] = InjectorControl(
                    control.well, control.fluid,
                    "SHUT" if reduced <= 0 else control.status,
                    control.mode, reduced or None, control.bhp,
                )

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
              limits: Limits,
              fluids: Fluids | None = None) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Ограничения по составу фонда — три формы, все названы организаторами.

    1. **Обязательные остановки** (`forced_shut_wells`): скважина закрывается
       независимо от того, что предложил цикл. Форма самая жёсткая и самая
       вероятная: «эти скважины в ремонте, ими не управлять».
    2. **Максимум действующих** (`max_active_wells`): если открытых больше
       предела, закрываются худшие по удельной марже — тем же правилом, каким
       планировщик режет отбор при нехватке мощности. Без нормативов правило
       решать нечем, и это отказ, а не молчаливое «закроем первых попавшихся».

       **Выбирают только добывающие, и это не вкусовое решение.** Удельная маржа
       определена через поднятый объём (`_margin_per_liquid`), а у нагнетательной
       объёма нет — для неё та же функция возвращает суммарную маржу в млн руб.,
       то есть величину другого масштаба: −0,26 млн против ±0,0005 млн/м³ у
       добывающих. В общем сорте нагнетательные оказывались худшими всегда, и
       предел числа действующих скважин первым делом рубил закачку — при том что
       срез закачки на Модели Z обрушивает нефть вчетверо (замер 20.08). Если
       добывающих не хватило, чтобы уйти под предел, закачка закрывается, но
       после них и с самой мелкой: крупная закачка держит пластовое давление
       (ревизия 03.09, 2.1 #5).
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
    # плотности кейса, а не умолчания Модели Z: маржа считается в тоннах через
    # них, и подмена меняет порядок скважин под пределом (2.1 #12)
    margins = _margin_per_liquid(forecast, norms, fluids or Fluids()) if norms else {}
    decisions: list[Allocation] = []
    reported: set[str] = set()
    result: list[ScheduleStep] = []
    # Накопительное состояние фонда: в деке шаг несёт только изменившиеся
    # скважины, и та, что на этом шаге не упомянута, продолжает работать. До
    # 02.09 ограничения фонда правили только записи шага — то есть скважина без
    # записи оставалась вне контроля, а проверка соглашалась, потому что смотрела
    # туда же (найдено репетицией приёмки).
    working: dict[str, tuple[bool, bool]] = {}
    rates: dict[str, tuple[str, float, float | None]] = {}
    fluids_of: dict[str, str] = {}

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
                # нехватке мощности по жидкости — иначе выбор был бы произвольным.
                # Ранжируются только добывающие: у нагнетательной удельной маржи
                # нет, и в общем сорте она оказывалась худшей всегда (Р-163)
                excess = len(active) - limit
                producing = {control.well for control in producers
                             if control.status == "OPEN"}
                worst = sorted(producing, key=lambda well: margins.get(well, 0.0))
                closing = set(worst[:excess])
                if len(closing) < excess:
                    # добывающих не хватило: остаток берём с самой мелкой закачки
                    pumping = {control.well: control.rate or 0.0 for control in injectors
                               if control.status == "OPEN"}
                    closing |= set(sorted(pumping, key=lambda well: pumping[well])
                                   [: excess - len(closing)])
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
                             f"действующих {len(active)} > предела {limit}; "
                             "добывающих не хватило, закрыта самая мелкая закачка",
                             control.rate or 0.0)
                        injectors[index] = InjectorControl(control.well, control.fluid,
                                                           "SHUT", control.mode, None,
                                                           control.bhp)

        # обновляем состояние по записям шага
        for control in producers:
            producing = control.status == "OPEN"
            working[control.well] = (producing,
                                     working.get(control.well, (False, False))[1]
                                     and not producing)
            if producing:
                rates[control.well] = ("producer", control.liquid_rate or 0.0, control.bhp)
        for control in injectors:
            injecting = control.status == "OPEN"
            working[control.well] = (working.get(control.well, (False, False))[0]
                                     and not injecting, injecting)
            if injecting:
                rates[control.well] = ("injector", control.rate or 0.0, control.bhp)
            fluids_of[control.well] = control.fluid

        # скважина, которую надо закрыть, но записи о ней на шаге нет: дописываем
        # её сами — иначе ограничение фонда не действует вовсе
        named = {control.well for control in producers} | {control.well for control in injectors}
        silent = {well for well, (producing, injecting) in working.items()
                  if (producing or injecting) and well not in named and well in shut_now}
        for well in sorted(silent):
            kind, rate, bhp = rates.get(well, ("producer", 0.0, None))
            shut(well, *_shut_reason(well, forced, active_windows), rate,
                 _shut_mark(well, forced, active_windows))
            if kind == "producer":
                producers.append(ProducerControl(well, "SHUT", "LRAT", None, bhp))
            else:
                # фаза берётся из последней записи скважины, а не выдумывается:
                # «WATER» верно на Модели Z и неверно вообще (ревизия 02.09)
                injectors.append(InjectorControl(well, fluids_of.get(well, "WATER"),
                                                 "SHUT", "RATE", None, bhp))
            working[well] = (False, False)

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
            # потолок ноль — это остановка: адресный ноль конфигурация принимает
            # («не давать этой скважине работать»), а открытая запись с нулевым
            # дебитом делает расписание непригодным
            producers.append(ProducerControl(
                control.well, "SHUT" if cap <= 0 else control.status,
                control.mode, cap or None, control.bhp,
            ))
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
            injectors.append(InjectorControl(
                control.well, control.fluid, "SHUT" if cap <= 0 else control.status,
                control.mode, cap or None, control.bhp,
            ))
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
                    unit="атм",
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
                    unit="атм",
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


def _horizon_budget(steps: list[ScheduleStep], limit_m3: float,
                    *, producers: bool) -> dict[int, float]:
    """Разложить объём **за весь горизонт** по годам: ранние берут всё, поздние остаток.

    Объём за горизонт фиксирован, значит каждый кубометр можно поднять рано или
    поздно — и дисконт отвечает однозначно: рано. Поэтому предел раздаётся годам
    по порядку: год берёт столько, сколько просит, пока предел не кончится;
    следующим достаётся остаток, дальше — ноль.

    Пропорциональная раскладка, стоявшая здесь до 02.09, была ошибкой ровно в
    этом месте: она отдавала тот же объём равномерно и теряла около трети
    дисконтированного отбора (замер ревизии: 3 302 815 против 4 324 619 м³
    приведённых при одинаковом суммарном объёме). Довод «не съедать последние
    годы» перевёрнут: съесть их — это и есть максимум при данном пределе.

    Годовой предел раскладкой не занимается вовсе: там корзина одна на год, и
    внутри года режется по марже.
    """
    volume: dict[int, float] = {}
    for step in steps:
        days = monthrange(step.date.year, step.date.month)[1]
        controls = step.producers if producers else step.injectors
        for control in controls:
            if control.status != "OPEN":
                continue
            rate = (control.liquid_rate if producers else control.rate) or 0.0
            volume[step.date.year] = volume.get(step.date.year, 0.0) + rate * days
    if sum(volume.values()) <= 0:
        return {}
    budget: dict[int, float] = {}
    left = limit_m3
    for year in sorted(volume):
        take = min(volume[year], left)
        budget[year] = take
        left -= take
    return budget


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
            injectors.append(InjectorControl(
                control.well, control.fluid,
                "SHUT" if reduced <= 0 else control.status,
                control.mode, reduced or None, control.bhp,
            ))
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
            producers.append(ProducerControl(
                control.well, "SHUT" if reduced <= 0 else control.status,
                control.mode, reduced or None, control.bhp,
            ))

        # Год, в котором бюджет отбора равен нулю, закрывает и закачку. Иначе
        # получается год без добычи, но с закачкой: вода идёт в пласт, OPEX
        # тратится, выручки нет вовсе — прямое следствие раздачи объёма ранним
        # годам (Р-147). Компенсировать в такой год нечего: отбора нет, а правило
        # баланса воды считает закачку от отбора (ревизия 03.09, 2.1 #6).
        injectors = list(step.injectors)
        if factor <= 0.0:
            for index, control in enumerate(injectors):
                if control.status != "OPEN":
                    continue
                decisions.append(Allocation(
                    well=control.well, resource=f"{reason}, {step.date.year}",
                    before=control.rate or 0.0, after=0.0,
                    reason=(f"за {step.date.year} отбор закрыт целиком: закачка без "
                            "добычи тратит OPEX без выручки, компенсировать нечего"),
                ))
                injectors[index] = InjectorControl(control.well, control.fluid, "SHUT",
                                                   control.mode, None, control.bhp)
        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result, decisions


def expand_deltas(steps: list[ScheduleStep] | tuple[ScheduleStep, ...]
                  ) -> list[ScheduleStep]:
    """Развернуть расписание-дельты в полные шаги: каждый шаг несёт весь фонд.

    В деке запись действует до следующей записи по той же скважине, поэтому шаг,
    где скважина не упомянута, означает «режим не менялся». Расписание Модели Z
    именно такое: 84…103 записи при 103 скважинах. Все ограничения объекта,
    группы и фонда считались **по записям шага** — то есть не видели скважину,
    которая работает с прошлого шага. Три «high» из ревизии 02.09 — это одна и та
    же ошибка в шести функциях: предел по закачке 800 м³/сут не срабатывал на трёх
    нагнетателях по 400, открытых в три разных месяца, и проверка соглашалась.

    Чинится не в шести местах, а здесь: на вход планировщика приходит полное
    расписание, и дальше вся арифметика по шагу верна сама собой. Развёрнутое
    расписание эквивалентно исходному по смыслу (то же состояние фонда в каждый
    месяц) и явнее его: режим каждой скважины виден в каждом шаге.
    """
    result: list[ScheduleStep] = []
    producers: dict[str, ProducerControl] = {}
    injectors: dict[str, InjectorControl] = {}
    for step in steps:
        for control in step.producers:
            producers[control.well] = control
            # запись добычи снимает запись закачки — и когда открывает скважину, и
            # когда закрывает: `WCONPROD … SHUT` в деке закрывает скважину целиком,
            # а не одну её роль. Без этого развёртка не идемпотентна и остановка
            # превращается в перевод под закачку (ревизия 02.09)
            injectors.pop(control.well, None)
        for control in step.injectors:
            injectors[control.well] = control
            producers.pop(control.well, None)
        result.append(ScheduleStep(
            step.date,
            tuple(producers[well] for well in sorted(producers)),
            tuple(injectors[well] for well in sorted(injectors)),
        ))
    return result


def fund_states(steps: list[ScheduleStep]) -> list[dict[str, tuple[bool, bool]]]:
    """Состояние фонда после каждого шага: `{скважина: (добывает, закачивает)}`.

    Состояние **накопительное**, и это не деталь реализации, а семантика деков:
    запись `WCONPROD` действует до следующей записи по той же скважине, а шаг, где
    скважина не упомянута, означает «режим не менялся». Расписание Модели Z
    именно такое — размер шага гуляет от 84 до 103 записей, а последний шаг
    (01.09.2025, конец расчёта) пуст вовсе.

    До 02.09 события считались сравнением **составов соседних шагов**, и из этого
    следовало два молчаливых дефекта: пустой финальный шаг выглядел как остановка
    всего фонда (103 «мероприятия» в 2025 году), а скважина, не упомянутая на шаге,
    считалась остановленной и снова пущенной. Найдено репетицией приёмки: проверка
    показала 91 мероприятие при квоте 30, а планировщик не отменил ни одного —
    два счётчика одной величины меряли разное.
    """
    states: list[dict[str, tuple[bool, bool]]] = []
    current: dict[str, tuple[bool, bool]] = {}
    for step in steps:
        for control in step.producers:
            # Запись добычи снимает закачку той же скважины: одновременно и то и
            # другое невозможно, и именно так выглядит возврат из ВНС. Запись
            # `SHUT` снимает **обе** роли: в деке она закрывает скважину, а не
            # только её добычу (ревизия 02.09 — иначе остановленная скважина
            # уезжала в файл нагнетательной).
            current[control.well] = ((True, False) if control.status == "OPEN"
                                     else (False, False))
        for control in step.injectors:
            current[control.well] = ((False, True) if control.status == "OPEN"
                                     else (False, False))
        states.append(dict(current))
    return states


def events_of(steps: list[ScheduleStep]) -> set[tuple[str, Any]]:
    """Мероприятия расписания: `{(скважина, дата шага)}`.

    Событие — изменение накопительного состояния скважины (см. `fund_states`), а
    не разница составов соседних шагов. Нужно, чтобы отличить мероприятие, которое
    предложил цикл, от порождённого срезкой самого планировщика: предел,
    доведённый до нуля, — это остановка, и отменять её квотой нельзя.
    """
    events: set[tuple[str, Any]] = set()
    states = fund_states(steps)
    for index, step in enumerate(steps):
        if index == 0:
            continue
        before, after = states[index - 1], states[index]
        for well, state in after.items():
            if _event_kind(before.get(well, (False, False)), state) is not None:
                events.add((well, step.date))
    return events


def _shut_below_minimum(steps: list[ScheduleStep],
                        minimum: float) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Остановить добывающие, чей режим ниже минимального рентабельного.

    Останавливаем, а не поднимаем до порога: поднятая уставка — обещание дебита,
    которого пласт может не дать. Нагнетательных правило не касается: у закачки
    своя экономика, и «минимальная рентабельная приёмистость» — не та же форма.

    **Остановка окончательна до конца горизонта.** Без этого скважина у порога
    дёргается каждый месяц: уставка колеблется вокруг минимума, и каждое
    пересечение стоит мероприятия — на реальных данных ревизия насчитала
    одиннадцать пусков и остановок одной скважины за год, то есть 11 млн руб. на
    ровном месте. Трактовка «дебит ниже рентабельного — скважина выводится»
    экономически осмысленна и не требует выдуманного коэффициента гистерезиса;
    если кейс потребует иного, это станет отдельным ключом.
    """
    decisions: list[Allocation] = []
    result: list[ScheduleStep] = []
    reported: set[str] = set()
    retired: set[str] = set()

    for step in steps:
        producers = []
        for control in step.producers:
            rate = control.liquid_rate or 0.0
            if control.status == "OPEN" and control.well in retired:
                producers.append(ProducerControl(control.well, "SHUT", control.mode,
                                                 None, control.bhp))
                continue
            if control.status != "OPEN" or rate >= minimum:
                producers.append(control)
                continue
            retired.add(control.well)
            if control.well not in reported:
                reported.add(control.well)
                decisions.append(Allocation(
                    well=control.well, resource="минимальный дебит",
                    before=rate, after=0.0,
                    reason=(f"режим {rate:,.4g} < минимального рентабельного "
                            f"{minimum:,.4g} м³/сут — скважина останавливается, "
                            "а не поднимается до порога"),
                ))
            producers.append(ProducerControl(control.well, "SHUT", control.mode,
                                             None, control.bhp))
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))
    return result, decisions


def _events_holding_the_fund(steps: list[ScheduleStep],
                             limits: Limits) -> set[tuple[str, Any]]:
    """Остановки, отмена которых нарушила бы уже соблюдённый предел.

    Квота отменяет мероприятие возвратом скважины в прежнее состояние, а отмена
    остановки возвращает и работу, и добычу: ломается и предел числа действующих
    скважин, и любой объёмный — годовой, горизонтный, суточный. Перепроверять
    после квоты некому, она идёт последней.

    Поэтому при **любом** активном пределе такого рода остановки защищаются
    целиком. Это осознанно грубо: точная проверка «нарушит ли именно эта отмена
    именно этот предел» стоит прохода по всему расписанию на каждое мероприятие,
    а результат тот же — квота уступает допустимости (Р-139). Уступка при этом
    печатается решением, а не подразумевается.
    """
    guarded = any(value is not None for value in (
        limits.max_active_wells,
        limits.max_field_liquid_m3_per_day, limits.max_field_injection_m3_per_day,
        limits.max_field_liquid_m3_per_year, limits.max_field_injection_m3_per_year,
        limits.max_field_liquid_m3_total, limits.max_field_injection_m3_total,
        limits.max_field_oil_t_per_day, limits.max_field_water_t_per_day,
        limits.min_producer_rate_m3_per_day,
    ))
    if not guarded:
        return set()
    # Защищаются мероприятия, отмена которых **увеличит** нагрузку: остановка и
    # перевод под закачку (обе прекращают добычу), конец закачки. Пуск и возврат
    # в добычу квота режет как прежде: их отмена нагрузку только снижает, и
    # пределы от неё не страдают.
    holding: set[tuple[str, Any]] = set()
    states = fund_states(steps)
    for index, step in enumerate(steps):
        if index == 0:
            continue
        before, after = states[index - 1], states[index]
        for well, (producing, injecting) in after.items():
            was_producing, was_injecting = before.get(well, (False, False))
            if (was_producing and not producing) or (was_injecting and not injecting):
                holding.add((well, step.date))
    return holding


def _hold_regimes(steps: list[ScheduleStep], months: int,
                  protected: set[tuple[str, Any]]) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Не менять режим скважины чаще, чем раз в `months` месяцев.

    Изменение, пришедшее раньше срока, отменяется — скважина остаётся на прежней
    уставке. Смена статуса (остановка, пуск, перевод) удержанию не подлежит: это
    мероприятие, и его ограничивает квота, а не длительность режима.
    """
    decisions: list[Allocation] = []
    result: list[ScheduleStep] = []
    # ключ — скважина **и роль**: перевод под закачку меняет роль, а не статус, и
    # обе записи при этом `OPEN`. С общим ключом скважина, переведённая под
    # закачку на 5 м³/сут, уезжала бы в файл нагнетательной на 100 — столько она
    # добывала жидкости месяцем раньше (ревизия 02.09).
    held_since: dict[tuple[str, str], int] = {}
    current: dict[tuple[str, str], float | None] = {}

    for step in steps:
        # календарные месяцы, а не номера шагов: шаг отчётности в кейсе задают
        # организаторы (ТЗ п. 4.3), и на квартальном расписании счёт по шагам
        # отменял бы смену через девять месяцев как «раньше срока» при пороге
        # шесть (ревизия 02.09). `max_events_per_month` рядом считает по календарю
        # — две ручки одного коммита не должны читать время по-разному.
        moment = step.date.year * 12 + step.date.month
        producers, injectors = [], []
        for control in step.producers:
            rate, kept = _hold_one((control.well, "producer"), control.liquid_rate,
                                   control.status, moment, months, held_since, current,
                                   (control.well, step.date) in protected)
            if kept and rate != control.liquid_rate:
                decisions.append(Allocation(
                    well=control.well, resource=f"длительность режима, {step.date:%Y-%m}",
                    before=control.liquid_rate or 0.0, after=rate or 0.0,
                    reason=(f"режим держится {months} мес.: смена раньше срока отменена"),
                ))
                control = ProducerControl(control.well, control.status, control.mode,
                                          rate, control.bhp)
            producers.append(control)
        for control in step.injectors:
            rate, kept = _hold_one((control.well, "injector"), control.rate,
                                   control.status, moment, months, held_since, current,
                                   (control.well, step.date) in protected)
            if kept and rate != control.rate:
                decisions.append(Allocation(
                    well=control.well, resource=f"длительность режима, {step.date:%Y-%m}",
                    before=control.rate or 0.0, after=rate or 0.0,
                    reason=(f"режим держится {months} мес.: смена раньше срока отменена"),
                ))
                control = InjectorControl(control.well, control.fluid, control.status,
                                          control.mode, rate, control.bhp)
            injectors.append(control)
        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result, decisions


def _hold_one(well: tuple[str, str], asked: float | None, status: str, moment: int,
              months: int, held_since: dict[tuple[str, str], int],
              current: dict[tuple[str, str], float | None],
              protected: bool) -> tuple[float | None, bool]:
    """Какую уставку оставить скважине и было ли удержание.

    `moment` — календарный месяц как число (`год × 12 + месяц`), а не номер шага:
    расписание кейса может идти кварталами.
    """
    if status != "OPEN":
        # смена статуса — мероприятие, а не режим: держать её нечем. Сброс
        # состояния обязателен: иначе первая уставка после возврата в работу
        # удерживалась бы на доостановочной
        current[well] = None
        held_since[well] = moment
        return asked, False
    if well not in current or current[well] is None:
        current[well] = asked
        held_since[well] = moment
        return asked, False
    if asked == current[well]:
        return asked, False
    if protected or moment - held_since[well] >= months:
        current[well] = asked
        held_since[well] = moment
        return asked, False
    return current[well], True


def _fit_events_to_quota(
    steps: list[ScheduleStep],
    *, per_month: int | None = None, per_year: int | None = None,
    protected: set[tuple[str, Any]] | None = None,
) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Квота платных мероприятий на календарный год.

    Методика оценивает мероприятия поштучно — 1,0 млн руб. за остановку и
    столько же за пуск, 5,0 млн за перевод под закачку, — поэтому «не более N
    мероприятий в год» это естественная форма ограничения по фонду, и
    организаторы назвали фонд одним из трёх видов ограничений кейса.

    **Мероприятие считается по расписанию, а не по выгрузке.** Эталонный
    расчётчик выводит события из отклика (`WLPR`, `WWIR`) и знает ещё смену ЭЦН
    и изменение режима; здесь считается то, чем управляет сдаваемый файл:
    перемена **накопительного состояния** скважины (`fund_states`). Смены ЭЦН
    среди них нет — насос выбирается по дебиту, а дебит это прогноз, а не наша
    уставка. Поэтому квота ограничивает **наши** решения, и число мероприятий у
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

    **Отмена мероприятия — это снятие записей скважины с шага**, а не подстановка
    записи предыдущего шага. В деке отсутствие записи означает «режим не менялся»,
    поэтому снятая запись оставляет скважину ровно в том состоянии, в каком она
    была, — и вместе с ней снимается изменение уставки, которое эту запись и
    несло. Прежняя редакция подставляла запись предыдущего шага и на реальном
    деке не срабатывала вовсе: там шаг содержит только изменившиеся скважины
    (найдено репетицией приёмки 01.09).

    `protected` — мероприятия, порождённые срезками самого планировщика. Их
    отмена вернула бы скважину в состояние, нарушающее жёсткое ограничение,
    поэтому они квоту расходуют, но не отменяются, а превышение печатается
    решением (Р-139).
    """
    decisions: list[Allocation] = []
    result: list[ScheduleStep] = []
    spent: dict[str, int] = {}
    state: dict[str, tuple[bool, bool]] = {}
    rates: dict[str, float] = {}
    #: О ком уже сказано, что мероприятие перенесено. Без этого одна
    #: скважина получала по решению на каждый месяц переноса: запись
    #: снимается с шага, но на развёрнутом расписании то же назначение
    #: стоит и на следующем, квота там снова полна — и так по кругу (С13).
    postponed: set[tuple[str, str]] = set()

    def buckets(moment) -> list[tuple[str, int]]:
        """Корзины квот, которые расходует мероприятие этого шага."""
        out: list[tuple[str, int]] = []
        if per_year is not None:
            out.append((f"год {moment.year}", per_year))
        if per_month is not None:
            out.append((f"месяц {moment:%Y-%m}", per_month))
        return out

    for index, step in enumerate(steps):
        step_buckets = buckets(step.date)
        after = dict(state)
        for control in step.producers:
            producing = control.status == "OPEN"
            after[control.well] = (producing, after.get(control.well, (False, False))[1]
                                   and not producing)
        for control in step.injectors:
            injecting = control.status == "OPEN"
            after[control.well] = (after.get(control.well, (False, False))[0]
                                   and not injecting, injecting)

        cancelled: set[str] = set()
        # порядок обхода — как в файле: сначала добывающие, потом нагнетательные
        wells = [c.well for c in step.producers]
        wells += [c.well for c in step.injectors if c.well not in wells]

        for well in wells:
            if index == 0:
                continue  # первому шагу не с чем сравнивать: истории до горизонта нет
            kind = _event_kind(state.get(well, (False, False)),
                               after.get(well, (False, False)))
            if kind is None:
                continue
            # мероприятие проходит, только если укладывается **во все** корзины:
            # месячная и годовая квоты считаются одним проходом, иначе отмена одной
            # переносит событие на другой месяц и ломает вторую (найдено репетицией
            # на реальных данных 02.09)
            full = [(name, limit) for name, limit in step_buckets
                    if spent.get(name, 0) >= limit]
            if not full:
                for name, _ in step_buckets:
                    spent[name] = spent.get(name, 0) + 1
                continue
            if (well, step.date) in (protected or set()):
                for name, _ in step_buckets:
                    spent[name] = spent.get(name, 0) + 1
                name, limit = full[0]
                decisions.append(Allocation(
                    well=well, resource=f"квота мероприятий, {name}",
                    before=0.0, after=0.0,
                    reason=(f"{kind} оставлен вопреки квоте {limit} на {name}: он "
                            f"порождён ограничением объекта, и его отмена нарушила бы "
                            f"это ограничение. Мероприятий уже {spent[name]}"),
                ))
                continue
            cancelled.add(well)
            asked = next((c for c in step.producers if c.well == well), None) \
                or next((c for c in step.injectors if c.well == well), None)
            name, limit = full[0]
            # «Отменён» было неправдой: запись снимается **с этого шага**, а на
            # развёрнутом расписании то же назначение стоит и на следующем —
            # мероприятие происходит позже, а не исчезает. Объяснение, которое
            # обещало «скважина остаётся как была», обещало то, чего не будет
            # (С13). Точную дату не называем: она зависит от того, когда
            # освободится корзина, и выдумывать её здесь нечем.
            if (well, name) in postponed:
                continue
            postponed.add((well, name))
            decisions.append(Allocation(
                well=well, resource=f"квота мероприятий, {name}",
                before=_rate_of(asked), after=rates.get(well, 0.0),
                reason=(f"{kind} перенесён с {step.date:%Y-%m}: квота мероприятий "
                        f"на {name} исчерпана (предел {limit}). Запись снята с "
                        f"этого шага, но на следующем то же назначение стоит "
                        f"снова — мероприятие произойдёт позже, а не исчезнет"),
            ))

        if cancelled:
            producers = tuple(c for c in step.producers if c.well not in cancelled)
            injectors = tuple(c for c in step.injectors if c.well not in cancelled)
            step = ScheduleStep(step.date, producers, injectors)
            after = dict(state)
            for control in producers:
                producing = control.status == "OPEN"
                after[control.well] = (producing,
                                       after.get(control.well, (False, False))[1]
                                       and not producing)
            for control in injectors:
                injecting = control.status == "OPEN"
                after[control.well] = (after.get(control.well, (False, False))[0]
                                       and not injecting, injecting)

        for control in step.producers:
            rates[control.well] = control.liquid_rate or 0.0
        for control in step.injectors:
            rates[control.well] = control.rate or 0.0

        state = after
        result.append(step)

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
        stamp = str(row.get("DATA", ""))[:7]
        if len(stamp) == 7:
            stamps.add(stamp)
        # То же правило, что в коридоре компенсации и в аудите: строка с любым
        # отрицательным приращением выброшена методикой целиком.
        if not counted_by_methodology(row):
            continue
        oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]
        if liquid_t > 0 and 0 <= oil_t <= liquid_t:
            produced += (liquid_t - oil_t) * 1000.0 / fluids.water_density
        planned += max(row["WWIT_Diff"], 0.0)

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
    carried: tuple[float, float] | None = None,
    well_caps: dict[str, float] | None = None,
    default_cap: float | None = None,
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

    **Коридор подбирается по годам накопительно, и это не уточнение, а условие
    допустимости.** До 03.09 здесь считался один множитель на весь горизонт:
    суммы `base` и `planned` брались за все годы сразу. Проверяется же коридор
    иначе — `aios.balance.compensation_problems` ведёт бегущие накопленные суммы
    и сверяет отношение **на конец каждого года**. Числа расходились не в
    последнем знаке: закачка 1 800 м³ при отборе 1 000 в первый год и 300 при
    1 000 во второй дают в сумме 1,05 — планировщик молчал, — а на конец первого
    года 180 % против потолка 115 %. То есть расписание проходило планировщик и
    оставалось недопустимым, причём именно тем классом отказа, который не виден
    до полного расчёта (ревизия 03.09, 2.1 #2).

    Поэтому здесь тот же счёт, что в аудите: год за годом, накопленным итогом.
    Множитель у каждого года свой и подбирается так, чтобы накопленное отношение
    на конец этого года легло в коридор; перебор предыдущего года не забывается,
    а вычитается из того, что можно закачать в следующем.

    Предел источника остаётся годовым: мощность — расход, накопить его нельзя,
    поэтому он ограничивает **свой** год, а не сумму (Р-162 про тот же счёт по
    месяцам в аудите).
    """
    years: dict[int, dict[str, float]] = {}
    stamps_by_year: dict[int, set[str]] = {}
    strays = 0
    for row in forecast:
        stamp = str(row.get("DATA", ""))[:7]
        if len(stamp) != 7 or not stamp[:4].isdigit():
            strays += 1
            continue
        year = int(stamp[:4])
        bucket = years.setdefault(year, {"base": 0.0, "produced": 0.0, "planned": 0.0})
        stamps_by_year.setdefault(year, set()).add(stamp)
        # Строка с любым отрицательным приращением методикой выброшена целиком, и
        # аудит (`balance.by_year`) её не видит. Здесь она в отборе не считалась,
        # а закачка из неё считалась — планировщик резал закачку там, где аудит
        # нарушения не находит (замер: срезка в 4 раза при компенсации 1,07 и
        # потолке 1,3). У суррогата таких строк бывают тысячи.
        if not counted_by_methodology(row):
            continue
        oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]
        if liquid_t > 0 and 0 <= oil_t <= liquid_t:
            water_m3 = (liquid_t - oil_t) * 1000.0 / fluids.water_density
            bucket["produced"] += water_m3
            bucket["base"] += water_m3
            if rule == "compensation":
                bucket["base"] += oil_t * 1000.0 / fluids.oil_density
        bucket["planned"] += max(row["WWIT_Diff"], 0.0)

    if strays:
        # Тихо подставить 365 суток или свалить всё в один год нельзя: коридор
        # проверяется по календарным годам, и строка без даты означает, что счёт
        # не с чем сверить. Лимит источника тем более суточный.
        raise ValueError(
            "коридору баланса воды нужна колонка DATA в каждой строке прогноза: "
            f"без даты строку не отнести к году (строк без даты {strays})"
        )
    if not years:
        # Пустой прогноз — это «сверять нечего», а не «данные испорчены». Так
        # бывает на расписании без управляемых шагов: `NeuralSurrogate.predict`
        # возвращает пустой список, и до 03.09 функция здесь тихо выходила
        # (найдено критиком).
        return steps, []

    planned_total = sum(one["planned"] for one in years.values())
    base_total = sum(one["base"] for one in years.values())
    if planned_total <= 0 or base_total <= 0:
        return steps, []

    factors: dict[int, float] = {}
    reasons: dict[int, str] = {}
    # Накопленное до первого года горизонта: отобранное и закачанное за историю,
    # которой в прогнозе нет. Пусто — счёт начинается с нуля, как было до 04.09.
    base_c, injected_c = carried if carried is not None else (0.0, 0.0)
    for year in sorted(years):
        bucket = years[year]
        base_c += bucket["base"]
        planned_y = bucket["planned"]
        if planned_y <= 0:
            continue

        factor = 1.0
        reason = ""

        if base_c <= 0:
            # Закачка есть, а возмещать нечего: отбора не было ни в этом году, ни
            # раньше. Такое бывает на новом элементе заводнения — нагнетательная
            # пущена раньше добывающей. Аудит для этого случая говорит
            # «возмещать нечего» (`balance.compensation_problems`), и правильный
            # множитель здесь ноль, а не деление на ноль (найдено критиком 03.09).
            factors[year] = 0.0
            reasons[year] = (
                f"{year}: закачка {planned_y:,.0f} м³ при нулевом накопленном отборе — "
                "возмещать нечего"
            )
            continue

        # верхняя граница: накопленное на конец года не выше базы × ceiling
        room = base_c * ceiling - injected_c
        # Строгое сравнение и нестрогое здесь неразличимы: при `planned_y == room`
        # множитель выходит ровно `room / planned_y = 1.0`, а единичный множитель
        # ниже не записывается. Делить на ноль тоже не на чем — год без закачки
        # отсеян выше (`planned_y <= 0`). Мутационный прогон помечает подмену
        # эквивалентной, и это верный вердикт (Р-229); наблюдаемый итог —
        # «накопленное ровно на потолке не режется» — закреплён
        # `test_a_year_exactly_at_the_corridor_ceiling_is_not_cut`.
        if planned_y > room:
            factor = max(room / planned_y, 0.0)
            reason = (
                f"{year}: накопленная закачка {injected_c + planned_y:,.0f} м³ даёт "
                f"компенсацию {(injected_c + planned_y) / base_c:.2f} от отобранных "
                f"{base_c:,.0f} м³ при потолке {ceiling:.2f}"
                if rule == "compensation" else
                f"{year}: накопленная закачка {injected_c + planned_y:,.0f} м³ превышает "
                f"добытую воду {base_c:,.0f} м³; источника сверх неё нет"
            )

        # физика источника: своя вода этого года плюс мощность за его сутки.
        # Воды, которой нет, никакая норма не создаёт — этот предел бьёт и
        # верхнюю границу, и добор снизу.
        if source_m3_per_day is not None:
            days = balance.days_of(stamps_by_year[year])
            by_source = bucket["produced"] + source_m3_per_day * days
            if planned_y * factor > by_source:
                factor = max(by_source / planned_y, 0.0)
                reason = (
                    f"{year}: закачке нужно "
                    f"{(planned_y - bucket['produced']) / days:,.0f} м³/сут воды со "
                    f"стороны при мощности источника {source_m3_per_day:,.10g} м³/сут"
                )

        # нижняя граница коридора: добираем, но не выше того, что есть
        if floor is not None:
            required = base_c * floor - injected_c
            if planned_y * factor < required:
                wanted = required / planned_y
                if source_m3_per_day is not None:
                    days = balance.days_of(stamps_by_year[year])
                    ceiling_by_source = (
                        (bucket["produced"] + source_m3_per_day * days) / planned_y)
                    wanted = min(wanted, ceiling_by_source)
                # При равенстве присваивание тождественно, поэтому строгое
                # сравнение и нестрогое дают один результат при любых данных —
                # мутация здесь эквивалентна по определению, а не по замеру
                # (Р-229).
                if wanted > factor:
                    factor = wanted
                    reason = (
                        f"{year}: накопленная закачка {injected_c + planned_y:,.0f} м³ "
                        f"даёт компенсацию {(injected_c + planned_y) / base_c:.2f} от "
                        f"базы правила {base_c:,.0f} м³ — ниже границы коридора "
                        f"{floor:.2f}"
                    )

        injected_c += planned_y * factor
        # Единичный множитель не записывается — и это вторая из двух проверок:
        # `_scale_injection_by_year` пропускает `factor == 1.0` сама. Отсюда
        # мутационный прогон помечает условие эквивалентным (Р-225): убрать его
        # наблюдаемого поведения не изменит. Оставлено намеренно — пустой
        # `factors` даёт ранний выход без обхода расписания, а наблюдаемый итог
        # («год внутри коридора не порождает решения») закреплён
        # `test_a_year_inside_the_corridor_produces_no_decision`.
        if abs(factor - 1.0) > 1e-12:
            factors[year] = factor
            reasons[year] = reason

    if not factors:
        return steps, []
    return _scale_injection_by_year(steps, factors, reasons,
                                    well_caps=well_caps, default_cap=default_cap)

def _injecting(control: InjectorControl, rate: float) -> InjectorControl:
    """Запись закачки с новой приёмистостью; ноль закрывает скважину.

    Открытая нагнетательная с приёмистостью 0 — недопустимая запись: `WCONINJE`
    в режиме `RATE` требует уставку, и `schedule.validate` отвергает такую строку
    («открытая скважина не может закачивать ноль»). Множитель ноль появляется у
    коридора законно — например, в год, когда возмещать нечего (Р-178), — и
    оставлять после него открытую скважину значило бы собирать расписание,
    которое сами же браковали бы на выходе (Р-189).
    """
    if rate > 0:
        return InjectorControl(control.well, control.fluid, control.status,
                               control.mode, rate, control.bhp)
    return InjectorControl(control.well, control.fluid, "SHUT", control.mode,
                           None, control.bhp)


def _scale_injection_by_year(steps: list[ScheduleStep],
                             factors: dict[int, float],
                             reasons: dict[int, str],
                             well_caps: dict[str, float] | None = None,
                             default_cap: float | None = None,
                             ) -> tuple[list[ScheduleStep], list[Allocation]]:
    """Изменить закачку каждого года своим множителем.

    Отличие от `_scale_injection` — не в реализации, а в том, что множитель у
    коридора баланса воды **годовой**: накопленное отношение проверяется на конец
    каждого года, и один множитель на горизонт этому условию не отвечает (Р-170).
    Решение по каждому году печатается один раз на скважину: иначе на девятнадцати
    годах отчёт распухает в разы и перестаёт читаться. Имя ресурса остаётся
    «закачка» — год назван в обосновании, а по имени ресурса решения группируют
    и репетиция приёмки, и отчёт.
    """
    decisions: list[Allocation] = []
    seen: set[tuple[int, str]] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        factor = factors.get(step.date.year, 1.0)
        if factor == 1.0:
            result.append(step)
            continue
        injectors = []
        for control in step.injectors:
            if control.status != "OPEN" or control.rate is None:
                injectors.append(control)
                continue
            reduced = control.rate * factor
            # Добор снизу не имеет права поднять уставку выше потолка, который
            # объявил кейс. Коридор идёт последним, поэтому пересечь его после
            # добора уже некому: закачка 100 при отборе 1 000 и границе 0,85
            # давала ×8,5 — расписание просило у нагнетателя того, чего кейс ему
            # прямо запретил (С5). Своей физики приёмистости у нас нет и
            # выдумывать её нельзя; ограничиваем тем числом, которое дано.
            cap = (well_caps or {}).get(control.well, default_cap)
            if factor > 1.0 and cap is not None and reduced > cap:
                reduced = max(cap, min(control.rate, cap))
            key = (step.date.year, control.well)
            if key not in seen:
                seen.add(key)
                decisions.append(Allocation(
                    well=control.well, resource="закачка",
                    before=control.rate, after=reduced,
                    reason=reasons.get(step.date.year, ""),
                ))
            injectors.append(_injecting(control, reduced))
        result.append(ScheduleStep(step.date, step.producers, tuple(injectors)))

    return result, decisions


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
            injectors.append(_injecting(control, reduced))
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

    **Срезанная в ноль скважина остаётся закрытой до конца горизонта.** Мощность
    объекта колеблется от месяца к месяцу, и без этого правила скважина у края
    квоты открывается и закрывается каждый месяц: остановка и пуск — **два
    отдельных** платных события по 1,0 млн руб. каждое, плюс скважина то входит,
    то выходит из действующего фонда. На синтетическом примере (предел 100,
    сосед колеблется 90/100) выходило пять мероприятий за шесть месяцев — то
    есть 5 млн руб. за право поработать в трёх месяцах на 20 м³/сут (ревизия
    03.09, 2.1 #4). Правило то же, что у минимального рентабельного дебита
    (`_shut_below_minimum`), и по той же причине: коэффициент гистерезиса
    пришлось бы выдумывать, а «выведена из-под квоты» — экономически осмысленное
    состояние.

    Пропорциональные срезки (`_fit_phase_to_capacity`, коридор воды) этого
    правила не требуют: они уменьшают уставку, а не закрывают скважину, и
    события не порождают вовсе.
    """
    value = _margin_per_liquid(forecast, norms, fluids)
    decisions: list[Allocation] = []
    result: list[ScheduleStep] = []
    reported: set[str] = set()
    retired: set[str] = set()

    for step in steps:
        held = [
            ProducerControl(control.well, "SHUT", control.mode, None, control.bhp)
            if control.well in retired and control.status == "OPEN" else control
            for control in step.producers
        ]
        working = [
            control for control in held
            if control.status == "OPEN" and control.liquid_rate
        ]
        total = sum(control.liquid_rate or 0.0 for control in working)
        if total <= capacity:
            result.append(ScheduleStep(step.date, tuple(held), step.injectors))
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
        for control in held:
            if control.well not in cuts:
                producers.append(control)
                continue
            new_rate = cuts[control.well]
            if new_rate <= 0:
                retired.add(control.well)
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
