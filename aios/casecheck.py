"""Проверка расписания на соблюдение условий кейса — по факту, а не по процессу.

`aios.planner` **приводит** расписание к ограничениям. Этот модуль отвечает на
другой вопрос: соблюдено ли каждое объявленное условие в том расписании, которое
получилось. Разница важна по двум причинам.

**Первая — мёртвые ручки.** До 30.08 `max_events_per_year` разбирался
конфигурацией и не читался ни одной веткой планировщика: число подставлялось,
прогон «принимал» его и молча нарушал (Р-113). Проверка процесса такого не
находит — её проходит и планировщик, который ничего не сделал. Проверка
результата находит: условие объявлено, факт выше предела, решений нет.

**Вторая — приёмка кейса.** Протокол первых суток
(`docs/validation-case-checklist.md`) требует по каждому условию видеть
«фактическое значение, предел, период и скважину, а не „нарушений нет“». Здесь
это и считается — до планировщика и после него, одними и теми же формулами.

Чего модуль не делает: не пересчитывает пласт. Фазы (нефть, вода) берутся из
прогноза и масштабируются вместе с уставкой по жидкости — тем же допущением
«при неизменной обводнённости срез жидкости на k срезает фазу на k», которым
пользуется сам планировщик. Настоящая проверка фазы возможна только по факту
полного расчёта.
"""

from __future__ import annotations

from calendar import monthrange
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aios.deck import ScheduleStep
from aios.margin import counted_by_methodology
from aios.planner import Limits, events_of, expand_deltas, fund_states


@dataclass(frozen=True)
class LimitCheck:
    """Одно условие кейса и то, как расписание с ним обошлось."""

    key: str
    """Ключ конфигурации, которым условие выражено."""

    limit: float
    """Объявленный предел в единицах ключа."""

    value: float
    """Фактическое значение в расписании: максимум по периодам."""

    where: str
    """Где достигнут максимум: месяц, год, скважина или группа."""

    unit: str

    @property
    def satisfied(self) -> bool:
        # допуск на счёт с плавающей точкой: срезка в точности до предела даёт
        # значение, отличающееся в последнем знаке, и без допуска соблюдённое
        # условие показывалось бы нарушенным
        return self.value <= self.limit * (1 + 1e-9)

    def __str__(self) -> str:
        mark = "соблюдено" if self.satisfied else "НАРУШЕНО"
        return (f"{self.key}: {self.value:,.2f} против предела {self.limit:,.2f} {self.unit} "
                f"({self.where}) — {mark}")


def check(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
          forecast: list[dict[str, Any]],
          limits: Limits) -> list[LimitCheck]:
    """Проверить расписание по всем объявленным ограничениям.

    Возвращает по одной записи на объявленный ключ — в том числе на соблюдённый:
    «условие проверено и держится» такой же результат приёмки, как нарушение, и
    именно его требует протокол.

    **Шаги разворачиваются в состояние фонда, а не читаются как записи.** В деке
    запись действует до следующей записи по той же скважине: шаг, где скважина не
    упомянута, означает «режим не менялся», а не «скважина не работает». Пока
    развёртки здесь не было, объектные пределы считались по составу шага —
    расписание, где 2030-01 назначает двум скважинам по 200 м³/сут, а 2030-02
    третьей ещё 200, давало «400 при пределе 500 — соблюдено» вместо настоящих
    600, и годовой объём занижался на 38 %. `planner.enforce` разворачивает свой
    вход с 02.09 (там же в докстринге: «три high из ревизии — одна и та же ошибка
    в шести функциях»), а приёмка кейса — нет; внутри этого модуля `_active_wells`
    уже считал по фонду и потому в том же вызове честно видел три скважины.
    """
    steps = expand_deltas(list(steps))
    checks: list[LimitCheck] = []

    if limits.max_field_liquid_m3_per_day is not None:
        checks.append(_field_rate(steps, limits.max_field_liquid_m3_per_day,
                                  key="max_field_liquid_m3_per_day", producers=True,
                                  average=limits.field_limits_are_annual_average))
    if limits.max_field_injection_m3_per_day is not None:
        checks.append(_field_rate(steps, limits.max_field_injection_m3_per_day,
                                  key="max_field_injection_m3_per_day", producers=False,
                                  average=limits.field_limits_are_annual_average))
    if limits.max_well_liquid_m3_per_day is not None:
        checks.append(_well_rate(steps, limits.max_well_liquid_m3_per_day,
                                 addressed=frozenset(limits.well_caps_m3_per_day or {}),
                                 key="max_well_liquid_m3_per_day", producers=True))
    if limits.max_well_injection_m3_per_day is not None:
        checks.append(_well_rate(steps, limits.max_well_injection_m3_per_day,
                                 key="max_well_injection_m3_per_day", producers=False))
    if limits.max_field_liquid_m3_per_year is not None:
        checks.append(_annual_volume(steps, limits.max_field_liquid_m3_per_year,
                                     key="max_field_liquid_m3_per_year", producers=True))
    if limits.max_field_injection_m3_per_year is not None:
        checks.append(_annual_volume(steps, limits.max_field_injection_m3_per_year,
                                     key="max_field_injection_m3_per_year", producers=False))
    if limits.max_field_liquid_m3_total is not None:
        checks.append(_horizon_volume(steps, limits.max_field_liquid_m3_total,
                                      key="max_field_liquid_m3_total", producers=True))
    if limits.max_field_injection_m3_total is not None:
        checks.append(_horizon_volume(steps, limits.max_field_injection_m3_total,
                                      key="max_field_injection_m3_total", producers=False))
    if limits.max_field_oil_t_per_day is not None:
        checks.append(_phase(steps, forecast, limits.max_field_oil_t_per_day,
                             key="max_field_oil_t_per_day", phase="нефть"))
    if limits.max_field_water_t_per_day is not None:
        checks.append(_phase(steps, forecast, limits.max_field_water_t_per_day,
                             key="max_field_water_t_per_day", phase="вода"))
    if limits.max_events_per_year is not None:
        checks.append(_events_per_period(steps, limits.max_events_per_year, period="year"))
    if limits.max_events_per_month is not None:
        checks.append(_events_per_period(steps, limits.max_events_per_month, period="month"))
    if limits.min_producer_rate_m3_per_day is not None:
        checks.append(_minimum_rate(steps, limits.min_producer_rate_m3_per_day))
    if limits.max_active_wells is not None:
        checks.append(_active_wells(steps, limits.max_active_wells))
    if limits.forced_shut_wells:
        checks.append(_forced_shut(steps, limits.forced_shut_wells))
    if limits.forced_shut_windows:
        checks.append(_forced_windows(steps, limits.forced_shut_windows))
    if limits.well_caps_m3_per_day:
        checks.append(_well_caps(steps, limits.well_caps_m3_per_day))
    for group, cap in sorted((limits.group_liquid_caps_m3_per_day or {}).items()):
        checks.append(_group_rate(steps, (limits.well_groups or {}).get(group, frozenset()),
                                  cap, key=f"group_liquid_caps_m3_per_day[{group}]",
                                  producers=True))
    for group, cap in sorted((limits.group_injection_caps_m3_per_day or {}).items()):
        checks.append(_group_rate(steps, (limits.well_groups or {}).get(group, frozenset()),
                                  cap, key=f"group_injection_caps_m3_per_day[{group}]",
                                  producers=False))
    return checks


def violations(checks: list[LimitCheck]) -> list[LimitCheck]:
    """Только нарушенные условия — то, из-за чего расписание нельзя отдавать."""
    return [one for one in checks if not one.satisfied]


def _rates(step: ScheduleStep, *, producers: bool) -> dict[str, float]:
    """Уставки открытых скважин одного шага: `{скважина: м³/сут}`."""
    if producers:
        return {control.well: control.liquid_rate or 0.0 for control in step.producers
                if control.status == "OPEN"}
    return {control.well: control.rate or 0.0 for control in step.injectors
            if control.status == "OPEN"}


def _stamp(step: ScheduleStep) -> str:
    return f"{step.date.year:04d}-{step.date.month:02d}"


def _field_rate(steps: list[ScheduleStep], limit: float, *, key: str,
                producers: bool, average: bool) -> LimitCheck:
    """Суточный предел объекта: пик месяца или среднее за календарный год."""
    if not average:
        best, where = 0.0, "—"
        for step in steps:
            total = sum(_rates(step, producers=producers).values())
            if total > best:
                best, where = total, _stamp(step)
        return LimitCheck(key=key, limit=limit, value=best, where=where, unit="м³/сут")

    volume: dict[int, float] = {}
    days: dict[int, int] = {}
    for step, span in zip(steps, _spans(steps), strict=True):
        volume[step.date.year] = volume.get(step.date.year, 0.0) + \
            sum(_rates(step, producers=producers).values()) * span
        days[step.date.year] = days.get(step.date.year, 0) + span
    best, where = 0.0, "—"
    for year, total in volume.items():
        rate = total / days[year]
        if rate > best:
            best, where = rate, f"{year} (среднее за год)"
    return LimitCheck(key=key, limit=limit, value=best, where=where, unit="м³/сут")


def _well_rate(steps: list[ScheduleStep], limit: float, *, key: str,
               producers: bool, addressed: frozenset[str] = frozenset()) -> LimitCheck:
    """Потолок по типу — для скважин, у которых нет своего.

    `addressed` — скважины, названные в `well_caps_m3_per_day`. Проект решил
    (`docs/decisions.md`, «адресный список сильнее потолка по типу»), что у
    скважины бывает свой паспорт насоса и он побеждает; планировщик так и
    делает — `cap_for` возвращает адресное значение, даже когда оно выше. А
    приёмка применяла потолок по типу ко всем без исключения, и кейс, назвавший
    и общий паспорт, и адресное исключение над ним, объявлял недопустимым любого
    кандидата, включая бездействие (фаззер ревизии, 4 попадания из 400).

    Названные скважины проверяет `_well_caps` — своим, адресным числом.
    """
    best, where = 0.0, "—"
    for step in steps:
        for well, rate in _rates(step, producers=producers).items():
            if well in addressed:
                continue
            if rate > best:
                best, where = rate, f"скв. {well}, {_stamp(step)}"
    return LimitCheck(key=key, limit=limit, value=best, where=where, unit="м³/сут")


def _group_rate(steps: list[ScheduleStep], members: frozenset[str], limit: float,
                *, key: str, producers: bool) -> LimitCheck:
    best, where = 0.0, "—"
    for step in steps:
        total = sum(rate for well, rate in _rates(step, producers=producers).items()
                    if well in members)
        if total > best:
            best, where = total, _stamp(step)
    return LimitCheck(key=key, limit=limit, value=best, where=where, unit="м³/сут")


def _spans(steps: list[ScheduleStep]) -> list[int]:
    """Сколько суток действует каждый шаг: до следующего, а не до конца месяца.

    Уставка в деке держится до следующей записи. Считать шаг равным его
    календарному месяцу верно ровно тогда, когда шаг месячный, — а `--horizon`
    и чужой дек бывают квартальными и годовыми. На квартальном расписании при
    уставке 1 000 м³/сут годовой объём выходил 123 000 м³ вместо 365 000, и
    предел 200 000 объявлялся соблюдённым при превышении почти вдвое (Н-24).

    Последний шаг длится до конца своего месяца: чем он кончится, расписание не
    говорит, и месяц — самое короткое честное допущение.
    """
    spans: list[int] = []
    for index, step in enumerate(steps):
        if index + 1 < len(steps):
            spans.append(max((steps[index + 1].date - step.date).days, 0))
        else:
            spans.append(monthrange(step.date.year, step.date.month)[1]
                         - step.date.day + 1)
    return spans


def _annual_volume(steps: list[ScheduleStep], limit: float, *, key: str,
                   producers: bool) -> LimitCheck:
    volume: dict[int, float] = {}
    for step, span in zip(steps, _spans(steps), strict=True):
        volume[step.date.year] = volume.get(step.date.year, 0.0) + \
            sum(_rates(step, producers=producers).values()) * span
    best, where = 0.0, "—"
    for year, total in volume.items():
        if total > best:
            best, where = total, str(year)
    return LimitCheck(key=key, limit=limit, value=best, where=where, unit="м³/год")


def _horizon_volume(steps: list[ScheduleStep], limit: float, *, key: str,
                    producers: bool) -> LimitCheck:
    """Объём за весь горизонт — одна корзина вместо годовых."""
    total = 0.0
    for step, span in zip(steps, _spans(steps), strict=True):
        total += sum(_rates(step, producers=producers).values()) * span
    span_from = f"{steps[0].date:%Y-%m}" if steps else "—"
    span_to = f"{steps[-1].date:%Y-%m}" if steps else "—"
    return LimitCheck(key=key, limit=limit, value=total,
                      where=f"{span_from} … {span_to}", unit="м³ за горизонт")


def _phase(steps: list[ScheduleStep], forecast: list[dict[str, Any]], limit: float,
           *, key: str, phase: str) -> LimitCheck:
    """Фаза по прогнозу, масштабированная изменением уставки по жидкости.

    Прогноз считался для другого расписания — того, что было до срезки. Пересчёта
    пласта у нас нет, поэтому фаза каждой скважины умножается на отношение
    «уставка после / достигнутый в прогнозе дебит», и это то же допущение, с
    которым фазу режет планировщик.

    **Отношение ограничено единицей, и это отдельное решение.** Уставка — потолок,
    а не обещание пласта: опустив её ниже достигнутого, отклик гарантированно
    срежешь, а подняв выше — не получишь ничего, кроме надежды. Считать иначе
    значило бы повторить ошибку, за которую в этом проекте забракован форсаж на
    суррогате: «выгоден всегда, обоснован никогда». Найдено репетицией приёмки
    01.09: без ограничения проверка объявляла нарушенной фазу, которую планировщик
    уже уложил в предел, — просто потому, что исторические уставки Модели Z выше
    достигнутых дебитов в 1,47 раза.

    Скважина, которой в прогнозе нет, в проверку не входит: выдумывать ей отклик
    нельзя.
    """
    planned: dict[str, dict[str, float]] = {}
    for step in steps:
        planned[_stamp(step)] = _rates(step, producers=True)

    totals: dict[str, float] = {}
    for row in forecast:
        if not counted_by_methodology(row):
            continue
        stamp = str(row["DATA"])[:7]
        rates = planned.get(stamp)
        if rates is None:
            continue
        oil_t = max(row["WOMT_Diff"], 0.0)
        liquid_t = max(row["WLPT_Diff"], 0.0)
        value = oil_t if phase == "нефть" else max(liquid_t - oil_t, 0.0)
        reference = float(row["WLPR"]) or 0.0
        share = min(1.0, rates.get(str(row["well"]), 0.0) / reference) if reference > 0 else 0.0
        totals[stamp] = totals.get(stamp, 0.0) + value * share

    best, where = 0.0, "—"
    for stamp, tonnes in totals.items():
        year, month = int(stamp[:4]), int(stamp[5:7])
        rate = tonnes / monthrange(year, month)[1]
        if rate > best:
            best, where = rate, stamp
    return LimitCheck(key=key, limit=limit, value=best, where=where, unit="т/сут")


def _minimum_rate(steps: list[ScheduleStep], minimum: float) -> LimitCheck:
    """Наименьший режим работающей добывающей против минимального рентабельного.

    Единственная проверка «снизу»: у всех остальных нарушение — превышение, здесь
    наоборот. Поэтому значение и предел переставлены местами так, чтобы `satisfied`
    читалось тем же правилом «значение ≤ предел»: сравнивается недобор до порога.
    """
    worst, where = 0.0, "—"
    for step in steps:
        # открытая запись с нулевым (или пустым) дебитом тоже нарушает минимум:
        # такие производят объёмные срезки, и до 02.09 проверка их пропускала —
        # ровно то состояние, которое создаёт планировщик, оставалось невидимым
        for well, rate in _rates(step, producers=True).items():
            gap = minimum - rate
            if gap > worst:
                worst, where = gap, f"скв. {well}, {_stamp(step)} ({rate:,.4g} м³/сут)"
    return LimitCheck(key="min_producer_rate_m3_per_day", limit=0.0, value=worst,
                      where=where, unit="м³/сут недобора")


def _events_per_period(steps: list[ScheduleStep], limit: int, *, period: str) -> LimitCheck:
    """Платных мероприятий за календарный год — против квоты.

    Единственный ключ, который проверка до 02.09 не знала вовсе, — при том что
    именно он назван в трёх докстрингах как причина существования этого модуля.
    После правки Р-139 квота может быть превышена законно (мероприятие, порождённое
    жёстким ограничением, не отменяется), и превышение обязано быть **видно**, а
    не подразумеваться.

    Мероприятия считаются тем же кодом, что и в планировщике: два счётчика одной
    величины разъезжаются молча.
    """
    by_period: dict[str, int] = {}
    for _, moment in events_of(steps):
        key = str(moment.year) if period == "year" else f"{moment:%Y-%m}"
        by_period[key] = by_period.get(key, 0) + 1
    best, where = 0, "—"
    for stamp, count in sorted(by_period.items()):
        if count > best:
            best, where = count, stamp
    name = "max_events_per_year" if period == "year" else "max_events_per_month"
    return LimitCheck(key=name, limit=float(limit), value=float(best),
                      where=where, unit="мероприятий")


def announced_for(check: LimitCheck, decisions: Iterable[Any]) -> bool:
    """Объявил ли планировщик уступку **в том же периоде**, где нарушение.

    Совпадения по виду условия мало: годовая и месячная квоты считаются по
    разным периодам, и уступка в одном году ничего не говорит о другом. Пока
    цикл спрашивал только «была ли уступка вообще», одна объявленная уступка
    гасила квоту целиком: 22 мероприятия против предела 1 в 2019 проходили
    fail-closed, если уступка объявлена хоть в каком-нибудь году (Н-19).

    Правило одно на цикл и на репетицию: два ответа на один вопрос разошлись бы
    молча, а здесь цена расхождения — пропущенное нарушение условия кейса.
    """
    period = "год" if check.key.endswith("_per_year") else "месяц"
    needle = f"на {period} {check.where}"
    return any("вопреки квоте" in decision.reason and needle in decision.reason
               for decision in decisions)


def _active_wells(steps: list[ScheduleStep], limit: int) -> LimitCheck:
    """Действующий фонд по накопительному состоянию, а не по составу шага.

    В деке шаг несёт только изменившиеся скважины (`planner.fund_states`), поэтому
    считать открытые записи шага значит считать не фонд, а список изменений. На
    Модели Z разница невелика — шаги плотные, — но именно «невелика» и опасно:
    ограничение выглядело бы соблюдённым потому, что половина фонда в шаг не попала.
    """
    best, where = 0, "—"
    for step, state in zip(steps, fund_states(steps), strict=True):
        active = sum(1 for producing, injecting in state.values() if producing or injecting)
        if active > best:
            best, where = active, _stamp(step)
    return LimitCheck(key="max_active_wells", limit=float(limit), value=float(best),
                      where=where, unit="скважин")


def _open_wells(state: dict[str, tuple[bool, bool]]) -> set[str]:
    return {well for well, (producing, injecting) in state.items() if producing or injecting}


def _forced_shut(steps: list[ScheduleStep], wells: frozenset[str]) -> LimitCheck:
    """Сколько раз обязательно остановленная скважина осталась в работе.

    По накопительному состоянию: скважина, не упомянутая на шаге, продолжает
    работать, и до 02.09 это считалось соблюдением.
    """
    breaches, where = 0, "—"
    # `strict=True` во всех трёх обходах защитный: `fund_states` отдаёт ровно
    # по состоянию на шаг, поэтому расхождение недостижимо — мутационный прогон
    # помечает подмену эквивалентной, и это верный вердикт (Р-240). Оставлено
    # как страховка на случай, если разметка фонда начнёт пропускать шаги.
    for step, state in zip(steps, fund_states(steps), strict=True):
        left = sorted(wells & _open_wells(state))
        if left:
            breaches += len(left)
            if where == "—":
                where = f"скв. {left[0]}, {_stamp(step)}"
    return LimitCheck(key="forced_shut_wells", limit=0.0, value=float(breaches),
                      where=where, unit="месяцев в работе")


def _forced_windows(steps: list[ScheduleStep], windows) -> LimitCheck:
    breaches, where = 0, "—"
    for step, state in zip(steps, fund_states(steps), strict=True):
        working = _open_wells(state)
        for window in windows:
            if window.covers(step.date) and window.well in working:
                breaches += 1
                if where == "—":
                    where = f"скв. {window.well}, {_stamp(step)}"
    return LimitCheck(key="forced_shut_windows", limit=0.0, value=float(breaches),
                      where=where, unit="месяцев в работе")


def _well_caps(steps: list[ScheduleStep], caps: dict[str, float]) -> LimitCheck:
    """Адресные потолки: наибольшее превышение **своего** потолка, в долях от него.

    Сравнивать абсолютные ставки нельзя: скважина на 900 при потолке 1000 не
    нарушает ничего, а скважина на 100 при потолке 10 нарушает вдесятеро — и до
    02.09 первая закрывала вторую собой, а `violations()` возвращал пусто
    (найдено критиком 01.09). Поэтому ищется максимум отношения, а в отчёт идёт
    та пара «факт и его потолок», на которой отношение достигнуто.
    """
    worst_ratio, worst, where, limit_of_worst = 0.0, 0.0, "—", 0.0
    for step in steps:
        rates = _rates(step, producers=True) | _rates(step, producers=False)
        for well, cap in caps.items():
            rate = rates.get(well)
            if rate is None:
                continue
            if cap <= 0:
                # потолок ноль означает «не давать работать»: любая работа его
                # нарушает. Прежняя редакция такую скважину пропускала — то есть
                # не проверяла единственный случай, когда потолок абсолютен
                if rate > 0 and worst_ratio < float("inf"):
                    worst_ratio = float("inf")
                    worst, where, limit_of_worst = rate, f"скв. {well}, {_stamp(step)}", 0.0
                continue
            ratio = rate / cap
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst, where, limit_of_worst = rate, f"скв. {well}, {_stamp(step)}", cap
    return LimitCheck(key="well_caps_m3_per_day", limit=limit_of_worst or 0.0,
                      value=worst, where=where, unit="м³/сут")
