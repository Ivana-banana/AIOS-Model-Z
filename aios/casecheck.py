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
from dataclasses import dataclass
from typing import Any

from aios.deck import ScheduleStep
from aios.margin import counted_by_methodology
from aios.planner import Limits


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
    """
    steps = list(steps)
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
    if limits.max_field_oil_t_per_day is not None:
        checks.append(_phase(steps, forecast, limits.max_field_oil_t_per_day,
                             key="max_field_oil_t_per_day", phase="нефть"))
    if limits.max_field_water_t_per_day is not None:
        checks.append(_phase(steps, forecast, limits.max_field_water_t_per_day,
                             key="max_field_water_t_per_day", phase="вода"))
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
    for step in steps:
        span = monthrange(step.date.year, step.date.month)[1]
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
               producers: bool) -> LimitCheck:
    best, where = 0.0, "—"
    for step in steps:
        for well, rate in _rates(step, producers=producers).items():
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


def _annual_volume(steps: list[ScheduleStep], limit: float, *, key: str,
                   producers: bool) -> LimitCheck:
    volume: dict[int, float] = {}
    for step in steps:
        span = monthrange(step.date.year, step.date.month)[1]
        volume[step.date.year] = volume.get(step.date.year, 0.0) + \
            sum(_rates(step, producers=producers).values()) * span
    best, where = 0.0, "—"
    for year, total in volume.items():
        if total > best:
            best, where = total, str(year)
    return LimitCheck(key=key, limit=limit, value=best, where=where, unit="м³/год")


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


def _active_wells(steps: list[ScheduleStep], limit: int) -> LimitCheck:
    best, where = 0, "—"
    for step in steps:
        active = len(_rates(step, producers=True)) + len(_rates(step, producers=False))
        if active > best:
            best, where = active, _stamp(step)
    return LimitCheck(key="max_active_wells", limit=float(limit), value=float(best),
                      where=where, unit="скважин")


def _forced_shut(steps: list[ScheduleStep], wells: frozenset[str]) -> LimitCheck:
    """Сколько раз обязательно остановленная скважина осталась открытой."""
    breaches, where = 0, "—"
    for step in steps:
        open_now = set(_rates(step, producers=True)) | set(_rates(step, producers=False))
        left = sorted(wells & open_now)
        if left:
            breaches += len(left)
            if where == "—":
                where = f"скв. {left[0]}, {_stamp(step)}"
    return LimitCheck(key="forced_shut_wells", limit=0.0, value=float(breaches),
                      where=where, unit="открытых шагов")


def _forced_windows(steps: list[ScheduleStep], windows) -> LimitCheck:
    breaches, where = 0, "—"
    for step in steps:
        open_now = set(_rates(step, producers=True)) | set(_rates(step, producers=False))
        for window in windows:
            if window.covers(step.date) and window.well in open_now:
                breaches += 1
                if where == "—":
                    where = f"скв. {window.well}, {_stamp(step)}"
    return LimitCheck(key="forced_shut_windows", limit=0.0, value=float(breaches),
                      where=where, unit="открытых шагов")


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
            if rate is None or cap <= 0:
                continue
            ratio = rate / cap
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst, where, limit_of_worst = rate, f"скв. {well}, {_stamp(step)}", cap
    return LimitCheck(key="well_caps_m3_per_day", limit=limit_of_worst or 0.0,
                      value=worst, where=where, unit="м³/сут")
