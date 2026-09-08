"""Экономический предел скважины — тупой бейзлайн перед любым умным решением.

Маржа скважины за месяц считается точно и в один проход по нормативам методики.
Это не ЧДД: налог на прибыль и дисконтирование живут на уровне года и объекта, а
здесь — скрининг, который отвечает на вопрос «эта скважина приносит деньги или
проедает их». Итоговое число сценария всегда даёт эталонный расчётчик
(`aios.economics`), и никогда — этот модуль.

Зачем он нужен. Прежде чем учить сеть, надо знать, сколько даёт правило. Отключение
скважин с устойчиво отрицательной маржой — самое дешёвое улучшение ЧДД, какое
вообще существует в этой задаче, и любая нейросеть обязана его превзойти числом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MILLION = 1_000_000.0


@dataclass(frozen=True)
class Norms:
    """Нормативы, приведённые к млн руб. на единицу. Берутся из файла организаторов."""

    oil_price_m_per_t: float
    deductions_m_per_t: float
    oil_opex_m_per_t: float
    liquid_opex_m_per_t: float
    injection_opex_m_per_m3: float
    fund_m_per_well_year: float
    stop_start_m: float
    conversion_m: float
    wacc_rate: float
    """Ставка дисконтирования, доля (0.10 при 10 %)."""

    @classmethod
    def from_assumptions(cls, assumptions: dict[str, Any]) -> Norms:
        """Нормативы кейса — но те из них, что методика держит замком, берутся у неё.

        Эталонный расчётчик перед расчётом **перезаписывает** часть нормативов
        значениями `METHODOLOGY_LOCKS` (`chdd_model.py`): стоимость остановки и
        пуска — 1,0 млн руб., перевода под закачку — 5,0 млн руб., чего бы ни
        стояло в книге. Мы же читали книгу как есть, и на кейсе с другими
        числами (проверено книгой с 4,0 и 12,0) скрининг оценивал бы событие в
        4,0 при настоящей цене 1,0 — то есть искал бы максимум не той функции,
        которой считают деньги.

        Замки берутся у вендора, а не переписываются литералами: обновят архив
        с другими значениями — пойдём за ним, а не разойдёмся молча.
        """
        from aios.economics import chdd_model

        assumptions = {**assumptions, **chdd_model.METHODOLOGY_LOCKS}
        return cls(
            oil_price_m_per_t=assumptions["oilPriceRubT"] / MILLION,
            deductions_m_per_t=assumptions["deductionsRubT"] / MILLION,
            oil_opex_m_per_t=assumptions["oilOpexRubT"] / MILLION,
            liquid_opex_m_per_t=assumptions["liquidOpexRubT"] / MILLION,
            injection_opex_m_per_m3=assumptions["injectionOpexRubM3"] / MILLION,
            fund_m_per_well_year=assumptions["fundAnnualRubWell"] / MILLION,
            stop_start_m=assumptions["stopStartCostM"],
            conversion_m=assumptions["conversionBaseCostM"],
            wacc_rate=assumptions["waccRate"] / 100.0,
        )

    @classmethod
    def from_file(cls, norms_file: Any = None) -> Norms:
        """Нормативы из файла организаторов. Путь берётся из `[economics].norms_file`.

        Без аргумента читается файл действующей поставки. На валидационном кейсе
        (ТЗ п. 4.3) нормативы приходят их, и скрининг обязан считать по тому же
        файлу, по какому считается ЧДД, — иначе одна страница показывает маржу
        по одной цене нефти, а число по другой (замер 23.08).
        """
        from aios.economics import NORMS_FILE, load_norms

        assumptions, _ = load_norms(NORMS_FILE if norms_file is None else norms_file)
        return cls.from_assumptions(assumptions)

    @property
    def oil_netback_m_per_t(self) -> float:
        """Сколько остаётся с тонны нефти после отчислений и переменных затрат."""
        return self.oil_price_m_per_t - self.deductions_m_per_t - self.oil_opex_m_per_t

    @property
    def fund_m_per_well_month(self) -> float:
        return self.fund_m_per_well_year / 12.0


def month_margin(row: dict[str, Any], norms: Norms) -> float:
    """Маржа скважины за месяц, млн руб. (до налога на прибыль и дисконтирования).

    Учитывает всё, что зависит от режима: нефть, жидкость, закачку и содержание
    действующего фонда. Событийные затраты (пуск/остановка, ВНС, смена ЭЦН) сюда
    не входят — они разовые и учитываются при решении, а не в оценке месяца.
    """
    oil_t = row["WOMT_Diff"]
    liquid_t = row["WLPT_Diff"]
    injection_m3 = row["WWIT_Diff"]

    revenue = oil_t * norms.oil_price_m_per_t
    variable = (
        oil_t * norms.deductions_m_per_t
        + oil_t * norms.oil_opex_m_per_t
        + liquid_t * norms.liquid_opex_m_per_t
        + injection_m3 * norms.injection_opex_m_per_m3
    )
    fund = norms.fund_m_per_well_month if is_active(row) else 0.0
    return revenue - variable - fund


def counted_by_methodology(row: dict[str, Any]) -> bool:
    """Идёт ли строка в экономику вовсе.

    Строка с отрицательным `WLPT_Diff`, `WOMT_Diff` или `WWIT_Diff` исключается
    методикой **целиком** — не обнуляется и не клампится; так её считает
    эталонный расчётчик организаторов, и так же обязан считать наш скрининг.
    Иначе мы приписываем скважине убыток, которого в деньгах не существует.

    Это не пересчёт методики: число сценария по-прежнему даёт только
    `aios.economics`. Здесь мы лишь не принимаем решений по строкам, которые в
    итоговое число не войдут.

    Отрицательные строки на этом объекте не гипотетические: 102 штуки за 2025-09
    в выгрузке организаторов.
    """
    return not (row["WLPT_Diff"] < 0 or row["WOMT_Diff"] < 0 or row["WWIT_Diff"] < 0)


def is_active(row: dict[str, Any]) -> bool:
    """Действующий фонд по методике: WLPR > 0 или WWIR > 0."""
    return row["WLPR"] > 0 or row["WWIR"] > 0


def breakeven_oil_t(liquid_t: float, norms: Norms) -> float:
    """Сколько тонн нефти в месяц нужно, чтобы скважина вышла в ноль."""
    cost = liquid_t * norms.liquid_opex_m_per_t + norms.fund_m_per_well_month
    return cost / norms.oil_netback_m_per_t


def breakeven_watercut(liquid_t: float, norms: Norms) -> float:
    """Предельная обводнённость по массе при данном отборе жидкости, доля.

    Ниже неё скважина зарабатывает, выше — проедает. Возвращает 0.0, если отбор
    настолько мал, что скважина убыточна при любой обводнённости.
    """
    if liquid_t <= 0:
        return 0.0
    return max(0.0, 1.0 - breakeven_oil_t(liquid_t, norms) / liquid_t)


@dataclass(frozen=True)
class WellVerdict:
    """Итог скрининга по одной скважине."""

    well: str
    months: int
    active_months: int
    total_margin_m: float
    negative_months: int
    trailing_negative_months: int
    """Сколько последних месяцев подряд маржа отрицательна."""

    trailing_margin_m: float
    """Суммарная маржа за эти последние убыточные месяцы (отрицательная)."""

    oil_t: float
    liquid_t: float
    injection_m3: float

    @property
    def watercut(self) -> float:
        return 1.0 - self.oil_t / self.liquid_t if self.liquid_t > 0 else 0.0

    @property
    def average_margin_m(self) -> float:
        return self.total_margin_m / self.months if self.months else 0.0


def screen(rows: list[dict[str, Any]], norms: Norms) -> list[WellVerdict]:
    """Пройти по выгрузке и оценить каждую скважину. Сортировка — худшие первыми."""
    by_well: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_well.setdefault(row["well"], []).append(row)

    verdicts: list[WellVerdict] = []
    for well, well_rows in by_well.items():
        well_rows.sort(key=lambda item: item["DATA"])
        # Строки, исключённые методикой, в оценку не идут: расчётчик выбрасывает
        # их целиком, и без этого сумма месячных марж перестаёт совпадать с его
        # EBITDA. Замер 21.08: одна такая строка на двенадцать месяцев даёт
        # расхождение 1,427 млн руб., а страница прогона утверждает, что скважины
        # отбираются по тем же деньгам, по каким нас будут оценивать.
        #
        # Отбор один на весь вердикт, а не только на деньги: тонны, месяцы и
        # средняя маржа стоят на странице в одной строке с ними и обязаны быть
        # посчитаны по тому же материалу. Иначе получается «маржа без исключённой
        # строки, тонны вместе с ней» — и обводнённость выходит за 100 %.
        counted = [row for row in well_rows if counted_by_methodology(row)]
        margins = [month_margin(row, norms) for row in counted]

        trailing = 0
        for margin in reversed(margins):
            if margin < 0:
                trailing += 1
            else:
                break

        verdicts.append(
            WellVerdict(
                well=well,
                months=len(counted),
                active_months=sum(1 for row in counted if is_active(row)),
                total_margin_m=sum(margins),
                negative_months=sum(1 for margin in margins if margin < 0),
                trailing_negative_months=trailing,
                trailing_margin_m=sum(margins[len(margins) - trailing:]) if trailing else 0.0,
                oil_t=sum(row["WOMT_Diff"] for row in counted),
                liquid_t=sum(row["WLPT_Diff"] for row in counted),
                injection_m3=sum(row["WWIT_Diff"] for row in counted),
            )
        )

    verdicts.sort(key=lambda verdict: verdict.total_margin_m)
    return verdicts


def shutdown_candidates(rows: list[dict[str, Any]],
                        norms: Norms,
                        min_trailing_months: int = 6) -> list[WellVerdict]:
    """Скважины, которые стоит остановить: устойчиво убыточны и остановка окупается.

    Два условия, и оба обязательны:

    1. маржа отрицательна `min_trailing_months` месяцев подряд — единичный плохой
       месяц не повод дёргать фонд;
    2. накопленный убыток за эти месяцы перекрывает стоимость самой остановки.
       Остановка и пуск — два платных события по 1,0 млн руб., то есть решение
       «остановить и потом вернуть» стоит 2 млн; экономия обязана это окупить.
    """
    candidates: list[WellVerdict] = []
    for verdict in screen(rows, norms):
        if verdict.trailing_negative_months < min_trailing_months:
            continue
        if -verdict.trailing_margin_m < norms.stop_start_m:
            continue
        candidates.append(verdict)
    return candidates
