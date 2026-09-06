"""Достижим ли форсаж отбора — по той истории, которая у нас есть.

Заявляемый сценарий содержит множитель отбора, и на защите первым спросят, откуда
на зрелом объекте с обводнённостью 83 % берётся дополнительная жидкость. Ответ
«симулятор посчитал» жюри инженеров не устроит, а Р-66 уже показал, что множитель
`LRAT` при историческом пластовом давлении даёт ×1,01 — то есть ничего.

Здесь — критерии из §8.3 внешнего разбора 4-1 (`docs/manus/round4.md`), каждый из
которых проверяется **по помесячной выгрузке**, без промысловых исследований.
Модуль отвечает на один вопрос: что история говорит про запас по притоку у каждой
скважины и у объекта.

Три вещи, которые он делает намеренно:

1. **Разделяет «предел», «есть запас» и «нельзя сказать».** Третья категория
   обязана существовать. Без частоты ЭЦН, тока двигателя и календаря ремонтов
   отличить предел притока от предела оборудования по `q`, `P_wf` и `P_wh`
   нельзя — разбор говорит это прямо, и выдавать догадку за диагноз здесь
   дороже, чем промолчать.
2. **Считает, сколько прироста жидкости приходится на нефть.** Это главный
   вопрос экономики: жидкость стоит денег на подъём и подготовку, выручка идёт
   только с нефти. Арифметика объекта беспощадна — при обводнённости 83 % роста
   обводнённости всего на 2,77 п.п. достаточно, чтобы **весь** прирост жидкости
   оказался водой (`absorbing_watercut`).
3. **Называет то, что не проверяется нашим стендом.** `UNCHECKABLE` — не
   оговорка мелким шрифтом, а часть вывода: три из шести признаков разбора
   требуют данных, которых в выгрузке нет вовсе.

Чего здесь нет и почему. IPR Вогеля и коэффициент продуктивности `J` не
считаются: `J = q/(P_r − P_wf)` требует **пластового** давления, а выгрузка несёт
только забойное и устьевое. Оценка `J` по истории при дрейфующем `P_r` уже
проверялась (третий заход) и дала отрицательный наклон у 27 скважин из 41 — то
есть не работает на этом горизонте.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_BHP_TOLERANCE_ATM = 1.0
"""Насколько выше нижней уставки скважина ещё считается «стоящей на ней».

Симулятор держит забойное на пределе не точно: округление шага и переключение
`LRAT`→`BHP` дают разброс в доли атмосферы.
"""

MIN_PRODUCING_MONTHS = 12
"""Меньше года истории — диагноз не ставится, а называется несостоявшимся."""

UNCHECKABLE = (
    "частота ЭЦН, ток и температура двигателя — без них предел оборудования "
    "не отличается от предела притока (признаки 4 и 6 разбора 4-1)",
    "календарь ремонтов и вмешательств — без него прирост после КРС читается "
    "как отклик на закачку (признак 1)",
    "пластовое давление на едином датуме — без него коэффициент продуктивности "
    "J = q/(P_r − P_wf) не считается вовсе (признак 3)",
)
"""Чего в выгрузке нет и что поэтому не проверяется. Идёт в отчёт дословно."""


def absorbing_watercut(watercut: float, liquid_ratio: float) -> float:
    """При какой обводнённости весь прирост жидкости окажется водой.

    `watercut` — сегодняшняя обводнённость, доля; `liquid_ratio` — во сколько раз
    растёт отбор жидкости. Возвращает обводнённость, при которой добыча нефти
    остаётся прежней.

    Смысл числа. Объект 2024 года: обводнённость 0,83, форсаж ×1,19 — достаточно
    роста до 0,8577, то есть **на 2,77 процентного пункта**, чтобы прирост
    жидкости не дал ни тонны нефти. Это не прогноз обводнения, а мера того,
    насколько узок запас: заявляя прирост нефти от форсажа, мы неявно утверждаем,
    что обводнённость вырастет меньше чем на эти пункты.
    """
    if liquid_ratio <= 0:
        raise ValueError("множитель отбора должен быть положительным")
    return 1.0 - (1.0 - watercut) / liquid_ratio


@dataclass(frozen=True)
class WellForcing:
    """Что история говорит про запас по притоку одной скважины."""

    well: str
    months: int
    at_min_bhp: float
    """Доля месяцев работы на нижней уставке забойного давления."""

    oil_in_extra_liquid: float | None
    """Сколько нефти приходится на кубометр прироста жидкости, т/м³.

    Наклон регрессии месячных приращений `Δq_o` на `Δq_l`. Ноль означает, что
    добавка жидкости — вода целиком; `None` — жидкость не менялась и наклон не
    определён.
    """

    oil_in_flow: float
    """Нефти в самом потоке скважины, т/м³: `среднее WOMR / среднее WLPR`.

    Нужно рядом с наклоном, потому что **наклон сам по себе ничего не говорит**.
    Если скважину просто прикрывали и открывали, состав потока не менялся, и
    наклон получится равным этому отношению — при полном отсутствии информации о
    том, что даст форсаж. Значение имеет только разность.
    """

    watercut_change: float | None
    """На сколько долей выросла обводнённость за историю скважины."""

    verdict: str
    reasons: tuple[str, ...]

    @property
    def extra_against_flow(self) -> float | None:
        """Насколько добавка жидкости богаче нефтью, чем текущий поток, т/м³.

        Это и есть измеряемая величина. Ноль означает «режим меняли
        пропорционально, состав не тронут» — то есть история про форсаж молчит.
        Отрицательное — добавка беднее нефтью, чем поток: ровно то, чего боятся
        на объекте с обводнённостью 83 %.
        """
        if self.oil_in_extra_liquid is None:
            return None
        return self.oil_in_extra_liquid - self.oil_in_flow

    def __str__(self) -> str:
        share = f"{self.at_min_bhp:.0%}"
        gap = self.extra_against_flow
        against = "—" if gap is None else f"{gap:+.3f}"
        return (f"скв. {self.well}: {self.verdict}; на нижней уставке {share} месяцев, "
                f"добавка против потока {against} т/м³")


def diagnose(rows: list[dict[str, Any]], *, min_bhp_atm: float) -> list[WellForcing]:
    """Диагноз по каждой добывающей скважине выгрузки.

    `min_bhp_atm` — нижняя уставка забойного давления из дека (`WCONPROD`). Она
    параметр, а не константа: на валидационном кейсе она будет другой, а зашитое
    число превратило бы диагноз в фикцию молча.
    """
    by_well: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if float(row.get("WLPR") or 0.0) <= 0:
            continue
        by_well.setdefault(str(row["well"]), []).append(row)

    return [_one(well, sorted(months, key=lambda item: str(item["DATA"])), min_bhp_atm)
            for well, months in sorted(by_well.items(), key=lambda item: _key(item[0]))]


def _one(well: str, months: list[dict[str, Any]], min_bhp_atm: float) -> WellForcing:
    liquid = np.array([float(row["WLPR"]) for row in months])
    oil = np.array([float(row.get("WOMR") or 0.0) for row in months])
    bhp = np.array([float(row.get("BHP") or 0.0) for row in months])

    at_min = float((bhp <= min_bhp_atm + MIN_BHP_TOLERANCE_ATM).mean())
    slope = _slope(np.diff(liquid), np.diff(oil), float(liquid.mean()))
    # Защита от деления на ноль недостижима: `diagnose` отсеял месяцы с
    # `WLPR <= 0` ещё на входе, поэтому среднее по жидкости здесь всегда
    # положительно. Мутационный прогон помечает подмену границы эквивалентной, и
    # это верный вердикт (Р-244); ветка оставлена как страховка на случай, если
    # отсев однажды снимут.
    in_flow = float(oil.mean() / liquid.mean()) if liquid.mean() > 0 else 0.0
    change = _watercut_change(liquid, oil)

    if len(months) < MIN_PRODUCING_MONTHS:
        return WellForcing(well, len(months), at_min, slope, in_flow, change,
                           "нельзя сказать",
                           (f"истории {len(months)} мес., меньше {MIN_PRODUCING_MONTHS}",))

    gap = None if slope is None else slope - in_flow
    reasons: list[str] = []
    if at_min >= 0.7:
        reasons.append(f"на нижней уставке {at_min:.0%} месяцев — запаса по депрессии нет")
    # Строгое сравнение и нестрогое различаются только при изменении ровно на
    # 0,02: обводнённость считается отношением сумм, и попасть им в круглое
    # число нельзя — подмена эквивалентна (Р-244).
    if change is not None and change > 0.02:
        reasons.append(f"обводнённость выросла на {change * 100:.1f} п.п. за историю")

    if gap is None:
        reasons.append("режим не менялся заметно ни разу — про добавку истории нечего сказать")
        verdict = "нельзя сказать"
    elif at_min >= 0.7 and gap <= 0.0:
        reasons.append(f"добавка жидкости беднее нефтью, чем поток, на {-gap:.3f} т/м³")
        verdict = "предел"
    elif at_min < 0.3 and gap > MEANINGFUL_GAP:
        reasons.append(f"добавка жидкости богаче потока на {gap:+.3f} т/м³ "
                       f"при {at_min:.0%} месяцев на уставке")
        verdict = "есть запас"
    else:
        reasons.append(f"добавка против потока {gap:+.3f} т/м³ — признаки разошлись")
        verdict = "нельзя сказать"
    return WellForcing(well, len(months), at_min, slope, in_flow, change, verdict,
                       tuple(reasons))


def summary(verdicts: list[WellForcing]) -> str:
    """Свод по фонду плюс дословный список того, что не проверяется."""
    if not verdicts:
        return "добывающих скважин в выгрузке нет"

    counts: dict[str, int] = {}
    for one in verdicts:
        counts[one.verdict] = counts.get(one.verdict, 0) + 1

    lines = [f"добывающих скважин: {len(verdicts)}"]
    for verdict in ("предел", "есть запас", "нельзя сказать"):
        share = counts.get(verdict, 0) / len(verdicts)
        lines.append(f"  {verdict}: {counts.get(verdict, 0)} ({share:.0%})")

    at_min = [one.at_min_bhp for one in verdicts]
    lines.append(f"месяцев на нижней уставке, медиана по фонду: {np.median(at_min):.0%}")

    measured = [one for one in verdicts if one.extra_against_flow is not None]
    lines.append(f"режим менялся заметно у {len(measured)} скважин из {len(verdicts)}")
    if measured:
        gaps = [one.extra_against_flow for one in measured]
        lines.append(f"  добавка против потока, медиана: {np.median(gaps):+.3f} т/м³")
        # выборка смещена, и это надо видеть рядом с числом, а не в сноске:
        # менять режим есть смысл там, где скважина суше среднего
        lines.append(
            f"  нефти в потоке: у этих {np.median([one.oil_in_flow for one in measured]):.3f} "
            f"против {np.median([one.oil_in_flow for one in verdicts]):.3f} т/м³ по фонду — "
            "выборка смещена в сторону сухих скважин, режим меняли именно им"
        )

    lines.append("не проверяется нашим стендом:")
    lines.extend(f"  {one}" for one in UNCHECKABLE)
    return "\n".join(lines)


MEANINGFUL_CHANGE = 0.02
"""Изменение отбора считается изменением режима от 2 % среднего дебита скважины.

Порог не косметический. Без него в выборку попадают месяцы, где отбор дрейфует на
сотые доли куба — это истощение, а не управление, и нефть в них падает сама по
себе. На скв. 16 базового прогона таких месяцев 54 из 201, медиана `|Δq_l|` —
0,027 м³/сут при среднем дебите 23,2: почти вся выборка была шумом, и только
взвешивание на `x²` в методе наименьших квадратов спасало оценку от него.
"""

MEANINGFUL_GAP = 0.05
"""Насколько добавка должна быть богаче потока, чтобы это считалось запасом, т/м³.

Меньше — в пределах того, чем месячная выгрузка отличается от управляемого опыта:
на приращениях остаются и запуски после простоя, и изменения после ремонта,
календаря которых у нас нет.
"""

MIN_CHANGED_MONTHS = 6
"""Меньше шести настоящих изменений режима — наклон не оценивается.

Три точки давали число всегда, и это число было бы диагнозом по шуму.
"""


def _slope(delta_liquid: np.ndarray, delta_oil: np.ndarray,
           mean_liquid: float) -> float | None:
    """Наклон `Δq_o` на `Δq_l` методом наименьших квадратов через ноль.

    Через ноль намеренно: свободный член означал бы «нефть меняется сама по себе,
    без изменения отбора», а вопрос ровно в том, сколько нефти несёт **добавка**
    жидкости. Считается на приращениях, а не на уровнях, чтобы общий тренд
    истощения не выдавал себя за отклик на управление.

    Единицы разные и остаются разными: `Δq_o` в т/сут, `Δq_l` в м³/сут, поэтому
    наклон читается «тонн нефти на кубометр прироста жидкости». Приводить нефть к
    объёму здесь нельзя — плотность у скважин разная (`aios.pvt`), и общий
    множитель смешал бы регионы.
    """
    threshold = max(1e-9, MEANINGFUL_CHANGE * mean_liquid)
    mask = np.abs(delta_liquid) > threshold
    if mask.sum() < MIN_CHANGED_MONTHS:
        return None
    x, y = delta_liquid[mask], delta_oil[mask]
    denominator = float((x * x).sum())
    if denominator <= 0:
        return None
    return float((x * y).sum() / denominator)


def _watercut_change(liquid: np.ndarray, oil: np.ndarray) -> float | None:
    """Насколько выросла обводнённость: первый год против последнего."""
    window = min(12, len(liquid) // 2)
    if window < 3:
        return None
    early = _watercut(liquid[:window], oil[:window])
    late = _watercut(liquid[-window:], oil[-window:])
    if early is None or late is None:
        return None
    return late - early


def _watercut(liquid: np.ndarray, oil: np.ndarray) -> float | None:
    total = float(liquid.sum())
    return None if total <= 0 else 1.0 - float(oil.sum()) / total


def _key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
