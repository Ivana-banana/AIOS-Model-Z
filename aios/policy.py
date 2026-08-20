"""Политика управления фондом: от диагноза к управляющим действиям.

Здесь живёт бейзлайн-политика — правило, которое любая нейросеть обязана
превзойти числом ЧДД, иначе её не существует. Она же задаёт формат, в котором
агенты возвращают решения: список `Action` с причиной и ценой, а не молчаливое
изменение расписания.

Два свойства, которые здесь важнее самой эвристики:

* **Политика причинная.** Решение на дату `t` принимается только по данным до `t`.
  Правило, подглядывающее в будущее, даст красивый ЧДД на истории и развалится на
  валидационном кейсе — а именно на нём нас и считают.
* **Каждое действие объясняет себя.** `Action.reason` пишется в момент принятия
  решения тем же кодом, который решение принял. Слой интерпретируемости — прямое
  требование регламента, и объяснение, дописанное задним числом, ему не отвечает.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from aios.deck import ProducerControl, ScheduleStep
from aios.margin import Norms, counted_by_methodology, is_active, month_margin

SHUT = "shut"


@dataclass(frozen=True)
class Action:
    """Одно управляющее воздействие с обоснованием и ценой."""

    date: date
    """Дата, с которой действие вступает в силу."""

    well: str
    kind: str
    reason: str
    cost_m: float
    """Разовая стоимость события по методике, млн руб."""

    def __str__(self) -> str:
        return f"{self.date.isoformat()} скв. {self.well}: {self.kind} — {self.reason}"


def plan_shutdowns(rows: list[dict[str, Any]],
                   norms: Norms,
                   *,
                   control_start: date,
                   lookback: int = 6) -> list[Action]:
    """Остановить скважины, устойчиво работающие в убыток.

    Скважина останавливается с месяца, следующего за тем, в котором выполнились
    оба условия: `lookback` убыточных месяцев подряд и накопленный за них убыток
    больше стоимости самой остановки (1,0 млн руб.). Второе условие обязательно:
    остановить скважину, теряющую 100 тыс. руб. в месяц, — значит заплатить
    миллион за экономию, которая его не окупит.

    Раньше `control_start` действий не бывает: до этой даты история задана
    организаторами и мы ей не управляем.
    """
    months = sorted({row["DATA"] for row in rows})
    by_month: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_month.setdefault(row["DATA"], []).append(row)

    streak: dict[str, int] = {}
    loss: dict[str, float] = {}
    shut_wells: set[str] = set()
    actions: list[Action] = []

    for index, month in enumerate(months):
        for row in by_month[month]:
            well = row["well"]
            if well in shut_wells:
                continue

            # Строку, исключённую методикой, скрининг тоже не считает: убытка,
            # которого нет в деньгах, не должно быть и в решении. Счётчик при
            # этом сбрасывается, а не продолжается через пропуск: правило
            # говорит «убыточна lookback месяцев **подряд**», а через дыру
            # подряд не бывает. Сброс — сторона осторожная, и это верная
            # сторона: замер 20.08 показал, что остановки на этом объекте
            # разрушают стоимость.
            if not counted_by_methodology(row):
                streak[well] = 0
                loss[well] = 0.0
                continue

            # Месяц закачки правилом про добывающие не судится. У нагнетательной
            # выручки нет по построению, поэтому её месячная маржа отрицательна
            # всегда, и «убыточна lookback месяцев подряд» срабатывало на ней
            # автоматически. Расписание такое решение не исполняло — `apply`
            # выдаёт только ProducerControl SHUT, — то есть остановка попадала в
            # таблицу решений и в объяснение, не происходя (замер 20.08). В
            # базовом расписании Модели Z таких скважин 41.
            #
            # Остановка закачки — отдельное решение со своими последствиями для
            # энергии пласта, и принимать его побочным эффектом правила про
            # добывающие нельзя: замер того же дня показал, что срез закачки
            # обрушивает нефть вчетверо.
            if float(row["WWIR"]) > 0 and float(row["WLPR"]) <= 0:
                streak[well] = 0
                loss[well] = 0.0
                continue

            margin = month_margin(row, norms)
            if margin < 0 and is_active(row):
                streak[well] = streak.get(well, 0) + 1
                loss[well] = loss.get(well, 0.0) + margin
            else:
                streak[well] = 0
                loss[well] = 0.0
                continue

            if streak[well] < lookback or -loss[well] < norms.stop_start_m:
                continue
            if index + 1 >= len(months):
                continue  # останавливать в последнем месяце горизонта нечего

            effective = _as_date(months[index + 1])
            if effective < control_start:
                continue

            actions.append(
                Action(
                    date=effective,
                    well=well,
                    kind=SHUT,
                    reason=(
                        f"убыток {streak[well]} мес. подряд, накоплено "
                        f"{-loss[well]:.2f} млн руб.; обводнённость "
                        f"{_watercut(row):.1%}, нефть {row['WOMT_Diff']:.1f} т/мес"
                    ),
                    cost_m=norms.stop_start_m,
                )
            )
            shut_wells.add(well)

    return actions


def apply(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
          actions: list[Action],
          *,
          bhp: float | None = 50.0) -> list[ScheduleStep]:
    """Наложить действия на расписание.

    В Eclipse режим действует до следующего явного изменения, поэтому остановка
    записывается один раз — повторять `SHUT` на каждом шаге не нужно и вредно:
    лишние записи раздувают файл и мешают его читать глазами. Остановленная
    скважина после этого не получает управления вовсе.
    """
    shut_from: dict[str, date] = {}
    for action in actions:
        if action.kind != SHUT:
            raise ValueError(f"неизвестное действие: {action.kind}")
        current = shut_from.get(action.well)
        if current is None or action.date < current:
            shut_from[action.well] = action.date

    result: list[ScheduleStep] = []
    already_shut: set[str] = set()

    for step in steps:
        producers: list[ProducerControl] = []
        for control in step.producers:
            since = shut_from.get(control.well)
            if since is None or step.date < since:
                producers.append(control)
                continue
            if control.well not in already_shut:
                producers.append(
                    ProducerControl(
                        well=control.well,
                        status="SHUT",
                        mode=control.mode,
                        liquid_rate=None,
                        bhp=bhp if bhp is not None else control.bhp,
                    )
                )
                already_shut.add(control.well)
            # иначе скважина уже остановлена — управления ей не выдаём

        # скважина, остановленная в момент, когда её не было в WCONPROD этого шага
        for well, since in shut_from.items():
            if well in already_shut or step.date < since:
                continue
            if any(control.well == well for control in step.injectors):
                continue
            producers.append(ProducerControl(well, "SHUT", "LRAT", None, bhp))
            already_shut.add(well)

        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))

    return result


def explain(actions: list[Action]) -> str:
    """Человеко-читаемый отчёт о принятых решениях — в UI, лог и отчёт."""
    if not actions:
        return "Действий нет: убыточных скважин, окупающих остановку, не найдено."

    total = sum(action.cost_m for action in actions)
    lines = [
        f"Действий: {len(actions)}, разовые затраты {total:.1f} млн руб.",
        "",
    ]
    lines.extend(f"  {action}" for action in actions)
    return "\n".join(lines)


def _watercut(row: dict[str, Any]) -> float:
    liquid = row["WLPT_Diff"]
    return 1.0 - row["WOMT_Diff"] / liquid if liquid > 0 else 0.0


def _as_date(value: str) -> date:
    return date.fromisoformat(value)
