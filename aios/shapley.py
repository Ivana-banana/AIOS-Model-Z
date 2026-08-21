"""Вклад группы скважин в ЧДД — точно, а не приближённо.

Вопрос, вокруг которого мы ходили кругами: ЧДД считается **один раз на весь
объект**, а решения принимают группы. Кому приписать прирост? Обычные ответы —
маржа скважины, значения critic'а в MARL, факторизация `Q` — отвечают не на тот
вопрос: они говорят, что считает модель, а не что подтверждено эталонной
экономикой.

Разбор 4-8 (`docs/manus/round4.md`) снимает вопрос арифметикой. При `m` группах
управления **точные** значения Шепли стоят `2^m` полных расчётов, и приближение
(Kernel SHAP) бюджет не экономит вовсе — оно дороже при малом `m`. У нас четыре
группы (Р-64), значит 16 прогонов; те же 16 ячеек дают заодно все главные
эффекты, все условные парные синергии и взаимодействия высших порядков.

Что здесь считается:

* `shapley` — точное распределение прироста ЧДД по группам. Свойство, которое
  делает его пригодным для отчёта: сумма вкладов **равна** фактическому приросту,
  `Σφᵢ = v(N) − v(∅)`, и это проверяется, а не декларируется (`efficiency_gap`);
* `interaction` — парная синергия **при заданном состоянии остальных групп**.
  Разбор требует показывать обе условные синергии, а не усреднённую: если
  `I_AB(C=−1)` и `I_AB(C=+1)` различаются по знаку, среднее число врёт про обе;
* `missing` — какие коалиции ещё не посчитаны. Пока список непуст, никаких
  вкладов не существует: Шепли не определён на неполной игре.

Три ограничения, которые несёт сам метод и которые идут в отчёт вместе с числами:

1. **Вклад не является свойством пласта.** Он определён для конкретного
   горизонта, baseline, набора режимов, ограничений и версии дека. Смена
   baseline меняет и знак, и порядок: «остановить группу» и «вернуть группе
   базовый режим» — разные контрфакты.
2. **Значение коалиции обязано приходить из полного расчёта.** Суррогатное
   `v(S)` даёт суррогатные вклады, и маркировать их как денежные нельзя —
   `Coalition.source` существует ровно для этого.
3. **Один блок из `2^m` прогонов не даёт доверительного интервала.** Разбор
   предлагает шесть контекстов (`6×8 = 48` прогонов); ансамбля калиброванных
   пластовых сценариев у нас нет, поэтому выполняется один блок точно, а про
   разброс не говорится ничего.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Any

FULL_RUN = "полный расчёт"
SURROGATE = "суррогат"


@dataclass(frozen=True)
class Coalition:
    """Одна ячейка плана: какие группы переведены на кандидатную политику."""

    active: frozenset[str]
    npv_m: float
    source: str = FULL_RUN
    """Откуда число. Денежным вкладом можно называть только `FULL_RUN`."""

    def __str__(self) -> str:
        names = ", ".join(sorted(self.active)) if self.active else "все на базовом режиме"
        return f"{{{names}}}: {self.npv_m:,.1f} млн руб. ({self.source})"


def plan(groups: list[str]) -> list[frozenset[str]]:
    """Все `2^m` коалиций в детерминированном порядке.

    Порядок фиксирован (по размеру, затем по алфавиту), потому что план прогонов
    должен быть воспроизводим: «в порядке, в котором освободилась машина» — не
    план, а его отсутствие.
    """
    unique = sorted(set(groups))
    if len(unique) != len(groups):
        raise ValueError("имена групп повторяются: коалиции перестанут различаться")
    if not unique:
        raise ValueError("групп нет — игра не определена")
    return [
        frozenset(subset)
        for size in range(len(unique) + 1)
        for subset in combinations(unique, size)
    ]


def missing(groups: list[str], known: list[Coalition]) -> list[frozenset[str]]:
    """Коалиции, которых ещё нет. Пока список непуст, вкладов не существует."""
    have = {one.active for one in known}
    return [subset for subset in plan(groups) if subset not in have]


def shapley(groups: list[str], known: list[Coalition]) -> dict[str, float]:
    """Точные значения Шепли по всем `2^m` коалициям, млн руб.

    Формула стандартная — среднее предельного вклада по всем порядкам прихода:

        φᵢ = Σ_{S ⊆ N∖{i}}  |S|!·(m−|S|−1)!/m! · [v(S∪{i}) − v(S)]

    Приближать нечего: при `m = 4` слагаемых восемь на группу, и все `v(S)` уже
    посчитаны — иначе функция и не вызывается.
    """
    absent = missing(groups, known)
    if absent:
        raise ValueError(
            f"не посчитано коалиций: {len(absent)} из {2 ** len(set(groups))}; "
            "Шепли не определён на неполной игре"
        )
    value = {one.active: one.npv_m for one in known}
    names = sorted(set(groups))
    size = len(names)

    result: dict[str, float] = {}
    for name in names:
        rest = [other for other in names if other != name]
        total = 0.0
        for count in range(size):
            weight = factorial(count) * factorial(size - count - 1) / factorial(size)
            for subset in combinations(rest, count):
                without = frozenset(subset)
                total += weight * (value[without | {name}] - value[without])
        result[name] = total
    return result


def efficiency_gap(groups: list[str], known: list[Coalition],
                   contributions: dict[str, float]) -> float:
    """`Σφᵢ − (v(N) − v(∅))`. Обязан быть нулём с точностью до счёта.

    Это не украшение отчёта, а единственная проверка, что вклады сложились в тот
    самый прирост, который мы заявляем. Ненулевой зазор означает ошибку сборки
    коалиций, а не «приближение».
    """
    value = {one.active: one.npv_m for one in known}
    names = frozenset(set(groups))
    return sum(contributions.values()) - (value[names] - value[frozenset()])


def interaction(known: list[Coalition], first: str, second: str,
                held: dict[str, bool]) -> float:
    """Парная синергия `I = v(AB) − v(A) − v(B) + v(∅)` при заданных остальных.

    `held` — состояние прочих групп: `{имя: True}` означает «на кандидатной
    политике». Условность обязательна: разбор 4-8 требует показывать
    `I_AB(C=−1)` и `I_AB(C=+1)` **порознь**, потому что третья группа может не
    просто ослабить связь первых двух, а перевернуть её знак. Усреднённый
    коэффициент парной регрессии тогда описывает несуществующее среднее.
    """
    value = {one.active: one.npv_m for one in known}
    base = frozenset(name for name, active in held.items() if active)
    if first in base or second in base:
        raise ValueError("группа не может одновременно быть парой и фоном")
    try:
        return (value[base | {first, second}] - value[base | {first}]
                - value[base | {second}] + value[base])
    except KeyError as error:
        raise ValueError(f"нет коалиции {sorted(error.args[0])}") from error


def conditional_interactions(groups: list[str], known: list[Coalition],
                             first: str, second: str) -> dict[frozenset[str], float]:
    """Синергия пары при **каждом** состоянии остальных групп.

    Возвращает `{множество активных прочих групп: синергия}`. Разброс этих чисел
    и есть ответ на вопрос «устойчива ли связь»: одно усреднённое значение его
    прячет.
    """
    others = sorted(set(groups) - {first, second})
    return {
        frozenset(subset): interaction(
            known, first, second,
            {name: name in subset for name in others},
        )
        for size in range(len(others) + 1)
        for subset in combinations(others, size)
    }


def explain(groups: list[str], known: list[Coalition]) -> str:
    """Человеко-читаемый разбор вкладов — в отчёт, UI и на защиту."""
    absent = missing(groups, known)
    if absent:
        lines = [f"вкладов нет: не посчитано {len(absent)} коалиций из "
                 f"{2 ** len(set(groups))}"]
        lines.extend(f"  осталось: {sorted(one) or ['все на базовом режиме']}"
                     for one in absent[:8])
        return "\n".join(lines)

    contributions = shapley(groups, known)
    value = {one.active: one.npv_m for one in known}
    surrogate = [one for one in known if one.source != FULL_RUN]

    total = value[frozenset(set(groups))] - value[frozenset()]
    lines = [
        f"прирост ЧДД всех групп вместе: {total:+,.1f} млн руб.",
        "вклад по Шепли:",
    ]
    lines.extend(
        f"  {name}: {share:+,.1f} млн руб."
        for name, share in sorted(contributions.items(), key=lambda item: -item[1])
    )
    lines.append(f"проверка Σφ = v(N) − v(∅): зазор "
                 f"{efficiency_gap(groups, known, contributions):+.6f} млн руб.")

    names = sorted(set(groups))
    if len(names) >= 2:
        lines.append("парные синергии (при каждом состоянии прочих групп):")
        for first, second in combinations(names, 2):
            found = conditional_interactions(names, known, first, second)
            values = sorted(found.values())
            sign = "меняет знак" if values[0] < 0 < values[-1] else "знак устойчив"
            lines.append(f"  {first} × {second}: от {values[0]:+,.1f} до "
                         f"{values[-1]:+,.1f} млн руб. — {sign}")

    if surrogate:
        lines.append(
            f"ВНИМАНИЕ: {len(surrogate)} коалиций из {len(known)} посчитаны суррогатом — "
            "это прогноз, а не денежный вклад группы"
        )
    return "\n".join(lines)


def as_dict(groups: list[str], known: list[Coalition]) -> dict[str, Any]:
    """Машиночитаемый результат — рядом с прогонами, для отчёта и UI."""
    contributions = shapley(groups, known)
    return {
        "groups": sorted(set(groups)),
        "coalitions": [
            {"active": sorted(one.active), "npv_m": one.npv_m, "source": one.source}
            for one in sorted(known, key=lambda item: (len(item.active), sorted(item.active)))
        ],
        "shapley_m": contributions,
        "efficiency_gap_m": efficiency_gap(groups, known, contributions),
        "pairwise": {
            f"{first}×{second}": {
                ",".join(sorted(held)) or "все прочие на базовом": gap
                for held, gap in conditional_interactions(
                    sorted(set(groups)), known, first, second).items()
            }
            for first, second in combinations(sorted(set(groups)), 2)
        },
        "all_from_full_runs": all(one.source == FULL_RUN for one in known),
        "scope": (
            "вклад определён для этого горизонта, baseline, набора режимов и версии "
            "дека; смена baseline меняет знак и порядок вкладов"
        ),
        "confidence_interval": (
            "не считался: выполнен один блок коалиций, ансамбля пластовых сценариев нет"
        ),
    }
