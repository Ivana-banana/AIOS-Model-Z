"""Группы фонда: 12 управляющих чисел вместо 103 скважин.

Поиск по симулятору стоит 12–19 минут за вызов, бюджет — около 180 вызовов
(Р-63). Поскважинное управление в такой бюджет не помещается: 103 скважины ×
несколько периодов — тысячи переменных. Значит, оптимизатор двигает не скважины,
а **групповые множители**, а внутри группы веса скважин зафиксированы историей.

Правила разбиения взяты из внешнего обзора (`docs/manus/well-grouping.md`,
решение Р-64), и главное из них жёсткое: **добывающие и нагнетательные не
смешиваются в одной группе** — у них разные управляющие переменные, разные
ограничения и разный физический смысл множителя.

Что здесь считается и чего здесь нет:

* **связность «нагнетательная — добывающая»** оценивается кросс-корреляцией
  приростов закачки и приростов отбора жидкости с лагом 0–24 месяца. Это
  **screening**, а не доказательство фильтрационной связи: корреляция не
  отделяет общий тренд объекта от направленного влияния, и обзор говорит это
  прямо. Полноценный CRM с сегментацией по КРС мы не строим — календаря ремонтов
  у нас нет, а без него CRM припишет закачке эффект вмешательства;
* **экономическая роль** группы задаётся обводнённостью: сухие добывающие
  приносят деньги, обводнённые их тратят на подъём и подготовку воды; закачка,
  идущая в обводнённое окружение, возвращается той же водой. Обводнённость —
  законный признак роли группы, но не признак связности;
* **веса внутри группы** — медиана исторического дебита (приёмистости) за окно.
  Это `capacity baseline`: реализуемо и не требует статической модели, но
  закрепляет историческое распределение. Равные веса не используем: на Yibal
  фактическая закачка по скважинам различалась в 90 раз, и равные веса мгновенно
  упирают половину фонда в предел.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

MAX_LAG_MONTHS = 24
"""Верхний лаг отклика добывающей на закачку. Обзор: 0–24 месяца."""

MIN_LINK = 0.30
"""Порог |ρ|, ниже которого пара не считается кандидатом на связь.

Консервативный screening-порог из обзора, не физическая граница.
"""


@dataclass(frozen=True)
class Group:
    """Группа скважин с фиксированными весами. Множитель применяется ко всей группе."""

    name: str
    kind: str
    """`producer` или `injector`. Смешивать типы в одной группе запрещено."""

    weights: dict[str, float]
    """Скважина → её доля в группе. Сумма долей равна единице."""

    @property
    def wells(self) -> tuple[str, ...]:
        return tuple(sorted(self.weights, key=_well_order))


def link_matrix(
    rows: pd.DataFrame,
    since: str,
    max_lag: int = MAX_LAG_MONTHS,
    min_link: float = MIN_LINK,
) -> pd.DataFrame:
    """Матрица связности «нагнетательная × добывающая» по помесячной истории.

    Строки — нагнетательные, столбцы — добывающие, значение — максимум по лагу
    модуля корреляции приростов закачки и приростов отбора жидкости. Всё, что
    ниже `min_link`, обнуляется: это шум, из которого группу строить нельзя.
    """
    injection = _monthly(rows, "WWIR", since)
    liquid = _monthly(rows, "WLPR", since)
    if injection.empty or liquid.empty:
        return pd.DataFrame(index=injection.columns, columns=liquid.columns, dtype=float)

    best = np.zeros((injection.shape[1], liquid.shape[1]))
    for lag in range(max_lag + 1):
        rho = _correlate(injection.to_numpy()[: len(injection) - lag],
                         liquid.to_numpy()[lag:])
        best = np.maximum(best, np.abs(rho))

    matrix = pd.DataFrame(best, index=injection.columns, columns=liquid.columns)
    return matrix.where(matrix >= min_link, 0.0)


def watercut(rows: pd.DataFrame, since: str) -> pd.Series:
    """Обводнённость добывающей за окно: доля воды в поднятой жидкости."""
    window = rows[rows["DATA"] >= since]
    totals = window.groupby(_names(window))[["WLPT_Diff", "WOMT_Diff"]].sum()
    liquid = totals["WLPT_Diff"]
    share = (liquid - totals["WOMT_Diff"]) / liquid.where(liquid > 0)
    return share.dropna()


def build_groups(rows: pd.DataFrame, since: str, min_link: float = MIN_LINK) -> list[Group]:
    """Четыре группы: сухие и обводнённые добывающие, полезная и рециклящая закачка.

    Порог деления — медиана по фонду, а не круглое число: медиана делит фонд
    пополам и не требует калибровки под конкретный объект (в валидационном кейсе
    придут другие скважины). Нагнетательная без подтверждённых связей попадает в
    консервативную группу: усиливать закачку, эффект которой мы не видим в
    истории, оснований нет.
    """
    window = rows[rows["DATA"] >= since]
    produced = window.groupby(_names(window))["WLPT_Diff"].sum()
    injected = window.groupby(_names(window))["WWIT_Diff"].sum()

    producers = sorted(produced[produced > 0].index, key=_well_order)
    injectors = sorted(injected[injected > 0].index, key=_well_order)

    cut = watercut(rows, since).reindex(producers).fillna(1.0)
    links = link_matrix(rows, since, min_link=min_link).reindex(
        index=injectors, columns=producers).fillna(0.0)

    weight = links.to_numpy().sum(axis=1)
    surrounding = pd.Series(
        np.divide(links.to_numpy() @ cut.to_numpy(), weight,
                  out=np.ones_like(weight), where=weight > 0),
        index=injectors,
    )

    return [
        _group("добывающие сухие", "producer",
               [w for w in producers if cut[w] <= cut.median()], rows, since, "WLPR"),
        _group("добывающие обводнённые", "producer",
               [w for w in producers if cut[w] > cut.median()], rows, since, "WLPR"),
        _group("закачка в невыработанное", "injector",
               [w for w in injectors if surrounding[w] <= surrounding.median()],
               rows, since, "WWIR"),
        _group("закачка в обводнённое", "injector",
               [w for w in injectors if surrounding[w] > surrounding.median()],
               rows, since, "WWIR"),
    ]


def problems(groups: list[Group], rows: pd.DataFrame, since: str) -> list[str]:
    """Проверка «разбиение осмысленно, а не случайно». Пустой список — возражений нет.

    Пороги — стартовые screening-критерии из обзора, а не опубликованные нормы:
    четыре скважины в группе и десятая часть базового объёма на группу. Смысл
    один: групповой множитель обязан двигать заметную величину, иначе поиск
    тратит вызовы на переменную, которая ни на что не влияет.
    """
    found: list[str] = []
    window = rows[rows["DATA"] >= since]
    names = _names(window)

    kinds = {group.name: group.kind for group in groups}
    for group in groups:
        if len(group.weights) < 4:
            found.append(f"группа «{group.name}»: {len(group.weights)} скважин — "
                         "меньше четырёх, множитель управляет почти отдельной скважиной")

    for column, kind in (("WLPT_Diff", "producer"), ("WWIT_Diff", "injector")):
        volume = window.groupby(names)[column].sum()
        total = sum(volume.reindex(g.wells).fillna(0.0).sum()
                    for g in groups if g.kind == kind)
        if total <= 0:
            continue
        for group in groups:
            if kinds[group.name] != kind:
                continue
            share = volume.reindex(group.wells).fillna(0.0).sum() / total
            if share < 0.10:
                found.append(f"группа «{group.name}»: {share:.1%} базового объёма — "
                             "меньше десятой части, множитель почти ничего не двигает")

    return found


def stability(rows: pd.DataFrame, since: str, draws: int, seed: int) -> float:
    """Устойчивость разбиения к выборке месяцев: средний ARI по бутстрэпу.

    Обзор предлагает считать разбиение принятым при ARI ≥ 0,70 на выборках 80 %
    месяцев. Ниже — состав групп определяется не историей, а случайностью окна,
    и укрупнение теряет смысл.
    """
    reference = _labels(build_groups(rows, since))
    months = np.sort(rows["DATA"].unique())
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        keep = generator.choice(months, size=int(len(months) * 0.8), replace=False)
        sample = rows[rows["DATA"].isin(keep)]
        labels = _labels(build_groups(sample, since))
        shared = sorted(set(reference) & set(labels), key=_well_order)
        scores.append(adjusted_rand_score([reference[w] for w in shared],
                                          [labels[w] for w in shared]))
    return float(np.mean(scores))


def _group(name: str, kind: str, wells: list[str], rows: pd.DataFrame,
           since: str, column: str) -> Group:
    window = rows[(rows["DATA"] >= since) & _names(rows[rows["DATA"] >= since]).isin(wells)]
    median = window[window[column] > 0].groupby(_names(window))[column].median()
    median = median.reindex(wells).fillna(0.0)
    total = median.sum()
    # Скважина без истории режима в окне не должна обнулять всю группу.
    share = median / total if total > 0 else median
    return Group(name=name, kind=kind, weights={w: float(share[w]) for w in wells})


def _labels(groups: list[Group]) -> dict[str, int]:
    return {well: number for number, group in enumerate(groups) for well in group.wells}


def _monthly(rows: pd.DataFrame, column: str, since: str) -> pd.DataFrame:
    """Помесячная матрица «месяц × скважина» приростов, без скважин без режима.

    Работаем с приростами, а не с уровнями: общий тренд объекта коррелирует сам с
    собой и даёт связь там, где её нет.
    """
    window = rows[rows["DATA"] >= since]
    table = window.pivot_table(index="DATA", columns=_names(window), values=column,
                               aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.max() > 0]
    return table.diff().dropna(how="all").fillna(0.0)


def _correlate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Корреляция каждого столбца `left` с каждым столбцом `right`."""
    if len(left) < 3:
        return np.zeros((left.shape[1], right.shape[1]))
    centred_left = left - left.mean(axis=0)
    centred_right = right - right.mean(axis=0)
    norm = np.outer(np.linalg.norm(centred_left, axis=0),
                    np.linalg.norm(centred_right, axis=0))
    return np.divide(centred_left.T @ centred_right, norm,
                     out=np.zeros_like(norm), where=norm > 0)


def _names(rows: pd.DataFrame) -> pd.Series:
    return rows["well"].astype(str)


def _well_order(name: str) -> tuple[int, str]:
    return (int(name), "") if name.isdigit() else (10**9, name)
