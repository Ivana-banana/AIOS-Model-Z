"""Мера недоверия к суррогату: сценарное расстояние и запас против оптимизма.

Зачем это существует. У нас уже есть проверка области применимости
(`aios.surrogate.TrainingEnvelope`): каждый управляющий множитель обязан лежать
внутри диапазона обучения и рядом с обучающей точкой. Она покоординатная — и по
построению не видит **совместно** нетипичный режим: DoE двигал скважины вразнобой,
поэтому расписание, где половина фонда разом стоит у верхней границы, проходит
поскважинную проверку целиком, хотя такого сочетания в обучении не было ни разу.

Это не гипотеза. 21.08 поиск с Р-57 выбрал расписание, которое суррогат оценил
выше базового, а полный расчёт ГГДМ дал −14,46 %: **ошибка прогноза 3 225,0 млн
руб.**, и ни одна защита не сработала. Разбор 4-7 четвёртого захода Manus
формулирует причину точнее, чем мы: прямоугольник `B = ∏ⱼ[minⱼ, maxⱼ]` разрешает
комбинацию «каждая скважина по отдельности бывала в таком режиме, но все вместе —
никогда», и **нулевое число отбраковок при разреженном DoE — ожидаемый результат,
а не парадокс**.

Что здесь считается:

* **`TrustRegion`** — расстояние кандидата до `k`-го ближайшего **обучающего
  сценария** в стандартизованном пространстве управлений. Единица измерения —
  целый сценарий, а не строка «скважина × месяц»: 370 месяцев одного прогона
  зависимы через общее расписание и общую динамику пласта, и считать их
  независимыми точками значит завысить плотность выборки в 370 раз;
* **`OptimismMargin`** — односторонний конформный запас против завышения цели.
  Считается по парам «прогноз суррогата и полный расчёт», и только по ним;
* **`safe_value`** — цель поиска `J_safe = Ĵ − λ·B_J − ρ·max(0, d_k − τ)²`.
  Оптимизатор перестаёт получать прямую выгоду от того, что ошибается в нашу
  пользу.

Чего этот модуль **не** делает и о чём обязан говорить вслух:

1. **Конформный запас не является гарантией для найденного оптимума.** Покрытие
   даётся для сценария, обменяемого с обучающими. Кандидат, которого оптимизатор
   выбрал **за** высокое суррогатное значение, таковым не является — это
   feedback covariate shift. `J_safe` это guardrail, а не доказательство
   безопасности, и в отчёт идёт именно так.
2. **Порог не назначается.** `τ` выводится из самой выборки: насколько одинока
   самая одинокая её точка (leave-one-out по сценариям). Назначенное число
   выглядело бы как измерение, не будучи им.
3. **Мало калибровочных пар — значит «не калибровано».** При `n` парах
   односторонний квантиль уровня `1−α` существует только когда
   `⌈(n+1)(1−α)⌉ ≤ n`. Иначе `OptimismMargin.value` равен `None`, и кандидат
   обязан идти на полный расчёт: подставить сюда правдоподобное число значит
   заявить защиту, которой нет.

Полный контракт артефакта — `contract()`, он же `artifacts/trust-gate.json`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

DEFAULT_K = 5
"""Сколько соседей смотрим. Пять — стартовая настройка, а не результат.

Разбор 4-7 называет `k=5` отправной точкой и требует выбирать его по внешнему
сценарному тесту. Число объяснимо инженеру: «покажите пять ближайших режимов
DoE» — ровно то, что `TrustRegion.neighbours` и возвращает.
"""

DEFAULT_ALPHA = 0.1
"""Уровень односторонней конформной калибровки: запас покрывает 90 % оптимизма."""

GUARANTEE_SCOPE = (
    "маргинальное покрытие при сценарной обменяемости; "
    "не гарантия для адаптивно выбранного оптимизатором кандидата"
)
"""Область действия гарантии — строка в артефакт и в отчёт, слово в слово.

Разбор 4-7 требует её присутствия отдельным полем: без неё конформный интервал
читается как обещание, которым он не является.
"""


@dataclass(frozen=True)
class Neighbour:
    """Обучающий сценарий рядом с кандидатом — для объяснения решения человеку."""

    scenario: str
    distance: float


@dataclass(frozen=True)
class Distance:
    """Насколько кандидат далёк от обучающих сценариев."""

    value: float
    """`d_k` — расстояние до `k`-го ближайшего обучающего сценария."""

    threshold: float
    """`τ` — насколько одинока самая одинокая точка самой обучающей выборки."""

    neighbours: tuple[Neighbour, ...]
    """`k` ближайших обучающих сценариев по возрастанию расстояния."""

    @property
    def outside(self) -> bool:
        return self.value > self.threshold

    def __str__(self) -> str:
        if not self.neighbours:
            return ("управление, которого DoE не двигал ни разу: сравнивать не с чем, "
                    "режим вне обучающего плана")
        names = ", ".join(f"{n.scenario} ({n.distance:.2f})" for n in self.neighbours)
        verdict = "вне доверенной области" if self.outside else "внутри доверенной области"
        return (f"d{len(self.neighbours)} = {self.value:.2f} при пороге "
                f"{self.threshold:.2f} — {verdict}; ближайшие: {names}")


def profile(multipliers: dict[date, dict[str, float]]) -> dict[str, float]:
    """Свернуть помесячные управляющие множители в сценарное представление.

    Сырые 370 месяцев повторять нельзя — они зависимы. Сжатие фиксированное и
    заранее объявленное, по два числа на управление:

    * `<ключ>|уровень` — среднее множителя по горизонту: где режим стоял;
    * `<ключ>|размах` — среднеквадратичное отклонение по месяцам: насколько режим
      двигался внутри сценария.

    Второе слагаемое не украшение: DoE с периодами (`aios/doe.py`) отличает
    сценарий, поднявший скважину один раз навсегда, от сценария, качавшего её
    туда-обратно, — а по одному среднему они совпадают.

    Разбор 4-7 разрешает ровно такое сжатие («уровни в контрольных узлах,
    среднее, наклон, число переключений») и запрещает подавать повторённые
    месячные строки как независимые точки.
    """
    if not multipliers:
        raise ValueError("расписание не даёт ни одного месяца управления")

    keys = sorted({key for month in multipliers.values() for key in month})
    result: dict[str, float] = {}
    for key in keys:
        values = np.array([month.get(key, 1.0) for month in multipliers.values()], dtype=float)
        result[f"{key}|уровень"] = float(values.mean())
        result[f"{key}|размах"] = float(values.std())
    return result


@dataclass(frozen=True)
class TrustRegion:
    """Доверенная область: где обучающие сценарии стоят достаточно густо."""

    features: tuple[str, ...]
    """Порядок признаков сценарного представления. Часть контракта артефакта."""

    centre: tuple[float, ...]
    scale: tuple[float, ...]
    """Стандартизация, снятая **только** с обучающих сценариев."""

    points: tuple[tuple[float, ...], ...]
    scenarios: tuple[str, ...]
    k: int
    threshold: float
    quantile: float
    """Какой квантиль leave-one-out расстояний взят порогом. 1.0 — максимум."""

    frozen: tuple[tuple[str, float], ...] = ()
    """Признаки, которые DoE не двигал ни разу, и их единственное значение.

    Такой признак в метрику не входит и не может: расстояние измеряется в
    стандартных отклонениях выборки, а по этому направлению отклонения нет.
    Выборка не содержит о нём **никакой** информации, поэтому отклонение от
    единственного наблюдённого значения даёт бесконечное расстояние — не как
    трюк, а как честное «сказать нечего».

    Это же и граница между двумя контурами. Скважину, которой DoE не касался,
    первым отбрасывает жёсткое правило (`TrainingEnvelope.covers`: отрезок
    [1, 1] не содержит ничего, кроме единицы). Здесь то же решение принимается
    ещё раз и по другой причине — сценарной, а не поскважинной, — и это
    намеренное дублирование: контуры независимы, и падение одного не должно
    открывать проход.
    """

    @classmethod
    def fit(cls, profiles: dict[str, dict[str, float]], *,
            k: int = DEFAULT_K, quantile: float = 1.0) -> TrustRegion:
        """Собрать область по сценарным представлениям обучающих прогонов.

        `profiles` — `{идентификатор сценария: представление}`. Ключ словаря
        обязан быть сценарием: если сюда подать строки «скважина × месяц»,
        плотность выборки окажется завышенной в сотни раз, и порог станет
        бессмысленно узким.

        Порог считается **leave-one-out**: для каждого обучающего сценария
        расстояние до его `k`-го ближайшего **соседа среди остальных**. Порог —
        квантиль этих расстояний. Смысл: «кандидат дальше от выборки, чем самая
        одинокая точка самой выборки». Число получено из данных, а не назначено.
        """
        if k < 1:
            raise ValueError("k должно быть не меньше единицы")
        if not 0 < quantile <= 1:
            raise ValueError("quantile должен быть в (0, 1]")
        if len(profiles) <= k:
            raise ValueError(
                f"сценариев {len(profiles)}, а нужно больше k = {k}: "
                "расстояние до k-го соседа при таком наборе не определено"
            )

        scenarios = tuple(sorted(profiles))
        every = tuple(sorted({key for one in profiles.values() for key in one}))
        raw = np.array([[profiles[name].get(key, 1.0) for key in every]
                        for name in scenarios], dtype=float)

        spread = raw.std(axis=0)
        moving = spread > 0
        frozen = tuple((key, float(raw[0, position]))
                       for position, key in enumerate(every) if not moving[position])
        if not moving.any():
            raise ValueError(
                "ни одно управление не менялось между сценариями: "
                "по такой выборке расстояние не определено"
            )

        features = tuple(key for key, keep in zip(every, moving, strict=True) if keep)
        raw = raw[:, moving]
        centre = raw.mean(axis=0)
        scale = spread[moving]
        points = (raw - centre) / scale

        loo = np.sort(_pairwise(points, points), axis=1)[:, k]  # 0-й столбец — сам сценарий
        threshold = float(np.quantile(loo, quantile))

        return cls(
            features=features,
            centre=tuple(float(value) for value in centre),
            scale=tuple(float(value) for value in scale),
            points=tuple(tuple(float(value) for value in row) for row in points),
            scenarios=scenarios,
            k=k,
            threshold=threshold,
            quantile=quantile,
            frozen=frozen,
        )

    def distance(self, one: dict[str, float], *, tolerance: float = 1e-9) -> Distance:
        """Насколько далёк кандидат и какие обучающие сценарии к нему ближе всех."""
        for key, value in self.frozen:
            if abs(one.get(key, value) - value) > tolerance:
                return Distance(
                    value=float("inf"),
                    threshold=self.threshold,
                    neighbours=(),
                )

        vector = np.array([one.get(key, 1.0) for key in self.features], dtype=float)
        vector = (vector - np.array(self.centre)) / np.array(self.scale)
        gaps = _pairwise(vector[None, :], np.array(self.points))[0]
        order = np.argsort(gaps)
        return Distance(
            value=float(gaps[order[self.k - 1]]),
            threshold=self.threshold,
            neighbours=tuple(
                Neighbour(self.scenarios[index], float(gaps[index]))
                for index in order[: self.k]
            ),
        )


@dataclass(frozen=True)
class OptimismMargin:
    """Односторонний конформный запас против завышения цели суррогатом.

    Считается по парам «что обещал суррогат и что дал полный расчёт». Оценка
    `a_i = Ĵ_i − J_i` положительна там, где суррогат обещал больше, чем есть, —
    это и есть та ошибка, на которую оптимизатор давит целенаправленно.
    """

    value: float | None
    """Запас в тех же единицах, что и цель. `None` — калибровки не хватает."""

    alpha: float
    samples: int
    index: int
    """Номер порядковой статистики `⌈(n+1)(1−α)⌉`. Проверяемая часть вывода."""

    reason: str = ""
    """Почему запаса нет, если его нет."""

    @classmethod
    def fit(cls, pairs: list[tuple[float, float]], *,
            alpha: float = DEFAULT_ALPHA) -> OptimismMargin:
        """`pairs` — `[(прогноз суррогата, значение полного расчёта), …]`.

        Значение полного расчёта берётся у эталонного расчётчика, и только у
        него: калибровать защиту от ошибки суррогата по самому суррогату
        бессмысленно.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha должна быть в (0, 1)")
        samples = len(pairs)
        index = math.ceil((samples + 1) * (1 - alpha))
        if samples == 0:
            return cls(None, alpha, 0, index,
                       "нет ни одной пары «прогноз и полный расчёт»")
        if index > samples:
            return cls(
                None, alpha, samples, index,
                f"пар {samples}, а для уровня {1 - alpha:.0%} нужен {index}-й по величине "
                f"остаток — при таком наборе односторонний квантиль не существует",
            )
        scores = sorted(predicted - actual for predicted, actual in pairs)
        return cls(float(scores[index - 1]), alpha, samples, index)

    @property
    def calibrated(self) -> bool:
        return self.value is not None


def safe_value(predicted: float, distance: Distance | None, margin: OptimismMargin | None,
               *, penalty: float = 1.0, weight: float = 1.0) -> float:
    """Цель поиска вместо голого прогноза: `Ĵ − λ·B_J − ρ·max(0, d_k − τ)²`.

    `penalty` — `ρ`, цена удаления от обучающих сценариев; `weight` — `λ`, вес
    конформного запаса. Оба задаются снаружи и подбираются по внешней валидации,
    а не на тех же сценариях.

    **Некалиброванный запас не заменяется нулём.** `margin.value is None`
    означает не «риска нет», а «риск не измерен», и такой кандидат обязан идти на
    полный расчёт — это решает `routed_to_full_run`, а не эта функция. Здесь
    некалиброванный запас просто не вычитается: выдуманное число исказило бы и
    порядок кандидатов, и заявляемую величину.
    """
    value = predicted
    if margin is not None and margin.value is not None:
        value -= weight * margin.value
    if distance is not None and distance.outside:
        value -= penalty * (distance.value - distance.threshold) ** 2
    return value


def routed_to_full_run(distance: Distance | None, margin: OptimismMargin | None,
                       hard: list[str] | None = None) -> list[str]:
    """Причины, по которым кандидат обязан пойти на полный расчёт ГГДМ.

    Пустой список означает «суррогату можно доверить ранжирование» — и только
    ранжирование: финальное решение подтверждается полным расчётом всегда,
    независимо от того, что вернула эта функция.
    """
    reasons = list(hard or [])
    if distance is None:
        reasons.append("сценарное расстояние не считалось: доверенная область не собрана")
    elif distance.outside:
        reasons.append(
            f"совместный режим далёк от обучающих сценариев: {distance}"
        )
    if margin is None or not margin.calibrated:
        why = margin.reason if margin is not None else "запас не считался"
        reasons.append(f"мера доверия не калибрована ({why}); кандидат обязателен к полному расчёту")
    return reasons


def contract(region: TrustRegion | None, margin: OptimismMargin | None, *,
             dataset_hash: str = "", model_hash: str = "",
             penalty: float = 1.0, weight: float = 1.0) -> dict[str, Any]:
    """Минимальный контракт артефакта доверия — `artifacts/trust-gate.json`.

    Состав полей задан разбором 4-7: без любого из обязательных блоков интерфейс
    обязан показывать «мера доверия не калибрована», а не нейтральный зелёный
    статус. Поэтому `calibrated` считается здесь, а не решается на стороне UI.
    """
    return {
        "scenario_unit": "целый сценарий DoE; строка «скважина × месяц» единицей не является",
        "guarantee_scope": GUARANTEE_SCOPE,
        "feature_schema": list(region.features) if region else [],
        "training_scenario_ids": list(region.scenarios) if region else [],
        "distance_method": "евклидово расстояние до k-го соседа в стандартизованном пространстве",
        "k": region.k if region else None,
        "threshold_tau": region.threshold if region else None,
        "threshold_rule": (
            f"квантиль {region.quantile:.2f} leave-one-out расстояний обучающих сценариев"
            if region else None
        ),
        "conformal_alpha": margin.alpha if margin else None,
        "conformal_samples": margin.samples if margin else 0,
        "quantile_index": margin.index if margin else None,
        "optimism_margin": margin.value if margin else None,
        "margin_reason": margin.reason if margin else "запас не считался",
        "search_weights": {"lambda": weight, "rho": penalty},
        "dataset_hash": dataset_hash,
        "model_hash": model_hash,
        "calibrated": bool(region is not None and margin is not None and margin.calibrated),
    }


def _pairwise(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Матрица евклидовых расстояний. Точек десятки — считаем прямо, без индекса."""
    return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
