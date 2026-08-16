"""Выбор суррогата для цикла и проверенная загрузка обученного артефакта."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from aios.deck import ScheduleStep
from aios.neural import NeuralBundle, NeuralSurrogate
from aios.summary import Fluids
from aios.surrogate import DeclineSurrogate, Surrogate


class SurrogateLoadError(RuntimeError):
    """Артефакт модели отсутствует, повреждён или несовместим."""


@dataclass(frozen=True)
class CycleSurrogate:
    model: Surrogate
    name: str
    joint_rate_search: bool


def for_cycle(
    *,
    neural_path: Path | None,
    history: list[dict],
    reference_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    fluids: Fluids,
) -> CycleSurrogate:
    """Вернуть явно выбранную модель; `None` означает проверенный бейзлайн."""
    if neural_path is None:
        return CycleSurrogate(
            DeclineSurrogate.fit(history, fluids=fluids), "тривиальный бейзлайн", False
        )
    if not neural_path.is_file():
        raise SurrogateLoadError(f"модель не найдена: {neural_path}")

    try:
        bundle = joblib.load(neural_path)
        if not isinstance(bundle, NeuralBundle):
            raise TypeError(f"ожидался NeuralBundle, получен {type(bundle).__name__}")
        surrogate = NeuralSurrogate.create(bundle, reference_steps, fluids)
    except Exception as error:
        raise SurrogateLoadError(
            f"не удалось загрузить проверенный суррогат {neural_path}: {error}"
        ) from error
    return CycleSurrogate(surrogate, "MLP по сценариям DoE", True)
