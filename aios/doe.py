"""Планирование эксперимента: сценарии управления для обучения суррогата.

Данные для суррогатной модели не даны заранее — их надо породить, прогнав дорогой
симулятор в заранее выбранных точках пространства управления. От того, как
выбраны точки, зависит всё: чего суррогат не увидел, того он и не выучит, а на
незнакомом режиме будет уверенно врать.

Берём **латинский гиперкуб**: каждая скважина получает свой множитель дебита, и
выборка устроена так, что по каждой скважине значения покрывают весь диапазон
равномерно, без сгущений и провалов, которые даёт простой случайный выбор. При
сотне прогонов на сотню скважин это принципиально: случайная выборка такой
размерности оставляет дыры.

Вся случайность — из одного seed в `config.toml`. Регламент прямо запрещает
неконтролируемые случайные параметры, и это ровно тот случай, когда «у нас
получилось 96 %» через неделю невоспроизводимо.

Что здесь **не** варьируется и почему: режим каждой скважины постоянен на всём
горизонте. Это сознательное упрощение первого датасета — оно даёт чистую связь
«уровень отбора → отклик», на которой суррогат учится базовой физике. Временны́е
профили (циклическое заводнение, переводы под закачку) — следующий датасет, и
делать его надо, когда первый уже показал, что контур работает.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from aios.deck import ScheduleStep
from aios.optimise import scale
from aios.schedule import assert_valid

#: Диапазон множителей к дебиту базового сценария.
#:
#: Вниз — до полной остановки: это законный режим и его отклик суррогату нужен.
#: Вверх — умеренно: за историческим максимумом скважины начинается область, где
#: врёт уже сам симулятор (режим не адаптирован), да и физически форсаж упирается
#: в давление. Выборка должна покрывать то, чем мы реально собираемся управлять.
DEFAULT_RANGE = (0.0, 1.25)


@dataclass(frozen=True)
class Scenario:
    """Один сценарий управления: что покрутили и во что это собралось."""

    name: str
    seed: int
    multipliers: dict[str, float]
    steps: tuple[ScheduleStep, ...]

    @property
    def shut_wells(self) -> tuple[str, ...]:
        return tuple(sorted(well for well, value in self.multipliers.items() if value == 0.0))

    def summary(self) -> str:
        values = list(self.multipliers.values())
        average = sum(values) / len(values) if values else 0.0
        return (
            f"{self.name}: скважин {len(self.multipliers)}, "
            f"средний множитель {average:.2f}, остановлено {len(self.shut_wells)}"
        )


def latin_hypercube(samples: int, dimensions: int, seed: int) -> list[list[float]]:
    """Латинский гиперкуб в единичном кубе: `samples` точек, `dimensions` осей.

    По каждой оси диапазон делится на `samples` равных страт, и в каждой страте
    оказывается ровно одна точка. Порядок страт по осям перемешивается
    независимо — так покрытие остаётся равномерным по каждой переменной, сколько
    бы их ни было.
    """
    if samples < 1 or dimensions < 1:
        raise ValueError("нужен хотя бы один сценарий и хотя бы одна скважина")

    import numpy as np

    generator = np.random.default_rng(seed)
    strata = (np.arange(samples)[:, None] + generator.random((samples, dimensions))) / samples
    for axis in range(dimensions):
        strata[:, axis] = generator.permutation(strata[:, axis])
    return strata.tolist()


def design(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
           *,
           samples: int,
           seed: int,
           bounds: tuple[float, float] = DEFAULT_RANGE,
           snap_to_zero: float = 0.05) -> list[Scenario]:
    """Собрать `samples` сценариев управления вокруг базового.

    `snap_to_zero` — множители ниже этого порога округляются до нуля: дебит в
    полпроцента от исторического физически неотличим от остановки, а вот
    экономически различие огромно (остановленная скважина не тянет содержание
    фонда), и датасет не должен размазывать эту границу.

    Про покрытие остановок. При диапазоне по умолчанию на ноль приходится всего
    `snap_to_zero / (high − low)` — около 4 % сценариев на скважину. Для обучения
    отклика на остановку этого мало: `coverage()` покажет долю честно, и если она
    нужна выше, стоит отдельным заходом сделать выборку в узком диапазоне
    (`bounds=(0.0, 0.3)`) и слить её с основной. Разбираться с этим надо до
    прогонов, а не после: сотня расчётов по несколько часов — слишком дорогая
    цена за дыру в датасете.
    """
    wells = sorted(
        {control.well for step in base_steps for control in step.producers}, key=_well_key
    )
    if not wells:
        raise ValueError("в базовом расписании нет добывающих скважин")

    low, high = bounds
    if not 0.0 <= low < high:
        raise ValueError(f"некорректный диапазон множителей: {bounds}")

    points = latin_hypercube(samples, len(wells), seed)
    scenarios: list[Scenario] = []

    for index, point in enumerate(points):
        multipliers = {}
        for well, value in zip(wells, point, strict=True):
            multiplier = low + value * (high - low)
            multipliers[well] = 0.0 if multiplier < snap_to_zero else round(multiplier, 4)

        steps = _apply(base_steps, multipliers)
        assert_valid(steps)
        scenarios.append(
            Scenario(
                name=f"doe-{index:03d}",
                seed=seed,
                multipliers=multipliers,
                steps=tuple(steps),
            )
        )
    return scenarios


def write_cases(scenarios: list[Scenario],
                run_root: str | Path,
                model_dir: str | Path,
                *,
                control_start: date) -> list[Path]:
    """Собрать расчётный кейс под каждый сценарий. Возвращает пути к `.DATA`."""
    from aios.run import build_case

    root = Path(run_root)
    written: list[Path] = []
    for scenario in scenarios:
        case = build_case(
            root / scenario.name,
            model_dir,
            steps=list(scenario.steps),
            control_start=control_start,
        )
        written.append(case.data_file)
    return written


def coverage(scenarios: list[Scenario]) -> dict[str, Any]:
    """Насколько выборка покрыла пространство — смотреть до того, как считать.

    Сотня прогонов по несколько часов каждый стоит слишком дорого, чтобы
    обнаружить провал в покрытии после них.
    """
    if not scenarios:
        return {"scenarios": 0}

    per_well: dict[str, list[float]] = {}
    for scenario in scenarios:
        for well, multiplier in scenario.multipliers.items():
            per_well.setdefault(well, []).append(multiplier)

    spans = [max(values) - min(values) for values in per_well.values()]
    shut_share = [
        sum(1 for value in values if value == 0.0) / len(values) for values in per_well.values()
    ]
    return {
        "scenarios": len(scenarios),
        "wells": len(per_well),
        "min_span": min(spans),
        "mean_span": sum(spans) / len(spans),
        "min_shut_share": min(shut_share),
        "max_shut_share": max(shut_share),
    }


def _apply(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
           multipliers: dict[str, float]) -> list[ScheduleStep]:
    steps = list(base_steps)
    for well, multiplier in multipliers.items():
        if multiplier != 1.0:
            steps = scale(steps, multiplier, wells={well})
    return steps


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
