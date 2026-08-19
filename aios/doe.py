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

Роль управления выбирается явно: добывающие, нагнетательные или оба типа.

Горизонт делится на **периоды**, и внутри сценария режим меняется на их границах.
Без этого один прогон даёт суррогату ровно одну точку управления, сколько бы
месяцев в нём ни было: множитель постоянен, и 457 320 строк датасета несут
информацию двенадцати точек (измерено 19.08, `docs/experiments.md`). Пять
периодов вместо одного — пятикратный рост выборки при том же счёте, а расчёт
остаётся один. По умолчанию период один: так собраны уже посчитанные пакеты.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from aios.deck import ScheduleStep
from aios.optimise import convert_to_injection, scale, scale_injection
from aios.schedule import assert_valid

#: Диапазон множителей к дебиту базового сценария.
#:
#: Вниз — до полной остановки: это законный режим и его отклик суррогату нужен.
#: Вверх — умеренно: за историческим максимумом скважины начинается область, где
#: врёт уже сам симулятор (режим не адаптирован), да и физически форсаж упирается
#: в давление. Выборка должна покрывать то, чем мы реально собираемся управлять.
DEFAULT_RANGE = (0.0, 1.25)

#: Диапазон приёмистости переведённой скважины — доля от её же бывшего дебита.
#:
#: Ноль сюда попасть не может по смыслу: перевод под закачку с нулевой
#: приёмистостью это просто остановка, а она уже покрыта основным видом
#: управления. Половина и полтора — вокруг «закачиваем столько, сколько
#: поднимали», то есть вокруг компенсации, ради которой ВНС и делают.
CONVERSION_RANGE = (0.5, 1.5)

#: Какая доля добывающего фонда переводится под закачку в одном сценарии.
#:
#: Переводить весь фонд бессмысленно — добывать станет нечем; переводить одну
#: скважину на сотню сценариев значит не увидеть отклика соседей, ради которого
#: ВНС и делается. Доля разыгрывается отдельной осью, как и общий уровень отбора.
CONVERSION_SHARE = (0.05, 0.30)


@dataclass(frozen=True)
class ControlPeriod:
    """Режим, действующий с `start` и до начала следующего периода.

    `start` — первый месяц периода включительно; так же его читает сборщик
    датасета (`scripts/build_dataset.py`, поле `from` манифеста).
    """

    start: date
    multipliers: dict[str, float]
    injector_multipliers: dict[str, float]


@dataclass(frozen=True)
class Scenario:
    """Один сценарий управления: что покрутили и во что это собралось."""

    name: str
    seed: int
    periods: tuple[ControlPeriod, ...]
    steps: tuple[ScheduleStep, ...]

    converted_wells: tuple[str, ...] = ()
    """Скважины, переведённые из добычи под закачку в этом сценарии.

    У каждой из них в датасете стоят **обе** величины: `production_multiplier`
    равен нулю и `injection_multiplier` больше нуля. Это и есть контракт с
    моделью: колонка `inj:<скважина>` появляется только у той, у кого множитель
    закачки непустой, — без него предложение перевести скважину под закачку не
    доходит до входа вообще.

    Список собирается по факту: в нём та скважина, у которой расписание
    действительно изменилось, а не та, которую генератор собирался перевести.
    """

    @property
    def multipliers(self) -> dict[str, float]:
        """Множители первого периода — режим на начало управления.

        Оставлено ради манифеста: старые поля `multipliers` /
        `injector_multipliers` читаются сборщиком датасета до сих пор, и пакеты,
        посчитанные без периодов, обязаны собираться по-прежнему.
        """
        return self.periods[0].multipliers

    @property
    def injector_multipliers(self) -> dict[str, float]:
        return self.periods[0].injector_multipliers

    @property
    def shut_wells(self) -> tuple[str, ...]:
        return tuple(sorted(well for well, value in self.multipliers.items() if value == 0.0))

    @property
    def shut_injectors(self) -> tuple[str, ...]:
        return tuple(
            sorted(well for well, value in self.injector_multipliers.items() if value == 0.0)
        )

    def summary(self) -> str:
        values = [
            value
            for period in self.periods
            for value in (*period.multipliers.values(), *period.injector_multipliers.values())
        ]
        wells = set(self.multipliers) | set(self.injector_multipliers)
        average = sum(values) / len(values) if values else 0.0
        periods = f", периодов {len(self.periods)}" if len(self.periods) > 1 else ""
        return (
            f"{self.name}: скважин {len(wells)}, "
            f"средний множитель {average:.2f}, остановлено {len(self.shut_wells)}, "
            f"нагнетателей {len(self.injector_multipliers)}{periods}"
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
           snap_to_zero: float = 0.05,
           levels: tuple[float, float] | None = None,
           control_kind: str = "producers",
           periods: int = 1) -> list[Scenario]:
    """Собрать `samples` сценариев управления вокруг базового.

    `snap_to_zero` — множители ниже этого порога округляются до нуля: дебит в
    полпроцента от исторического физически неотличим от остановки, а вот
    экономически различие огромно (остановленная скважина не тянет содержание
    фонда), и датасет не должен размазывать эту границу.

    `levels` — диапазон **общего уровня отбора** сценария. Без него гиперкуб
    разыгрывает скважины независимо, и средний множитель по сценарию по закону
    больших чисел садится в середину `bounds` у всех точек сразу: выборка
    получается разнообразной внутри сценария и однообразной между сценариями.
    Измерено 16.08 на первых шести расчётах — разброс ЧДД вышел 5,9 % при
    множителях от 0 до 1.25 (`docs/experiments.md`). С `levels` уровень
    разыгрывается отдельной осью, а раскладка по скважинам ложится вокруг него.

    Про покрытие остановок. При диапазоне по умолчанию на ноль приходится всего
    `snap_to_zero / (high − low)` — около 4 % сценариев на скважину. Для обучения
    отклика на остановку этого мало: `coverage()` покажет долю честно, и если она
    нужна выше, стоит отдельным заходом сделать выборку в узком диапазоне
    (`bounds=(0.0, 0.3)`) и слить её с основной. Разбираться с этим надо до
    прогонов, а не после: сотня расчётов по несколько часов — слишком дорогая
    цена за дыру в датасете.

    `periods` — на сколько кусков делится горизонт. Внутри куска режим постоянен,
    на границе меняется, и каждый кусок разыгрывается своей частью гиперкуба
    вместе со своим уровнем отбора. Отсюда весь смысл: один прогон даёт суррогату
    `periods` точек управления вместо одной. Куски равны по числу месячных шагов,
    а не по календарю: шаги в деке месячные, разница несущественна, зато точки
    распределены по выборке ровно.
    """
    producer_wells = sorted(
        {control.well for step in base_steps for control in step.producers}, key=_well_key
    )
    injector_wells = sorted(
        {control.well for step in base_steps for control in step.injectors}, key=_well_key
    )
    if control_kind not in {"producers", "injectors", "all", "conversions"}:
        raise ValueError(f"неизвестный тип управления: {control_kind}")
    if control_kind == "conversions":
        if periods != 1:
            raise ValueError(
                "перевод под закачку — разовое событие, делить его на периодов "
                f"больше одного нечем (запрошено {periods})"
            )
        return _design_conversions(base_steps, producer_wells,
                                   samples=samples, seed=seed, bounds=bounds)

    starts = _period_starts(base_steps, periods)

    coordinates = []
    if control_kind in {"producers", "all"}:
        coordinates.extend(("production", well) for well in producer_wells)
    if control_kind in {"injectors", "all"}:
        coordinates.extend(("injection", well) for well in injector_wells)
    if not coordinates:
        role = "добывающих" if control_kind == "producers" else "нагнетательных"
        raise ValueError(f"в базовом расписании нет {role} скважин")

    low, high = bounds
    if not 0.0 <= low < high:
        raise ValueError(f"некорректный диапазон множителей: {bounds}")

    if levels is not None:
        level_low, level_high = levels
        if not 0.0 <= level_low < level_high:
            raise ValueError(f"некорректный диапазон уровней: {levels}")

    # +1 ось под общий уровень сценария: без неё гиперкуб независимо разыгрывает
    # каждую скважину, среднее по 70+ осям садится в середину диапазона, и все
    # сценарии оказываются на одном уровне отбора. Разнообразие получается внутри
    # сценария, но не между ними — а суррогату нужно именно второе.
    #
    # Своя такая пачка осей на каждый период: иначе режим на границе не изменится,
    # и периоды станут записью в манифесте, за которой ничего не стоит.
    width = len(coordinates) + 1
    points = latin_hypercube(samples, width * periods, seed)
    scenarios: list[Scenario] = []

    for index, point in enumerate(points):
        control: list[ControlPeriod] = []
        for number, start in enumerate(starts):
            axes = point[number * width:(number + 1) * width]
            level = 1.0
            if levels is not None:
                level_low, level_high = levels
                level = level_low + axes[-1] * (level_high - level_low)

            multipliers: dict[str, float] = {}
            injector_multipliers: dict[str, float] = {}
            for (kind, well), value in zip(coordinates, axes[:-1], strict=True):
                multiplier = (low + value * (high - low)) * level
                target = multipliers if kind == "production" else injector_multipliers
                target[well] = 0.0 if multiplier < snap_to_zero else round(multiplier, 4)
            control.append(ControlPeriod(start, multipliers, injector_multipliers))

        steps = _apply_periods(base_steps, control)
        assert_valid(steps)
        scenarios.append(
            Scenario(
                name=f"doe-{index:03d}",
                seed=seed,
                periods=tuple(control),
                steps=tuple(steps),
            )
        )
    return scenarios


def _design_conversions(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                        producer_wells: list[str],
                        *,
                        samples: int,
                        seed: int,
                        bounds: tuple[float, float]) -> list[Scenario]:
    """Сценарии перевода добывающих скважин под закачку.

    Три оси, и они разделены намеренно: какие скважины переводить, сколько их и с
    какой приёмистостью. Первое разыгрывается гиперкубом по скважинам — точка
    задаёт порядок, и переводятся первые в нём. Второе и третье — двумя
    отдельными осями на сценарий, поэтому «перевели пять скважин с приёмистостью
    ×1.2» читается прямо и повторяется по seed.

    Приёмистость одна на сценарий, а не своя у каждой скважины. Это осознанное
    упрощение первого датасета: суррогату сейчас нужен сам факт, что перевод
    вообще доходит до входа, а не тонкая раскладка по скважинам. Разные
    приёмистости внутри одного сценария — следующий заход, как и временные
    профили у основного вида управления.

    В кандидаты идёт не всякая скважина с записью `WCONPROD`, а только та, что
    в горизонте управления действительно добывает. В Модели Z 30 скважин
    переведены под закачку за историю, и у части из них в горизонте остаётся
    одна закрывающая запись «SHUT, дебит 0» — переводить там нечего, а
    `converted_wells` при этом называл их переведёнными.
    """
    producing = _producing(base_steps)
    candidates = [well for well in producer_wells if well in producing]
    if not candidates:
        raise ValueError("в горизонте управления нет работающих добывающих скважин")
    if bounds == DEFAULT_RANGE:
        bounds = CONVERSION_RANGE

    low, high = bounds
    if not 0.0 < low < high:
        raise ValueError(f"приёмистость перевода должна быть больше нуля: {bounds}")

    points = latin_hypercube(samples, len(candidates) + 2, seed)
    share_low, share_high = CONVERSION_SHARE
    scenarios: list[Scenario] = []

    for index, point in enumerate(points):
        share = share_low + point[-2] * (share_high - share_low)
        multiplier = round(low + point[-1] * (high - low), 4)
        count = max(1, round(share * len(candidates)))

        order = sorted(zip(candidates, point[:-2], strict=True), key=lambda item: item[1])
        chosen = {well for well, _ in order[:count]}

        steps = convert_to_injection(base_steps, chosen, multiplier=multiplier)
        assert_valid(steps)

        # Названа переведённой только та, у которой расписание действительно
        # стало другим. Сборщик датасета верит этому списку на слово и пишет по
        # нему `production_multiplier = 0` и `injection_multiplier > 0`; строка
        # «скважину перевели» с опорным откликом учит сеть тому, что перевод
        # ничего не даёт, — ровно наоборот тому, ради чего пакет считается.
        converted = tuple(sorted(_changed_wells(base_steps, steps), key=_well_key))
        multipliers = dict.fromkeys(converted, 0.0)
        injector_multipliers = dict.fromkeys(converted, multiplier)
        scenarios.append(
            Scenario(
                name=f"doe-{index:03d}",
                seed=seed,
                periods=(ControlPeriod(steps[0].date, multipliers, injector_multipliers),),
                steps=tuple(steps),
                converted_wells=converted,
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

    pooled: dict[str, list[float]] = {}
    by_period: list[dict[str, list[float]]] = [{} for _ in scenarios[0].periods]
    for scenario in scenarios:
        for number, period in enumerate(scenario.periods):
            for well, multiplier in period.multipliers.items():
                pooled.setdefault(f"prod:{well}", []).append(multiplier)
                by_period[number].setdefault(f"prod:{well}", []).append(multiplier)
            for well, multiplier in period.injector_multipliers.items():
                pooled.setdefault(f"inj:{well}", []).append(multiplier)
                by_period[number].setdefault(f"inj:{well}", []).append(multiplier)

    report = {
        "scenarios": len(scenarios),
        "converted_total": sum(len(s.converted_wells) for s in scenarios),
        **_spans(pooled),
    }
    # Общий размах по пакету скрывает провал в отдельном периоде: сценариев в
    # каждом столько же, сколько во всём пакете, а точек по скважине в нём — свои
    if len(by_period) > 1:
        report["periods"] = [
            {"from": scenarios[0].periods[number].start.isoformat(), **_spans(values)}
            for number, values in enumerate(by_period)
        ]
    return report


def _spans(per_well: dict[str, list[float]]) -> dict[str, Any]:
    """Размах и доля остановок по каждой скважине — свёрнутые в минимум и среднее."""
    spans = [max(values) - min(values) for values in per_well.values()]
    shut_share = [
        sum(1 for value in values if value == 0.0) / len(values) for values in per_well.values()
    ]
    return {
        "wells": len(per_well),
        "min_span": min(spans),
        "mean_span": sum(spans) / len(spans),
        "min_shut_share": min(shut_share),
        "max_shut_share": max(shut_share),
    }


def _period_starts(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   periods: int) -> list[date]:
    """Даты начала периодов: горизонт делится на равные по числу шагов куски.

    Первый период начинается на первом же шаге, то есть на `control_start`, —
    этого требует контракт манифеста: сборщик датасета берёт множители первого
    периода для всех месяцев до начала второго.
    """
    if periods < 1:
        raise ValueError(f"периодов должно быть хотя бы один, запрошено {periods}")

    dates = sorted({step.date for step in base_steps})
    if periods > len(dates):
        raise ValueError(
            f"периодов больше, чем шагов в горизонте: {periods} > {len(dates)}"
        )
    size = len(dates) / periods
    return [dates[int(number * size)] for number in range(periods)]


def _apply_periods(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   periods: list[ControlPeriod]) -> list[ScheduleStep]:
    """Применить режим каждого периода к его куску расписания.

    Период действует с своей даты и до начала следующего; последний — до конца
    горизонта. Куски собираются обратно в одно расписание по порядку.
    """
    ordered = sorted(base_steps, key=lambda step: step.date)
    result: list[ScheduleStep] = []
    for number, period in enumerate(periods):
        end = periods[number + 1].start if number + 1 < len(periods) else None
        chunk = [
            step for step in ordered
            if step.date >= period.start and (end is None or step.date < end)
        ]
        result.extend(_apply(chunk, period.multipliers, period.injector_multipliers))
    return result


def _apply(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
           multipliers: dict[str, float],
           injector_multipliers: dict[str, float]) -> list[ScheduleStep]:
    steps = list(base_steps)
    for well, multiplier in multipliers.items():
        if multiplier != 1.0:
            steps = scale(steps, multiplier, wells={well})
    for well, multiplier in injector_multipliers.items():
        if multiplier != 1.0:
            steps = scale_injection(steps, multiplier, wells={well})
    return steps


def _producing(steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> set[str]:
    """Скважины, которые в этом горизонте действительно добывают.

    Роль берётся по горизонту управления, а не по всей истории дека: скважина,
    переведённая под закачку до `control_start`, добывающей уже не является,
    сколько бы лет она ни качала нефть до того.
    """
    return {
        control.well
        for step in steps
        for control in step.producers
        if control.status == "OPEN" and control.liquid_rate
    }


def _changed_wells(base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   steps: list[ScheduleStep]) -> set[str]:
    """Скважины, у которых записи расписания отличаются от опорных.

    Сравнение прямое, запись к записи по ключу «дата × роль × скважина». Так
    «перевели» означает ровно то, что видно в файле, а не то, что собирался
    сделать генератор.
    """
    def records(items: Any) -> dict[tuple[date, str, str], Any]:
        return {
            (step.date, kind, control.well): control
            for step in items
            for kind, controls in (("prod", step.producers), ("inj", step.injectors))
            for control in controls
        }

    before = records(base_steps)
    return {
        key[-1] for key, control in records(steps).items() if before.get(key) != control
    }


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
