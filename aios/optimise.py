"""Подбор режимов работы скважин поверх суррогатной модели.

Остановить убыточную скважину — самое дешёвое улучшение ЧДД, но и самое грубое:
между «работает как есть» и «выключена» лежит весь диапазон режимов. Здесь
перебираются варианты дебита и выбирается лучший **по деньгам**, а не по
инженерному чутью.

## Почему перебор так дёшев

Тривиальный суррогат (`aios.surrogate.DeclineSurrogate`) не знает межскважинного
взаимодействия: обводнённость каждой скважины зависит только от её собственного
накопленного отбора. Из этого следует, что вклад скважины в результат **не
зависит от того, что делают остальные**, и оптимум по фонду распадается на
независимые оптимумы по скважинам. Поэтому вместо `скважины × варианты` прогонов
достаточно `варианты`: прогоняем весь фонд на каждом множителе дебита и по
каждой скважине берём лучший.

**Это свойство исчезнет вместе с тривиальным суррогатом.** Как только появится
нейросеть, обученная на прогонах ГГДМ, закачка соседа начнёт менять отклик
скважины, и раскладывать задачу на независимые куски будет нельзя — понадобится
согласование через планировщика. Здесь это записано явно, чтобы при переходе не
унаследовать молча неверное допущение.

## Чем меряем

Дисконтированной маржой скважины: сумма месячных марж (`aios.margin`), каждая со
своим годовым коэффициентом дисконтирования. Это ровно та величина, которая
складывается в дисконтированную EBITDA методики — закреплено тестом. Полный ЧДД
считается один раз, на собранном плане, и только эталонным расчётчиком.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from aios import economics
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms, month_margin
from aios.schedule import MAX_INJECTION_BHP_ATM, MAX_LIQUID_RATE_M3_PER_DAY
from aios.surrogate import Surrogate

#: Множители к дебиту базового сценария. Ноль — остановка, это тоже режим.
#:
#: Вверх по умолчанию не идём, и это не осторожность, а следствие честности
#: бейзлайна. Тривиальный суррогат не знает пластового давления: обводнённость у
#: него зависит только от накопленного отбора, поэтому удвоить дебит — значит
#: получить ту же нефть раньше, а дисконтирование за это доплачивает. Форсаж в
#: такой модели выгоден **всегда** и **всегда** необоснован: в жизни он упирается
#: в давление и резко обводняет скважину. Ставить в расписание решение, которого
#: модель не может подтвердить, — это ровно тот случай, когда суррогат ошибается
#: в нашу пользу, а платим за это мы.
DEFAULT_MULTIPLIERS = (0.0, 0.5, 0.75, 1.0)

#: Форсирование — только осознанно и только когда появится суррогат, обученный
#: на прогонах ГГДМ: `choose_rates(..., multipliers=DEFAULT_MULTIPLIERS + FORCING)`.
FORCING_MULTIPLIERS = (1.25, 1.5)
JOINT_MULTIPLIERS = DEFAULT_MULTIPLIERS + FORCING_MULTIPLIERS


@dataclass(frozen=True)
class RateChoice:
    """Выбранный режим скважины и то, чем он лучше исходного."""

    well: str
    multiplier: float
    baseline_margin_m: float
    chosen_margin_m: float

    @property
    def gain_m(self) -> float:
        return self.chosen_margin_m - self.baseline_margin_m

    @property
    def changed(self) -> bool:
        return self.multiplier != 1.0

    @property
    def before_m(self) -> float:
        return self.baseline_margin_m

    @property
    def after_m(self) -> float:
        return self.chosen_margin_m

    @property
    def joint(self) -> bool:
        return False

    @property
    def kind(self) -> str:
        return "production"

    def __str__(self) -> str:
        if not self.changed:
            return f"скв. {self.well}: режим без изменений"
        what = "остановка" if self.multiplier == 0 else f"дебит ×{self.multiplier:g}"
        return (
            f"скв. {self.well}: {what}, дисконтированная маржа "
            f"{self.baseline_margin_m:+.2f} → {self.chosen_margin_m:+.2f} млн руб."
        )


@dataclass(frozen=True)
class JointRateChoice:
    """Изменение режима, принятое по ЧДД всего взаимодействующего фонда."""

    well: str
    multiplier: float
    baseline_chdd_m: float
    chosen_chdd_m: float
    kind: str = "production"
    outside_training: tuple[float, ...] = ()
    """Кандидаты, отброшенные до выбора: таких режимов у этого управления в обучении не было."""

    @property
    def gain_m(self) -> float:
        return self.chosen_chdd_m - self.baseline_chdd_m

    @property
    def changed(self) -> bool:
        return self.multiplier != 1.0

    @property
    def before_m(self) -> float:
        return self.baseline_chdd_m

    @property
    def after_m(self) -> float:
        return self.chosen_chdd_m

    @property
    def joint(self) -> bool:
        return True

    def __str__(self) -> str:
        control = "закачка" if self.kind == "injection" else "дебит"
        what = "без изменений" if not self.changed else (
            "остановка" if self.multiplier == 0 else f"{control} ×{self.multiplier:g}"
        )
        text = (
            f"скв. {self.well}: {what}, ЧДД фонда "
            f"{self.baseline_chdd_m:+.2f} → {self.chosen_chdd_m:+.2f} млн руб."
        )
        if not self.outside_training:
            return text
        skipped = ", ".join(f"×{value:g}" for value in self.outside_training)
        return f"{text}; вне обучения не рассматривались: {skipped}"


def discounted_margin_by_well(rows: list[dict[str, Any]],
                              norms: Norms,
                              start_year: int) -> dict[str, float]:
    """Дисконтированная маржа по скважинам, млн руб.

    Коэффициент дисконтирования — годовой, как в методике: `1/(1+WACC)^(y−y₀)`.
    Годы раньше стартового не дисконтируются в обратную сторону: они просто не
    относятся к расчётному горизонту, и их вклад берётся с коэффициентом 1.
    """
    totals: dict[str, float] = {}
    for row in rows:
        year = int(str(row["DATA"])[:4])
        factor = 1.0 / (1.0 + norms.wacc_rate) ** max(year - start_year, 0)
        totals[row["well"]] = totals.get(row["well"], 0.0) + month_margin(row, norms) * factor
    return totals


def choose_rates(surrogate: Surrogate,
                 base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                 norms: Norms,
                 *,
                 start_date: date,
                 multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS) -> list[RateChoice]:
    """Перебрать множители дебита и выбрать по каждой скважине лучший.

    Возвращает решение по каждой добывающей скважине, включая «оставить как
    есть» — так видно и то, что система рассмотрела и отвергла.
    """
    if 1.0 not in multipliers:
        raise ValueError("в наборе множителей обязан быть 1.0 — режим «как есть» это база")

    scores: dict[float, dict[str, float]] = {}
    for multiplier in multipliers:
        rows = surrogate.predict(scale(base_steps, multiplier))
        scores[multiplier] = discounted_margin_by_well(rows, norms, start_date.year)

    wells = sorted(
        {control.well for step in base_steps for control in step.producers},
        key=_well_key,
    )

    choices: list[RateChoice] = []
    for well in wells:
        baseline = scores[1.0].get(well, 0.0)
        best_multiplier, best_margin = 1.0, baseline
        for multiplier in multipliers:
            margin = scores[multiplier].get(well, 0.0)
            if margin > best_margin + 1e-9:
                best_multiplier, best_margin = multiplier, margin
        choices.append(
            RateChoice(
                well=well,
                multiplier=best_multiplier,
                baseline_margin_m=baseline,
                chosen_margin_m=best_margin,
            )
        )
    return choices


def choose_joint_rates(
    surrogate: Surrogate,
    base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    *,
    start_date: date,
    multipliers: tuple[float, ...] = JOINT_MULTIPLIERS,
) -> list[JointRateChoice]:
    """Координатный поиск для суррогата с межскважинными связями.

    Скважины рассматриваются последовательно. Каждый режим проверяется при уже
    принятых режимах остальных скважин и оценивается полным эталонным ЧДД фонда.
    Это не гарантирует глобальный оптимум, но не использует неверное допущение о
    независимости скважин и ни одним шагом не ухудшает текущего кандидата.

    **Кандидат вне обучающего диапазона до выбора не доходит.** Раньше область
    применимости проверялась в `aios.loop` по уже собранному расписанию, то есть
    пометка приходила после того, как решение принято, — а поиск ищет максимум и
    тянет решение ровно туда, где суррогат ошибается в нашу пользу (`CLAUDE.md`).
    Отброшенное записывается в `JointRateChoice.outside_training` и попадает в
    объяснение решения: молча сузить перебор нельзя.
    """
    if 1.0 not in multipliers:
        raise ValueError("в наборе множителей обязан быть 1.0 — режим «как есть» это база")

    producer_wells = sorted(
        {control.well for step in base_steps for control in step.producers}, key=_well_key
    )
    injector_wells = sorted(
        {control.well for step in base_steps for control in step.injectors}, key=_well_key
    )
    controls = [
        *(("injection", well) for well in injector_wells),
        *(("production", well) for well in producer_wells),
    ]
    selected = dict.fromkeys(controls, 1.0)
    current_steps = _scale_individually(base_steps, selected)
    current_chdd = _chdd(surrogate.predict(current_steps), start_date)
    choices: list[JointRateChoice] = []

    for kind, well in controls:
        before = current_chdd
        covered, outside = _covered_multipliers(surrogate, kind, well, multipliers)
        best_multiplier = selected[(kind, well)]
        best_steps = current_steps
        best_chdd = current_chdd
        for multiplier in covered:
            candidate_multipliers = selected | {(kind, well): multiplier}
            candidate = _scale_individually(base_steps, candidate_multipliers)
            candidate_chdd = _chdd(surrogate.predict(candidate), start_date)
            if candidate_chdd > best_chdd + 1e-9:
                best_multiplier = multiplier
                best_steps = candidate
                best_chdd = candidate_chdd

        selected[(kind, well)] = best_multiplier
        current_steps = best_steps
        current_chdd = best_chdd
        choices.append(
            JointRateChoice(well, best_multiplier, before, best_chdd, kind, outside)
        )
    return choices


def _covered_multipliers(
    surrogate: Surrogate,
    kind: str,
    well: str,
    multipliers: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Разделить кандидатов на покрытых обучением и отброшенных.

    Из двух возможных правил — «не рассматривать» и «рассматривать с пометкой» —
    выбрано первое (`docs/decisions.md`, Р-43): прогноз за границей обучения это
    не менее точное число, а число без оснований, и проверить его нечем — полный
    расчёт у организаторов один, повторной попытки нет.

    Покрытие спрашивается через `getattr`, а не по типу суррогата: у тривиального
    бейзлайна `covers_control` нет вовсе, и это значит «покрытие неизвестно» —
    тогда поиск работает как раньше. Множитель 1.0 остаётся всегда: это базовый
    режим расписания и точка отсчёта, а не кандидат.
    """
    covers = getattr(surrogate, "covers_control", None)
    if covers is None:
        return multipliers, ()
    covered = tuple(value for value in multipliers
                    if value == 1.0 or covers(kind, well, value))
    return covered, tuple(value for value in multipliers if value not in covered)


def scale(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
          multiplier: float,
          wells: set[str] | None = None) -> list[ScheduleStep]:
    """Умножить дебит добывающих скважин на множитель.

    Ноль означает остановку, а не открытую скважину с нулевым дебитом: последнее
    методика считает некорректной записью, да и физического смысла в нём нет.
    Верхняя граница 500 м³/сут соблюдается здесь же — расписание, которое её
    нарушает, не должно даже собираться.
    """
    result: list[ScheduleStep] = []
    for step in steps:
        producers = []
        for control in step.producers:
            if wells is not None and control.well not in wells:
                producers.append(control)
                continue
            producers.append(_scaled(control, multiplier))
        result.append(ScheduleStep(step.date, tuple(producers), step.injectors))
    return result


def scale_injection(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    multiplier: float,
    wells: set[str] | None = None,
) -> list[ScheduleStep]:
    """Умножить приёмистость выбранных нагнетательных скважин."""
    result: list[ScheduleStep] = []
    for step in steps:
        injectors = tuple(
            control
            if wells is not None and control.well not in wells
            else _scaled_injector(control, multiplier)
            for control in step.injectors
        )
        result.append(ScheduleStep(step.date, step.producers, injectors))
    return result


def cyclic_injection(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    phase_a: set[str],
    phase_b: set[str],
    *,
    half_period: int,
    amplitude: float = 1.0,
) -> list[ScheduleStep]:
    """Циклическое заводнение: две группы нагнетателей работают противофазно.

    Организаторы рекомендуют этот режим прямо. Смысл его не в том, что закачка
    становится переменной, — так средняя закачка просто падает. Смысл в
    **перепаде давления между участками**: пока одна группа нагнетает, соседняя
    стоит, фронт вытеснения перекашивается, и вода идёт в поры, мимо которых при
    постоянной закачке проходила бы по промытым путям.

    Отсюда и устройство: на каждом полупериоде одна группа открыта, другая
    закрыта, на границе они меняются местами. Синхронная пульсация обеих групп
    отвергается — это уже не заводнение, а равномерное снижение закачки.

    Скважины вне обеих групп не трогаются: их режим задаётся другим видом
    управления, и смешивать два воздействия в одном сценарии значит не понять,
    какое из них дало отклик.

    `amplitude` умножает приёмистость работающей группы. Единица означает «та же
    закачка, но в полтора раза реже» — суммарный объём падает вдвое; чтобы
    сохранить его, нужна амплитуда около двух.
    """
    if half_period < 1:
        raise ValueError(f"полупериод задаётся в месяцах и не может быть меньше одного: {half_period}")
    overlap = phase_a & phase_b
    if overlap:
        raise ValueError(
            "группы обязаны быть в противофазе, а не пересекаться: "
            + ", ".join(sorted(overlap))
        )

    result: list[ScheduleStep] = []
    for index, step in enumerate(steps):
        first_group_works = (index // half_period) % 2 == 0
        injectors = []
        for control in step.injectors:
            if control.well in phase_a:
                works = first_group_works
            elif control.well in phase_b:
                works = not first_group_works
            else:
                injectors.append(control)
                continue
            injectors.append(_scaled_injector(control, amplitude if works else 0.0))
        result.append(ScheduleStep(step.date, step.producers, tuple(injectors)))
    return result


def convert_to_injection(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    wells: set[str],
    *,
    multiplier: float = 1.0,
    since: date | None = None,
    bhp: float | None = None,
) -> list[ScheduleStep]:
    """Перевести добывающие скважины под закачку — ВНС.

    Отличие от `scale` и `scale_injection` принципиальное: те меняют уже
    существующий режим, а здесь режим появляется. У добывающей скважины записи
    `WCONINJE` в базовом расписании нет вовсе, поэтому масштабировать нечего.

    Приёмистость берётся из её же дебита на момент перевода, умноженного на
    `multiplier`: скважина закачивает примерно столько, сколько поднимала. Это
    единственная величина, которую можно вывести из расписания, ничего не
    выдумывая; скважина, ни разу не работавшая на добычу, не переводится вовсе.

    Добыча закрывается на **каждом** последующем шаге: дек переоткрывает её
    своей записью на каждом шаге, и одной остановки не хватит. Запись закачки,
    наоборот, ставится один раз — Eclipse держит режим до следующей записи.

    Зачем это нужно суррогату. Признаки собираются по ролям, и колонка
    `inj:<скважина>` появляется только у той, у кого есть непустой множитель
    закачки. У добывающей его нет ни в одном сценарии DoE, поэтому предложение
    перевести её под закачку сегодня не искажается, а **исчезает**: модель видит
    «добыча остановлена» и предсказывает ноль, а прирост по соседям, ради
    которого ВНС и делается, не оценивается ничем.
    """
    if multiplier <= 0:
        raise ValueError(f"множитель перевода должен быть больше нуля, получено {multiplier:g}")

    pressure = bhp if bhp is not None else _injection_pressure(steps)
    last_rate: dict[str, float] = {}
    opened: set[str] = set()
    result: list[ScheduleStep] = []

    for step in steps:
        for control in step.producers:
            if control.well in wells and control.status == "OPEN" and control.liquid_rate:
                last_rate.setdefault(control.well, control.liquid_rate)

        active = since is None or step.date >= since
        producers: list[ProducerControl] = []
        injectors = list(step.injectors)
        for control in step.producers:
            if not active or control.well not in wells:
                producers.append(control)
                continue
            producers.append(_scaled(control, 0.0) if control.status == "OPEN" else control)
            rate = last_rate.get(control.well)
            if rate and control.well not in opened:
                opened.add(control.well)
                injectors.append(InjectorControl(
                    control.well, "WATER", "OPEN", "RATE", round(rate * multiplier, 4), pressure
                ))
        result.append(ScheduleStep(step.date, tuple(producers), tuple(injectors)))
    return result


def _injection_pressure(steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> float:
    """Предел давления для новой нагнетательной — как у остальных в этом деке."""
    seen = [control.bhp for step in steps for control in step.injectors if control.bhp]
    return max(seen) if seen else MAX_INJECTION_BHP_ATM


def apply_choices(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                  choices: list[RateChoice] | list[JointRateChoice]) -> list[ScheduleStep]:
    """Собрать расписание из выбранных по скважинам режимов."""
    producers = {
        choice.well: choice.multiplier for choice in choices if choice.kind == "production"
    }
    injectors = {
        choice.well: choice.multiplier for choice in choices if choice.kind == "injection"
    }
    result: list[ScheduleStep] = []
    for step in steps:
        scaled_producers = tuple(
            _scaled(control, producers.get(control.well, 1.0)) for control in step.producers
        )
        scaled_injectors = tuple(
            _scaled_injector(control, injectors.get(control.well, 1.0))
            for control in step.injectors
        )
        result.append(ScheduleStep(step.date, scaled_producers, scaled_injectors))
    return result


def explain(choices: list[RateChoice] | list[JointRateChoice]) -> str:
    """Отчёт о подборе — в лог, UI и презентацию."""
    changed = [choice for choice in choices if choice.changed]
    if not changed:
        return f"Режимы не менялись: рассмотрено {len(choices)} скважин, лучше исходного нет."

    gain = sum(choice.gain_m for choice in changed)
    stopped = sum(1 for choice in changed if choice.multiplier == 0)
    objective = (
        "прирост ЧДД фонда"
        if isinstance(choices[0], JointRateChoice)
        else "прирост дисконтированной маржи"
    )
    lines = [
        f"Изменено режимов: {len(changed)} из {len(choices)} "
        f"(из них остановок {stopped}); {objective} {gain:+.1f} млн руб.",
        "",
    ]
    # Считаем по ВСЕМ решениям, а не только по изменённым: управление, у которого
    # кандидатов отбросили, а режим остался прежним, иначе не попадёт в отчёт —
    # а знать, что выбор был неполным, важно именно там
    withheld = [c for c in choices if getattr(c, "outside_training", ())]
    if withheld:
        lines.insert(1, f"  управлений, где часть кандидатов вне обучения не "
                        f"рассматривалась: {len(withheld)} из {len(choices)}")
    lines.extend(f"  {choice}" for choice in sorted(changed, key=lambda item: -item.gain_m))
    return "\n".join(lines)


def _scale_individually(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    multipliers: dict[tuple[str, str], float],
) -> list[ScheduleStep]:
    result: list[ScheduleStep] = []
    for step in steps:
        producers = tuple(
            _scaled(control, multipliers.get(("production", control.well), 1.0))
            for control in step.producers
        )
        injectors = tuple(
            _scaled_injector(control, multipliers.get(("injection", control.well), 1.0))
            for control in step.injectors
        )
        result.append(ScheduleStep(step.date, producers, injectors))
    return result


def _chdd(rows: list[dict[str, Any]], start_date: date) -> float:
    return economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date=start_date.isoformat()
    ).chdd_m


def _scaled(control: ProducerControl, multiplier: float) -> ProducerControl:
    if multiplier == 1.0 or control.status != "OPEN":
        return control
    if multiplier == 0.0:
        return ProducerControl(control.well, "SHUT", control.mode, None, control.bhp)
    rate = (control.liquid_rate or 0.0) * multiplier
    if rate <= 0:
        return ProducerControl(control.well, "SHUT", control.mode, None, control.bhp)
    return ProducerControl(
        control.well,
        control.status,
        control.mode,
        min(rate, MAX_LIQUID_RATE_M3_PER_DAY),
        control.bhp,
    )


def _scaled_injector(control: InjectorControl, multiplier: float) -> InjectorControl:
    if multiplier == 1.0 or control.status != "OPEN":
        return control
    if multiplier == 0.0:
        return InjectorControl(
            control.well, control.fluid, "SHUT", control.mode, None, control.bhp
        )
    rate = (control.rate or 0.0) * multiplier
    if rate <= 0:
        return InjectorControl(
            control.well, control.fluid, "SHUT", control.mode, None, control.bhp
        )
    return InjectorControl(
        control.well, control.fluid, control.status, control.mode, rate, control.bhp
    )


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
