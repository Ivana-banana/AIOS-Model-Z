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

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from aios import economics, planner, trust
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms, counted_by_methodology, month_margin
from aios.schedule import MAX_INJECTION_BHP_ATM, MAX_LIQUID_RATE_M3_PER_DAY
from aios.summary import Fluids
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

TRUST_PENALTY = 1.0
"""`ρ` — цена удаления кандидата от обучающих сценариев, млн руб. за (d − τ)².

Единица — не результат измерения, а честное «вес не подбирали». Разбор 4-7
требует подбирать `λ` и `ρ` по внешней валидации, а не на тех же сценариях; до
этого замера значение единичное и записано в артефакт доверия, чтобы влияние
штрафа на найденный режим можно было пересчитать.
"""

TRUST_WEIGHT = 1.0
"""`λ` — вес конформного запаса. Единица означает «вычитаем весь измеренный запас»."""


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

    since: date | None = None
    """С какой даты действует режим. `None` — весь горизонт, один режим на скважину.

    Появляется, когда поиск ведётся по периодам: границы те же, что у DoE
    (`aios.doe`), поэтому найденное расписание выражается в тех же величинах, на
    которых обучалась модель.
    """

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
        period = f" с {self.since:%Y-%m-%d}" if self.since is not None else ""
        text = (
            f"скв. {self.well}{period}: {what}, ЧДД фонда "
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
        # та же причина, что и в `margin.screen`: строку, которую методика
        # выбрасывает целиком, нельзя класть в целевую функцию поиска
        if not counted_by_methodology(row):
            continue
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
    periods: int = 1,
    limits: planner.Limits | None = None,
    norms: Norms | None = None,
    fluids: Fluids | None = None,
    region: trust.TrustRegion | None = None,
    margin: trust.OptimismMargin | None = None,
    penalty: float = TRUST_PENALTY,
    weight: float = TRUST_WEIGHT,
    norms_file: Path = economics.NORMS_FILE,
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

    **Общий ресурс учитывается до выбора, а не после.** `limits` — те же
    ограничения уровня объекта, которыми `aios.planner` согласует предложения
    скважинных агентов. Когда они переданы, кандидат оценивается по ЧДД
    расписания, **уже приведённого** к ним: сначала срезка, потом деньги. Без
    этого поиск выбирает режим по числу, которого не будет, — вода на закачку
    принадлежит объекту целиком, и прибавка одной скважины отнимается у соседей.

    Замер 21.08 на Модели Z (обученный суррогат, 113 управлений): поиск выбрал
    режимы на 110 391,5 млн руб., после планировщика осталось 84 740,8 — минус
    25 650,8, то есть 23,2 % заявленного прироста. Планировщик изменил 23
    нагнетательных режима, 22 из них поиск считал своим решением
    (`docs/experiments.md`, Р-53).

    Цена согласования — второй `predict` на кандидата: расписание считается до
    срезки (планировщику нужен отклик, чтобы знать, сколько воды поднято) и
    после. Без `limits` поведение прежнее, лишних расчётов нет.

    **`periods` — на сколько кусков горизонта режим может различаться.** Единица,
    значение по умолчанию, означает сегодняшнее поведение: один множитель на все
    19 лет. Между тем пакет `doe-periods` учил модель управлению, меняющемуся по
    пяти периодам, — то есть пространство поиска было беднее обучающего, и часть
    того, что суррогат умеет предсказывать, поиску была недоступна. Границы
    берутся тем же делением, что у DoE, иначе найденный режим не выражался бы в
    величинах, на которых модель обучалась.

    Счёт растёт прямо пропорционально: управлений × периодов × множителей полных
    прогнозов фонда. Замер 21.08 на Модели Z: 113 × 6 прогнозов — 509 с, то есть
    пять периодов стоят около сорока минут, а с согласованием — вдвое дороже.
    """
    if 1.0 not in multipliers:
        raise ValueError("в наборе множителей обязан быть 1.0 — режим «как есть» это база")
    if limits is not None and norms is None:
        raise ValueError(
            "с ограничениями объекта нужны нормативы: планировщик режет по удельной "
            "марже, и без неё ему нечем решать, кого"
        )

    producer_wells = sorted(
        {control.well for step in base_steps for control in step.producers}, key=_well_key
    )
    injector_wells = sorted(
        {control.well for step in base_steps for control in step.injectors}, key=_well_key
    )
    starts = _period_starts(base_steps, periods) if periods > 1 else [None]
    spans = _period_spans(base_steps, starts)
    controls = [
        (kind, well, start)
        for start in starts
        for kind, well in (
            *(("injection", well) for well in injector_wells),
            *(("production", well) for well in producer_wells),
        )
    ]
    selected = dict.fromkeys(controls, 1.0)
    current_steps = _scale_individually(base_steps, selected)
    current_chdd = _value_of(surrogate, current_steps, start_date, limits, norms, fluids,
                             region, margin, penalty, weight, norms_file)
    choices: list[JointRateChoice] = []

    for kind, well, start in controls:
        before = current_chdd
        # покрытие спрашивается про тот кусок расписания, который масштабирует этот
        # кандидат (Р-50): множитель периода меняет режим только внутри его месяцев
        covered, outside = _covered_multipliers(
            surrogate, kind, well, multipliers, spans[start]
        )
        best_multiplier = selected[(kind, well, start)]
        best_steps = current_steps
        best_chdd = current_chdd
        for multiplier in covered:
            candidate_multipliers = selected | {(kind, well, start): multiplier}
            candidate = _scale_individually(base_steps, candidate_multipliers)
            candidate_chdd = _value_of(surrogate, candidate, start_date, limits, norms,
                                       fluids, region, margin, penalty, weight, norms_file)
            if candidate_chdd > best_chdd + 1e-9:
                best_multiplier = multiplier
                best_steps = candidate
                best_chdd = candidate_chdd

        selected[(kind, well, start)] = best_multiplier
        current_steps = best_steps
        current_chdd = best_chdd
        choices.append(
            JointRateChoice(well, best_multiplier, before, best_chdd, kind, outside, start)
        )
    return choices


def _value_of(
    surrogate: Surrogate,
    steps: list[ScheduleStep],
    start_date: date,
    limits: planner.Limits | None,
    norms: Norms | None,
    fluids: Fluids | None,
    region: trust.TrustRegion | None = None,
    margin: trust.OptimismMargin | None = None,
    penalty: float = TRUST_PENALTY,
    weight: float = TRUST_WEIGHT,
    norms_file: Path = economics.NORMS_FILE,
) -> float:
    """Чего кандидат стоит — уже с учётом общего ресурса, если он задан.

    Порядок именно такой: планировщику нужен отклик на предложенное расписание,
    чтобы знать, сколько воды поднято, поэтому прогноз считается дважды — до
    срезки и после неё. Деньги берутся со второго: первое расписание
    неисполнимо, и его ЧДД не значит ничего.
    """
    forecast = surrogate.predict(steps)
    if limits is None:
        value, judged = _chdd(forecast, start_date, norms_file), steps
    else:
        fitted, _decisions = planner.enforce(steps, forecast, norms, limits, fluids)
        value, judged = _chdd(surrogate.predict(fitted), start_date, norms_file), fitted
    return trust.safe_value(value, _distance_of(surrogate, judged, region), margin,
                            penalty=penalty, weight=weight)


def _distance_of(surrogate: Surrogate, steps: list[ScheduleStep],
                 region: trust.TrustRegion | None) -> trust.Distance | None:
    """Насколько совместный режим кандидата далёк от обучающих сценариев.

    Спрашивается у суррогата тем же способом, что и покрытие, — через `getattr`:
    у тривиального бейзлайна сценарного представления нет, и это значит «мера не
    считается», а не «кандидат близок». Без области поиск работает как раньше.

    Меряется расписание **после** планировщика, если ограничения заданы: именно
    оно уходит в расчёт, и именно его совместный режим имеет значение. До срезки
    расписание может быть сколь угодно экзотическим — оно всё равно не исполнится.
    """
    if region is None:
        return None
    build = getattr(surrogate, "control_profile", None)
    if build is None:
        return None
    one = build(steps)
    return region.distance(one) if one else None


def _covered_multipliers(
    surrogate: Surrogate,
    kind: str,
    well: str,
    multipliers: tuple[float, ...],
    base_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
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

    Спрашивается покрытие **про масштабируемое расписание**, а не про голый
    множитель: поиск отсчитывает его от `base_steps`, а обучающие множители DoE
    отсчитаны от опорного расписания дека. Совпадают они только когда цикл идёт
    поверх исторического расписания; поверх форсированного прогона голый
    множитель означал не тот режим, и фильтр пропускал экстраполяцию, которую
    детектор на собранном расписании затем честно помечал.
    """
    covers = getattr(surrogate, "covers_control", None)
    if covers is None:
        return multipliers, ()
    covered = tuple(value for value in multipliers
                    if value == 1.0 or covers(kind, well, value, base_steps))
    return covered, tuple(value for value in multipliers if value not in covered)


#: Режимы, в которых поле расхода — уставка, а не предел. Только к ним множитель
#: применим буквально: под `LRAT` симулятор держит заданный дебит жидкости, под
#: `RATE` — заданную приёмистость. В остальных режимах (`BHP`, `THP`, `GRUP`,
#: `ORAT`, `WRAT`, `RESV`) то же поле означает верхнюю границу, и умножение
#: ослабляет ограничение, ничего не форсируя.
SETPOINT_MODES = {"producer": "LRAT", "injector": "RATE"}


@dataclass(frozen=True)
class ScalingReport:
    """Что на самом деле стало с записями режимов при масштабировании.

    Ярлык «×1.5» описывает намерение, а не файл. На Модели Z разницы нет: все
    записи под `LRAT`/`RATE` и ни одна не упирается в предел (замер 23.08 —
    13 382 добывающих и 8 212 нагнетательных записей, все с уставкой). На чужом
    деке валидационного кейса (ТЗ п. 4.3) она появляется сразу, поэтому доля
    буквально применённых записей — величина, которую проверяет машина, а не
    строка, которую человек прочитает в логе (Б29).
    """

    records: int
    """Записей, к которым множитель применялся (открытые, множитель ≠ 1)."""

    literal: int
    """Из них получивших ровно заявленный множитель."""

    capped: int
    """Упёрлись в предел дебита жидкости — множитель вышел меньше заявленного."""

    off_mode: dict[str, int]
    """Режим не уставочный: поле расхода — предел. Ключ — режим."""

    without_rate: dict[str, int]
    """Поля расхода в записи нет вовсе: умножать нечего, запись оставлена как есть."""

    @property
    def share(self) -> float:
        """Доля записей, к которым множитель применён буквально. 1.0 — все."""
        return self.literal / self.records if self.records else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "literal": self.literal,
            "capped": self.capped,
            "off_mode": dict(sorted(self.off_mode.items())),
            "without_rate": dict(sorted(self.without_rate.items())),
            "share": self.share,
        }


def scaling_report(before: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   after: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   *, production: float | Mapping[str, float],
                   injection: float | Mapping[str, float]) -> ScalingReport:
    """Сравнить расписание до и после масштабирования запись за записью.

    Считаются только открытые записи и только те, к которым множитель вообще
    применялся: при множителе 1.0 сравнивать нечего.

    Множитель может быть общим числом или **картой «скважина → множитель»**:
    сценарий «просить по достигнутому» (Р-90) назначает каждой скважине свой,
    и проверять его общим числом значило бы считать умышленное снижение
    просьбы потерей множителя. Скважина, которой карта ничего не назначила,
    в отчёт не входит — к ней множитель не применялся.
    """
    records = literal = capped = 0
    off_mode: dict[str, int] = {}
    without_rate: dict[str, int] = {}

    def note(counter: dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    for step_before, step_after in zip(before, after, strict=True):
        pairs: list[tuple[str, float, object, object]] = [
            ("producer", _multiplier_of(production, control_before.well),
             control_before, control_after)
            for control_before, control_after in zip(step_before.producers,
                                                     step_after.producers, strict=True)
        ]
        pairs += [
            ("injector", _multiplier_of(injection, control_before.well),
             control_before, control_after)
            for control_before, control_after in zip(step_before.injectors,
                                                     step_after.injectors, strict=True)
        ]

        for kind, multiplier, control_before, control_after in pairs:
            if control_before.status != "OPEN" or multiplier == 1.0:
                continue
            records += 1
            rate_before = (control_before.liquid_rate if kind == "producer"
                           else control_before.rate)
            rate_after = (control_after.liquid_rate if kind == "producer"
                          else control_after.rate)
            if rate_before is None:
                note(without_rate, f"{control_before.mode}")
                continue
            if control_before.mode != SETPOINT_MODES[kind]:
                note(off_mode, f"{control_before.mode}")
                continue
            if rate_after is not None and rate_before * multiplier > rate_after + 1e-9:
                capped += 1
                continue
            literal += 1

    return ScalingReport(records=records, literal=literal, capped=capped,
                         off_mode=off_mode, without_rate=without_rate)


def _multiplier_of(spec: float | Mapping[str, float], well: str) -> float:
    """Множитель записи: общий или назначенный этой скважине картой."""
    if isinstance(spec, Mapping):
        return float(spec.get(well, 1.0))
    return float(spec)

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
    selected = {
        (choice.kind, choice.well, getattr(choice, "since", None)): choice.multiplier
        for choice in choices
    }
    return _scale_individually(steps, selected)


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
    multipliers: dict[tuple[str, str, date | None], float],
) -> list[ScheduleStep]:
    """Применить выбранные множители, каждый — со своей даты и до следующей."""
    by_control: dict[tuple[str, str], list[tuple[date | None, float]]] = {}
    for (kind, well, start), multiplier in multipliers.items():
        by_control.setdefault((kind, well), []).append((start, multiplier))
    for entries in by_control.values():
        entries.sort(key=lambda entry: entry[0] or date.min)

    result: list[ScheduleStep] = []
    for step in steps:
        producers = tuple(
            _scaled(control, _multiplier_at(by_control, "production", control.well, step.date))
            for control in step.producers
        )
        injectors = tuple(
            _scaled_injector(
                control, _multiplier_at(by_control, "injection", control.well, step.date)
            )
            for control in step.injectors
        )
        result.append(ScheduleStep(step.date, producers, injectors))
    return result


def _multiplier_at(
    by_control: dict[tuple[str, str], list[tuple[date | None, float]]],
    kind: str,
    well: str,
    when: date,
) -> float:
    """Множитель, действующий на эту дату: последний из назначенных не позже неё."""
    value = 1.0
    for start, multiplier in by_control.get((kind, well), ()):
        if start is None or start <= when:
            value = multiplier
    return value


def _period_spans(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
    starts: list[date | None],
) -> dict[date | None, list[ScheduleStep]]:
    """Куски расписания по периодам: с даты начала и до следующей."""
    if starts == [None]:
        return {None: list(steps)}
    bounds = [*starts[1:], None]
    return {
        start: [step for step in steps
                if step.date >= start and (end is None or step.date < end)]
        for start, end in zip(starts, bounds, strict=True)
    }


def _period_starts(steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                   periods: int) -> list[date]:
    """Начала периодов — тем же делением, что у DoE.

    Функция берётся из `aios.doe`, а не пишется заново: границы периодов обязаны
    совпадать с теми, по которым размечена обучающая выборка, иначе поиск
    выражает режим не в тех величинах, в которых учился суррогат. Две копии
    одного правила разошлись бы молча.
    """
    from aios.doe import _period_starts as doe_period_starts

    return doe_period_starts(steps, periods)


def _chdd(rows: list[dict[str, Any]], start_date: date,
          norms_file: Path = economics.NORMS_FILE) -> float:
    """ЧДД кандидата. Нормативы — те же, по которым будет считаться заявляемое число.

    Поиск обязан оптимизировать ту экономику, в которой нас потом посчитают: на
    валидационном кейсе (ТЗ п. 4.3) нормативы приходят от организаторов, и поиск
    по вендоренному файлу выбирал бы режимы под чужую цену нефти.
    """
    return economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date=start_date.isoformat(),
        norms_file=norms_file,
    ).chdd_m


def _scaled(control: ProducerControl, multiplier: float) -> ProducerControl:
    if multiplier == 1.0 or control.status != "OPEN":
        return control
    if multiplier == 0.0:
        return ProducerControl(control.well, "SHUT", control.mode, None, control.bhp)
    if control.liquid_rate is None:
        # Записи без поля дебита масштабировать нечем: под `ORAT`, `WRAT` и
        # `RESV` уставка стоит в своей колонке, а колонка LRAT пуста. Раньше
        # такая запись превращалась в `0 × множитель = 0`, то есть работающая
        # скважина молча **останавливалась**, а печатная нота при этом говорила
        # «поле дебита это предел, а не уставка» — ровно наоборот тому, что
        # произошло (замер 23.08, Б29). Оставляем запись как есть: множитель к
        # ней неприменим, и это честнее, чем закрыть скважину.
        return control
    rate = control.liquid_rate * multiplier
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
    if control.rate is None:
        # то же, что у добывающих: под `RESV` приёмистость задана в своей
        # колонке, а колонка расхода пуста — умножать нечего, останавливать
        # нельзя
        return control
    rate = control.rate * multiplier
    if rate <= 0:
        return InjectorControl(
            control.well, control.fluid, "SHUT", control.mode, None, control.bhp
        )
    return InjectorControl(
        control.well, control.fluid, control.status, control.mode, rate, control.bhp
    )


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
