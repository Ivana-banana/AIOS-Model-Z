"""Планирование эксперимента: выборка сценариев для обучения суррогата."""

from __future__ import annotations

import shutil
from datetime import date

import pytest

from aios import doe, economics
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.schedule import MAX_LIQUID_RATE_M3_PER_DAY, validate

SEED = 20260817


def plan(wells, months: int = 6, rate: float = 40.0):
    return [
        ScheduleStep(
            date(2007, month, 1),
            tuple(ProducerControl(well, "OPEN", "LRAT", rate, 50.0) for well in wells),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, 300.0),),
        )
        for month in range(1, months + 1)
    ]


def test_latin_hypercube_puts_one_point_in_every_stratum():
    """Свойство, ради которого он и берётся: по каждой оси нет ни дыр, ни сгущений."""
    samples, dimensions = 20, 5
    points = doe.latin_hypercube(samples, dimensions, SEED)

    assert len(points) == samples
    for axis in range(dimensions):
        strata = sorted(int(row[axis] * samples) for row in points)
        assert strata == list(range(samples))


def test_latin_hypercube_is_reproducible_from_the_seed():
    """Регламент запрещает неконтролируемую случайность."""
    assert doe.latin_hypercube(8, 3, SEED) == doe.latin_hypercube(8, 3, SEED)
    assert doe.latin_hypercube(8, 3, SEED) != doe.latin_hypercube(8, 3, SEED + 1)


def test_latin_hypercube_rejects_empty_designs():
    with pytest.raises(ValueError):
        doe.latin_hypercube(0, 3, SEED)


def test_design_is_reproducible():
    base = plan(["1", "2", "3"])
    first = doe.design(base, samples=5, seed=SEED)
    again = doe.design(base, samples=5, seed=SEED)

    assert [s.multipliers for s in first] == [s.multipliers for s in again]
    assert [s.name for s in first] == ["doe-000", "doe-001", "doe-002", "doe-003", "doe-004"]


def test_multipliers_stay_inside_the_bounds():
    scenarios = doe.design(plan(["1", "2"]), samples=12, seed=SEED, bounds=(0.2, 1.1))
    values = [value for s in scenarios for value in s.multipliers.values()]

    assert min(values) >= 0.2
    assert max(values) <= 1.1


def test_near_zero_multipliers_snap_to_a_real_shutdown():
    """Полпроцента дебита физически неотличимы от остановки, экономически — нет.

    Остановленная скважина выпадает из действующего фонда и не платит за
    содержание; открытая с околонулевым дебитом платит. Датасет не должен
    размазывать эту границу.
    """
    scenarios = doe.design(plan(["1"]), samples=30, seed=SEED, bounds=(0.0, 1.0))
    values = {value for s in scenarios for value in s.multipliers.values()}

    assert 0.0 in values
    assert not any(0.0 < value < 0.05 for value in values)


def test_shut_wells_are_really_shut():
    scenarios = doe.design(plan(["1", "2", "3"]), samples=20, seed=SEED)
    scenario = next(s for s in scenarios if s.shut_wells)

    shut = set(scenario.shut_wells)
    for step in scenario.steps:
        for control in step.producers:
            if control.well in shut:
                assert control.status == "SHUT"
                assert control.liquid_rate is None


def test_every_scenario_passes_the_schedule_validator():
    """Невалидный сценарий — это потерянные часы расчёта, ловим до симулятора."""
    for scenario in doe.design(plan(["1", "2", "3"], rate=450.0), samples=10, seed=SEED):
        assert validate(list(scenario.steps)) == []
        rates = [
            control.liquid_rate
            for step in scenario.steps
            for control in step.producers
            if control.liquid_rate is not None
        ]
        assert max(rates) <= MAX_LIQUID_RATE_M3_PER_DAY


def test_injectors_are_not_touched_by_the_design():
    scenarios = doe.design(plan(["1", "2"]), samples=6, seed=SEED)
    for scenario in scenarios:
        assert scenario.steps[0].injectors[0].rate == 30.0


def test_injector_design_varies_injection_and_keeps_production_fixed():
    scenarios = doe.design(
        plan(["1", "2"]), samples=8, seed=SEED, bounds=(0.5, 1.5),
        control_kind="injectors",
    )

    rates = {scenario.steps[0].injectors[0].rate for scenario in scenarios}

    assert len(rates) == 8
    assert all(scenario.steps[0].producers[0].liquid_rate == 40.0 for scenario in scenarios)
    assert all(scenario.multipliers == {} for scenario in scenarios)
    assert all(set(scenario.injector_multipliers) == {"9"} for scenario in scenarios)
    assert all(validate(list(scenario.steps)) == [] for scenario in scenarios)


def test_all_control_design_keeps_roles_separate_for_a_converted_well():
    steps = [
        ScheduleStep(
            date(2007, 1, 1),
            (ProducerControl("9", "OPEN", "LRAT", 40.0, 50.0),),
            (InjectorControl("9", "WATER", "SHUT", "RATE", None, 300.0),),
        ),
        ScheduleStep(
            date(2007, 2, 1),
            (ProducerControl("9", "SHUT", "LRAT", None, 50.0),),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, 300.0),),
        ),
    ]

    scenario = doe.design(
        steps, samples=2, seed=SEED, bounds=(0.5, 1.5), control_kind="all"
    )[0]

    assert set(scenario.multipliers) == {"9"}
    assert set(scenario.injector_multipliers) == {"9"}
    assert validate(list(scenario.steps)) == []


def test_scenarios_actually_differ():
    scenarios = doe.design(plan(["1", "2", "3"]), samples=8, seed=SEED)
    signatures = {tuple(sorted(s.multipliers.items())) for s in scenarios}
    assert len(signatures) == 8


def test_design_needs_producers():
    steps = [ScheduleStep(date(2007, 1, 1), (), ())]
    with pytest.raises(ValueError, match="добывающих скважин"):
        doe.design(steps, samples=3, seed=SEED)


def test_design_rejects_a_broken_range():
    with pytest.raises(ValueError, match="диапазон"):
        doe.design(plan(["1"]), samples=3, seed=SEED, bounds=(1.0, 0.5))


def test_coverage_reports_the_span_per_well():
    scenarios = doe.design(plan(["1", "2", "3"]), samples=25, seed=SEED)
    report = doe.coverage(scenarios)

    assert report["scenarios"] == 25
    assert report["wells"] == 3
    # латинский гиперкуб обязан покрыть почти весь диапазон по каждой скважине
    assert report["min_span"] > 0.9 * (doe.DEFAULT_RANGE[1] - doe.DEFAULT_RANGE[0])


def test_coverage_of_nothing_is_not_a_crash():
    assert doe.coverage([]) == {"scenarios": 0}


def test_scenario_summary_is_readable():
    scenario = doe.design(plan(["1", "2"]), samples=4, seed=SEED)[0]
    text = scenario.summary()

    assert "doe-000" in text
    assert "скважин 2" in text


def test_levels_spread_scenarios_apart_not_just_wells():
    """Без общего уровня все сценарии садятся на один уровень отбора.

    Латинский гиперкуб разыгрывает каждую скважину независимо, поэтому средний
    множитель по сценарию — это среднее десятков независимых величин, и по закону
    больших чисел оно почти одинаково у всех точек выборки. Разнообразие
    получается внутри сценария, но не между сценариями, а суррогату нужно именно
    второе: он должен предсказывать отклик объекта на изменение уровня отбора.

    Поймано на реальных расчётах 16.08: шесть сценариев дали разброс ЧДД 5,9 %
    при множителях по скважинам от 0 до 1.25 (`docs/experiments.md`).
    """
    steps = [
        ScheduleStep(
            date=date(2007, 1, 1),
            producers=tuple(
                ProducerControl(well=str(number), status="OPEN", mode="LRAT",
                                     liquid_rate=100.0, bhp=50.0)
                for number in range(1, 41)
            ),
            injectors=(),
        ),
    ]

    def spread(scenarios):
        averages = [
            sum(s.multipliers.values()) / len(s.multipliers) for s in scenarios
        ]
        return max(averages) - min(averages)

    flat = doe.design(steps, samples=12, seed=1, bounds=(0.0, 1.25))
    layered = doe.design(steps, samples=12, seed=1, bounds=(0.8, 1.2), levels=(0.5, 1.5))

    # сравнение относительное, а не по абсолютному порогу: разброс средних падает
    # с ростом числа скважин (на 72 скважинах Модели Z он вдвое у́же, чем здесь),
    # и абсолютный порог означал бы разные требования на разных фондах
    assert spread(layered) > 2 * spread(flat), (
        f"уровни не развели сценарии: без уровней {spread(flat):.3f}, "
        f"с уровнями {spread(layered):.3f}"
    )


# ── третий вид управления: перевод под закачку ───────────────────────────────
#
# Контракт между машинами: сценарий с ВНС обязан записать по переведённой
# скважине **обе** величины — добычу в ноль и закачку больше нуля. Только тогда
# у неё появится колонка `inj:` в датасете, и вход суррогата перестанет быть
# немым. Без этого предложение перевести скважину не искажается, а исчезает.

def test_conversion_scenarios_record_both_multipliers():
    scenarios = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=7,
                           control_kind="conversions")

    converted = [
        (s, well) for s in scenarios for well in s.converted_wells
    ]
    assert converted, "хотя бы один сценарий обязан кого-то перевести"
    for scenario, well in converted:
        assert scenario.multipliers[well] == 0.0, "добыча переведённой — ноль"
        assert scenario.injector_multipliers[well] > 0.0, "а закачка больше нуля"


def test_conversion_actually_changes_the_schedule():
    scenarios = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=7,
                           control_kind="conversions")
    scenario = next(s for s in scenarios if s.converted_wells)
    well = scenario.converted_wells[0]

    injectors = {c.well for step in scenario.steps for c in step.injectors}
    assert well in injectors, "переведённая обязана появиться среди нагнетательных"
    assert all(c.status == "SHUT" for step in scenario.steps
               for c in step.producers if c.well == well)


def test_conversion_scenarios_stay_valid():
    for scenario in doe.design(plan(["1", "2", "3"]), samples=3, seed=1,
                               control_kind="conversions"):
        assert validate(list(scenario.steps)) == []


def test_conversion_share_is_reproducible_from_the_seed():
    first = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=11,
                       control_kind="conversions")
    second = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=11,
                        control_kind="conversions")
    assert [s.converted_wells for s in first] == [s.converted_wells for s in second]


def test_coverage_reports_conversions():
    scenarios = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=7,
                           control_kind="conversions")
    report = doe.coverage(scenarios)
    assert report["converted_total"] >= 1


# ── пустые переводы: скважина, уже работающая под закачкой ────────────────────
#
# Измерено на настоящем деке 19.08 (`docs/experiments.md`): в кандидаты на ВНС
# попадает скважина, которая в горизонте управления не добывает ни месяца, — в
# Модели Z 30 скважин переведены под закачку за историю. Расписание при этом
# корректно, переводить нечего; портится манифест: сборщик датасета запишет такой
# скважине `production_multiplier = 0` и `injection_multiplier > 0`, и в обучающей
# выборке появится строка «скважину перевели», отклик у которой в точности
# опорный. Сеть выучит, что перевод под закачку не даёт ничего — ровно на том
# классе решений, ради которого пакет и затевается.

def already_injecting(months: int = 6):
    """Расписание, где скважина 102 к началу управления уже под закачкой.

    Списано со скважины 102 Модели Z: одна закрывающая запись `WCONPROD` на
    01.2007 (`SHUT`, дебит 0) и закачка 90 м³/сут во всех месяцах после неё.
    Скважина 1 рядом — настоящий кандидат на перевод.
    """
    steps = []
    for month in range(1, months + 1):
        producers = [ProducerControl("1", "OPEN", "LRAT", 40.0, 50.0)]
        if month == 1:
            producers.append(ProducerControl("102", "SHUT", "LRAT", 0.0, 50.0))
        steps.append(ScheduleStep(
            date(2007, month, 1),
            tuple(producers),
            (InjectorControl("102", "WATER", "OPEN", "RATE", 90.0, 300.0),),
        ))
    return steps


def records_of(steps, well: str):
    """Все записи расписания по одной скважине — с чем сравнивать «изменилось»."""
    return [
        (step.date, kind, control)
        for step in steps
        for kind, controls in (("prod", step.producers), ("inj", step.injectors))
        for control in controls
        if control.well == well
    ]


def test_a_well_already_under_injection_is_not_offered_for_conversion():
    """Переводить под закачку ту, что и так качает, нечего."""
    for scenario in doe.design(already_injecting(), samples=6, seed=SEED,
                               control_kind="conversions"):
        assert "102" not in scenario.converted_wells
        assert "102" not in scenario.multipliers
        assert "102" not in scenario.injector_multipliers


def test_converted_wells_lists_only_wells_whose_schedule_changed():
    """Контракт манифеста: названа переведённой — значит, расписание другое."""
    base = already_injecting()
    for scenario in doe.design(base, samples=4, seed=SEED, control_kind="conversions"):
        assert scenario.converted_wells, "сценарий обязан кого-то перевести"
        for well in scenario.converted_wells:
            assert records_of(scenario.steps, well) != records_of(base, well)


def test_coverage_counts_only_real_conversions():
    """Отчёт о покрытии — число настоящих переводов, а не заявленных."""
    scenarios = doe.design(already_injecting(), samples=5, seed=SEED,
                           control_kind="conversions")
    report = doe.coverage(scenarios)

    assert report["converted_total"] == sum(len(s.converted_wells) for s in scenarios)
    assert report["converted_total"] == len(scenarios), "кандидат здесь ровно один"


def test_conversions_need_a_well_that_actually_produces():
    """Фонд, целиком работающий под закачкой, — отказ, а не пустой пакет."""
    steps = [
        ScheduleStep(
            date(2007, 1, 1),
            (ProducerControl("102", "SHUT", "LRAT", 0.0, 50.0),),
            (InjectorControl("102", "WATER", "OPEN", "RATE", 90.0, 300.0),),
        ),
    ]
    with pytest.raises(ValueError, match="добывающих скважин"):
        doe.design(steps, samples=2, seed=SEED, control_kind="conversions")


# ── режим по периодам внутри сценария ────────────────────────────────────────
#
# Измерено 19.08 на собранном датасете: множитель постоянен по месяцам, поэтому
# пакет из 12 прогонов даёт суррогату 12 независимых точек управления, а не
# 457 320 строк. Периоды — способ получить N точек с одного прогона: режим
# меняется внутри сценария, и один расчёт даёт столько точек, сколько в нём
# периодов. Схема манифеста закреплена со стороны сборщика (`tests/test_dataset.py`),
# здесь проверяется, что генератор её соблюдает.

def rates_of(steps, well: str) -> list[float | None]:
    return [
        control.liquid_rate
        for step in steps
        for control in step.producers
        if control.well == well
    ]


def test_periods_split_the_horizon_and_start_at_the_first_step():
    """Первый период совпадает с началом управления — это контракт манифеста."""
    scenario = doe.design(plan(["1", "2"], months=12), samples=4, seed=SEED,
                          bounds=(0.5, 1.5), periods=3)[0]

    starts = [period.start for period in scenario.periods]

    assert starts == [date(2007, 1, 1), date(2007, 5, 1), date(2007, 9, 1)]
    assert starts == sorted(starts)


def test_periods_actually_change_the_regime_inside_one_scenario():
    """Иначе периоды — запись в манифесте, за которой ничего не стоит."""
    scenario = doe.design(plan(["1", "2"], months=12), samples=4, seed=SEED,
                          bounds=(0.5, 1.5), periods=3)[0]

    rates = rates_of(scenario.steps, "1")

    assert len(rates) == 12
    assert len(set(rates[:4])) == 1, "внутри периода режим постоянен"
    assert len(set(rates[4:8])) == 1
    assert len({rates[0], rates[4], rates[8]}) > 1, "на границе периода режим тот же"


def test_every_period_has_its_own_multipliers_for_every_well():
    scenario = doe.design(plan(["1", "2"], months=12), samples=4, seed=SEED,
                          bounds=(0.5, 1.5), periods=4)[0]

    assert len(scenario.periods) == 4
    for period in scenario.periods:
        assert set(period.multipliers) == {"1", "2"}


def test_one_period_covers_the_whole_horizon():
    """Умолчание не меняется: пакеты без периодов собираются как были."""
    scenario = doe.design(plan(["1", "2"]), samples=4, seed=SEED)[0]

    assert len(scenario.periods) == 1
    assert scenario.periods[0].start == date(2007, 1, 1)
    assert scenario.periods[0].multipliers == scenario.multipliers


def test_periods_stay_inside_the_schedule_validator():
    for scenario in doe.design(plan(["1", "2"], months=12, rate=450.0), samples=6,
                               seed=SEED, bounds=(0.5, 1.1), periods=3):
        assert validate(list(scenario.steps)) == []
        assert len(scenario.steps) == 12, "куски периодов обязаны сойтись в целое"


def test_more_periods_than_steps_is_a_refusal():
    with pytest.raises(ValueError, match="периодов"):
        doe.design(plan(["1"], months=3), samples=2, seed=SEED, periods=5)


def test_conversions_do_not_take_periods():
    """Перевод под закачку — разовое событие, дробить его по периодам нечем."""
    with pytest.raises(ValueError, match="периодов"):
        doe.design(plan(["1", "2"], months=12), samples=2, seed=SEED,
                   control_kind="conversions", periods=3)


def test_coverage_reports_every_period_separately():
    """Без разбивки по периодам dry-run не показывает, где выборка дырявая."""
    scenarios = doe.design(plan(["1", "2"], months=12), samples=20, seed=SEED,
                           bounds=(0.0, 1.25), periods=3)
    report = doe.coverage(scenarios)

    assert len(report["periods"]) == 3
    assert [entry["from"] for entry in report["periods"]] == [
        "2007-01-01", "2007-05-01", "2007-09-01",
    ]
    for entry in report["periods"]:
        assert entry["min_span"] > 0.9 * (doe.DEFAULT_RANGE[1] - doe.DEFAULT_RANGE[0])


def test_coverage_without_periods_says_nothing_about_them():
    report = doe.coverage(doe.design(plan(["1", "2"]), samples=6, seed=SEED))
    assert "periods" not in report


# ── манифест пакета: схему читает сборщик датасета, а не человек ─────────────

DECK_DATA = """RUNSPEC

DIMENS
 2 2 2 /

PROPS
INCLUDE
'p.inc' /

REGIONS
INCLUDE
'r.inc' /

SUMMARY
INCLUDE
's.inc' /

SCHEDULE
INCLUDE
'h.inc' /

END
"""

DECK_SCHEDULE = """WELSPECS
 '1' 'G' 1 1 1* 'OIL' /
 '2' 'G' 2 2 1* 'OIL' /
/
""" + "".join(
    f"""
DATES
 01 {month} 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
 '2' 'OPEN' 'LRAT' 1* 1* 1* 20.0 1* 50 1* 1* /
/
"""
    for month in ("JAN", "FEB", "MAR", "APR", "MAY", "JUN")
)


TINY_CONFIG = """[case]
model_dir = "deck"
run_dir = "runs"
case_name = "RUN"
control_start = 2007-01-01

[economics]
start_date = 2007-01-01
norms_file = "norms.xlsx"

[fluids]
oil_density = 913.0765883224138
water_density = 1172.0

[limits]
max_field_liquid_m3_per_day = 0
max_field_injection_m3_per_day = 0
enforce_water_balance = true
max_external_water_m3_per_day = 0
water_balance_rule = "compensation"
water_tolerance = 1.0

[explain]
enabled = false
model = "claude-opus-5"
max_tokens = 1024
effort = "low"
cache_dir = "artifacts"

[policy]
lookback_months = 6

[run]
seed = 20260817
processes = 1
"""


def tiny_project(tmp_path):
    """Крошечный дек и конфиг под него — шесть месячных шагов и две скважины.

    Модель Z в репозиторий не кладётся (`GIT.md`), а проверяющий запускает тесты
    на чистом клоне: тест, берущий дек из корневого `config.toml`, у него упадёт.
    """
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "FIELD.DATA").write_text(DECK_DATA, encoding="utf-8")
    (deck / "p.inc").write_text("-- пусто", encoding="utf-8")
    (deck / "r.inc").write_text("-- пусто", encoding="utf-8")
    (deck / "h.inc").write_text(DECK_SCHEDULE, encoding="utf-8")

    # Нормативы кладутся рядом настоящие: с 03.09 сборка пакета берёт из них
    # предел дебита жидкости, а из `[limits]` — предел давления нагнетания, и
    # проверяет сценарии этими числами, а не умолчаниями Модели Z (С12).
    # Подставлять сюда пустышку нельзя: тогда тест проверял бы, что пределы не
    # доходят, — ровно то состояние, которое чинится.
    shutil.copy(economics.NORMS_FILE, tmp_path / "norms.xlsx")

    config = tmp_path / "config.toml"
    config.write_text(TINY_CONFIG, encoding="utf-8")
    return config


def test_rebuilding_a_package_invalidates_completed_cases(tmp_path, capsys):
    """Новый `RUN.DATA` нельзя сочетать с отметкой от прежнего расчёта.

    Повторная сборка может поменять расписание сценария, но оставляет каталог
    тем же. Если не снять `.doe-done`, `run_doe.py` увидит старый `.UNSMRY` и
    пропустит расчёт уже другого входного дека.
    """
    from scripts import build_doe, run_doe

    root = tmp_path / "pack"
    arguments = [
        "--samples", "2", "--run-root", str(root),
        "--config", str(tiny_project(tmp_path)),
    ]
    assert build_doe.main(arguments) == 0

    cases = sorted(root.glob("doe-*"))
    for case in cases:
        (case / "RUN.UNSMRY").write_text("старый результат", encoding="utf-8")
        (case / ".doe-done").write_text("2026-08-18", encoding="utf-8")

    assert build_doe.main(arguments) == 0
    assert all((case / "RUN.UNSMRY").is_file() for case in cases)
    assert not any((case / ".doe-done").exists() for case in cases)

    capsys.readouterr()
    assert run_doe.main([str(root), "2", "--dry-run"]) == 0
    assert "готово: 0, в очереди: 2" in capsys.readouterr().out


def test_the_manifest_carries_the_periods_the_dataset_builder_expects(tmp_path):
    """Схема стыка: `from` ISO-датой, периоды по возрастанию, первый — control_start.

    Читает её `scripts/build_dataset.py` (`_periods`, `_at`), и она уже закреплена
    с той стороны. Прошлый стык — перевод под закачку — сломался ровно потому, что
    контракт проверялся только на одной стороне.
    """
    import json

    from scripts import build_doe

    root = tmp_path / "pack"
    status = build_doe.main([
        "--samples", "3", "--periods", "3", "--run-root", str(root),
        "--config", str(tiny_project(tmp_path)),
    ])

    assert status == 0
    manifest = json.loads((root / "doe.json").read_text(encoding="utf-8"))
    assert manifest["periods"] == 3
    for scenario in manifest["scenarios"]:
        periods = scenario["periods"]
        assert [entry["from"] for entry in periods] == ["2007-01-01", "2007-03-01", "2007-05-01"]
        for entry in periods:
            assert set(entry["multipliers"]) == {"1", "2"}
            assert "injector_multipliers" in entry
        # старые поля остаются: уже посчитанные пакеты собираются по-прежнему
        assert scenario["multipliers"] == periods[0]["multipliers"]


def test_dry_run_shows_the_coverage_of_every_period(tmp_path, capsys):
    from scripts import build_doe

    status = build_doe.main([
        "--samples", "6", "--periods", "3", "--dry-run",
        "--config", str(tiny_project(tmp_path)),
    ])
    output = capsys.readouterr().out

    assert status == 0
    assert "периодов:" in output
    for start in ("2007-01-01", "2007-03-01", "2007-05-01"):
        assert start in output


# ── покрытие остановок: отдельная ось, а не побочный эффект границ ───────────
#
# Доля нулей в выборке была следствием арифметики: множитель ниже `snap_to_zero`
# округляется до нуля, значит на остановку приходится snap_to_zero / (high - low)
# — при границах по умолчанию около 4 % точек на скважину, а по части скважин
# ноль. Отключение убыточной скважины при этом — эвристика первого уровня
# (`CLAUDE.md`, принцип 3), то есть класс решений, который оптимизатор предложит
# раньше всех. Суррогат обязан видеть отклик на него, а не угадывать.

def shut_counts(scenario) -> list[int]:
    """Сколько скважин остановлено в каждом периоде сценария."""
    return [
        sum(1 for value in (*period.multipliers.values(),
                            *period.injector_multipliers.values()) if value == 0.0)
        for period in scenario.periods
    ]


def test_the_shut_axis_puts_the_requested_share_of_wells_on_zero():
    wells = [str(number) for number in range(11, 21)]  # 9 занята нагнетателем в plan()
    scenarios = doe.design(plan(wells), samples=10, seed=SEED,
                           bounds=(0.5, 1.25), shut_share=(0.2, 0.4))

    counts = [count for scenario in scenarios for count in shut_counts(scenario)]

    assert min(counts) >= 2, "нижняя граница доли не соблюдена"
    assert max(counts) <= 4, "верхняя граница доли не соблюдена"
    assert len(set(counts)) > 1, "доля разыгрывается осью, а не задана числом"


def test_without_the_axis_some_well_never_sees_a_shutdown():
    """Ради чего ось и заведена: на границах по умолчанию нулей почти нет."""
    wells = [str(number) for number in range(11, 21)]  # 9 занята нагнетателем в plan()
    steps = plan(wells)

    without = doe.coverage(doe.design(steps, samples=8, seed=SEED))
    with_axis = doe.coverage(doe.design(steps, samples=8, seed=SEED,
                                        shut_share=(0.2, 0.4)))

    assert without["min_shut_share"] == 0.0
    assert with_axis["min_shut_share"] > 0.0, (
        "по каждой скважине в выборке обязан быть хотя бы один месяц остановки"
    )


def test_the_shut_axis_works_inside_every_period():
    """Остановка в одном периоде и работа в следующем — это пуск из простоя.

    Экономика считает остановку и пуск двумя отдельными событиями по 1,0 млн руб.
    (`docs/method/`), и отклик на такую пару суррогату нужен не меньше, чем на
    ровный режим.
    """
    wells = [str(number) for number in range(11, 21)]  # 9 занята нагнетателем в plan()
    scenarios = doe.design(plan(wells, months=12), samples=8, seed=SEED,
                           bounds=(0.5, 1.25), shut_share=(0.1, 0.4), periods=3)

    restarted = [
        scenario for scenario in scenarios
        if any(scenario.periods[0].multipliers[well] == 0.0
               and scenario.periods[1].multipliers[well] > 0.0 for well in wells)
    ]

    assert all(len(shut_counts(scenario)) == 3 for scenario in scenarios)
    assert restarted, "ни в одном сценарии скважина не запускается из простоя"


def test_shut_wells_from_the_axis_are_really_shut_in_the_schedule():
    wells = [str(number) for number in range(11, 21)]  # 9 занята нагнетателем в plan()
    scenario = doe.design(plan(wells), samples=6, seed=SEED,
                          bounds=(0.5, 1.25), shut_share=(0.2, 0.4))[0]

    assert scenario.shut_wells
    for step in scenario.steps:
        for control in step.producers:
            if control.well in set(scenario.shut_wells):
                assert control.status == "SHUT"
                assert control.liquid_rate is None
    assert validate(list(scenario.steps)) == []


def test_shut_share_rejects_a_broken_range():
    with pytest.raises(ValueError, match="доля остановок"):
        doe.design(plan(["1", "2"]), samples=3, seed=SEED, shut_share=(0.6, 0.2))


def test_shut_share_rejects_a_share_above_the_whole_fund():
    with pytest.raises(ValueError, match="доля остановок"):
        doe.design(plan(["1", "2"]), samples=3, seed=SEED, shut_share=(0.5, 1.5))


def test_conversions_do_not_take_the_shut_axis():
    """У перевода множители заданы самим переводом: ноль добычи и закачка."""
    with pytest.raises(ValueError, match="доля остановок"):
        doe.design(plan(["1", "2", "3"]), samples=3, seed=SEED,
                   control_kind="conversions", shut_share=(0.2, 0.4))


def test_the_shut_axis_does_not_change_the_design_when_it_is_off():
    """Умолчание не трогается: три посчитанных пакета остаются воспроизводимыми."""
    steps = plan(["1", "2", "3"])
    assert ([s.multipliers for s in doe.design(steps, samples=5, seed=SEED)]
            == [s.multipliers for s in doe.design(steps, samples=5, seed=SEED,
                                                  shut_share=None)])


def test_the_cli_needs_both_shut_bounds(tmp_path, capsys):
    from scripts import build_doe

    status = build_doe.main([
        "--samples", "3", "--shut-low", "0.2", "--dry-run",
        "--config", str(tiny_project(tmp_path)),
    ])

    assert status == 1
    assert "задаются вместе" in capsys.readouterr().err


# ── циклическое заводнение ───────────────────────────────────────────────────

def plan_with_injectors(injectors, months: int = 6, rate: float = 40.0):
    """Базовое расписание с несколькими нагнетательными — иначе противофазы нет."""
    return [
        ScheduleStep(
            date(2007, month, 1),
            (ProducerControl("1", "OPEN", "LRAT", rate, 50.0),),
            tuple(InjectorControl(w, "WATER", "OPEN", "RATE", 30.0, 300.0) for w in injectors),
        )
        for month in range(1, months + 1)
    ]


def test_cyclic_scenarios_alternate_two_groups():
    """Смысл режима в перепаде давления, значит группы не работают вместе."""
    scenarios = doe.design(plan_with_injectors(["9", "8", "7", "6"]), samples=4, seed=5,
                           control_kind="cyclic")

    for scenario in scenarios:
        for step in scenario.steps:
            working = [c.well for c in step.injectors if c.status == "OPEN"]
            resting = [c.well for c in step.injectors if c.status != "OPEN"]
            assert working and resting, "на каждом шаге кто-то работает, а кто-то стоит"


def test_cyclic_scenario_is_written_as_periods():
    """Один множитель на скважину такой сценарий описать не может."""
    scenario = doe.design(plan_with_injectors(["9", "8"]), samples=2, seed=5,
                          control_kind="cyclic")[0]

    assert len(scenario.periods) > 1, "режим обязан меняться во времени"
    assert scenario.periods[0].injector_multipliers != scenario.periods[1].injector_multipliers


def test_cyclic_needs_at_least_two_injectors():
    """Одна скважина в противофазе сама с собой — это не заводнение."""
    with pytest.raises(ValueError, match="двух"):
        doe.design(plan_with_injectors(["9"]), samples=2, seed=5, control_kind="cyclic")


def test_cyclic_scenarios_stay_valid():
    for scenario in doe.design(plan_with_injectors(["9", "8"]), samples=3, seed=1,
                               control_kind="cyclic"):
        assert validate(list(scenario.steps)) == []


def test_periods_and_the_direct_transformation_agree():
    """Два представления одного режима обязаны совпадать, иначе они разойдутся.

    В датасете цикл записывается периодами — их понимает сборщик. Агенту нужен
    прямой приём `optimise.cyclic_injection`. Если бы они расходились, датасет
    учил бы сеть одному, а предлагала бы она другое.
    """
    from aios import optimise

    base = plan_with_injectors(["9", "8"], months=6)
    scenario = doe.design(base, samples=2, seed=5, control_kind="cyclic")[0]

    groups: dict[float, set[str]] = {}
    for well, value in scenario.periods[0].injector_multipliers.items():
        groups.setdefault(value, set()).add(well)
    working = groups[max(groups)]
    resting = groups[min(groups)]

    direct = optimise.cyclic_injection(
        base, working, resting,
        half_period=scenario.cyclic_half_period, amplitude=max(groups),
    )
    assert [step.injectors for step in direct] == [step.injectors for step in scenario.steps]


def test_coverage_survives_scenarios_with_different_period_layouts():
    """У циклических сценариев полупериод свой, и «период №3» — разные месяцы.

    Раньше подсчёт покрытия брал число периодов у первого сценария и падал по
    выходу за границы списка на втором. Складывать такие периоды в одну корзину
    нельзя и по смыслу: сравнивать имеет смысл только совпадающие границы.
    """
    scenarios = doe.design(plan_with_injectors(["9", "8", "7"]), samples=5, seed=3,
                           control_kind="cyclic")
    layouts = {tuple(p.start for p in s.periods) for s in scenarios}
    assert len(layouts) > 1, "иначе тест не проверяет того, ради чего написан"

    report = doe.coverage(scenarios)

    assert report["scenarios"] == 5
    assert "periods" not in report, "поштучное сравнение здесь бессмысленно"
    assert report["period_layouts"] == len(layouts)


def test_design_validates_scenarios_by_the_limits_of_the_case():
    """Сценарий DoE тоже пишется в файл, который считает симулятор.

    С12: до правки `design` проверял умолчаниями Модели Z, поэтому на кейсе с
    ЭЦН до 200 м³/сут пакет собрался бы целиком, а отвергся бы шлюзом — после
    того, как расчёты уже запущены.
    """
    base = plan(("1", "2"), rate=150.0)

    # умолчание 500 — сценарии проходят
    assert doe.design(base, samples=2, seed=SEED)

    with pytest.raises(ValueError):
        doe.design(base, samples=2, seed=SEED, limits={"max_liquid_rate": 1.0})


def test_cyclic_refuses_parameters_it_cannot_use():
    """Неприменимый параметр отвергается, а не принимается молча.

    Циклическое заводнение задаётся периодами противофазной пульсации закачки, и
    ни диапазон множителей, ни округление к нулю, ни уровень сценария, ни доля
    остановок, ни число периодов сюда не входят. Пока они принимались молча, два
    вызова с совершенно разными параметрами давали побайтово одинаковые
    сценарии — воспроизводимость по конфигу обещана регламентом, а конфиг ни на
    что не влиял. `conversions` рядом отвергает неприменимое явно; теперь и этот
    режим (Р-70).
    """
    steps = plan_with_injectors(["i1", "i2"])

    for extra in ({"levels": (0.8, 1.2)}, {"shut_share": (0.1, 0.2)},
                  {"periods": 3}, {"bounds": (0.5, 1.5)}, {"snap_to_zero": 0.2}):
        with pytest.raises(ValueError, match="неприменимы"):
            doe.design(steps, samples=4, seed=1, control_kind="cyclic", **extra)

    # без них режим по-прежнему работает
    assert doe.design(steps, samples=4, seed=1, control_kind="cyclic")
