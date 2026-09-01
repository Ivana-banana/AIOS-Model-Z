"""Конфигурация прогона — обязательный по регламенту артефакт."""

from __future__ import annotations

from datetime import date

import pytest

from aios import config

MINIMAL = """
[case]
model_dir = "vendor/Model_Z_OPM"
run_dir = "runs/base"
case_name = "RUN"
control_start = 2007-01-01

[economics]
start_date = 2007-01-01
norms_file = "vendor/CHDD_PYTHON/input/n.xlsx"

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
cache_dir = "artifacts/explanations"

[policy]
lookback_months = 6

[run]
seed = 20260817
processes = 2
"""


def test_repository_config_loads():
    """Файл в корне репозитория обязан читаться — по нему воспроизводят прогон."""
    loaded = config.load()

    assert loaded.case.control_start == date(2007, 1, 1)
    assert loaded.economics.start_date == date(2007, 1, 1)
    assert loaded.case.model_dir.name == "Model_Z_OPM"
    assert loaded.fluids.oil_density == pytest.approx(913.0765883224138)
    assert loaded.run.seed == 20260817


def test_paths_are_resolved_relative_to_the_config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(MINIMAL, encoding="utf-8")

    loaded = config.load(path)

    assert loaded.case.model_dir == tmp_path / "vendor/Model_Z_OPM"
    assert loaded.economics.norms_file == tmp_path / "vendor/CHDD_PYTHON/input/n.xlsx"
    assert loaded.source == path


def test_missing_section_fails_loudly(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(MINIMAL.replace("[policy]", "[unused]"), encoding="utf-8")

    with pytest.raises(KeyError, match=r"\[policy\]"):
        config.load(path)


def test_missing_key_names_the_key_and_the_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(MINIMAL.replace("seed = 20260817\n", ""), encoding="utf-8")

    with pytest.raises(KeyError, match="seed"):
        config.load(path)


def test_no_silent_defaults(tmp_path):
    """Пропущенный ключ не подменяется «разумным значением» — иначе прогон
    невоспроизводим по одному файлу, а регламент требует именно этого."""
    path = tmp_path / "config.toml"
    path.write_text(MINIMAL.replace("start_date = 2007-01-01\n", ""), encoding="utf-8")

    with pytest.raises(KeyError, match="start_date"):
        config.load(path)


def test_control_start_matches_the_organisers_export_script():
    """01.01.2007 — та же дата, с которой начинает выгрузку script_1.py."""
    loaded = config.load()
    assert loaded.case.control_start == date(2007, 1, 1)
    assert loaded.economics.start_date == loaded.case.control_start


# ── значения, а не только наличие ключей ─────────────────────────────────────
#
# В валидационном кейсе (ТЗ п. 4.3) ограничения организаторов лягут именно сюда,
# и править файл будет человек. Опечатка в знаке уходит прямо в ЧДД: предел −1
# м³/сут — это не «предела нет», а требование срезать весь фонд, и получившийся
# прогон выглядит совершенно обычным.

def _written(tmp_path, text: str):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_zero_limit_still_means_no_limit(tmp_path):
    """Ноль — задокументированное «предела нет», его ломать нельзя."""
    loaded = config.load(_written(tmp_path, MINIMAL))

    assert loaded.limits.max_field_liquid_m3_per_day is None
    assert loaded.limits.max_field_injection_m3_per_day is None


def test_a_negative_limit_is_refused(tmp_path):
    """Отрицательный предел проходит как настоящий и срезает фонд в ноль."""
    text = MINIMAL.replace("max_field_liquid_m3_per_day = 0",
                           "max_field_liquid_m3_per_day = -1")
    with pytest.raises(ValueError, match="max_field_liquid_m3_per_day"):
        config.load(_written(tmp_path, text))


def test_a_non_positive_density_is_refused(tmp_path):
    """Плотность — множитель в выручке; ноль превращает нефть в ничто."""
    text = MINIMAL.replace("oil_density = 913.0765883224138", "oil_density = 0")
    with pytest.raises(ValueError, match="oil_density"):
        config.load(_written(tmp_path, text))


def test_a_non_positive_water_tolerance_is_refused(tmp_path):
    text = MINIMAL.replace("water_tolerance = 1.0", "water_tolerance = 0")
    with pytest.raises(ValueError, match="water_tolerance"):
        config.load(_written(tmp_path, text))


def test_reservoir_pressure_limits_are_optional(tmp_path):
    """Как и остальные ручки кейса: ключа нет — предела нет."""
    limits = config.load(_written(tmp_path, MINIMAL)).limits
    assert limits.min_reservoir_pressure_atm is None
    assert limits.max_reservoir_pressure_atm is None


def test_reservoir_pressure_limits_are_read_from_the_config(tmp_path):
    text = MINIMAL.replace(
        "water_tolerance = 1.0",
        "water_tolerance = 1.0\nmin_reservoir_pressure_atm = 50.0\n"
        "max_reservoir_pressure_atm = 300.0",
    )
    limits = config.load(_written(tmp_path, text)).limits
    assert limits.min_reservoir_pressure_atm == 50.0
    assert limits.max_reservoir_pressure_atm == 300.0


def test_the_corridor_floor_is_optional(tmp_path):
    """Ключа нет — снизу закачка не ограничена; так получено заявляемое число."""
    assert config.load(_written(tmp_path, MINIMAL)).limits.water_tolerance_min is None


def test_the_corridor_floor_is_read_from_the_config(tmp_path):
    text = MINIMAL.replace("water_tolerance = 1.0",
                           "water_tolerance = 1.15\nwater_tolerance_min = 0.85")
    assert config.load(_written(tmp_path, text)).limits.water_tolerance_min == 0.85


@pytest.mark.parametrize(("line", "broken", "key"), [
    ("processes = 2", "processes = 0", "processes"),
    ("lookback_months = 6", "lookback_months = 0", "lookback_months"),
    ("max_tokens = 1024", "max_tokens = 0", "max_tokens"),
])
def test_counts_must_be_at_least_one(tmp_path, line, broken, key):
    with pytest.raises(ValueError, match=key):
        config.load(_written(tmp_path, MINIMAL.replace(line, broken)))


def test_the_second_doe_config_does_not_repeat_the_seed():
    """Два пакета DoE с одним seed — это один пакет, растянутый дважды.

    Измерено 19.08 перед постановкой пяти часов счёта: латинский гиперкуб при
    равном seed разыгрывает один и тот же нормированный узор, поэтому пакет с
    другими границами повторяет первый — корреляция узоров вышла **0.925**, а
    средние уровни по сценариям совпали до сотых. С разными seed стало 0.107.

    Ошибка воспроизводится тривиально: конфиг копируют ради других границ и
    забывают seed. Тогда вторые 2,5 часа машинного времени уходят на почти те же
    точки, и заметить это можно только по метрикам суррогата.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    main = config.load(root / "config.toml")
    wide = config.load(root / "config-doe-wide.toml")

    assert wide.run.seed != main.run.seed, (
        "второй пакет DoE обязан иметь свой seed, иначе он копия первого"
    )
