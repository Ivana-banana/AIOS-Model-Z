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
water_tolerance = 1.0

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
