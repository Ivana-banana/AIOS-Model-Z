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

[trust]
pairs_file = "docs/data/margin-pairs-search.csv"
pairs_family = "search"
search_seeds = [1, 2, 3]

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
    assert loaded.trust.pairs_family == "search"
    assert loaded.trust.search_seeds == (20260830, 20260831, 20260901, 20260902, 20260903,
                                         20260904, 20260905, 20260906, 20260907, 20260908)


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


def test_trust_seeds_are_required_and_distinct(tmp_path):
    with pytest.raises(ValueError, match="search_seeds"):
        config.load(_written(tmp_path, MINIMAL.replace("search_seeds = [1, 2, 3]", "search_seeds = []")))
    with pytest.raises(ValueError, match="повторяются"):
        config.load(_written(tmp_path, MINIMAL.replace("search_seeds = [1, 2, 3]", "search_seeds = [1, 1]")))


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
# ── формы ограничений кейса, заведённые 01.09 ────────────────────────────────

WINDOWS = """
[[limits.forced_shut_windows]]
well = "102"
start = 2027-01-01
end = 2027-06-30

[[limits.forced_shut_windows]]
well = 305
start = 2028-02-01
end = 2028-02-29
"""


def test_forced_shut_windows_are_read_as_dates(tmp_path):
    """Окно остановки приходит датами TOML, а номер скважины может быть числом."""
    loaded = config.load(_written(tmp_path, MINIMAL + WINDOWS))
    windows = loaded.limits.forced_shut_windows

    assert len(windows) == 2
    assert windows[0].well == "102"
    assert windows[0].start == date(2027, 1, 1) and windows[0].end == date(2027, 6, 30)
    assert windows[1].well == "305", "номер, записанный числом, обязан стать строкой"


def test_forced_shut_window_with_a_string_date_is_refused(tmp_path):
    """Строка `\"2027-13-01\"` прошла бы разбор и сравнивалась бы лексикографически."""
    text = MINIMAL + WINDOWS.replace("start = 2027-01-01", 'start = "2027-01-01"')

    with pytest.raises(ValueError, match="должно быть датой TOML"):
        config.load(_written(tmp_path, text))


def test_forced_shut_window_without_end_is_refused(tmp_path):
    """Окно без конца — это не «до конца горизонта», это недописанная запись."""
    text = MINIMAL + WINDOWS.replace("end = 2027-06-30", "")

    with pytest.raises(ValueError, match="не хватает полей: end"):
        config.load(_written(tmp_path, text))


def test_reversed_window_is_refused_on_load(tmp_path):
    """Проверка окна живёт в самой структуре и срабатывает при чтении конфига."""
    text = MINIMAL + WINDOWS.replace("end = 2027-06-30", "end = 2026-06-30")

    with pytest.raises(ValueError, match="раньше начала"):
        config.load(_written(tmp_path, text))


def test_annual_liquid_volume_and_average_reading_are_optional(tmp_path):
    """Обе новые ручки необязательны: конфиг кейса не обязан их содержать."""
    loaded = config.load(_written(tmp_path, MINIMAL))

    assert loaded.limits.max_field_liquid_m3_per_year is None
    assert loaded.limits.field_limits_are_annual_average is False
    assert loaded.limits.forced_shut_windows is None


def test_annual_liquid_volume_and_average_reading_are_read(tmp_path):
    # вторая секция [limits] в TOML запрещена — дописываем ключи в существующую
    text = MINIMAL.replace(
        "water_tolerance = 1.0",
        "water_tolerance = 1.0\nmax_field_liquid_m3_per_year = 1000000\n"
        "field_limits_are_annual_average = true",
    )
    loaded = config.load(_written(tmp_path, text))

    assert loaded.limits.max_field_liquid_m3_per_year == 1_000_000.0
    assert loaded.limits.field_limits_are_annual_average is True
GROUPS = """
[limits.well_groups]
"куст-1" = ["102", 305]

[limits.group_liquid_caps_m3_per_day]
"куст-1" = 400

[limits.group_injection_caps_m3_per_day]
"куст-1" = 300
"""


def test_well_groups_and_caps_are_read(tmp_path):
    loaded = config.load(_written(tmp_path, MINIMAL + GROUPS))

    assert loaded.limits.well_groups == {"куст-1": frozenset({"102", "305"})}
    assert loaded.limits.group_liquid_caps_m3_per_day == {"куст-1": 400.0}
    assert loaded.limits.group_injection_caps_m3_per_day == {"куст-1": 300.0}


def test_empty_group_is_refused(tmp_path):
    """Пустая группа не выражает ограничения, а прячет его."""
    text = MINIMAL + GROUPS.replace('"куст-1" = ["102", 305]', '"куст-1" = []')

    with pytest.raises(ValueError, match="пуста"):
        config.load(_written(tmp_path, text))


def test_phase_limits_are_optional_and_read(tmp_path):
    """Мощность УПН и УПСВ: пределы по фазе, а не по жидкости."""
    assert config.load(_written(tmp_path, MINIMAL)).limits.max_field_oil_t_per_day is None

    text = MINIMAL.replace(
        "water_tolerance = 1.0",
        "water_tolerance = 1.0\nmax_field_oil_t_per_day = 1200\n"
        "max_field_water_t_per_day = 9000",
    )
    loaded = config.load(_written(tmp_path, text))

    assert loaded.limits.max_field_oil_t_per_day == 1200.0
    assert loaded.limits.max_field_water_t_per_day == 9000.0
def test_optional_keys_documented_in_config_toml_actually_load(tmp_path):
    """Примеры необязательных ручек в `config.toml` обязаны читаться, а не украшать.

    Блок закомментированных ключей — единственное место, где человек узнаёт, что
    ручка есть. Пример, который не грузится (опечатка в имени, устаревший ключ,
    неверный тип), хуже отсутствующего: он обещает ограничение, которого не будет.
    """
    lines = config.DEFAULT_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines)
                 if "необязательные ручки под валидационный кейс" in line)
    end = next(i for i, line in enumerate(lines[start:], start) if line.startswith("[policy]"))

    flat, tables = [], []
    current = flat
    for line in lines[start:end]:
        stripped = line.lstrip("#").strip()
        if not stripped or stripped.startswith("---"):
            continue
        if stripped.startswith("["):
            current = tables
        elif "=" not in stripped:
            continue
        if "=" in stripped or stripped.startswith("["):
            current.append(stripped.split("  #")[0].strip())

    assert flat and tables, "блок примеров пуст — проверка потеряла смысл"

    text = MINIMAL.replace("water_tolerance = 1.0",
                           "water_tolerance = 1.0\n" + "\n".join(flat))
    text += "\n" + "\n".join(tables) + "\n"
    loaded = config.load(_written(tmp_path, text))

    assert loaded.limits.max_field_oil_t_per_day == 1200.0
    assert loaded.limits.well_groups == {"куст-1": frozenset({"102", "305"})}
    assert loaded.limits.forced_shut_windows[0].well == "102"
    assert loaded.limits.max_events_per_year == 12
def test_a_typo_in_a_limits_key_is_refused(tmp_path):
    """Опечатка читалась бы как «ограничения нет» — и условие кейса терялось молча.

    Найдено критиком 01.09: `max_field_oil_t_per_dai = 700` проходил разбор,
    прогон шёл, репетиция печатала «пройдена», а предел не применялся вовсе.
    """
    text = MINIMAL.replace("water_tolerance = 1.0",
                           "water_tolerance = 1.0\nmax_field_oil_t_per_dai = 700")

    with pytest.raises(ValueError, match="max_field_oil_t_per_dai"):
        config.load(_written(tmp_path, text))


def test_every_key_of_the_repository_config_is_known():
    """Сам `config.toml` обязан проходить ту же проверку — иначе она бесполезна."""
    loaded = config.load()

    assert loaded.limits is not None
def test_zero_event_quota_is_a_limit_not_its_absence(tmp_path):
    """«Мероприятий в этом году не делать» — условие, а не отсутствие условия."""
    text = MINIMAL.replace("water_tolerance = 1.0",
                           "water_tolerance = 1.0\nmax_events_per_year = 0")

    assert config.load(_written(tmp_path, text)).limits.max_events_per_year == 0


def test_a_share_outside_zero_to_one_is_refused(tmp_path):
    """`max_watercut = 90` вместо `0.9` молча отключало бы ограничение."""
    text = MINIMAL.replace("water_tolerance = 1.0",
                           "water_tolerance = 1.0\nmax_watercut = 90")

    with pytest.raises(ValueError, match="долей в диапазоне"):
        config.load(_written(tmp_path, text))

    good = MINIMAL.replace("water_tolerance = 1.0",
                           "water_tolerance = 1.0\nmax_watercut = 0.9")
    assert config.load(_written(tmp_path, good)).limits.max_watercut == 0.9
