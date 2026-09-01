"""Чтение `config.toml`.

Типизированные структуры и явные поля — не конфиг-движок. Никакого слияния
слоёв, наследования профилей и значений по умолчанию, размазанных по коду:
пропущенный ключ здесь — ошибка, а не «возьмём разумное значение». Иначе
воспроизвести прогон по одному файлу невозможно, а регламент требует именно
этого.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

from aios.explain import Settings as ExplainSettings
from aios.planner import Limits, ShutWindow
from aios.summary import Fluids

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class CaseConfig:
    model_dir: Path
    run_dir: Path
    case_name: str
    control_start: date


@dataclass(frozen=True)
class EconomicsConfig:
    start_date: date
    norms_file: Path


@dataclass(frozen=True)
class PolicyConfig:
    lookback_months: int


@dataclass(frozen=True)
class RunConfig:
    seed: int
    processes: int


@dataclass(frozen=True)
class Config:
    case: CaseConfig
    economics: EconomicsConfig
    explain: ExplainSettings
    fluids: Fluids
    limits: Limits
    policy: PolicyConfig
    run: RunConfig
    source: Path
    """Откуда прочитано — идёт в отчёт вместе с числом ЧДД."""


def load(path: str | Path = DEFAULT_PATH) -> Config:
    """Прочитать конфигурацию. Отсутствие ключа — ошибка с внятным текстом."""
    source = Path(path)
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    root = source.parent
    case = _section(raw, "case", source)
    economics = _section(raw, "economics", source)
    explain = _section(raw, "explain", source)
    fluids = _section(raw, "fluids", source)
    limits = _section(raw, "limits", source)
    _reject_unknown_limits(limits, source)
    policy = _section(raw, "policy", source)
    run = _section(raw, "run", source)

    return Config(
        case=CaseConfig(
            model_dir=root / _value(case, "model_dir", "case", source),
            run_dir=root / _value(case, "run_dir", "case", source),
            case_name=_value(case, "case_name", "case", source),
            control_start=_date(case, "control_start", "case", source),
        ),
        economics=EconomicsConfig(
            start_date=_date(economics, "start_date", "economics", source),
            norms_file=root / _value(economics, "norms_file", "economics", source),
        ),
        explain=ExplainSettings(
            enabled=bool(_value(explain, "enabled", "explain", source)),
            # Единственные два ключа с умолчанием, и это осознанное исключение из
            # правила «пропущенный ключ — ошибка». Оба не влияют на ЧДД вовсе, а
            # их отсутствие означает ровно то, что было до 23.08: поставщик
            # Anthropic и никакого каталога. Требовать их значило бы сломать
            # чужой конфиг ради настройки, которая на число не действует.
            provider=str(explain.get("provider", "anthropic")),
            folder_id=str(explain.get("folder_id", "")),
            model=str(_value(explain, "model", "explain", source)),
            max_tokens=_at_least_one(explain, "max_tokens", "explain", source),
            effort=str(_value(explain, "effort", "explain", source)),
            cache_dir=root / _value(explain, "cache_dir", "explain", source),
        ),
        fluids=Fluids(
            oil_density=_positive(fluids, "oil_density", "fluids", source),
            water_density=_positive(fluids, "water_density", "fluids", source),
        ),
        limits=Limits(
            # ноль в конфигурации означает «предела нет»: в TOML нет пустого значения,
            # а держать магическую строку хуже, чем один задокументированный ноль
            max_field_liquid_m3_per_day=(
                _non_negative(limits, "max_field_liquid_m3_per_day", "limits", source) or None
            ),
            max_field_injection_m3_per_day=(
                _non_negative(limits, "max_field_injection_m3_per_day", "limits", source) or None
            ),
            enforce_water_balance=bool(
                _value(limits, "enforce_water_balance", "limits", source)
            ),
            max_external_water_m3_per_day=(
                _non_negative(limits, "max_external_water_m3_per_day", "limits", source) or None
            ),
            water_balance_rule=str(
                _value(limits, "water_balance_rule", "limits", source)
            ),
            water_tolerance=_positive(limits, "water_tolerance", "limits", source),
            # Нижняя граница коридора — опциональная: ключа нет, значит снизу
            # закачка не ограничена. Обязательной её делать нельзя, иначе каждый
            # существующий конфиг стал бы нечитаемым, а заявляемое число
            # изменилось бы от одной правки схемы.
            water_tolerance_min=_optional_positive(limits, "water_tolerance_min",
                                                   "limits", source),
            # Ручки под валидационный кейс (Р-110). У всех умолчание «ключа нет —
            # ограничения нет»: это единственные ключи в конфигурации, которые
            # можно не писать, и так сделано намеренно. Кейс придёт со своим
            # набором ограничений, и требовать все восемь от каждого конфига
            # значило бы ломать чужой файл ради настройки, которой в нём нет.
            min_producer_bhp_atm=_optional_positive(limits, "min_producer_bhp_atm",
                                                    "limits", source),
            max_injector_bhp_atm=_optional_positive(limits, "max_injector_bhp_atm",
                                                    "limits", source),
            min_reservoir_pressure_atm=_optional_positive(
                limits, "min_reservoir_pressure_atm", "limits", source),
            max_reservoir_pressure_atm=_optional_positive(
                limits, "max_reservoir_pressure_atm", "limits", source),
            max_well_injection_m3_per_day=_optional_positive(
                limits, "max_well_injection_m3_per_day", "limits", source),
            max_well_liquid_m3_per_day=_optional_positive(
                limits, "max_well_liquid_m3_per_day", "limits", source),
            well_caps_m3_per_day=_caps(limits, "well_caps_m3_per_day", "скважина", source),
            well_groups=_well_groups(limits, source),
            group_liquid_caps_m3_per_day=_caps(
                limits, "group_liquid_caps_m3_per_day", "группа", source),
            group_injection_caps_m3_per_day=_caps(
                limits, "group_injection_caps_m3_per_day", "группа", source),
            max_field_oil_t_per_day=_optional_positive(
                limits, "max_field_oil_t_per_day", "limits", source),
            max_field_water_t_per_day=_optional_positive(
                limits, "max_field_water_t_per_day", "limits", source),
            controllable_wells=_well_set(limits, "controllable_wells", source),
            forced_shut_wells=_well_set(limits, "forced_shut_wells", source),
            forced_shut_windows=_shut_windows(limits, source),
            max_active_wells=_optional_count(limits, "max_active_wells", "limits", source),
            max_field_injection_m3_per_year=_optional_positive(
                limits, "max_field_injection_m3_per_year", "limits", source),
            max_field_liquid_m3_per_year=_optional_positive(
                limits, "max_field_liquid_m3_per_year", "limits", source),
            field_limits_are_annual_average=bool(
                limits.get("field_limits_are_annual_average", False)),
            max_events_per_year=_optional_count(limits, "max_events_per_year",
                                                "limits", source),
            max_events_per_month=_optional_count(limits, "max_events_per_month",
                                                 "limits", source),
            min_regime_months=_optional_count(limits, "min_regime_months",
                                              "limits", source),
            min_producer_rate_m3_per_day=_optional_positive(
                limits, "min_producer_rate_m3_per_day", "limits", source),
            max_watercut=_optional_positive(limits, "max_watercut", "limits", source),
            max_field_liquid_m3_total=_optional_positive(
                limits, "max_field_liquid_m3_total", "limits", source),
            max_field_injection_m3_total=_optional_positive(
                limits, "max_field_injection_m3_total", "limits", source),
        ),
        policy=PolicyConfig(
            lookback_months=_at_least_one(policy, "lookback_months", "policy", source),
        ),
        run=RunConfig(
            seed=int(_value(run, "seed", "run", source)),
            processes=_at_least_one(run, "processes", "run", source),
        ),
        source=source,
    )


def _optional_positive(section: dict, key: str, name: str, source: Path) -> float | None:
    """Необязательная величина: ключа нет или ноль — ограничения нет.

    Отличается от `_non_negative` тем, что отсутствие ключа законно. Так сделано
    только для ручек валидационного кейса: конфиг организаторов не обязан их
    содержать, а требовать — значит отвергать исправный файл.
    """
    if key not in section:
        return None
    value = float(section[key])
    if value < 0:
        raise ValueError(
            f"В {source} значение {key} в секции [{name}] не может быть отрицательным "
            f"(получено {value:g}); «ограничения нет» записывается нулём или пропуском"
        )
    return value or None


def _optional_count(section: dict, key: str, name: str, source: Path) -> int | None:
    """То же для счётчика: скважин, мероприятий."""
    if key not in section:
        return None
    value = int(section[key])
    if value < 0:
        raise ValueError(
            f"В {source} значение {key} в секции [{name}] не может быть отрицательным "
            f"(получено {value})"
        )
    return value or None


def _well_set(section: dict, key: str, source: Path) -> frozenset[str] | None:
    """Список скважин из конфигурации. Пустой список — ограничения нет.

    Имена приводятся к строкам: в TOML номер скважины легко написать числом, и
    тогда `102` не совпало бы с `"102"` из дека — молча и на всём фонде.
    """
    if key not in section:
        return None
    values = section[key]
    if not isinstance(values, list):
        raise ValueError(f"В {source} значение {key} в секции [limits] должно быть списком")
    names = frozenset(str(value) for value in values)
    return names or None


def _reject_unknown_limits(section: dict, source: Path) -> None:
    """Незнакомый ключ в `[limits]` — ошибка, а не «настройка на будущее».

    Умолчание каждой ручки кейса — «ограничения нет», и это делает опечатку
    невидимой: `max_field_oil_t_per_dai = 700` читается как «предела по нефти нет»,
    прогон проходит, репетиция печатает «пройдена», а условие кейса не применено
    вовсе. Найдено критиком 01.09. Поэтому список ключей закрыт: имена берутся из
    самой структуры `Limits`, то есть новая ручка разрешается автоматически, а
    опечатка — никогда.
    """
    known = {field.name for field in fields(Limits)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(
            f"В {source} в секции [limits] неизвестные ключи: {', '.join(unknown)}. "
            "Проверьте написание: незнакомый ключ читался бы как «ограничения нет», "
            f"и условие кейса не применилось бы молча. Известные ключи: "
            f"{', '.join(sorted(known))}"
        )


def _shut_windows(section: dict, source: Path) -> tuple[ShutWindow, ...] | None:
    """Обязательные остановки с окном дат. Пустой список — таких нет.

    Формат — список таблиц, потому что окон у одной скважины может быть несколько:

        [[limits.forced_shut_windows]]
        well = "102"
        start = 2027-01-01
        end = 2027-06-30

    Даты читаются нативным типом TOML, а не строкой: строка `"2027-13-01"`
    прошла бы разбор и сравнивалась бы лексикографически — тихо и на всём
    горизонте.
    """
    key = "forced_shut_windows"
    if key not in section:
        return None
    entries = section[key]
    if not isinstance(entries, list):
        raise ValueError(
            f"В {source} значение {key} в секции [limits] должно быть списком таблиц "
            "вида {well, start, end}"
        )
    windows: list[ShutWindow] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"В {source} запись {key} должна быть таблицей {{well, start, end}}, "
                f"получено {entry!r}"
            )
        missing = [field for field in ("well", "start", "end") if field not in entry]
        if missing:
            raise ValueError(
                f"В {source} в записи {key} не хватает полей: {', '.join(missing)}"
            )
        for field in ("start", "end"):
            if not isinstance(entry[field], date):
                raise ValueError(
                    f"В {source} поле {field} записи {key} должно быть датой TOML "
                    f"(2027-01-01), получено {entry[field]!r}"
                )
        windows.append(ShutWindow(well=str(entry["well"]),
                                  start=entry["start"], end=entry["end"]))
    return tuple(windows) or None


def _caps(section: dict, key: str, whose: str, source: Path) -> dict[str, float] | None:
    """Карта «имя → потолок, м³/сут». Пустая таблица — ограничения нет.

    Одна функция на скважинные и групповые потолки: разница между ними только в
    том, чьё имя стоит слева, а проверки и способ ошибиться одни и те же.
    """
    if key not in section:
        return None
    table = section[key]
    if not isinstance(table, dict):
        raise ValueError(
            f"В {source} значение {key} в секции [limits] должно быть таблицей "
            f"вида {{{whose} = потолок}}"
        )
    caps: dict[str, float] = {}
    for name, cap in table.items():
        value = float(cap)
        if value < 0:
            raise ValueError(
                f"В {source} потолок ({whose} {name}) отрицателен ({value:g})"
            )
        caps[str(name)] = value
    return caps or None


def _well_groups(section: dict, source: Path) -> dict[str, frozenset[str]] | None:
    """Группы скважин: `{"куст-1" = ["102", "305"]}`. Пустая таблица — групп нет.

    Скважина может входить в несколько групп: куст и блок заводнения — разные
    разрезы одного фонда, и запрещать пересечения значило бы выражать не тот
    объект. Имена приводятся к строкам по той же причине, что и везде: номер
    скважины в TOML легко написать числом.
    """
    key = "well_groups"
    if key not in section:
        return None
    table = section[key]
    if not isinstance(table, dict):
        raise ValueError(
            f"В {source} значение {key} в секции [limits] должно быть таблицей "
            "вида {группа = [скважины]}"
        )
    groups: dict[str, frozenset[str]] = {}
    for name, wells in table.items():
        if not isinstance(wells, list):
            raise ValueError(
                f"В {source} состав группы {name} должен быть списком скважин, "
                f"получено {wells!r}"
            )
        members = frozenset(str(well) for well in wells)
        if not members:
            raise ValueError(
                f"В {source} группа {name} пуста: пустая группа не выражает ограничения, "
                "а прячет его"
            )
        groups[str(name)] = members
    return groups or None


def _positive(section: dict, key: str, name: str, source: Path) -> float:
    """Величина, для которой ноль и минус не имеют смысла: плотность, допуск."""
    value = float(_value(section, key, name, source))
    if value <= 0:
        raise ValueError(
            f"В {source} значение {key} в секции [{name}] должно быть больше нуля, "
            f"получено {value:g}"
        )
    return value


def _non_negative(section: dict, key: str, name: str, source: Path) -> float:
    """Предел уровня объекта: ноль означает «предела нет», минус — опечатка.

    Отрицательный предел не отвергается сам собой: планировщик примет его за
    настоящий и срежет фонд в ноль, а прогон будет выглядеть обычным.
    """
    value = float(_value(section, key, name, source))
    if value < 0:
        raise ValueError(
            f"В {source} значение {key} в секции [{name}] не может быть отрицательным "
            f"(получено {value:g}); «предела нет» записывается нулём"
        )
    return value


def _at_least_one(section: dict, key: str, name: str, source: Path) -> int:
    """Счётчик: процессы, месяцы, токены. Ноль здесь — не «выключено», а поломка."""
    value = int(_value(section, key, name, source))
    if value < 1:
        raise ValueError(
            f"В {source} значение {key} в секции [{name}] должно быть не меньше единицы, "
            f"получено {value}"
        )
    return value


def _section(raw: dict, name: str, source: Path) -> dict:
    if name not in raw:
        raise KeyError(f"В {source} нет секции [{name}]")
    return raw[name]


def _value(section: dict, key: str, name: str, source: Path):
    if key not in section:
        raise KeyError(f"В {source} в секции [{name}] нет ключа {key}")
    return section[key]


def _date(section: dict, key: str, name: str, source: Path) -> date:
    value = _value(section, key, name, source)
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
