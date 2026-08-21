"""Чтение `config.toml`.

Типизированные структуры и явные поля — не конфиг-движок. Никакого слияния
слоёв, наследования профилей и значений по умолчанию, размазанных по коду:
пропущенный ключ здесь — ошибка, а не «возьмём разумное значение». Иначе
воспроизвести прогон по одному файлу невозможно, а регламент требует именно
этого.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aios.explain import Settings as ExplainSettings
from aios.planner import Limits
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
