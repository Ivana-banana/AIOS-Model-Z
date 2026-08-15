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
    fluids: Fluids
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
    fluids = _section(raw, "fluids", source)
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
        fluids=Fluids(
            oil_density=float(_value(fluids, "oil_density", "fluids", source)),
            water_density=float(_value(fluids, "water_density", "fluids", source)),
        ),
        policy=PolicyConfig(
            lookback_months=int(_value(policy, "lookback_months", "policy", source)),
        ),
        run=RunConfig(
            seed=int(_value(run, "seed", "run", source)),
            processes=int(_value(run, "processes", "run", source)),
        ),
        source=source,
    )


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
