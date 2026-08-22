"""Замороженный внешний тест: сценарии, которых модель не видит никогда.

Зачем отдельно от train/test. Наше разбиение `split_scenarios` делит сценарии на
обучающие и отложенные по seed — и этого достаточно, чтобы мерить обобщение
**одной** модели. Но как только мы начинаем выбирать между моделями,
признаками, архитектурами и порогами по этой же отложенной части, она перестаёт
быть тестом: каждый выбор понемногу подгоняет решение под неё. К 22.08 мы
сделали таких выборов десятки — `raw` против `watercut`, `all` против `own`,
MLP против линейной, четыре порога допуска.

Разбор 4-6 (`docs/manus/round4.md`) формулирует требование прямо: заморозить
часть сценариев **до** экспериментов и не использовать их ни для выбора модели,
ни для настройки ядра, ни для выбора следующей точки DoE. Только тогда сравнение
«48 сценариев против 48 + новых» честно, и только тогда фраза «адаптивные точки
улучшили суррогат» вообще может быть проверена.

Как это устроено здесь:

* список сценариев лежит **файлом** (`data/holdout.json`), а не выводится из
  seed при каждом запуске: seed мы меняем, а замороженный тест обязан пережить
  смену seed, иначе он не заморожен;
* файл создаётся один раз командой и после этого не пересобирается — попытка
  пересобрать поверх существующего отвергается;
* `train` исключает эти сценарии целиком: они не идут ни в обучение, ни в ту
  отложенную часть, по которой считаются метрики допуска.

Чего файл **не** делает: он не мешает посчитать на замороженных сценариях
финальную ошибку — для этого он и нужен. Он мешает делать это **до** того, как
все решения приняты, и мешает случайно обучиться на них.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PATH = Path("data/holdout.json")

MIN_REMAINING = 12
"""Сколько сценариев обязано остаться на обучение и отбор после заморозки.

Меньше — и разбиение train/test внутри оставшегося перестаёт иметь смысл:
`split_scenarios` требует минимум трёх, но три сценария на 112 управляющих
признаков не измеряют ничего.
"""


@dataclass(frozen=True)
class Holdout:
    """Замороженный внешний тест и обстоятельства его заморозки."""

    scenarios: frozenset[str]
    seed: int
    frozen_at: str
    source: tuple[str, ...]
    """Из какого набора сценариев отбирали — чтобы было видно, что он не менялся."""

    note: str = ""

    def excludes(self, scenario: str) -> bool:
        return scenario in self.scenarios

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenarios": sorted(self.scenarios),
            "seed": self.seed,
            "frozen_at": self.frozen_at,
            "source": list(self.source),
            "note": self.note,
            "rule": (
                "эти сценарии не участвуют ни в обучении, ни в выборе модели, "
                "признаков и порогов; они считаются один раз, когда все решения приняты"
            ),
        }


def freeze(scenarios: list[str], *, seed: int, count: int,
           note: str = "") -> Holdout:
    """Отобрать `count` сценариев во внешний тест. Отбор случайный по seed.

    Случайный, а не «самые непохожие»: выбор по непохожести — уже решение,
    принятое по данным, и замороженный тест перестал бы быть независимым от них.
    """
    unique = sorted(set(scenarios))
    if count < 1:
        raise ValueError("замораживать ноль сценариев бессмысленно")
    if len(unique) - count < MIN_REMAINING:
        raise ValueError(
            f"сценариев {len(unique)}, заморозить просят {count} — на обучение и "
            f"отбор осталось бы меньше {MIN_REMAINING}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(unique, dtype=object), size=count, replace=False)
    return Holdout(
        scenarios=frozenset(str(value) for value in chosen),
        seed=seed,
        frozen_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        source=tuple(unique),
        note=note,
    )


def save(holdout: Holdout, path: Path = DEFAULT_PATH) -> None:
    """Записать файл. Поверх существующего — отказ, и это главное свойство модуля.

    Замороженный тест, который можно пересобрать, не заморожен. Если он
    действительно устарел — набор сценариев вырос, — файл удаляется руками, и это
    видимое решение с отметкой в журнале, а не побочный эффект команды.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} уже существует: замороженный тест не пересобирают. "
            "Если он устарел — удалите файл руками и запишите это в docs/experiments.md"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(holdout.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load(path: Path = DEFAULT_PATH) -> Holdout | None:
    """Прочитать файл. `None` — внешнего теста нет, и это надо говорить вслух."""
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Holdout(
        scenarios=frozenset(str(name) for name in raw["scenarios"]),
        seed=int(raw["seed"]),
        frozen_at=str(raw.get("frozen_at", "")),
        source=tuple(str(name) for name in raw.get("source", [])),
        note=str(raw.get("note", "")),
    )


def leaked(holdout: Holdout | None, used: list[str] | frozenset[str]) -> list[str]:
    """Какие замороженные сценарии просочились в обучение. Пусто — не просочились.

    Проверка существует потому, что утечка бесшумна: модель, обучённая на
    замороженном сценарии, покажет на нём прекрасную ошибку и подтвердит любое
    улучшение, которого нет.
    """
    if holdout is None:
        return []
    return sorted(holdout.scenarios & frozenset(used))
