"""Сколько пометок «вне обучающей выборки» даёт суррогат на расписании.

    python scripts/surrogate_warnings.py --model artifacts/surrogate.joblib
    python scripts/surrogate_warnings.py --model artifacts/surrogate.joblib --scenario conversions

Пометки — единственное, что отделяет «суррогат оценил» от «суррогат промолчал и
выдал число». Принцип 6 `CLAUDE.md`: сеть не падает, а тихо врёт, поэтому решение
за пределами обучающей выборки обязано уходить на полный расчёт, а не приниматься
по прогнозу.

Мерить их надо на **настоящем** расписании Модели Z, а не на синтетике из тестов,
и по двум крайностям сразу:

* опорное расписание — здесь пометок быть не должно вовсе. Любая означает ложное
  срабатывание, а защита, которая шумит на обычном режиме, перестаёт читаться;
* сценарий с переводом под закачку — здесь пометки обязаны быть: этот класс
  решений DoE не покрывает, и суррогат должен сказать «не оценивал».

Число печатается и по строкам, и по различным причинам: 188 строк «то же самое
про 188 скважино-месяцев» и 188 разных причин — это разные новости.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from aios import config as config_module  # noqa: E402
from aios import doe  # noqa: E402
from aios import surrogate as surrogate_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout, missing_deck_refusal  # noqa: E402
from aios.neural import NeuralBundle, NeuralSurrogate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument("--scenario", default=None,
                        choices=("producers", "injectors", "all", "conversions"),
                        help="дополнительно померить на сценарии DoE этого вида")
    arguments = parser.parse_args(argv)

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(arguments.config)
    if not arguments.model.is_file():
        # артефакт лежит в `artifacts/`, а этот каталог в git не ведётся: промах
        # мимо него на чистом клоне — обычный первый запуск, а не случайность
        print(f"нет обученного суррогата: {arguments.model}", file=sys.stderr)
        return 1
    bundle = joblib.load(arguments.model)
    if not isinstance(bundle, NeuralBundle):
        print(f"это не обученный суррогат: {type(bundle).__name__}", file=sys.stderr)
        return 1

    try:
        reference = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        print(missing_deck_refusal(config.case.model_dir, error), file=sys.stderr)
        return 1
    controlled = [step for step in reference.steps if step.date >= config.case.control_start]
    # порог качества здесь намеренно не применяется: мерить пометки надо и у
    # модели, которая в цикл не попадёт, — иначе про неё нечего сказать
    surrogate = NeuralSurrogate.create(
        bundle, controlled, config.fluids, require_quality=False
    )

    print(f"модель:    {arguments.model} ({bundle.report.kind})")
    print(f"контролей: {len(bundle.control_wells)}")
    print(f"шагов с {config.case.control_start}: {len(controlled)}")
    print()
    _report("опорное расписание (пометок быть не должно)",
            surrogate.in_training_range(controlled))

    if arguments.scenario:
        scenario = doe.design(
            reference.steps, samples=1, seed=config.run.seed,
            control_kind=arguments.scenario,
        )[0]
        planned = [step for step in scenario.steps if step.date >= config.case.control_start]
        _report(f"сценарий вида «{arguments.scenario}» (пометки обязаны быть)",
                surrogate.in_training_range(planned))
    return 0


def _report(title: str, warnings: list[str]) -> None:
    print(f"--- {title} ---")
    print(f"строк: {len(warnings)}")
    kinds = Counter(_kind(warning) for warning in warnings)
    for kind, count in kinds.most_common():
        print(f"  {count:>4}  {kind}")
    for warning in warnings[:3]:
        print(f"  пример: {warning}")
    print()


#: Классификатор причин переехал в `aios.surrogate`: сворачивать одинаково
#: должны и командная строка, и страница цикла (Б33). Имя оставлено — на нём
#: стоит тест, проверяющий, что ни одна настоящая пометка не попадает в «прочее».
_kind = surrogate_module.warning_kind


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
