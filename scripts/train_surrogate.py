"""Обучить MLP-суррогат и сохранить модель с метриками held-out сценариев.

    python scripts/train_surrogate.py --dataset data/dataset.csv data/dataset-levels.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from aios import config as config_module  # noqa: E402
from aios import holdout as holdout_module  # noqa: E402
from aios.neural import (  # noqa: E402
    FEATURE_KINDS,
    TrainingReport,
    assert_quality,
    train,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", nargs="+", default=["data/dataset.csv", "data/dataset-levels.csv"],
        help="один или несколько CSV; одинаковые имена сценариев разделяются по имени файла",
    )
    parser.add_argument("--out", default="artifacts/surrogate.joblib")
    parser.add_argument("--report", default="artifacts/surrogate-metrics.json")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument(
        "--targets", choices=("raw", "watercut"), default="raw",
        help="как представлять нефть: своей величиной или через долю воды",
    )
    parser.add_argument(
        "--features", choices=FEATURE_KINDS, default="own",
        help="что видит модель скважины: весь фонд (all) или свои столбцы плюс "
             "агрегаты по фонду (own)",
    )
    parser.add_argument(
        "--model", choices=("mlp", "linear"), default="mlp",
        help="какая из двух обученных моделей поедет в цикл; меряются обе в любом случае",
    )
    parser.add_argument(
        "--holdout", default=str(holdout_module.DEFAULT_PATH),
        help="файл замороженного внешнего теста; его сценарии исключаются целиком",
    )
    arguments = parser.parse_args(argv)

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    settings = config_module.load(arguments.config)
    frame = load_datasets(arguments.dataset)
    frozen = holdout_module.load(Path(arguments.holdout))
    if frozen is None:
        print(f"внешнего теста нет ({arguments.holdout}): сравнение моделей между "
              "собой сделано на той же выборке, по которой их и выбирали")
    else:
        print(f"замороженный внешний тест: {len(frozen.scenarios)} сценариев, "
              f"заморожен {frozen.frozen_at}")
    # метрики считаются только на управляемой части горизонта: история у всех
    # сценариев одна и по входу, и по отклику, поэтому по ней меряется память
    bundle = train(frame, seed=settings.run.seed, kind=arguments.model,
                   control_start=settings.case.control_start,
                   targets=arguments.targets, features=arguments.features,
                   holdout=frozen.scenarios if frozen else None)
    metadata = {
        "seed": settings.run.seed,
        "datasets": dataset_fingerprints(arguments.dataset),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }
    bundle.metadata.update(metadata)

    output = Path(arguments.out)
    report_path = Path(arguments.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    report = {**asdict(bundle.report), "training": metadata}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"модель: {output}")
    print(f"отчёт:  {report_path}")
    print(f"train/test сценариев: {len(bundle.report.train_scenarios)}/"
          f"{len(bundle.report.test_scenarios)}")
    print(f"в модель сохранён: {bundle.report.kind}, цели: {bundle.report.targets}, "
          f"признаки: {bundle.report.features}")
    print(f"метрики считаны на {bundle.report.metric_rows} строках"
          + (f" с {bundle.report.metric_start}" if bundle.report.metric_start else ""))
    for dataset in metadata["datasets"]:
        print(f"данные: {dataset['path']} ({dataset['bytes']} байт), sha256={dataset['sha256']}")
    print(f"порог качества: {quality_verdict(bundle.report)}")
    for target, metric in bundle.report.metrics.items():
        print(
            f"{target}: MLP MAE={metric.neural_mae:.4g}, "
            f"RMSE={metric.neural_rmse:.4g}, R²={metric.neural_r2:.4g}; "
            f"linear MAE={metric.linear_mae:.4g}, RMSE={metric.linear_rmse:.4g}, "
            f"R²={metric.linear_r2:.4g}"
        )
    print("R² выше — по цели целиком, и в ней около 92 % занимает уровень скважины "
          "(замер 21.08): его высокое значение получает и модель, которая про "
          "управление не знает вовсе. Ниже — та же ошибка, отнесённая к дисперсии "
          "отклика, то есть к той части, по градиенту которой ходит оптимизатор.")
    for target, metric in bundle.report.metrics.items():
        print(
            f"{target}: уровень R²={_r2(metric.level_r2)}; отклик R² — "
            f"MLP={_r2(metric.neural_response_r2)}, "
            f"linear={_r2(metric.linear_response_r2)}, "
            f"наклон={_r2(metric.slope_response_r2)}"
        )
    print(f"планка отклика: {response_verdict(bundle.report)}")
    return 0


def _r2(value: float | None) -> str:
    """`None` — это «не измерено»: у постоянной цели знаменатель `R²` нулевой."""
    return "нет" if value is None else f"{value:.4g}"


def response_verdict(report: TrainingReport) -> str:
    """Бьёт ли уехавшая в цикл модель тупой бейзлайн «свой наклон» — одной строкой.

    Порог этого не требует: `assert_quality` отвечает за «не хуже, чем ничего».
    Но принцип 3 `CLAUDE.md` требует, чтобы каждое усложнение доказывало
    превосходство над предыдущим уровнем числом, а не намерением, — и число это
    видно должно быть при обучении, а не восстанавливаться потом из журнала.
    """
    field = "linear_response_r2" if report.kind == "linear" else "neural_response_r2"
    lines = []
    for target in ("WLPT_Diff", "WOMT_Diff"):
        metric = report.metrics[target]
        model = getattr(metric, field, None)
        slope = getattr(metric, "slope_response_r2", None)
        if model is None or slope is None:
            lines.append(f"{target}: отклик не измерен")
            continue
        lines.append(f"{target}: {model:.4g} против {slope:.4g} — "
                     + ("бьёт" if model > slope else "НЕ бьёт"))
    return "; ".join(lines)


def quality_verdict(report: TrainingReport) -> str:
    """Проходит ли сеть порог против линейного бейзлайна — одной строкой.

    Сам порог применяется не здесь, а при загрузке артефакта в цикл
    (`NeuralSurrogate.create`), и это правильно: непрошедшую модель полезно
    сохранить и разобрать. Но узнавать об отказе при запуске цикла — поздно:
    метрики этого прогона уже уйдут в журнал как «сеть обучена». Поэтому вердикт
    печатается сразу, рядом с метриками.
    """
    try:
        assert_quality(report)
    except ValueError as error:
        return f"НЕ ПРОЙДЕН — {error}; в цикл такая модель не загрузится"
    if report.kind == "linear":
        return ("пройден, линейный суррогат объясняет отложенные сценарии и отклик "
                "на управление (порог «лучше бейзлайна» к нему неприменим — он и "
                "есть бейзлайн)")
    return ("пройден, MLP лучше линейного бейзлайна и объясняет отклик на "
            "управление по нефти и жидкости")


def load_datasets(paths: list[str]) -> pd.DataFrame:
    """Объединить пакеты DoE, сохранив уникальную идентичность их сценариев."""
    ordered = sorted(paths, key=lambda value: Path(value).as_posix().casefold())
    labels = [Path(path).stem for path in ordered]
    if len(labels) != len(set(labels)):
        raise ValueError("имена файлов датасетов должны различаться")

    frames = []
    for path, label in zip(ordered, labels, strict=True):
        frame = pd.read_csv(path)
        if "scenario" not in frame:
            raise ValueError(f"в {path} нет колонки scenario")
        frame["scenario"] = label + ":" + frame["scenario"].astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("не передан ни один датасет")
    return pd.concat(frames, ignore_index=True)


def dataset_fingerprints(paths: list[str]) -> list[dict[str, str | int]]:
    """Хэши исходных CSV для связи модели и отчёта с точной версией данных."""
    fingerprints = []
    for value in sorted(paths, key=lambda item: Path(item).as_posix().casefold()):
        path = Path(value)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprints.append(
            {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    return fingerprints


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
