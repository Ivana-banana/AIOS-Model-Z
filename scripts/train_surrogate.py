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
from aios.neural import TrainingReport, assert_quality, train  # noqa: E402


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
        "--model", choices=("mlp", "linear"), default="mlp",
        help="какая из двух обученных моделей поедет в цикл; меряются обе в любом случае",
    )
    arguments = parser.parse_args(argv)

    settings = config_module.load(arguments.config)
    frame = load_datasets(arguments.dataset)
    # метрики считаются только на управляемой части горизонта: история у всех
    # сценариев одна и по входу, и по отклику, поэтому по ней меряется память
    bundle = train(frame, seed=settings.run.seed, kind=arguments.model,
                   control_start=settings.case.control_start)
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
    print(f"в модель сохранён: {bundle.report.kind}")
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
    return 0


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
        return ("пройден, линейный суррогат объясняет отложенные сценарии "
                "(порог «лучше бейзлайна» к нему неприменим — он и есть бейзлайн)")
    return "пройден, MLP лучше линейного бейзлайна по нефти и жидкости"


def load_datasets(paths: list[str]) -> pd.DataFrame:
    """Объединить пакеты DoE, сохранив уникальную идентичность их сценариев."""
    labels = [Path(path).stem for path in paths]
    if len(labels) != len(set(labels)):
        raise ValueError("имена файлов датасетов должны различаться")

    frames = []
    for path, label in zip(paths, labels, strict=True):
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
    for value in paths:
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
