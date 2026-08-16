"""Обучить MLP-суррогат и сохранить модель с метриками held-out сценариев.

    python scripts/train_surrogate.py --dataset data/dataset.csv data/dataset-levels.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from aios import config as config_module  # noqa: E402
from aios.neural import train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", nargs="+", default=["data/dataset.csv", "data/dataset-levels.csv"],
        help="один или несколько CSV; одинаковые имена сценариев разделяются по имени файла",
    )
    parser.add_argument("--out", default="artifacts/surrogate.joblib")
    parser.add_argument("--report", default="artifacts/surrogate-metrics.json")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    settings = config_module.load(arguments.config)
    frame = load_datasets(arguments.dataset)
    bundle = train(frame, seed=settings.run.seed)

    output = Path(arguments.out)
    report_path = Path(arguments.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    report_path.write_text(
        json.dumps(asdict(bundle.report), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"модель: {output}")
    print(f"отчёт:  {report_path}")
    print(f"train/test сценариев: {len(bundle.report.train_scenarios)}/"
          f"{len(bundle.report.test_scenarios)}")
    for target, metric in bundle.report.metrics.items():
        print(f"{target}: MLP MAE={metric.neural_mae:.4g}, linear MAE={metric.linear_mae:.4g}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
