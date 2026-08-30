"""Ночью собрать поисковые пары для конформного запаса одной командой.

    python scripts/calibrate_search.py --model artifacts/surrogate.joblib \
        --flow-args=--parsing-strictness=low

Каждый seed из `[trust].search_seeds` задаёт порядок координат совместного
поиска, поэтому получается отдельный кандидат оптимизатора. Для каждого
кандидата выполняется весь `run_all.py` (включая полный ГГДМ), затем выгрузка
и, в конце, `collect_pairs.py --append`. Ручные сценарии сюда не попадают.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402


def commands(*, model: str, base: Path, out_root: Path, config: str,
             seeds: tuple[int, ...], processes: int, flow_args: str,
             pairs: Path, family: str) -> list[list[str]]:
    """План отделён от subprocess: ночной набор проверяется без симулятора."""
    result: list[list[str]] = []
    runs: list[str] = []
    for seed in seeds:
        run = out_root / f"search-seed-{seed}"
        runs.append(str(run))
        result.append([sys.executable, str(ROOT / "scripts" / "run_all.py"),
                       "--skip-base", "--base", str(base), "--out", str(run),
                       "--config", config, "--surrogate-model", model,
                       "--search-seed", str(seed), "--processes", str(processes),
                       *([f"--flow-args={flow_args}"] if flow_args else [])])
        result.append([sys.executable, str(ROOT / "scripts" / "export_rows.py"), str(run),
                       "--config", config])
    result.append([sys.executable, str(ROOT / "scripts" / "collect_pairs.py"),
                   "--model", model, "--runs", *runs,
                   "--family", *(family for _ in runs), "--out", str(pairs),
                   "--append", "--config", config])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="тот же обученный суррогат, что ведёт поиск")
    parser.add_argument("--base", default=None, help="готовый базовый расчёт; по умолчанию [case].run_dir")
    parser.add_argument("--out-root", default="runs/search-calibration")
    parser.add_argument("--flow-args", default="")
    parser.add_argument("--config", default=str(config_module.DEFAULT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = config_module.load(args.config)
    if not settings.trust.search_seeds:
        print("в конфиге нет [trust].search_seeds: ночной поиск нечем воспроизвести",
              file=sys.stderr)
        return 1
    plan = commands(model=args.model, base=Path(args.base) if args.base else settings.case.run_dir,
                    out_root=Path(args.out_root), config=args.config,
                    seeds=settings.trust.search_seeds,
                    processes=settings.run.processes, flow_args=args.flow_args,
                    pairs=settings.trust.pairs_file, family=settings.trust.pairs_family)
    for command in plan:
        print(" ".join(command), flush=True)
        if not args.dry_run and subprocess.run(command, cwd=ROOT).returncode:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
