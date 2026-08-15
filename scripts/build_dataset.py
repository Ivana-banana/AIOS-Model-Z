"""Собрать датасет «управление → отклик» из посчитанных сценариев DoE.

    python scripts/build_dataset.py --run-root runs/doe --out data/dataset.csv

Одна строка = скважина × месяц одного сценария: то же, что отдаёт симулятор в
контракте 14 колонок расчётчика, плюс два столбца управления — множитель дебита
этой скважины в этом сценарии и признак остановки. Именно на этом учится
суррогат: вход — управление и состояние, выход — отклик объекта.

Множители берутся из манифеста `doe.json`, который пишет `build_doe.py`. Без
манифеста связать отклик с управлением нечем: имя каталога не несёт значений.

Сценарии, которые не досчитались, пропускаются с явным сообщением — молча
недособранный датасет хуже, чем его отсутствие.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, summary  # noqa: E402

CONTROL_COLUMNS = ["scenario", "multiplier", "is_shut"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", default="runs/doe")
    parser.add_argument("--out", default="data/dataset.csv")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    root = Path(arguments.run_root)
    manifest_path = root / "doe.json"
    if not manifest_path.is_file():
        print(f"нет манифеста {manifest_path}: пересоберите пакет через scripts/build_doe.py",
              file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    multipliers = {s["name"]: s["multipliers"] for s in manifest["scenarios"]}

    target = Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTROL_COLUMNS + economics.REQUIRED_COLUMNS)
        writer.writeheader()

        for name in sorted(multipliers):
            case = root / name / config.case.case_name
            if not case.with_suffix(".UNSMRY").is_file():
                print(f"{name}: не досчитан, пропускаю")
                skipped += 1
                continue

            rows = summary.monthly_rows(case, fluids=config.fluids)
            problems = summary.sanity_problems(rows)
            if problems:
                print(f"{name}: санити-чеки — нарушений {len(problems)}, первое: {problems[0]}")

            scenario_multipliers = multipliers[name]
            for row in rows:
                multiplier = scenario_multipliers.get(row["well"])
                writer.writerow({
                    "scenario": name,
                    # нагнетательных скважин в DoE не крутили: у них множителя нет,
                    # и подставлять 1.0 нельзя — это разные вещи, «не управляли» и
                    # «управляли, оставив как есть»
                    "multiplier": "" if multiplier is None else multiplier,
                    "is_shut": "" if multiplier is None else int(multiplier == 0.0),
                    **row,
                })
            written += 1
            print(f"{name}: {len(rows)} строк")

    size_mb = target.stat().st_size / 1e6
    print()
    print(f"датасет:   {target} ({size_mb:.1f} МБ)")
    print(f"сценариев: {written} собрано, {skipped} пропущено")
    if skipped:
        print("пропущенные сценарии придётся досчитать: суррогат, обученный на дырявой "
              "выборке, врёт именно в дырах")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
