"""Собрать датасет «управление → отклик» из посчитанных сценариев DoE.

    python scripts/build_dataset.py --run-root runs/doe --out data/dataset.csv

Одна строка = скважина × месяц одного сценария: то же, что отдаёт симулятор в
контракте 14 колонок расчётчика, плюс столбцы управления добычей и закачкой
этой скважины в этом сценарии. Именно на этом учится
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
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, summary  # noqa: E402
from aios.deck import ScheduleStep, parse_schedule  # noqa: E402

CONTROL_COLUMNS = [
    "scenario", "multiplier", "production_multiplier", "injection_multiplier", "is_shut"
]


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
    controls = {
        scenario["name"]: (
            scenario.get("multipliers", {}), scenario.get("injector_multipliers", {})
        )
        for scenario in manifest["scenarios"]
    }
    reference = parse_schedule(config.case.model_dir / "Model_Z_sch.inc")

    target = Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTROL_COLUMNS + economics.REQUIRED_COLUMNS)
        writer.writeheader()

        for name in sorted(controls):
            case = root / name / config.case.case_name
            if not case.with_suffix(".UNSMRY").is_file():
                print(f"{name}: не досчитан, пропускаю")
                skipped += 1
                continue

            rows = summary.monthly_rows(case, fluids=config.fluids)
            roles = control_roles(reference.steps, {str(row["DATA"]) for row in rows})
            problems = summary.sanity_problems(rows)
            if problems:
                print(f"{name}: санити-чеки — нарушений {len(problems)}, первое: {problems[0]}")

            producer_multipliers, injector_multipliers = controls[name]
            for row in rows:
                role = roles.get((str(row["DATA"]), row["well"]))
                producer = (
                    producer_multipliers.get(row["well"]) if role == "production" else None
                )
                injector = (
                    injector_multipliers.get(row["well"]) if role == "injection" else None
                )
                multiplier = producer if producer is not None else injector
                writer.writerow({
                    "scenario": name,
                    "multiplier": "" if multiplier is None else multiplier,
                    "production_multiplier": "" if producer is None else producer,
                    "injection_multiplier": "" if injector is None else injector,
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


def control_roles(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...], dates: set[str]
) -> dict[tuple[str, str], str]:
    """Активная роль скважины по исходному расписанию, включая нулевой отклик."""
    ordered_steps = sorted(steps, key=lambda step: step.date)
    producer_status: dict[str, str] = {}
    injector_status: dict[str, str] = {}
    last_role: dict[str, str] = {}
    result: dict[tuple[str, str], str] = {}
    index = 0

    for value in sorted(dates):
        current = date.fromisoformat(value[:10])
        while index < len(ordered_steps) and ordered_steps[index].date <= current:
            step = ordered_steps[index]
            for control in step.producers:
                producer_status[control.well] = control.status
                if control.status == "OPEN":
                    last_role[control.well] = "production"
            for control in step.injectors:
                injector_status[control.well] = control.status
                if control.status == "OPEN":
                    last_role[control.well] = "injection"
            index += 1

        for well in producer_status.keys() | injector_status.keys():
            if producer_status.get(well) == "OPEN" and injector_status.get(well) != "OPEN":
                role = "production"
            elif injector_status.get(well) == "OPEN" and producer_status.get(well) != "OPEN":
                role = "injection"
            else:
                role = last_role.get(well)
            if role is not None:
                result[(value, well)] = role
    return result


if __name__ == "__main__":
    raise SystemExit(main())
