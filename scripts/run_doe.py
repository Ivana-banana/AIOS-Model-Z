"""Возобновляемый пакетный запуск подготовленных сценариев DoE.

    python scripts/run_doe.py runs/doe-injection 4 -- --parsing-strictness=low
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_FLOW = ROOT / "scripts" / "run_flow.sh"


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_arguments:
        separator = raw_arguments.index("--")
        flow_args = raw_arguments[separator + 1:]
        raw_arguments = raw_arguments[:separator]
    else:
        flow_args = []

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_root", help="каталог с doe.json и подпапками doe-*")
    parser.add_argument("processes", type=int, nargs="?", default=1)
    parser.add_argument("--dry-run", action="store_true", help="только показать очередь")
    arguments = parser.parse_args(raw_arguments)

    run_root = Path(arguments.run_root)
    try:
        cases = discover_cases(run_root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    pending = [case for case in cases if not case.with_suffix(".UNSMRY").is_file()]
    completed = len(cases) - len(pending)
    print(f"пакет: {run_root}")
    print(f"готово: {completed}, в очереди: {len(pending)}, всего: {len(cases)}")
    if arguments.dry_run:
        for case in pending:
            print(case)
        return 0

    failures: list[dict[str, str | int]] = []
    for index, case in enumerate(pending, start=1):
        print(f"\n=== [{index}/{len(pending)}] {case.parent.name} ===", flush=True)
        command = [
            "bash", str(RUN_FLOW), str(case), str(arguments.processes), *flow_args
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0 or not case.with_suffix(".UNSMRY").is_file():
            failures.append({"case": case.parent.name, "returncode": result.returncode})

    report = {
        "updated_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "total": len(cases),
        "completed": sum(case.with_suffix(".UNSMRY").is_file() for case in cases),
        "failures": failures,
        "flow_args": flow_args,
    }
    report_path = run_root / "doe-status.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nстатус: {report_path}")
    print(f"готово: {report['completed']}/{report['total']}, ошибок: {len(failures)}")
    return 1 if failures or report["completed"] != report["total"] else 0


def discover_cases(run_root: Path) -> list[Path]:
    if not (run_root / "doe.json").is_file():
        raise ValueError(f"нет {run_root / 'doe.json'}: сначала запустите build_doe.py")
    cases = sorted(run_root.glob("doe-*/*.DATA"))
    if not cases:
        raise ValueError(f"в {run_root} нет собранных doe-*/*.DATA")
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
