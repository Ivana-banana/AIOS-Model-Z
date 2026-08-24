"""Сверить аннотированный тег заявки с текущим манифестом отправки."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_TAG = re.compile(r"`(submit-\d{4}-\d{2}-\d{2}(?:\.\d+)?)`")


@dataclass(frozen=True)
class Manifest:
    npv: str
    base: str
    uplift: str
    uplift_percent: str
    schedule: str
    md5: str


def _capture(pattern: str, text: str, label: str) -> tuple[str, ...]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"в docs/submission.md не найдено поле: {label}")
    return match.groups()


def declared_manifest(text: str) -> Manifest:
    """Извлечь ровно те поля, которые должны быть закреплены тегом заявки."""
    npv, = _capture(r"^\| \*\*ЧДД\*\* \| \*\*([\d ]+,\d) млн руб\.", text, "ЧДД")
    base, = _capture(
        r"^\| базовый сценарий[^\n]*\| ([\d ]+,\d) млн руб\.", text, "база"
    )
    uplift, uplift_percent = _capture(
        r"^\| прирост \| \*\*\+([\d ]+,\d) млн руб\. \(\+([\d,]+) %\)",
        text,
        "прирост",
    )
    schedule, md5 = _capture(
        r"^\| сдаваемый файл \| `(runs/[\w.-]+/wells_schedule\.inc)`[^\n]*"
        r"`md5 ([0-9a-fA-F]{32})`",
        text,
        "файл и md5",
    )
    return Manifest(npv, base, uplift, uplift_percent, schedule, md5.lower())


def manifest_problem(manifest: Manifest, tag_message: str) -> str | None:
    expected = (
        f"{manifest.npv} млн руб.",
        f"База: {manifest.base}",
        f"+{manifest.uplift} (+{manifest.uplift_percent} %)",
        manifest.schedule,
        f"md5 {manifest.md5}",
    )
    missing = [fragment for fragment in expected if fragment not in tag_message]
    if missing:
        return "тег расходится с заявкой; не найдено: " + ", ".join(missing)
    return None


def _tag_message(root: Path, tag: str) -> str:
    result = subprocess.run(
        ["git", "for-each-ref", f"refs/tags/{tag}", "--format=%(objecttype)%00%(contents)"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    kind, separator, message = result.stdout.partition("\0")
    if not separator:
        raise ValueError(f"тег {tag} не найден")
    if kind != "tag":
        raise ValueError(f"тег {tag} не аннотированный")
    return message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="тег заявки; по умолчанию берётся из паспорта")
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    try:
        submission = (root / "docs" / "submission.md").read_text(encoding="utf-8")
        tag = arguments.tag
        if tag is None:
            passport = (root / "docs" / "submission-day.md").read_text(encoding="utf-8")
            match = ACTIVE_TAG.search(passport)
            if match is None:
                raise ValueError("в паспорте не найден активный тег submit-<дата>")
            tag = match.group(1)
        problem = manifest_problem(declared_manifest(submission), _tag_message(root, tag))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"[НЕТ] тег заявки: {error}")
        return 1

    if problem:
        print(f"[НЕТ] тег заявки: {problem}")
        return 1
    print(f"[ГОТОВО] тег заявки: {tag} совпадает с docs/submission.md")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
