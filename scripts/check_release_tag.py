"""Сверить аннотированный тег заявки с текущим манифестом отправки."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Действующий тег берётся по ЯВНОМУ якорю паспорта, а не по первому упоминанию:
# в паспорте по делу названы и прежние теги (протокол холостого прогона 04.09,
# строка про отозванную заявку), и порядок строк действующим тег не делает.
ACTIVE_TAG = re.compile(
    r"\*\*Действующий тег заявки — `(submit-\d{4}-\d{2}-\d{2}(?:\.\d+)?)`\*\*")


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
        r"^\| сдаваемый файл \| `(runs/[\w.-]+(?:/[\w.-]+)*/wells_schedule\.inc)`[^\n]*"
        r"`md5 ([0-9a-fA-F]{32})`",
        text,
        "файл и md5",
    )
    return Manifest(npv, base, uplift, uplift_percent, schedule, md5.lower())


def tagged_manifest(text: str) -> Manifest:
    """Прочитать декларативную шапку тега, не принимая числа из его истории."""
    npv, = _capture(
        r"^Заявляемый ЧДД: ([\d ]+,\d) млн руб\.", text, "ЧДД в теге"
    )
    base, uplift, uplift_percent = _capture(
        r"^База: ([\d ]+,\d)\. Прирост: \+([\d ]+,\d) \(\+([\d,]+) %\)\.",
        text,
        "база и прирост в теге",
    )
    schedule, md5 = _capture(
        r"^Файл: (runs/[\w.-]+(?:/[\w.-]+)*/wells_schedule\.inc), md5 ([0-9a-fA-F]{32})\.",
        text,
        "файл и md5 в теге",
    )
    return Manifest(npv, base, uplift, uplift_percent, schedule, md5.lower())


def manifest_problem(manifest: Manifest, tag_message: str) -> str | None:
    try:
        tagged = tagged_manifest(tag_message)
    except ValueError as error:
        return str(error)
    differences = [
        name
        for name in Manifest.__dataclass_fields__
        if getattr(manifest, name) != getattr(tagged, name)
    ]
    if differences:
        return "тег расходится с заявкой; поля: " + ", ".join(differences)
    return None


def reproduction_problem(manifest: Manifest, submission: str) -> str | None:
    """Проверить, что инструкция снимка тега строит объявленный сценарий."""
    try:
        commands = submission.split("## 3.", maxsplit=1)[1]
        commands = commands.split("## 4.", maxsplit=1)[0]
    except IndexError:
        return "в снимке тега нет полного раздела 3 инструкции воспроизведения"
    run = manifest.schedule.rsplit("/", maxsplit=1)[0]
    # С 08.09 сдаётся хвост за период управления, и он лежит в подкаталоге
    # `<прогон>/submission/` (Р-249). Кейс же собирается в самом каталоге
    # прогона, и инструкция воспроизведения называет именно его `RUN.DATA`.
    if run.endswith("/submission"):
        run = run[: -len("/submission")]
    if f"{run}/RUN.DATA" not in commands:
        return f"инструкция снимка тега не считает заявленный сценарий {run}"
    return None


def tagged_file_problem(root: Path, tag: str, manifest: Manifest) -> str | None:
    """Проверить, что заявленный файл лежит в самом теге и это он же.

    До 08.09 сторож сверял только тексты: манифест заявки, сообщение тега и
    инструкцию воспроизведения. До файла он не доходил — и когда форма сдачи
    сменилась на хвост за период управления, а исключение `.gitignore` осталось
    на прежнем пути, тег `submit-2026-09-08` объявил файл, которого в теге нет.
    Сторож при этом отвечал «манифест и инструкция совпадают» (Н-04).
    """
    result = subprocess.run(
        ["git", "show", f"{tag}:{manifest.schedule}"],
        cwd=root,
        capture_output=True,
    )
    if result.returncode != 0:
        return (
            f"в теге {tag} нет заявленного файла {manifest.schedule} — "
            "он не отслеживается git, и клон проверяющего его не получит"
        )
    actual = hashlib.md5(result.stdout).hexdigest()
    if actual != manifest.md5:
        return (
            f"файл в теге {tag} не тот, что заявлен: md5 в теге {actual}, "
            f"в заявке {manifest.md5}"
        )
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


def _tag_file(root: Path, tag: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tag}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


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
                raise ValueError(
                    "в паспорте нет строки «**Действующий тег заявки — `submit-<дата>`**»"
                )
            tag = match.group(1)
        current = declared_manifest(submission)
        tagged_submission = _tag_file(root, tag, "docs/submission.md")
        tagged = declared_manifest(tagged_submission)
        problem = (
            "манифест в снимке тега расходится с текущей заявкой"
            if tagged != current
            else reproduction_problem(tagged, tagged_submission)
            or manifest_problem(current, _tag_message(root, tag))
            or tagged_file_problem(root, tag, current)
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"[НЕТ] тег заявки: {error}")
        return 1

    if problem:
        print(f"[НЕТ] тег заявки: {problem}")
        return 1
    print(
        f"[ГОТОВО] тег заявки: {tag}, манифест и инструкция совпадают "
        "с docs/submission.md"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
