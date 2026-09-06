"""Проверить проект глазами проверяющего: настоящий `git clone`, потом `pytest`.

    python scripts/clean_clone_check.py

Зачем отдельный скрипт. Числа README про чистый клон — «1 592 теста, 0 падений,
21 пропуск» — на машине разработки взять неоткуда: у неё есть выгрузки `runs/`,
распакованный дек и всё, что накопилось за месяц. Поэтому клон делали руками и
редко: 23.08, 04.09, 06.09. И **три падающих на клоне теста** нашлись ровно в
тот день, когда клон сделали снова — притом что прежние протоколы шли по
инструкции, а проверяющий по ней идти не обязан (Р-236).

Что меряется. Два состояния клона, потому что проверяющий бывает в обоих:

* **дек не распакован** — он набрал `pytest` первым, чтобы посмотреть, живой ли
  проект. Падений быть не должно: тест без предпосылки обязан пропускаться с
  названной причиной, а не краснеть;
* **дек распакован** — он прошёл инструкцию по шагам. Это и есть числа README.

Разницы между «упало» и «пропущено» здесь достаточно: код возврата 1, если на
клоне есть хоть одно падение или если числа разошлись с README.

Чего скрипт не проверяет. Установку зависимостей: клон запускается **тем же**
интерпретатором, что и сам скрипт, то есть уже готовым окружением. Значит,
`pip install -r requirements-dev.txt` на чистой машине этим прогоном не
подтверждается — это делает протокол `docs/clean-clone-check.md` вручную.
Клонируется **закоммиченное** состояние (`git clone` берёт HEAD источника), а не
рабочее дерево: незакоммиченная правка в замер не попадёт, и это правильно —
проверяющий получит именно то, что запушено.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def readme_expectations(text: str) -> dict[str, int]:
    """Числа, которые README обещает проверяющему.

    Берутся из текста, а не из константы: разойтись должно быть невозможно, а
    не «замечено при вычитке».
    """
    expected: dict[str, int] = {}

    with_deck = re.search(
        r"\*\*([\d\s ]+) тест\S*, 0 падений, ([\d\s ]+) пропуск", text)
    if with_deck:
        expected["tests"] = int(re.sub(r"\D", "", with_deck.group(1)))
        expected["skipped_with_deck"] = int(re.sub(r"\D", "", with_deck.group(2)))

    no_deck = re.search(r"до распаковки дека, пропусков становится ([\d\s ]+)", text)
    if no_deck:
        expected["skipped_without_deck"] = int(re.sub(r"\D", "", no_deck.group(1)))

    return expected


def counts(report: Path) -> dict[str, int]:
    """Итоги прогона из `--junitxml`, а не из хвоста вывода.

    Хвост `pytest -q` на этой платформе теряет строку итога при выводе кириллицы
    — числа приходилось выцарапывать глазами. XML такого не умеет.
    """
    root = ET.parse(report).getroot()
    suite = root[0] if root.tag == "testsuites" else root
    return {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]) + int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
    }


def verdict(measured: dict[str, dict[str, int]], expected: dict[str, int]) -> list[str]:
    """Расхождения замера с README. Пустой список — всё сошлось."""
    problems = []

    for stage, result in measured.items():
        if result["failures"]:
            problems.append(
                f"{stage}: {result['failures']} падений на чистом клоне — "
                "проверяющий увидит красное")

    if "with-deck" in measured and "tests" in expected:
        if measured["with-deck"]["tests"] != expected["tests"]:
            problems.append(
                f"README обещает {expected['tests']} тестов, на клоне "
                f"{measured['with-deck']['tests']}")
        if measured["with-deck"]["skipped"] != expected.get("skipped_with_deck"):
            problems.append(
                f"README обещает {expected.get('skipped_with_deck')} пропусков с "
                f"деком, на клоне {measured['with-deck']['skipped']}")

    if ("no-deck" in measured and "skipped_without_deck" in expected
            and measured["no-deck"]["skipped"] != expected["skipped_without_deck"]):
        problems.append(
            f"README обещает {expected['skipped_without_deck']} пропусков без "
            f"дека, на клоне {measured['no-deck']['skipped']}")

    return problems


def deck_steps(text: str) -> tuple[str, str, str, str]:
    """Разобрать шаг распаковки дека из README: архив, куда, что во что переименовать.

    Имена не зашиты в скрипт намеренно: валидационный кейс придёт с другим деком,
    а проверка на зашитое имя `Model_Z` в коде — отдельный сторож проекта. Здесь
    же важнее второе: скрипт обязан делать **то же самое**, что написано
    проверяющему. Разойдись он с инструкцией — мерил бы не её.
    """
    unpack = re.search(r"unzip [^\n]*?(\S+\.zip) -d (\S+)", text)
    rename = re.search(r"\nmv (\S+) (\S+)", text)
    if not unpack or not rename:
        raise ValueError("в README не нашлись команды распаковки дека (раздел 2)")
    return unpack.group(1), unpack.group(2), rename.group(1), rename.group(2)


def unpack_deck(clone: Path, steps: tuple[str, str, str, str]) -> None:
    """Шаг 2 инструкции, выполненный ровно так, как он записан в README."""
    archive_name, destination, unpacked, target = steps
    with zipfile.ZipFile(clone / archive_name) as archive:
        archive.extractall(clone / destination)
    shutil.rmtree(clone / destination / "__MACOSX", ignore_errors=True)
    (clone / unpacked).rename(clone / target)


def run_pytest(clone: Path, report: Path, python: str) -> dict[str, int]:
    """Прогон изнутри клона — как его запустит проверяющий."""
    subprocess.run(
        [python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--junitxml={report}"],
        cwd=clone, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    return counts(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="интерпретатор для прогона в клоне")
    parser.add_argument("--stage", choices=("all", "deck", "nodeck"), default="all",
                        help="что мерить: оба состояния, только с деком, только без")
    parser.add_argument("--keep", action="store_true",
                        help="не удалять клон — пригодится, чтобы посмотреть падение")
    args = parser.parse_args()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    steps = deck_steps(readme)

    workspace = Path(tempfile.mkdtemp(prefix="aios-clean-clone-"))
    clone = workspace / "clone"
    measured: dict[str, dict[str, int]] = {}
    try:
        done = subprocess.run(
            ["git", "clone", "--quiet", str(ROOT), str(clone)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            print(f"клон не сделался: {done.stderr.strip()}")
            return 1

        head = subprocess.run(
            ["git", "log", "--oneline", "-1"], cwd=clone,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"клон: {head.stdout.strip()}")

        if args.stage in ("all", "nodeck"):
            measured["no-deck"] = run_pytest(
                clone, workspace / "no-deck.xml", args.python)
        if args.stage in ("all", "deck"):
            unpack_deck(clone, steps)
            measured["with-deck"] = run_pytest(
                clone, workspace / "with-deck.xml", args.python)
    finally:
        if args.keep:
            print(f"клон оставлен: {clone}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    print()
    print(f"{'состояние клона':<24} {'тестов':>7} {'падений':>8} {'пропусков':>10}")
    for stage, result in measured.items():
        print(f"{stage:<24} {result['tests']:>7} {result['failures']:>8} "
              f"{result['skipped']:>10}")

    problems = verdict(measured, readme_expectations(readme))
    print()
    if problems:
        for one in problems:
            print(f"РАСХОЖДЕНИЕ: {one}")
        return 1

    print("клон сходится с README")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
