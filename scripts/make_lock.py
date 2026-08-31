"""Собрать `requirements.lock` с хэшами из отчёта resolver'а pip.

    python -m pip install --dry-run --ignore-installed --report report.json \
        -r requirements.txt
    python scripts/make_lock.py report.json --out requirements.lock

Зачем. Прямые зависимости у нас запинены все, но resolver тянет ещё три десятка
транзитивных — включая NumPy, pandas и SciPy, которые приходят через `resdata` и
`scikit-learn`. Без их фиксации один и тот же `requirements.txt` в разные дни
даёт разную среду, а значит и разные числа. Внешняя проверка воспроизводимости
(ответ 4-10) назвала это отдельным пунктом «мешает».

**Файл платформенно-зависим, и это не оговорка, а свойство.** Хэши в отчёте —
хэши конкретных колёс: колесо NumPy для `manylinux_x86_64` и для Windows разные
файлы с разными хэшами. Поэтому lock снимается там же, где считается поставка —
в Linux той же версии, что и базовый образ, — а заголовок файла обязан называть
платформу, версию Python, дату и индекс. Lock, снятый на Windows, описывал бы не
ту поставку и был бы хуже отсутствующего: он выглядел бы как гарантия.

Ставится в сборке так:

    python -m pip install --require-hashes -r requirements.lock
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path


def entries(report: dict) -> list[tuple[str, str, list[str]]]:
    """`(имя, версия, хэши)` по каждому пакету разрешённого графа."""
    found = []
    for item in report.get("install", []):
        metadata = item["metadata"]
        hashes = item.get("download_info", {}).get("archive_info", {}).get("hashes", {})
        digests = [f"{kind}:{value}" for kind, value in sorted(hashes.items())]
        found.append((metadata["name"], metadata["version"], digests))
    return sorted(found, key=lambda one: one[0].lower())


def render(report: dict, source: str) -> str:
    found = entries(report)
    unhashed = [name for name, _version, digests in found if not digests]
    if unhashed:
        raise ValueError(
            "у этих пакетов в отчёте нет хэшей, а lock без них бесполезен: "
            + ", ".join(unhashed)
        )

    header = [
        "# Сгенерирован scripts/make_lock.py — руками не править.",
        f"# источник:  {source}",
        f"# платформа: {platform.system()} {platform.machine()}",
        f"# python:    {platform.python_version()}",
        f"# снят:      {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"# индекс:    {report.get('environment', {}).get('index_url', 'PyPI по умолчанию')}",
        "#",
        "# Хэши здесь — хэши конкретных колёс, поэтому файл описывает ЭТУ платформу.",
        "# Для другой ОС нужен свой lock; см. docstring scripts/make_lock.py.",
        "#",
        "# Установка:  python -m pip install --require-hashes -r requirements.lock",
        "",
    ]
    lines = list(header)
    for name, version, digests in found:
        lines.append(f"{name}=={version} \\")
        lines.extend(f"    --hash={digest} \\" for digest in digests[:-1])
        lines.append(f"    --hash={digests[-1]}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="JSON отчёта `pip install --report`")
    parser.add_argument("--out", type=Path, default=Path("requirements.lock"))
    arguments = parser.parse_args(argv)

    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    try:
        text = render(report, arguments.report.name)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    arguments.out.write_text(text, encoding="utf-8")
    print(f"пакетов: {len(entries(report))}")
    print(f"записан: {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
