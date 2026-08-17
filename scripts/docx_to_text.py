"""Извлечение текста из .docx организаторов.

Материалы приходят в .docx (методика, техническое задание). Чтобы на них можно
было ссылаться из кода, тестов и отчёта, текст кладётся в `docs/method/*.txt`
рядом с остальными выжимками.

Зависимостей нет: .docx — это zip с XML, стандартной библиотеки достаточно.
Отдельная библиотека здесь была бы лишней строкой в поставке решения.

    python scripts/docx_to_text.py materials/ТЗ.docx docs/method/tz-full.txt
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def paragraph_text(paragraph: ET.Element) -> str:
    """Текст абзаца со сборкой из отдельных прогонов (`w:r`)."""
    parts: list[str] = []
    for node in paragraph.iter():
        tag = node.tag.split("}")[-1]
        if tag == "t":
            parts.append(node.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag == "br":
            parts.append("\n")
    return "".join(parts)


def table_lines(table: ET.Element) -> list[str]:
    """Таблица в виде строк «ячейка | ячейка», с явными границами блока."""
    lines = ["[ТАБЛИЦА]"]
    for row in table.findall("w:tr", NS):
        cells = []
        for cell in row.findall("w:tc", NS):
            cells.append(
                " ".join(
                    paragraph_text(p).strip() for p in cell.findall("w:p", NS)
                ).strip()
            )
        lines.append(" | ".join(cells))
    lines.append("[/ТАБЛИЦА]")
    return lines


def docx_to_text(path: Path) -> str:
    """Текст документа: абзацы и таблицы в порядке следования в теле."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    body = ET.fromstring(xml).find("w:body", NS)
    if body is None:
        raise ValueError(f"{path}: в документе нет тела w:body")

    out: list[str] = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            out.append(paragraph_text(child))
        elif tag == "tbl":
            out.extend(table_lines(child))

    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    source, target = Path(argv[1]), Path(argv[2])
    # Кодировка задаётся явно: на Windows умолчание — cp1251, и текст поедет.
    target.write_text(docx_to_text(source), encoding="utf-8")
    print(f"{source} → {target}")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv))
