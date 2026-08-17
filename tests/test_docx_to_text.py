"""Извлечение текста из .docx организаторов.

Материалы приходят в .docx, а ссылаться на них нужно из отчёта и решений. Разбор
проверяется на собранном в памяти документе, а не на файле организаторов: тест
обязан работать в чистом клоне, где `materials/` может быть не распакован.

Отдельно закреплена кодировка вывода: на Windows умолчание — cp1251, и текст
методики в UTF-8-файле превращается в мусор молча, без ошибки.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from docx_to_text import docx_to_text, main  # noqa: E402

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def build_docx(path: Path, body: str) -> Path:
    """Минимальный .docx: zip с одним `word/document.xml`."""
    document = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    return path


def paragraph(*runs: str) -> str:
    inner = "".join(f"<w:r><w:t>{run}</w:t></w:r>" for run in runs)
    return f"<w:p>{inner}</w:p>"


def test_paragraphs_are_extracted_in_order(tmp_path: Path):
    path = build_docx(tmp_path / "d.docx", paragraph("Первый") + paragraph("Второй"))
    assert docx_to_text(path).splitlines() == ["Первый", "Второй"]


def test_split_runs_do_not_lose_characters(tmp_path: Path):
    """Word рвёт предложение на прогоны по любому изменению форматирования."""
    path = build_docx(tmp_path / "d.docx", paragraph("макси", "мизация ", "NPV"))
    assert docx_to_text(path).strip() == "максимизация NPV"


def test_table_rows_are_flattened_with_markers(tmp_path: Path):
    table = (
        f'<w:tbl><w:tr><w:tc>{paragraph("Критерий")}</w:tc>'
        f'<w:tc>{paragraph("Вес")}</w:tc></w:tr></w:tbl>'
    )
    lines = docx_to_text(build_docx(tmp_path / "d.docx", table)).splitlines()
    assert lines == ["[ТАБЛИЦА]", "Критерий | Вес", "[/ТАБЛИЦА]"]


def test_body_is_required(tmp_path: Path):
    path = tmp_path / "broken.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{W}"/>')
    with pytest.raises(ValueError, match="w:body"):
        docx_to_text(path)


def test_output_is_written_as_utf8(tmp_path: Path):
    """Кириллица обязана пережить запись на Windows, где умолчание — cp1251."""
    source = build_docx(tmp_path / "d.docx", paragraph("Скважина №1, ЧДД"))
    target = tmp_path / "out.txt"
    assert main(["docx_to_text.py", str(source), str(target)]) == 0
    assert target.read_text(encoding="utf-8").strip() == "Скважина №1, ЧДД"


def test_wrong_argument_count_is_reported(tmp_path: Path):
    assert main(["docx_to_text.py"]) == 2


def test_extracted_tz_is_present_in_repo():
    """ТЗ разобрано и лежит в docs/: на его пункты ссылается docs/tz-compliance.md."""
    tz = Path(__file__).resolve().parent.parent / "docs" / "method" / "tz-full.txt"
    assert tz.is_file(), "нет docs/method/tz-full.txt"
    text = tz.read_text(encoding="utf-8")
    # Требование, которое легко потерять при обновлении ТЗ, — LLM в ядре.
    assert "языковую (мультимодальную) модель" in text
    assert "wells_schedule.inc" in text
