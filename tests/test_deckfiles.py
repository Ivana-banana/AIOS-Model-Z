"""Состав дека определяется из дека, а не из зашитых имён файлов.

По ТЗ п. 4.3 заявляемый ЧДД считается на тестовом кейсе организаторов, где дек
будет называться иначе, чем Модель Z. Поэтому все проверки идут на деках с
чужими именами: если что-то в коде снова привяжется к `Model_Z_*`, эти тесты
упадут.
"""

from __future__ import annotations

import pytest

from aios.deckfiles import find_data_file, layout, parse_layout

FOREIGN = """\
-- Дек стороннего месторождения: имена не имеют ничего общего с Моделью Z.
RUNSPEC

DIMENS
 10 20 5 /

GRID
INCLUDE
'geometry.grdecl' /

PROPS
INCLUDE
'fluid_tables.inc' /

REGIONS
INCLUDE
'zones.inc' /

SUMMARY
INCLUDE
'vectors.inc' /

SCHEDULE
INCLUDE
'history.inc' /

END
"""


@pytest.fixture
def foreign_deck(tmp_path):
    directory = tmp_path / "FieldX"
    directory.mkdir()
    (directory / "FIELD_X.DATA").write_text(FOREIGN, encoding="utf-8")
    for name in ("geometry.grdecl", "fluid_tables.inc", "zones.inc", "vectors.inc", "history.inc"):
        (directory / name).write_text("-- заглушка\n", encoding="utf-8")
    return directory


# --- поиск корневого файла --------------------------------------------------

def test_data_file_is_found_regardless_of_its_name(foreign_deck):
    assert find_data_file(foreign_deck).name == "FIELD_X.DATA"


def test_lowercase_extension_is_found_too(tmp_path):
    directory = tmp_path / "deck"
    directory.mkdir()
    (directory / "model.data").write_text(FOREIGN, encoding="utf-8")
    assert find_data_file(directory).name == "model.data"


def test_a_directory_without_a_deck_fails_loudly(tmp_path):
    (tmp_path / "readme.txt").write_text("не дек", encoding="utf-8")
    with pytest.raises(ValueError, match="нет ни одного .DATA"):
        find_data_file(tmp_path)


def test_two_decks_are_a_question_for_a_human(tmp_path):
    """Выбор «какую модель считать» не должен зависеть от порядка сортировки."""
    (tmp_path / "a.DATA").write_text(FOREIGN, encoding="utf-8")
    (tmp_path / "b.DATA").write_text(FOREIGN, encoding="utf-8")
    with pytest.raises(ValueError, match="несколько .DATA"):
        find_data_file(tmp_path)


def test_missing_directory_is_named_in_the_error(tmp_path):
    with pytest.raises(ValueError, match="нет каталога дека"):
        find_data_file(tmp_path / "нет-такого")


# --- отнесение INCLUDE к секции ---------------------------------------------

def test_includes_are_attributed_to_their_section(foreign_deck):
    deck = layout(foreign_deck)
    assert deck.one("SCHEDULE").name == "history.inc"
    assert deck.one("SUMMARY").name == "vectors.inc"
    assert deck.one("PROPS").name == "fluid_tables.inc"
    assert deck.one("REGIONS").name == "zones.inc"
    assert deck.one("GRID").name == "geometry.grdecl"


def test_section_is_decided_by_position_not_by_file_name(foreign_deck):
    """Ни одно имя здесь не намекает на назначение — работает только порядок секций."""
    deck = layout(foreign_deck)
    assert "Model_Z" not in str(deck.one("SCHEDULE"))


def test_include_paths_are_absolute(foreign_deck):
    assert layout(foreign_deck).one("SCHEDULE").is_absolute()


def test_a_section_without_include_fails_loudly(tmp_path):
    (tmp_path / "d.DATA").write_text("RUNSPEC\n\nSCHEDULE\n\nEND\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ожидался ровно один INCLUDE"):
        layout(tmp_path).one("SCHEDULE")


def test_two_includes_in_one_section_fail_loudly(tmp_path):
    text = "SCHEDULE\nINCLUDE\n'a.inc' /\nINCLUDE\n'b.inc' /\n"
    (tmp_path / "d.DATA").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="найдено 2"):
        layout(tmp_path).one("SCHEDULE")


def test_commented_include_is_not_picked_up(tmp_path):
    text = "SCHEDULE\n-- INCLUDE\n-- 'старое.inc' /\nINCLUDE\n'новое.inc' /\n"
    (tmp_path / "d.DATA").write_text(text, encoding="utf-8")
    assert layout(tmp_path).one("SCHEDULE").name == "новое.inc"


def test_lowercase_keywords_are_understood(tmp_path):
    text = "schedule\ninclude\n'h.inc' /\n"
    (tmp_path / "d.DATA").write_text(text, encoding="utf-8")
    assert layout(tmp_path).one("SCHEDULE").name == "h.inc"


# --- размеры сетки ----------------------------------------------------------

def test_dimensions_come_from_the_deck(foreign_deck):
    assert layout(foreign_deck).dimensions == (10, 20, 5)
    assert layout(foreign_deck).cells == 10 * 20 * 5


def test_absent_dimens_is_not_an_error_by_itself(tmp_path):
    """`DIMENS` нужен только тем, кто адресует ячейку; сборке кейса — нет."""
    (tmp_path / "d.DATA").write_text("SCHEDULE\nINCLUDE\n'h.inc' /\n", encoding="utf-8")
    assert layout(tmp_path).dimensions is None


def test_asking_for_the_grid_without_dimens_fails_loudly(tmp_path):
    (tmp_path / "d.DATA").write_text("SCHEDULE\nINCLUDE\n'h.inc' /\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нет DIMENS"):
        layout(tmp_path).grid()


def test_dimens_on_the_same_line_as_the_keyword_is_not_required(tmp_path):
    text = "RUNSPEC\nDIMENS\n 7 8 9 /\n"
    (tmp_path / "d.DATA").write_text(text, encoding="utf-8")
    assert parse_layout(tmp_path / "d.DATA").dimensions == (7, 8, 9)


# --- защита от возвращения хардкода -----------------------------------------

def test_no_module_hardcodes_a_model_z_file_name():
    """Имя файла Модели Z в исполняемой строке — отказ на кейсе организаторов.

    Проверяется код, а не комментарии: ссылка на поставку в документации — это
    полезный контекст, а `open("Model_Z_sch.inc")` — дефект, который проявится
    ровно один раз и ровно там, где его нельзя исправить.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "aios"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        # строки-документация — это контекст, а не поведение: пропускаем их
        documentation = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                first = node.body[0] if node.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    documentation.add(id(first.value))
            # отдельные строки-абзацы после полей dataclass — тоже документация
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                documentation.add(id(node.value))

        for node in ast.walk(tree):
            if id(node) in documentation:
                continue
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "Model_Z" in node.value):
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, "имена файлов дека зашиты в коде:\n" + "\n".join(offenders)
