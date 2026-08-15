"""Среда решения: версии пинуются, образ собирается из того же, что мы тестируем.

Регламент требует воспроизводимого результата в идентичной среде. Разъезд между
«протестировано» и «развёрнуто» — самый скучный способ потерять баллы, поэтому он
проверяется тестом, а не вниманием.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "docker" / "Dockerfile"
REQUIREMENTS = ROOT / "requirements.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"


def _packages(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-r"):
            lines.append(line)
    return lines


def test_every_runtime_dependency_is_pinned():
    for requirement in _packages(REQUIREMENTS):
        assert "==" in requirement, f"не закреплена версия: {requirement}"


def test_every_dev_dependency_is_pinned():
    for requirement in _packages(DEV_REQUIREMENTS):
        assert "==" in requirement, f"не закреплена версия: {requirement}"


def test_runtime_stays_lean():
    """В рантайме только то, без чего решение не работает.

    Расшифровка записей и разбор pdf — инструменты подготовки, им в образе,
    который считает сценарии, делать нечего.
    """
    names = {re.split(r"[=<>\[]", line)[0].lower() for line in _packages(REQUIREMENTS)}
    assert names == {"openpyxl", "resdata", "fastapi", "uvicorn", "jinja2"}
    assert not names & {"pytest", "ruff", "pymupdf", "faster-whisper", "av", "httpx"}


def test_dockerfile_never_installs_a_named_package_without_a_version_source():
    """`pip install <имя>` в образе — это несовпадение с тем, что мы тестировали."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    for match in re.finditer(r"pip install ([^\n\\]*)", text):
        arguments = match.group(1)
        assert "-r " in arguments or "==" in arguments, f"pip без источника версии: {arguments}"


def test_dockerfile_verifies_the_simulator_at_build_time():
    """Сборка обязана падать здесь, а не на первом прогоне в облаке."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "flow --version" in text
    assert "import resdata" in text


def test_dockerfile_does_not_bake_in_the_organisers_deck():
    """Дек — их, весит 78 МБ и монтируется томом, а не копируется в образ."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY vendor/Model_Z" not in text
    # эталонный расчётчик ЧДД, наоборот, обязан быть внутри: он источник числа
    assert "COPY vendor/CHDD_PYTHON" in text


def test_build_context_excludes_heavy_material():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in ("vendor/Model_Z_OPM/", "*.webm", "runs/", "Lib/"):
        assert pattern in ignore, f"не исключено из контекста сборки: {pattern}"


@pytest.mark.parametrize("path", [DOCKERFILE, ROOT / ".dockerignore", REQUIREMENTS])
def test_packaging_files_exist(path: Path):
    assert path.is_file()
