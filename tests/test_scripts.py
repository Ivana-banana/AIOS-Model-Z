"""Скрипты запуска обязаны быть в репозитории — и оставаться в нём.

История вопроса. До 16.08 каталога `scripts/` в репозитории не было вовсе, хотя
на него ссылались `README.md`, обе инструкции и `docker/Dockerfile`. Причина —
строка `/Scripts/` в `.gitignore`: она писалась под venv для Windows, а на
регистронезависимой файловой системе git применил её к нашему `scripts/` и молча
выкинул все скрипты из коммитов. На чистом клоне не работала ни одна команда из
инструкции запуска, а `docker build` падал на `COPY scripts`.

Инструкция запуска — обязательный пункт допуска к оценке, поэтому проверяется
тестом, а не вниманием.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

#: Скрипты, на которые ссылаются документация и Dockerfile.
REQUIRED = ("build_case.py", "run_flow.sh", "export_rows.py", "build_doe.py", "run_doe.py", "run_cycle.py",
            "transcribe.py", "build_dataset.py", "train_surrogate.py")


@pytest.mark.parametrize("name", REQUIRED)
def test_documented_script_exists(name: str):
    assert (SCRIPTS / name).is_file(), f"нет scripts/{name}, а на него ссылается документация"


@pytest.mark.parametrize("name", REQUIRED)
def test_script_is_tracked_by_git(name: str):
    """Файл на диске ничего не значит: важно, что он попадёт в клон другой машины."""
    listed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", f"scripts/{name}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if listed.returncode != 0 and "not a git repository" in listed.stderr.lower():
        pytest.skip("не git-репозиторий")
    assert listed.returncode == 0, f"scripts/{name} не под контролем версий: {listed.stderr.strip()}"


def test_scripts_are_not_ignored_on_case_insensitive_filesystems():
    """Ровно тот случай, который стоил проекту всех скриптов запуска."""
    checked = subprocess.run(
        ["git", "-c", "core.ignorecase=true", "check-ignore", "-v",
         *(f"scripts/{name}" for name in REQUIRED)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if "not a git repository" in checked.stderr.lower():
        pytest.skip("не git-репозиторий")
    # check-ignore возвращает 0, если хоть один путь игнорируется, и печатает правило
    assert checked.returncode != 0, (
        "scripts/ игнорируется при core.ignorecase=true — под Windows скрипты снова "
        f"выпадут из репозитория:\n{checked.stdout}"
    )


def test_scripts_reach_the_docker_image():
    """`COPY scripts` в образе работает, только если контекст сборки их не исключает."""
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    patterns = [line.strip() for line in ignore.splitlines()
                if line.strip() and not line.startswith("#")]
    for pattern in patterns:
        assert pattern.rstrip("/").lower() != "scripts", (
            f"'{pattern}' в .dockerignore исключит scripts/ на регистронезависимой ФС"
        )
    assert "COPY scripts" in (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")


def test_shell_scripts_are_executable():
    for path in SCRIPTS.glob("*.sh"):
        assert os.access(path, os.X_OK), f"{path.name} не исполняемый — инструкция запуска врёт"


def _fake_flow(tmp_path: Path, body: str) -> dict[str, str]:
    """PATH с подставным `flow`, чтобы проверять поведение обвязки без симулятора."""
    binary = tmp_path / "bin"
    binary.mkdir()
    script = binary / "flow"
    script.write_text(
        '#!/usr/bin/env bash\n'
        '[[ "$1" == "--version" ]] && { echo "flow 0.0-fake"; exit 0; }\n'
        f"{body}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}


def test_run_flow_fails_when_the_result_is_missing(tmp_path: Path):
    """Симулятор умеет выйти с нулём, не создав `.UNSMRY`.

    Такой прогон обязан быть отказом: при пакетном счёте DoE иначе в датасет
    молча попадают пустые кейсы, и обнаруживается это через сотню расчётов.
    """
    case = tmp_path / "case"
    case.mkdir()
    (case / "RUN.DATA").write_text("", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_flow.sh"), str(case / "RUN.DATA"), "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        env=_fake_flow(tmp_path, "exit 0"),
    )
    assert result.returncode != 0, (
        "результата нет, а скрипт доложил об успехе:\n" + result.stdout[-800:]
    )


def test_run_flow_reports_skipped_keywords(tmp_path: Path):
    """Блок пропущенных ключевых слов — главная защита от тихой подмены физики.

    Flow не падает на том, чего не понял: он это игнорирует и считает дальше.
    Если блок скажет «нет» при непустом списке, прогон будет принят за
    корректный, хотя посчитана другая модель.
    """
    case = tmp_path / "case"
    case.mkdir()
    (case / "RUN.DATA").write_text("", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_flow.sh"), str(case / "RUN.DATA"), "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        env=_fake_flow(
            tmp_path,
            'echo "Error: Unsupported keywords or keyword items:"\n'
            'echo "  PINCHREG: keyword not supported"\n'
            f'touch "{case}/RUN.UNSMRY"\n'
            "exit 0",
        ),
    )
    assert "PINCHREG" in result.stdout, (
        "пропущенное ключевое слово не попало в отчёт скрипта:\n" + result.stdout[-800:]
    )
    block = result.stdout.split("--- пропущенные и непонятые ключевые слова ---", 1)[-1]
    assert not block.lstrip().startswith("нет"), (
        "блок доложил «нет» при непустом списке:\n" + block[:400]
    )


@pytest.mark.parametrize("name", [name for name in REQUIRED if name.endswith(".py")])
def test_python_scripts_run_from_a_clean_checkout(name: str):
    """`python scripts/<имя>.py --help` обязан работать без установки пакета.

    Скрипты запускаются из корня репозитория, где `aios` не установлен в
    окружение, поэтому каждый добавляет корень в `sys.path` сам.
    """
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"{name} --help упал:\n{result.stderr}"
