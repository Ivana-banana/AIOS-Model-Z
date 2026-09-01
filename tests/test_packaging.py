"""Среда решения: версии пинуются, образ собирается из того же, что мы тестируем.

Регламент требует воспроизводимого результата в идентичной среде. Разъезд между
«протестировано» и «развёрнуто» — самый скучный способ потерять баллы, поэтому он
проверяется тестом, а не вниманием.
"""

from __future__ import annotations

import re
import shlex
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
    assert names == {
        "openpyxl", "resdata", "fastapi", "uvicorn", "jinja2", "scikit-learn", "joblib",
        # слой интерпретируемости — требование ТЗ п. 3.1, а не удобство: языковая
        # модель обязана быть в ядре слоя. Без ключа пакет работает по шаблону.
        "anthropic",
    }
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


def test_every_local_copy_source_exists_and_is_tracked():
    """Локальный файл не попадёт в чистый Docker-контекст только потому, что он есть здесь."""
    import subprocess

    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("COPY "):
            continue
        tokens = shlex.split(line)
        sources = [token for token in tokens[1:-1] if not token.startswith("--")]
        for source in sources:
            path = ROOT / source
            assert path.exists(), f"источник COPY отсутствует: {source}"
            listed = subprocess.run(
                ["git", "ls-files", source],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            assert listed.stdout.strip(), f"источник COPY не отслеживается Git: {source}"


@pytest.mark.parametrize("path", [DOCKERFILE, ROOT / ".dockerignore", REQUIREMENTS])
def test_packaging_files_exist(path: Path):
    assert path.is_file()


LICENSES = ROOT / "LICENSES"
PINCH_DOCKERFILE = ROOT / "docker" / "Dockerfile.pinch935"


def test_base_image_is_pinned_by_digest_and_both_files_agree():
    """Тег `ubuntu:24.04` — движущаяся цель: его пересобирают, и «тот же
    Dockerfile» через месяц даёт другой образ.

    До 22.08 по digest пинуется только `Dockerfile.pinch935`, а основной образ —
    тот, в котором посчитаны все числа проекта, — брал плавающий тег. Внешняя
    проверка назвала это отдельным пунктом (`docs/manus/round4.md`, 4-10).
    Digest у двух файлов обязан быть один: два разных базовых образа в одной
    поставке — это две разные среды, о чём бы ни говорил отчёт.
    """
    digests = set()
    for path in (DOCKERFILE, PINCH_DOCKERFILE):
        text = path.read_text(encoding="utf-8")
        found = re.findall(r"^FROM ubuntu:24\.04@(sha256:[0-9a-f]{64})", text, re.M)
        assert found, f"база не закреплена по digest: {path.name}"
        digests.update(found)
    assert len(digests) == 1, f"базовые образы разошлись: {sorted(digests)}"


@pytest.mark.parametrize("name", [
    "LICENSES/GPL-3.0.txt",
    "THIRD_PARTY_NOTICES.md",
    "MODIFICATIONS.md",
    "PROVENANCE_AND_RIGHTS.md",
])
def test_licence_package_is_in_place(name: str):
    """Передача образа с Flow — это conveying по GPLv3 §6, а не просто отправка файла.

    Требуется приложить текст лицензии, сохранить notices, обозначить свои правки
    GPL-компонентов и назвать, откуда взять Corresponding Source. Четыре файла
    выше — минимум, без которого пакет отдавать нельзя; до 22.08 не было ни
    одного.
    """
    assert (ROOT / name).is_file(), f"нет файла лицензионного пакета: {name}"


def test_gpl_text_is_the_real_thing_not_a_placeholder():
    text = (LICENSES / "GPL-3.0.txt").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 29 June 2007" in text
    assert "Corresponding Source" in text, "текст обрезан: §6 обязателен"


def test_report_does_not_reopen_closed_licence_work():
    report = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")

    assert "Наш код с 24.08 объявлен под MIT" in report
    assert "Транзитивные Python-зависимости зафиксированы" in report
    assert "Лицензия нашего собственного кода не объявлена" not in report
    assert "Транзитивные зависимости не зафиксированы" not in report


def test_submission_names_our_licence_and_only_current_open_risks():
    submission = (ROOT / "docs" / "submission.md").read_text(encoding="utf-8")

    assert "| `LICENSE` | MIT на наш код" in submission
    assert "лицензия нашего собственного\nкода не объявлена" not in submission
    assert "состав финального образа ещё не снят" in submission


def test_every_pinned_runtime_package_has_its_licence_text():
    """Лицензия — свойство версии, а не названия пакета.

    Поэтому файлы в `LICENSES/` названы `<пакет>-<версия>-<лицензия>.txt`, и
    подъём пина обязан ронять этот тест: новый пин без нового текста означает,
    что в поставке лежит лицензия от другой версии.
    """
    texts = [path.name for path in LICENSES.glob("*.txt")]
    for requirement in _packages(REQUIREMENTS):
        name, version = requirement.split("==")
        prefix = f"{name}-{version}-"
        assert any(text.startswith(prefix) for text in texts), \
            f"нет текста лицензии для {requirement} (ожидался {prefix}*.txt)"


def test_the_image_carries_the_licence_texts_but_not_our_rights_register():
    """GPLv3 §4: копия лицензии передаётся вместе с программой.

    Получатель образа получает Flow и `resdata` под GPL — значит тексты лицензий
    должны быть внутри образа, а не только в репозитории, из которого его
    собирали. `PROVENANCE_AND_RIGHTS.md` — наоборот: внутренний реестр прав, где
    записано, чего мы не подтвердили; получателю образа он не адресован.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    copied = "\n".join(re.findall(r"^COPY .*", text, re.M))

    assert "COPY LICENSES" in copied
    assert "COPY LICENSE " in copied
    assert "THIRD_PARTY_NOTICES.md" in copied
    # проверяются именно инструкции COPY: упоминание файла в комментарии рядом —
    # это объяснение, почему его тут нет, и оно как раз уместно
    assert "PROVENANCE_AND_RIGHTS.md" not in copied


def test_the_licence_boundary_names_every_top_level_entry():
    """Каждый каталог и файл верхнего уровня отнесён к покрытым или к исключённым.

    Первая редакция `LICENSE` перечисляла покрытое и исключённое выборочно, и
    треть репозитория не попадала никуда: 17 файлов в корне, `docker/`,
    `LICENSES/`, `manus-answered/`. Хуже того, `docs/screenshots/` — снимки
    постов организаторов — формально уходили под нашу MIT, потому что лежат
    внутри покрытого `docs/`. Нашла машина Б в Б39, граница закрыта (Р-109).

    Тест держит именно это свойство: не текст формулировок, а то, что ни одна
    запись верхнего уровня не осталась без ответа «наше или чужое».
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    licence = (root / "LICENSE").read_text(encoding="utf-8")
    listed = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                            text=True, encoding="utf-8", check=True).stdout.splitlines()

    top = {name.strip('"').split("/")[0] for name in listed if name.strip()}
    # архивы и служебные точечные файлы называются в лицензии явным списком
    missing = [name for name in sorted(top) if name not in licence]

    assert not missing, (
        "в LICENSE не отнесены ни к покрытым, ни к исключённым: " + ", ".join(missing))


def test_service_archives_stay_out_of_tracking():
    """П. 8.3 ТЗ: «не содержать избыточных материалов, затрудняющих проверку».

    `data.7z` (49 МБ) и `to-machine-b.7z` (3,4 МБ) — передача между нашими же
    машинами, к решению отношения не имеющая. Оба были в git до 01.09: правило
    `/data.7z` в `.gitignore` появилось **после** коммита файла, а на
    отслеживаемые файлы `.gitignore` не действует — то есть само по себе правило
    возврат не остановит, остановит этот тест (Р-131).
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    listed = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                            text=True, encoding="utf-8", check=True).stdout.split()

    archives = sorted(name for name in listed if name.endswith(".7z"))

    assert not archives, (
        "служебные архивы вернулись в отслеживание: " + ", ".join(archives) +
        " — снимать через git rm --cached, файл на диске остаётся"
    )
    for name in ("/data.7z", "/to-machine-b.7z"):
        assert name in (root / ".gitignore").read_text(encoding="utf-8"), (
            f"{name} не закрыт правилом .gitignore"
        )


def test_readme_does_not_expand_mit_beyond_the_licence_boundary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`docs/screenshots/`, `manus-answered/`" in readme
    assert "`docs/` (кроме\n`docs/method/`)" not in readme
