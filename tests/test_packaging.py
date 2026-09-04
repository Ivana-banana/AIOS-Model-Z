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


#: Порог «тяжёлого» файла и причина, по которой каждый такой файл здесь лежит.
#: Порог не про место на диске: п. 8.3 ТЗ запрещает «избыточные или несвязанные
#: материалы, затрудняющие проверку решения», а тяжёлый файл в репозитории
#: решения — ровно тот случай, который обязан быть объяснён словами. Причина
#: пишется здесь, а не в комментарии к коммиту: коммит через месяц никто не
#: найдёт, а этот список читается вместе с тестом.
HEAVY_BYTES = 1_000_000
HEAVY_ALLOWED = {
    "materials/Model_Z_final_OPM.zip":
        "действующая поставка дека — вход воспроизведения (README, разд. 6)",
    "materials/Model_Z.zip":
        "первая поставка, тНавигатор — источник разбора COMPDATMD (aios/pvt.py)",
    "materials/3. AIOS_Лекция_Разработка.pdf":
        "лекция организаторов; выжимка — docs/method/lecture-3-reservoir.txt",
    "materials/AIOS_Лекция.pptx":
        "лекция организаторов; выжимка — docs/method/lecture-2-ml.txt",
    "materials/Техническое_задание_хакатона_«AIOS».docx":
        "ТЗ хакатона — сама постановка задачи",
    "runs/matched-0995/wells_schedule.inc":
        "сдаваемый файл — заявленный сценарий, на нём считается заявляемый ЧДД",
    "artifacts/surrogate.joblib":
        "обученный суррогат — часть комплекта поставки; без него допуск падал "
        "9 из 10 на клоне проверяющего (Д4, Р-197)",
}

#: Потолок веса рабочего дерева. Смысл не в точном числе, а в том, чтобы прибавка
#: мегабайтов замечалась сразу, а не через ревизию.
#:
#: Поднят с 60 до 70 МБ 04.09 — осознанно и ровно по правилу, которое печатает сам
#: этот тест: в git лёг обученный суррогат (5,4 МБ), без которого допуск падал
#: 9 из 10 на клоне проверяющего (Д4, Р-197). Дерево — 62,1 МБ в 329 файлах.
TRACKED_CEILING_BYTES = 70 * 1024 * 1024


def _tracked_files() -> list[tuple[int, str]]:
    """Отслеживаемые файлы и их размеры. `-z`, потому что иначе git цитирует имена.

    Без `-z` git отдаёт кириллические имена в кавычках и восьмеричными кодами
    вместо букв. Сравнить такое имя с записью списка нельзя, и проверка молча
    перестаёт что-либо находить — худший исход для сторожа.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True,
                            check=True).stdout.decode("utf-8").split('\x00')
    found = []
    for name in listed:
        if not name:
            continue
        path = root / name
        if path.is_file():
            found.append((path.stat().st_size, name))
    return found


def test_every_heavy_tracked_file_has_a_written_reason():
    """Тяжёлый файл в репозитории обязан быть объяснён, иначе он груз (п. 8.3 ТЗ).

    Прежний сторож ловил `.7z` по расширению — то есть ровно те два архива,
    которые уже нашли. Любой другой тяжёлый файл (`.zip`, датасет, веса) вошёл бы
    так же тихо, как в своё время `data.7z`: правило `.gitignore` на уже
    отслеживаемый файл не действует, а по расширению его никто не ждал.
    Здесь проверяется вес, а не расширение.
    """
    heavy = sorted(name for size, name in _tracked_files() if size >= HEAVY_BYTES)
    unexplained = [name for name in heavy if not HEAVY_ALLOWED.get(name)]

    assert not unexplained, (
        "отслеживается тяжёлый файл без записанной причины: " + ", ".join(unexplained)
        + " — либо снять через git rm --cached, либо дописать причину в HEAVY_ALLOWED"
    )


def test_the_list_of_heavy_files_does_not_outlive_them():
    """Список причин не превращается в кладбище: снятый файл убирается из него.

    Иначе через месяц он описывает не репозиторий, а память о нём — и разрешает
    вернуть файл, про который решили обратное.
    """
    tracked = {name for _, name in _tracked_files()}
    stale = sorted(name for name in HEAVY_ALLOWED if name not in tracked)

    assert not stale, (
        "в HEAVY_ALLOWED остались файлы, которых больше нет в отслеживании: "
        + ", ".join(stale))


def test_tracked_weight_stays_under_the_ceiling():
    """Вес рабочего дерева под потолком — защита от прибавки по мегабайту.

    Отдельно от проверки выше: сто файлов по 900 КБ каждый порог «тяжёлого» не
    берут, а проверку решения затрудняют так же.
    """
    files = _tracked_files()
    total = sum(size for size, _ in files)

    assert total <= TRACKED_CEILING_BYTES, (
        f"отслеживается {total / 1048576:.1f} МБ в {len(files)} файлах при потолке "
        f"{TRACKED_CEILING_BYTES / 1048576:.0f} МБ — либо груз лишний, либо потолок "
        f"поднимается осознанно и с записью в docs/decisions.md"
    )


#: Предел длины отслеживаемого пути. Не про красоту: `git clone` на Windows
#: обрывается по `MAX_PATH` (260 символов) вместе с путём каталога, в который
#: клонируют. `docs/clean-clone-check.md` описывает это как первую находку
#: протокола: клон в глубокий путь «succeeded», а рабочее дерево пустое —
#: `git status` показал 289 удалённых файлов. Организаторы работают в
#: тНавигаторе, то есть с высокой вероятностью на Windows (С14).
TRACKED_PATH_LIMIT = 80


def test_no_tracked_path_is_long_enough_to_break_a_windows_clone():
    """Длинный путь ломает не наш клон, а клон проверяющего — и молча.

    Хуже всего то, как это выглядит: `git clone` печатает «succeeded», а рабочее
    дерево остаётся пустым. Проверяющий видит пустой репозиторий, а не ошибку.

    **`vendor/` исключён, и это не поблажка себе.** Там лежит расчётчик
    организаторов, вендоренный побайтово; переименовать его файлы значит нарушить
    правило `CLAUDE.md` «вендор оставляем как есть» и сломать сверку отпечатков.
    Два самых длинных пути там — 92 символа
    (`vendor/CHDD_PYTHON/output/Расчет_ЧДД_…_OPTIMIZED_AlenkinDA.json` и `.xlsx`),
    и это остаточный риск, который мы снять не можем: он называется здесь, а не
    прячется.
    """
    too_long = sorted((name for _, name in _tracked_files()
                       if len(name) > TRACKED_PATH_LIMIT
                       and not name.startswith("vendor/")), key=len, reverse=True)

    assert not too_long, (
        "путь длиннее " + str(TRACKED_PATH_LIMIT) + " символов: "
        + "; ".join(f"{len(name)} — {name}" for name in too_long[:5])
    )


def test_every_manus_path_named_in_the_docs_exists():
    """Ссылка на разбор внешнего агента обязана указывать на живой файл.

    Файлы переименованы 03.09 из-за длины пути (С14). Ссылка, оставшаяся от
    прежнего имени, ведёт в никуда — и заметно это только тому, кто по ней пошёл.
    Разбор без регулярного выражения намеренно: имя обрывается по первому
    разделителю, и читать такую проверку проще, чем шаблон с экранированием.
    """
    root = Path(__file__).resolve().parent.parent
    stop = set("`)]*<>|") | {chr(34), chr(39), chr(10), chr(9), " "}
    broken: list[str] = []

    for path in sorted(root.rglob("*.md")):
        if ".venv" in path.parts or ".git" in path.parts or "__MACOSX" in path.parts:
            continue
        if path.name.startswith("._"):
            continue  # ресурсные вилки macOS из распакованного архива — не текст
        text = path.read_text(encoding="utf-8", errors="replace")
        for chunk in text.split("manus-answered/")[1:]:
            name = ""
            for char in chunk:
                if char in stop:
                    break
                name += char
            if name.endswith(".md") and not (root / "manus-answered" / name).is_file():
                broken.append(f"{path.relative_to(root)} -> {name}")

    assert not broken, "ссылки в никуда: " + "; ".join(broken[:5])


def test_readme_does_not_expand_mit_beyond_the_licence_boundary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`docs/screenshots/`, `manus-answered/`" in readme
    assert "`docs/` (кроме\n`docs/method/`)" not in readme
def test_launch_check_survives_a_reworded_readme():
    """Сторож инструкции запуска ищет смысл, а не буквальные заголовки.

    02.09 README переоформили под защиту — разделы стали нумерованными, — и
    проверка объявила пакет неполным при полностью сохранившейся инструкции.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.check_package import _readme_has_launch

    root = Path(__file__).resolve().parent.parent
    assert _readme_has_launch(root / "README.md"), "настоящий README обязан проходить"


def test_launch_check_still_fails_without_the_single_entry_point(tmp_path):
    """И всё же проверяет: README без единой команды прогона не проходит."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.check_package import _readme_has_launch

    readme = tmp_path / "README.md"
    readme.write_text("# Проект\n\n## Установка\n\n## Запуск\n", encoding="utf-8")

    assert not _readme_has_launch(readme)


def test_the_readme_explains_the_skips_a_clean_clone_will_show():
    """Проверяющий увидит 18 пропусков и должен понимать, что это не поломка.

    Замер 04.09 на чистом клоне: 1 452 теста, 0 падений, **18 пропусков** против
    6 на машине с выгрузками. Пропуск здесь означает «проверять нечего на этой
    машине», а не «проверка не работает», — и без объяснения в README это
    читается как треть молчащих проверок.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "18 тестов пропускаются" in readme, "README не готовит к пропускам"
    assert "-rs" in readme, "не сказано, как увидеть причины пропусков"
    for source in ("runs/base", "portability.md", "core.js"):
        assert source in readme, f"не названа причина пропусков: {source}"
