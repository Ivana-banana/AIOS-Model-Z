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
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

#: Скрипты, на которые ссылаются документация и Dockerfile.
REQUIRED = ("build_case.py", "run_flow.sh", "export_rows.py", "build_doe.py", "run_doe.py", "run_cycle.py", "calibrate_search.py",
            "transcribe.py", "build_dataset.py", "train_surrogate.py", "run_all.py",
            "docx_to_text.py", "profiles.py", "pack_npv.py",
            "surrogate_warnings.py", "build_scaled_case.py", "check_deck.py",
            "forcing_check.py", "check_package.py", "check_release_tag.py",
            "check_submission.py")
PYTHON_ENTRYPOINTS = tuple(sorted(path.name for path in SCRIPTS.glob("*.py")))


@pytest.mark.parametrize("name", REQUIRED)
def test_documented_script_exists(name: str):
    assert (SCRIPTS / name).is_file(), f"нет scripts/{name}, а на него ссылается документация"


@pytest.mark.parametrize("name", REQUIRED)
def test_script_is_tracked_by_git(name: str):
    """Файл на диске ничего не значит: важно, что он попадёт в клон другой машины."""
    listed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", f"scripts/{name}"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if listed.returncode != 0 and "not a git repository" in listed.stderr.lower():
        pytest.skip("не git-репозиторий")
    assert listed.returncode == 0, f"scripts/{name} не под контролем версий: {listed.stderr.strip()}"


def test_scripts_are_not_ignored_on_case_insensitive_filesystems():
    """Ровно тот случай, который стоил проекту всех скриптов запуска."""
    checked = subprocess.run(
        ["git", "-c", "core.ignorecase=true", "check-ignore", "-v",
         *(f"scripts/{name}" for name in REQUIRED)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
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


def test_collect_pairs_example_updates_the_tracked_report():
    text = (SCRIPTS / "collect_pairs.py").read_text(encoding="utf-8")

    assert "--out docs/data/margin-pairs.csv" in text
    assert 'default="docs/data/margin-pairs.csv"' in text
    # шестая колонка — sha256 артефакта: без неё переобученный суррогат молча
    # наследует калибровку модели, которой больше нет
    assert "model_sha = checksum(Path(arguments.model))" in text
    assert "model_sha]" in text


def test_shell_scripts_are_executable():
    for path in SCRIPTS.glob("*.sh"):
        assert os.access(path, os.X_OK), f"{path.name} не исполняемый — инструкция запуска врёт"


def _bash_path(path: Path) -> str:
    """Путь для bash — относительно корня репозитория, через прямые слэши.

    Под Windows `subprocess` находит `Git\\usr\\bin\\bash.exe`: это MSYS-bash без
    настроенного отображения дисков, и абсолютные пути ему недоступны вообще —
    ни `C:\\Users\\...` (съедает обратные слэши как escape), ни `C:/Users/...`,
    ни `/c/Users/...`. Относительный путь от рабочего каталога видят все три
    варианта bash, поэтому все вызовы идут с `cwd=ROOT` и путями от него.

    Цена ошибки — не падение, а ложное прохождение: bash не находил скрипт,
    возвращал 127, и проверка «код возврата не ноль» засчитывала это за успех.
    Защита от повтора — явная проверка на 127 в самих тестах.
    """
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:  # другой диск под Windows — общего относительного пути нет
        pytest.skip(f"{path} на другом диске относительно {ROOT}")


def _fake_flow(tmp_path: Path, body: str) -> dict[str, str]:
    """PATH с подставным `flow`, чтобы проверять поведение обвязки без симулятора."""
    binary = tmp_path / "bin"
    binary.mkdir()
    script = binary / "flow"
    # newline="\n" обязателен: под Windows перевод строки по умолчанию — CRLF,
    # и shebang превращается в «#!/usr/bin/env bash\r», которого нет в системе.
    script.write_text(
        '#!/usr/bin/env bash\n'
        '[[ "$1" == "--version" ]] && { echo "flow 0.0-fake"; exit 0; }\n'
        f"{body}\n",
        encoding="utf-8", newline="\n",
    )
    script.chmod(0o755)
    return binary


def _with_fake_flow(binary: Path, command: list[str]) -> list[str]:
    """Запустить команду так, чтобы подставной `flow` нашёлся любым bash.

    Через переменную окружения это не работает и молча: `subprocess` на Windows
    резолвит `bash` через CreateProcess, а тот находит `C:\\Windows\\System32\\bash.exe`
    (WSL) раньше Git Bash — и Windows-PATH внутрь WSL не переносится. После
    установки OPM Flow в WSL эти тесты начали запускать **настоящий** симулятор
    вместо подставного, то есть проверяли не то, что написано.

    Поэтому PATH задаётся внутри самой команды, а путь к каталогу — относительный
    от `cwd`, который понимают и MSYS-bash, и WSL-bash.
    """
    inner = " ".join(f'"{part}"' for part in command)
    return ["bash", "-c", f'PATH="$PWD/{_bash_path(binary)}:$PATH"; exec {inner}']


def test_run_flow_fails_when_the_result_is_missing(tmp_path: Path):
    """Симулятор умеет выйти с нулём, не создав `.UNSMRY`.

    Такой прогон обязан быть отказом: при пакетном счёте DoE иначе в датасет
    молча попадают пустые кейсы, и обнаруживается это через сотню расчётов.
    """
    case = tmp_path / "case"
    case.mkdir()
    (case / "RUN.DATA").write_text("", encoding="utf-8")

    result = subprocess.run(
        _with_fake_flow(
            _fake_flow(tmp_path, "exit 0"),
            ["bash", _bash_path(SCRIPTS / "run_flow.sh"), _bash_path(case / "RUN.DATA"), "1"],
        ),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode != 0, (
        "результата нет, а скрипт доложил об успехе:\n" + result.stdout[-800:]
    )
    # Отказ должен быть отказом обвязки, а не «bash не нашёл файл».
    assert result.returncode != 127, f"скрипт не запустился вовсе:\n{result.stderr[-400:]}"


def test_run_flow_reports_skipped_keywords(tmp_path: Path):
    """Блок пропущенных ключевых слов — главная защита от тихой подмены физики.

    Flow не падает на том, чего не понял: он это игнорирует и считает дальше.
    Если блок скажет «нет» при непустом списке, прогон будет принят за
    корректный, хотя посчитана другая модель.
    """
    case = tmp_path / "case"
    case.mkdir()
    (case / "RUN.DATA").write_text("", encoding="utf-8")

    fake = _fake_flow(
        tmp_path,
        'echo "Error: Unsupported keywords or keyword items:"\n'
        'echo "  PINCHREG: keyword not supported"\n'
        f'touch "$PWD/{_bash_path(case)}/RUN.UNSMRY"\n'
        "exit 0",
    )
    result = subprocess.run(
        _with_fake_flow(
            fake,
            ["bash", _bash_path(SCRIPTS / "run_flow.sh"), _bash_path(case / "RUN.DATA"), "1"],
        ),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert "PINCHREG" in result.stdout, (
        "пропущенное ключевое слово не попало в отчёт скрипта:\n" + result.stdout[-800:]
    )
    block = result.stdout.split("--- пропущенные и непонятые ключевые слова ---", 1)[-1]
    assert not block.lstrip().startswith("нет"), (
        "блок доложил «нет» при непустом списке:\n" + block[:400]
    )


@pytest.mark.parametrize("name", PYTHON_ENTRYPOINTS)
def test_python_scripts_run_from_a_clean_checkout(name: str):
    """`python scripts/<имя>.py --help` обязан работать без установки пакета.

    Скрипты запускаются из корня репозитория, где `aios` не установлен в
    окружение, поэтому каждый добавляет корень в `sys.path` сам.
    """
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name), "--help"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, f"{name} --help упал:\n{result.stderr}"


@pytest.mark.parametrize("name", PYTHON_ENTRYPOINTS)
def test_python_scripts_survive_a_cp1251_console(name: str):
    """Вывод не должен зависеть от кодировки консоли.

    Настоящий дефект, найденный 17.08: `build_dataset.py --help` падал с
    `UnicodeEncodeError` на символе «→», потому что консоль Windows по умолчанию
    cp1251. Цена ошибки несимметрична: расчёт к этому моменту уже выполнен, и
    рушится именно печать результата — то есть теряется то, ради чего запускали.

    ТЗ (п. 3.5) запрещает скрытые зависимости, препятствующие проверке решения;
    кодировка консоли проверяющего — ровно такая зависимость. `PYTHONIOENCODING`
    здесь воспроизводит чужую консоль на любой платформе.
    """
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name), "--help"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "cp1251"},
    )
    assert result.returncode == 0, (
        f"{name} --help падает в консоли cp1251:\n{result.stderr[-600:]}"
    )
    assert "UnicodeEncodeError" not in result.stderr


def test_export_rows_labels_two_different_totals_blocks(tmp_path, monkeypatch, capsys):
    """Слово «нефть» печаталось дважды с разным смыслом — и числа уже путались.

    Верхний блок считает всю выгрузку целиком, нижний — эталонный расчётчик, то
    есть с даты дисконтирования и без строк с отрицательными приращениями. Это
    разные периоды и разные фильтры, а подпись была одна. Дважды числа за разные
    периоды попадали в одну строку журнала — поэтому период теперь стоит в
    заголовке, а жидкость и закачка печатаются в обоих блоках, а не только в
    верхнем.
    """
    from conftest import make_row

    from aios import pvt, summary
    from scripts import export_rows

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "RUN.UNSMRY").write_bytes(b"")
    rows = [
        make_row("2013-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2014-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2014-02-01", "2", wwir=50.0, injection=1500.0),
    ]
    monkeypatch.setattr(summary, "monthly_rows", lambda *a, **k: rows)
    monkeypatch.setattr(export_rows.summary, "monthly_rows", lambda *a, **k: rows)
    monkeypatch.setattr(
        export_rows.pvt, "fluids_by_well",
        lambda *a, **k: pvt.DeckFluids({}, "плотности: подставные, тест"),
    )

    code = export_rows.main([str(run_dir), "--case-name", "RUN"])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "итоги по всей выгрузке" in printed, "верхний блок обязан назвать свой период"
    assert printed.count("жидкость") >= 2, "жидкости не было в блоке ЧДД вовсе"
    assert printed.count("закачка") >= 2, "закачки не было в блоке ЧДД вовсе"


def test_export_rows_says_whether_the_run_obeys_our_own_water_rule(tmp_path, monkeypatch,
                                                                    capsys):
    """Сдаваемый сценарий собран тупым бейзлайном и правило по воде не проходил.

    До 23.08 ограничение по воде жило только в цикле планировщика, а
    `build_scaled_case.py` его не звал вовсе. Проверенный этой проверкой прогон
    ×1.5/×1.6 нарушает наш же потолок компенсации в десяти годах — и узнали мы
    это только когда проверку поставили в путь, которым считается выгрузка.
    """
    from conftest import make_row

    from aios import pvt, summary
    from scripts import export_rows

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "RUN.UNSMRY").write_bytes(b"")
    rows = [
        make_row("2013-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        # закачка вчетверо выше отобранного объёма — потолок компенсации пробит
        make_row("2013-02-01", "2", wwir=200.0, injection=6000.0),
    ]
    monkeypatch.setattr(summary, "monthly_rows", lambda *a, **k: rows)
    monkeypatch.setattr(export_rows.summary, "monthly_rows", lambda *a, **k: rows)
    monkeypatch.setattr(
        export_rows.pvt, "fluids_by_well",
        lambda *a, **k: pvt.DeckFluids({}, "плотности: подставные, тест"),
    )

    code = export_rows.main([str(run_dir), "--case-name", "RUN"])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "баланс воды" in printed, "выгрузка молчит о нашем же ограничении"
    assert "выше предела" in printed, "нарушение потолка компенсации не названо"


def test_water_notes_follow_the_rule_named_in_the_config(tmp_path):
    """Правило выбирается конфигом: подменять одно другим молча значит менять ЧДД."""
    from dataclasses import replace

    from conftest import make_row

    from aios import config as config_module
    from scripts import export_rows

    configuration = config_module.load(config_module.DEFAULT_PATH)
    rows = [
        make_row("2013-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2013-02-01", "2", wwir=200.0, injection=6000.0),
    ]

    strict = replace(configuration.limits, water_balance_rule="produced_water")
    off = replace(configuration.limits, enforce_water_balance=False)

    assert export_rows.water_notes(rows, replace(configuration, limits=strict))
    assert export_rows.water_notes(rows, replace(configuration, limits=off)) == [
        "проверка выключена (`enforce_water_balance = false`)"
    ]


def test_pack_npv_skips_an_unfinished_scenario(tmp_path, monkeypatch, capsys):
    """Оборванный сценарий обязан выпасть из таблицы, а не встать в неё числом.

    У прогона, убитого на середине, `.UNSMRY` читается — он просто короче. ЧДД по
    нему посчитается и будет выглядеть настоящим, хотя это ЧДД начала истории.
    Отметка `.doe-done` — единственное, что отличает досчитанный сценарий от
    обрезка (`scripts/run_doe.py`), поэтому таблица опирается на неё.
    """
    import json

    from conftest import make_row

    from scripts import pack_npv

    root = tmp_path / "pack"
    for name, done in (("doe-000", True), ("doe-001", False)):
        (root / name).mkdir(parents=True)
        (root / name / "RUN.UNSMRY").write_text("", encoding="utf-8")
        if done:
            (root / name / ".doe-done").write_text("2026-08-19", encoding="utf-8")
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [
            {"name": "doe-000", "multipliers": {"1": 1.2}},
            {"name": "doe-001", "multipliers": {"1": 0.8}},
        ]}),
        encoding="utf-8",
    )
    rows = [make_row("2007-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0)]
    monkeypatch.setattr(pack_npv.summary, "monthly_rows", lambda *a, **k: rows)

    code = pack_npv.main([str(root), "--config", str(_tiny_pack_config(tmp_path))])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "doe-000" in printed
    assert "досчитано 1 из 2" in printed
    assert "пропущен doe-001" in printed


def _tiny_pack_config(tmp_path) -> Path:
    """Конфиг с крошечным деком: таблица ЧДД к содержимому дека не привязана."""
    from test_dataset import _tiny_project

    return _tiny_project(tmp_path)


def test_surrogate_warnings_names_a_missing_model(tmp_path, capsys):
    """Промах мимо артефакта — строка и код 1, а не traceback из joblib.

    Детектор диапазона стоит вторым шагом покандидатного протокола рядом со
    шлюзом доверия (`HANDOFF.md`), а артефакт живёт в `artifacts/`, который в
    git не ведётся: на чистом клоне промах мимо него — обычный первый запуск.
    Шлюз это уже говорит словами, детектор ронял `FileNotFoundError` из недр
    `joblib.numpy_pickle`.
    """
    from scripts import surrogate_warnings

    code = surrogate_warnings.main(["--model", str(tmp_path / "нет.joblib")])
    assert code == 1
    assert "нет обученного суррогата" in capsys.readouterr().err


def test_surrogate_warnings_counts_reasons_not_just_lines(tmp_path, capsys):
    """Пометки надо считать по причинам, а не по строкам.

    «188 строк» и «188 разных причин» — разные новости: первое означает, что одна
    и та же причина повторена по скважино-месяцам, второе — что модель не знает
    почти ничего. Свёртка причин (`7163c71`) сделана ровно поэтому, и проверка
    здесь на том, что счётчик её показывает.

    Порог качества скриптом намеренно не применяется: мерить пометки надо и у
    модели, которая в цикл не попадёт, иначе про неё нечего сказать.
    """
    import joblib
    import pandas as pd
    from test_dataset import _tiny_project

    from aios.neural import NeuralBundle, TargetMetric, TrainingReport
    from aios.surrogate import NumericRange, TrainingEnvelope
    from scripts import surrogate_warnings

    metrics = {target: TargetMetric(3.0, 1.0) for target in ("WLPT_Diff", "WOMT_Diff",
                                                            "WWIT_Diff", "THP", "BHP")}
    bundle = NeuralBundle(
        ("prod:1",), {}, pd.Timestamp("2007-01-01"), 12,
        TrainingEnvelope({"prod:1": NumericRange(0.5, 1.5)},
                         NumericRange(0.5, 1.5), NumericRange(0.0, 0.0)),
        # метрики заведомо проигрышные: скрипт обязан работать и на такой модели
        TrainingReport(("a",), ("b",), 1, 2, metrics),
    )
    model = tmp_path / "surrogate.joblib"
    joblib.dump(bundle, model)

    code = surrogate_warnings.main(
        ["--model", str(model), "--config", str(_tiny_project(tmp_path))]
    )
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "опорное расписание" in printed
    assert "строк:" in printed


def test_every_warning_kind_is_recognised_not_lumped_into_other():
    """Счётчик причин обязан знать все причины, которые система порождает.

    Иначе он молча сваливает самую многочисленную группу в «прочее» и
    показывает `20925 прочее` — то есть ровно ту бесполезную сводку, ради
    замены которой писался. Проверяется не список маркеров, а само свойство:
    ни одна настоящая пометка не должна остаться неопознанной.
    """
    from datetime import date

    from aios.surrogate import NumericRange, TrainingEnvelope
    from scripts.surrogate_warnings import _kind

    envelope = TrainingEnvelope.fit([
        {"prod:1": 1.0, "prod:2": 1.0, "inj:9": 1.0},
        {"prod:1": 0.8, "prod:2": 1.2, "inj:9": 1.2},
        {"prod:1": 1.2, "prod:2": 0.8, "inj:9": 0.8},
    ])
    produced = envelope.warnings({date(2007, 1, 1): {
        "prod:1": 1.9,        # вне своего диапазона
        "prod:2": 1.2,        # внутри своего, но фонд целиком — уже нет
        "inj:9": 1.0,
        "prod:НОВАЯ": 1.0,    # такого управления в обучении не было
    }})
    # артефакт, обученный до перевода детектора на множители, — тоже причина
    produced += TrainingEnvelope(
        {}, NumericRange(0.0, 0.0), NumericRange(0.0, 0.0)
    ).warnings({date(2007, 1, 1): {"prod:1": 1.0}})
    assert produced, "проверка бессмысленна, если предупреждений нет"

    unrecognised = sorted({warning for warning in produced if _kind(warning) == "прочее"})
    assert not unrecognised, "неопознанные причины:\n" + "\n".join(unrecognised)


def test_pack_npv_shows_the_whole_scenario_not_only_its_first_period(tmp_path, monkeypatch,
                                                                    capsys):
    """Колонка множителей обязана описывать сценарий целиком, а не его начало.

    `Scenario.multipliers` — это множители **первого периода**: поле оставлено
    ради совместимости манифеста и так и описано в `aios/doe.py`. Таблица брала
    его как «средний отбор сценария». На 4-периодном сценарии генератора это
    давало 0,384 при среднем по сценарию 0,453 и размахе 0,326…0,743 (замер
    20.08); на пакете с десятью периодами колонка описывала бы десятую часть
    прогона, а читалась бы как весь.
    """
    import json

    from conftest import make_row

    from scripts import pack_npv

    root = tmp_path / "pack"
    (root / "doe-000").mkdir(parents=True)
    (root / "doe-000" / "RUN.UNSMRY").write_text("", encoding="utf-8")
    (root / "doe-000" / ".doe-done").write_text("2026-08-20", encoding="utf-8")
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [{
            "name": "doe-000",
            "periods": [
                {"from": "2007-01-01", "multipliers": {"1": 0.80}, "injector_multipliers": {}},
                {"from": "2012-01-01", "multipliers": {"1": 1.30}, "injector_multipliers": {}},
            ],
            "multipliers": {"1": 0.80},
            "injector_multipliers": {},
        }]}),
        encoding="utf-8",
    )
    rows = [make_row("2007-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0)]
    monkeypatch.setattr(pack_npv.summary, "monthly_rows", lambda *a, **k: rows)

    code = pack_npv.main([str(root), "--config", str(_tiny_pack_config(tmp_path))])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "0.80…1.30" in printed, "показан только первый период"
    assert "период" in printed, "не сказано, что режим внутри сценария меняется"


def test_run_cycle_main_survives_its_own_call_sites(tmp_path, monkeypatch, capsys):
    """Цикл обязан доходить до конца на рабочем пути, а не только в юнит-тестах.

    Настоящий случай 20.08, найденный первым же сквозным прогоном на ГГДМ:
    `run_cycle.py` звал `for_cycle(..., well_fluids=…)`, а такого параметра у
    функции нет — `TypeError` прилетал **после** одиннадцати минут расчёта
    базового сценария. Ошибка прожила двое суток: юнит-тесты проверяли
    `loop.run_cycle` и `model.for_cycle` порознь, и ни один не звал `main`
    скрипта, то есть ровно ту склейку, которая и разошлась.

    Тест собирает крошечный прогон и проходит цикл целиком на тривиальном
    бейзлайне — без симулятора, но через тот же `main`, что запускает
    `run_all.py`.
    """
    import csv

    from conftest import make_row
    from test_dataset import _tiny_project

    from scripts import run_cycle

    config = _tiny_project(tmp_path)
    # нормативы берём настоящие: предмет теста — склейка вызовов, а не экономика
    norms = ROOT / "vendor" / "CHDD_PYTHON" / "input" / "Нормативы_ЧДД.xlsx"
    if not norms.is_file():
        pytest.skip("нет файла нормативов организаторов")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'norms_file = "norms.xlsx"', f'norms_file = "{norms.as_posix()}"'
        ),
        encoding="utf-8",
    )
    source = tmp_path / "runs" / "RUN"
    source.mkdir(parents=True)
    # с 24.08 цикл проверяет наличие результата расчёта до чтения (иначе resdata
    # бросает OSError вместо отказа, Б31). Здесь выгрузка подменена, поэтому
    # файлу достаточно существовать: предмет теста — склейка вызовов
    source.with_suffix(".UNSMRY").write_bytes(b"")
    rows = [
        make_row(f"{year}-{month:02d}-01", "1", wlpr=30.0, womr=20.0,
                 liquid=900.0, oil=600.0)
        for year in (2007, 2008) for month in range(1, 13)
    ]
    with (source.parent / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(run_cycle.summary, "monthly_rows", lambda *a, **k: rows)

    code = run_cycle.main([
        "--from", str(source.parent), "--out", str(tmp_path / "out"),
        "--no-tune-rates", "--config", str(config),
    ])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "суррогат:" in printed

    # Хвост Р-47: кейс уезжает считаться на другую машину, а `rows.csv` в его
    # каталоге не появится — его пишет `export_rows.py` уже по результату
    # полного расчёта. Поэтому плотности, на которых цикл ВЫБИРАЛ решения,
    # подписаны здесь же, рядом с `RUN.DATA`, а не в прокрутке терминала.
    note = tmp_path / "out" / "fluids.source.txt"
    assert note.is_file(), printed
    assert "прогноз цикла считался одной плотностью" in note.read_text(encoding="utf-8")
    assert "на каких плотностях:" in printed

    monkeypatch.setattr(run_cycle.summary, "sanity_problems", lambda _rows: ["отрицательный объём"])
    code = run_cycle.main([
        "--from", str(source.parent), "--confirm", str(tmp_path / "out"),
        "--no-tune-rates", "--config", str(config),
    ])

    assert code == 1
    assert "подтверждать ЧДД" in capsys.readouterr().err


def test_pack_npv_skips_a_case_without_our_summary_section(tmp_path, monkeypatch, capsys):
    """Негодная выгрузка — пропуск с названной причиной, а не падение всей таблицы.

    `monthly_rows` теперь отказывается читать расчёт, к которому не подключена
    SUMMARY-секция: до 20.08 он давал таблицу нулей и ЧДД 0,000 млн руб. как
    обычный результат. В пакете из 48 сценариев один такой не должен ни уносить
    таблицу по остальным сорока семи, ни молча становиться в ней нулём.
    """
    import json

    from scripts import pack_npv

    root = tmp_path / "pack"
    for name in ("doe-000", "doe-001"):
        (root / name).mkdir(parents=True)
        (root / name / "RUN.UNSMRY").write_text("", encoding="utf-8")
        (root / name / ".doe-done").write_text("2026-08-20", encoding="utf-8")
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [
            {"name": "doe-000", "multipliers": {"1": 1.2}},
            {"name": "doe-001", "multipliers": {"1": 0.8}},
        ]}),
        encoding="utf-8",
    )

    from conftest import make_row

    def rows_or_refuse(case, *args, **kwargs):
        if "doe-001" in str(case):
            raise ValueError("нет ни одного вектора по скважинам: SUMMARY-секция не подключена")
        return [make_row("2007-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0)]

    monkeypatch.setattr(pack_npv.summary, "monthly_rows", rows_or_refuse)

    code = pack_npv.main([str(root), "--config", str(_tiny_pack_config(tmp_path))])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "досчитано 1 из 2" in printed
    assert "SUMMARY-секция не подключена" in printed


def _dotted(node) -> str | None:
    """Имя вызываемого через точку: `loop.run_cycle`, `summary.monthly_rows`."""
    import ast

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _keyword_mismatches(path: Path) -> list[str]:
    """Ключевые аргументы вызовов в `aios`, которых нет в сигнатуре вызываемого."""
    import ast
    import importlib
    import inspect

    def signature_of(dotted: str):
        parts = dotted.split(".")
        for split in range(len(parts) - 1, 0, -1):
            try:
                target = importlib.import_module(".".join(parts[:split]))
            except Exception:  # noqa: BLE001 — не наш модуль, судить не о чем
                continue
            try:
                for attr in parts[split:]:
                    target = getattr(target, attr)
                return inspect.signature(target)
            except (AttributeError, TypeError, ValueError):
                return None
        return None

    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("aios"):
            aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
        elif isinstance(node, ast.Import):
            aliases.update({a.asname or a.name.split(".")[0]: a.name
                            for a in node.names if a.name.startswith("aios")})

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        head = name.split(".")[0] if name else None
        if head not in aliases:
            continue
        signature = signature_of(aliases[head] + name[len(head):])
        if signature is None:
            continue
        parameters = signature.parameters
        if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            continue
        found.extend(
            f"{path.name}:{node.lineno}: {name}(...) не принимает «{keyword.arg}»"
            for keyword in node.keywords
            if keyword.arg and keyword.arg not in parameters
        )
    return found


def test_no_call_passes_an_argument_the_callee_does_not_accept():
    """Склейка «скрипт → aios» обязана сходиться по именам аргументов.

    Дефект `aa88244`: `run_cycle.py` звал `for_cycle(..., well_fluids=…)` и
    `loop.run_cycle(..., well_fluids=…)` — такого параметра нет ни у той, ни у
    другой. `TypeError` прилетал **после одиннадцати минут** расчёта базового
    сценария, и прожил дефект двое суток: юнит-тесты проверяли обе стороны
    порознь, но ни один не звал `main` скрипта, то есть ровно ту склейку, которая
    и разошлась.

    Дозваться до всех склеек тестами дорого — на рабочем пути нужен симулятор.
    Эта проверка статическая и потому дешёвая: она смотрит ключевые аргументы
    каждого вызова против настоящей сигнатуры. Проверена на том самом коммите до
    правки — находит оба вызова.
    """
    modules = sorted(Path(__file__).resolve().parent.parent.glob("aios/**/*.py"))
    # скрипты берём списком, а не маской: на Windows каталог venv `Scripts/` —
    # для файловой системы тот же путь, что наш `scripts/`
    scripts = [SCRIPTS / name for name in PYTHON_ENTRYPOINTS]

    mismatches = [line for path in modules + scripts for line in _keyword_mismatches(path)]

    assert mismatches == [], "\n".join(mismatches)


def test_search_calibration_runs_every_config_seed_through_full_cycle(tmp_path):
    from scripts.calibrate_search import commands

    plan = commands(model="artifacts/surrogate.joblib", base=Path("runs/base"),
                    out_root=tmp_path, config="config.toml", seeds=(11, 12),
                    processes=2, flow_args="--parsing-strictness=low",
                    pairs=Path("docs/data/pairs.csv"), family="search")

    runs = [command for command in plan if command[1].endswith("run_all.py")]
    assert [command[command.index("--search-seed") + 1] for command in runs] == ["11", "12"]
    assert all("--skip-base" in command and "--surrogate-model" in command for command in runs)
    assert plan[-1][-2:] == ["--config", "config.toml"]
    assert "--append" in plan[-1]


def test_run_all_confirms_with_the_same_surrogate_it_planned_with():
    """Шаг подтверждения обязан считать «обещание» той моделью, что принимала решения.

    Настоящий случай 20.08, найденный первым прогоном на обученном суррогате:
    шаг 3 шёл на линейной модели («суррогат: линейный суррогат по сценариям
    DoE»), а шаг 5 — на тривиальном бейзлайне, потому что `--surrogate-model`
    ему не передавался. Подтверждённый ЧДД при этом верен — он из полного
    расчёта, — но строка «прогноз суррогата обещал» относилась к **другой**
    модели, и «ошибка прогноза» сравнивала расчёт с чужим прогнозом.

    Цена такой путаницы прямая: именно этим числом мы судим, можно ли доверять
    суррогату.
    """
    from scripts.run_all import plan

    steps = plan(
        base_dir=Path("runs/base"), policy_dir=Path("runs/policy"),
        processes=4, flow_args=[], config_path="config.toml",
        surrogate_model="artifacts/surrogate.joblib", case_name="RUN", skip_base=False,
    )
    cycle_steps = [command for _, command in steps if "run_cycle.py" in " ".join(command)]

    assert len(cycle_steps) == 2, [command for _, command in steps]
    for command in cycle_steps:
        assert "--surrogate-model" in command, (
            "шаг подтверждения считает обещание другой моделью: " + " ".join(command)
        )


def test_scaled_case_scales_our_steps_and_leaves_history_alone(tmp_path, monkeypatch):
    """Масштабируется управление, а не история — иначе мы переписываем прошлое.

    Команда появилась 21.08 потому, что лучший подтверждённый результат проекта
    (форсаж ×1,5 с закачкой ×2,0, ЧДД 13 642,4) собирался разовым кодом: прогон
    исчез вместе со средой, а команды, которой его повторить, в репозитории не
    осталось. Заявлять ЧДД, который не можешь пересчитать, запрещено, поэтому
    сценарий обязан строиться командой и быть покрыт тестом.
    """
    from test_dataset import _tiny_project

    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from scripts import build_scaled_case

    captured = {}

    def fake_build_case(run_dir, model_dir, steps=None, control_start=None,
                        wells=None, case_name="RUN"):
        captured["steps"] = list(steps or [])
        captured["control_start"] = control_start

        class Case:
            data_file = Path(run_dir) / f"{case_name}.DATA"

        return Case()

    history = ScheduleStep(date(2006, 1, 1),
                           (ProducerControl("1", "OPEN", "LRAT", 10.0, 50.0),), ())
    ours = ScheduleStep(date(2007, 1, 1),
                        (ProducerControl("1", "OPEN", "LRAT", 20.0, 50.0),),
                        (InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 250.0),))

    class Parsed:
        steps = (history, ours)
        wells = ("1", "2")

    monkeypatch.setattr(build_scaled_case, "build_case", fake_build_case)
    monkeypatch.setattr(build_scaled_case, "parse_schedule", lambda *a, **k: Parsed())
    monkeypatch.setattr(build_scaled_case, "find_data_file", lambda *a, **k: None)
    monkeypatch.setattr(build_scaled_case, "layout",
                        lambda *a, **k: type("L", (), {"one": lambda self, name: Path("s.inc")})())

    code = build_scaled_case.main([
        str(tmp_path / "run"), "--production", "1.5", "--injection", "2.0",
        "--config", str(_tiny_project(tmp_path)),
    ])

    assert code == 0
    assert len(captured["steps"]) == 1, "история не должна попадать в наши шаги"
    scaled = captured["steps"][0]
    assert scaled.producers[0].liquid_rate == pytest.approx(30.0)
    assert scaled.injectors[0].rate == pytest.approx(80.0)


def test_build_scaled_case_reports_where_the_multiplier_was_not_applied_literally():
    """Ярлык «×1.5» обязан быть честным или снабжён оговоркой.

    Масштабирование молча делает две вещи: обрезает дебит по пределу методики
    500 м³/сут и применяет множитель к скважине, которая управляется не дебитом
    (у неё это поле — предел, а не уставка). На Модели Z ни то ни другое не
    срабатывает (замер 21.08: максимальный дебит 110 м³/сут, все 22 008 записей
    `LRAT`), но заявляемое число считается на **чужом** деке валидационного кейса,
    где так может не быть.

    Поведение намеренно не меняется — сдаваемый файл обязан остаться тем, на
    котором посчитано заявляемое число. Меняется только видимость.
    """
    from datetime import date

    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.optimise import scale, scale_injection, scaling_report
    from scripts.build_scaled_case import _honest_multiplier_notes

    before = [ScheduleStep(date(2007, 1, 1), (
        ProducerControl("предел", "OPEN", "LRAT", 400.0, 50.0),
        ProducerControl("давление", "OPEN", "BHP", 100.0, 90.0),
        ProducerControl("обычная", "OPEN", "LRAT", 100.0, 50.0),
        ProducerControl("без дебита", "OPEN", "ORAT", None, 50.0),
    ), (
        InjectorControl("закачка", "WATER", "OPEN", "RATE", 100.0, 250.0),
        InjectorControl("объёмная", "WATER", "OPEN", "RESV", None, 250.0),
    ))]
    after = scale_injection(scale(before, 1.5), 1.6)
    report = scaling_report(before, after, production=1.5, injection=1.6)
    notes = _honest_multiplier_notes(report)

    assert any("упёрлись в предел" in note for note in notes), notes
    assert any("не под уставочным режимом" in note and "BHP" in note
               for note in notes), notes
    # нагнетательные считаются наравне с добывающими: раньше нота смотрела
    # только на добывающих и говорила «все под LRAT», когда нагнетательная под
    # RESV молча останавливалась (Б29)
    assert any("без поля расхода" in note and "RESV" in note and "ORAT" in note
               for note in notes), notes
    assert report.records == 6 and report.literal == 2, report


def test_build_scaled_case_says_so_when_the_multiplier_was_applied_literally():
    """Обратная сторона: на исправном материале оговорок быть не должно.

    Иначе предупреждение, звучащее всегда, перестаёт читаться — та же болезнь,
    что была у детектора обучающего диапазона.
    """
    from datetime import date

    from aios.deck import ProducerControl, ScheduleStep
    from aios.optimise import scale, scaling_report
    from scripts.build_scaled_case import _honest_multiplier_notes

    before = [ScheduleStep(date(2007, 1, 1),
                           (ProducerControl("1", "OPEN", "LRAT", 110.0, 50.0),), ())]
    report = scaling_report(before, scale(before, 1.5), production=1.5, injection=1.0)
    notes = _honest_multiplier_notes(report)

    assert notes == ["ни одна запись не потеряла заявленный множитель: "
                     "предел не сработал, все режимы уставочные"], notes
    assert report.share == 1.0


def test_run_cycle_names_the_missing_simulator_output(tmp_path, monkeypatch, capsys):
    """Без выгрузки ГГДМ цикл обязан отказать строкой, а не трассировкой resdata.

    Замер 24.08 (Б31): команда из выданной задачи —
    `run_cycle.py --from runs/base --out … --surrogate-model …` — на машине без
    расчёта падала `OSError: Failed to create summary instance`. Это законный
    исход (расчёта нет), но узнавать о нём надо строкой с ценой: тот же класс
    дефекта, что чинили в Б27 и Б30.
    """
    from test_dataset import _tiny_project

    from scripts import run_cycle

    config = _tiny_project(tmp_path)
    monkeypatch.setattr(run_cycle.pvt, "fluids_by_well",
                        lambda *a, **k: type("F", (), {"note": "тест", "by_well": {}})())
    source = tmp_path / "runs" / "empty"
    source.mkdir(parents=True)

    code = run_cycle.main(["--from", str(source), "--out", str(tmp_path / "out"),
                            "--config", str(config)])

    assert code == 1
    printed = capsys.readouterr().err
    assert "нет результата расчёта" in printed
    assert "run_flow.sh" in printed, "отказ обязан называть, чем выгрузка получается"


def test_run_cycle_passes_the_multiplier_set_and_refuses_one_without_unity(tmp_path,
                                                                          monkeypatch,
                                                                          capsys):
    """Набор множителей поиска задаётся снаружи — иначе его нечем варьировать.

    Б31 велит поварьировать «разные стартовые множители», но у команды такого
    ключа не было вовсе: набор жил константой в `aios.optimise`. Узкий набор
    держит поиск ближе к обучающим сценариям, широкий уводит из области доверия,
    и оба замера нужны — на этой машине они и делаются.

    Единица в наборе обязательна: без неё у поиска нет режима «как есть», то
    есть не с чем сравнивать кандидатов.
    """
    from conftest import make_row
    from test_dataset import _tiny_project

    from scripts import run_cycle

    config = _tiny_project(tmp_path)
    monkeypatch.setattr(run_cycle.pvt, "fluids_by_well",
                        lambda *a, **k: type("F", (), {"note": "тест", "by_well": {}})())
    source = tmp_path / "runs" / "RUN"
    source.mkdir(parents=True)
    source.with_suffix(".UNSMRY").write_bytes(b"")

    code = run_cycle.main(["--from", str(source.parent), "--out", str(tmp_path / "out"),
                            "--multipliers", "0.9,1.1", "--config", str(config)])

    assert code == 1
    assert "обязан быть 1.0" in capsys.readouterr().err

    seen = {}

    def remember(*args, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(run_cycle.summary, "monthly_rows",
                        lambda *a, **k: [make_row("2007-01-01", "1", wlpr=30.0,
                                                  womr=20.0, liquid=900.0, oil=600.0)])
    monkeypatch.setattr(run_cycle.loop, "run_cycle", remember)

    with pytest.raises(SystemExit):
        run_cycle.main(["--from", str(source.parent), "--out", str(tmp_path / "out"),
                        "--multipliers", "0.75,1.0,1.25", "--config", str(config)])

    assert seen["multipliers"] == (0.75, 1.0, 1.25)



def test_the_cycle_records_which_surrogate_built_the_case(tmp_path):
    """Кейс обязан помнить, чем считался его прогноз.

    Найдено 24.08 на первом же кандидате поиска: кейс собран обученным
    артефактом, а подтверждение шло тривиальным бейзлайном — и печатало
    «прогноз обещал» от одной модели рядом с фактом, полученным расписанием
    другой. Та же ловушка, что пара «файл от одного прогона, число от другого».
    """
    from scripts import run_cycle

    artefact = tmp_path / "surrogate.joblib"
    artefact.write_bytes("веса".encode())

    plain = run_cycle.surrogate_identity(None, "тривиальный бейзлайн")
    trained = run_cycle.surrogate_identity(str(artefact), "MLP")

    assert plain == {"surrogate": "тривиальный бейзлайн"}
    assert trained["surrogate"] == "MLP"
    assert trained["model"] == str(artefact)
    assert len(trained["sha256"]) == 64, "отпечаток артефакта обязан быть записан"
    assert plain != trained


def test_a_case_built_before_the_marker_appeared_is_not_refused(tmp_path):
    """Кейс без `cycle.json` собран до Р-100 — это не повод отказывать."""
    from scripts import run_cycle

    assert run_cycle.written_identity(tmp_path) is None

    (tmp_path / "cycle.json").write_text('{"surrogate_identity": {"surrogate": "MLP"}}',
                                         encoding="utf-8")
    assert run_cycle.written_identity(tmp_path) == {"surrogate": "MLP"}

    (tmp_path / "cycle.json").write_text("не json", encoding="utf-8")
    assert run_cycle.written_identity(tmp_path) is None, "битый файл не должен ронять прогон"


def test_base_schedule_starts_the_search_from_the_scenario_not_from_the_deck(tmp_path,
                                                                            monkeypatch,
                                                                            capsys):
    """`--base-schedule` выражает сценарий множителями, а опорное не подменяет.

    Обученная модель принимает режим множителями к опорному расписанию дека, и
    подстановка другого опорного меняла бы смысл её входа, а не точку старта:
    замер 24.08 — один и тот же файл `runs/srg-narrow` оценивался в 12 664,9 как
    кандидат и в 11 201,4 как опорное, разница 1 463,5 млн руб. Поэтому сценарий
    переводится в множители к деку, и поиск стартует из них; на Модели Z это
    воспроизвело прогноз сценария с точностью 0,017 % (12 664,9 → 12 662,8).
    """
    from conftest import make_row
    from test_dataset import _tiny_project

    from scripts import run_cycle

    config = _tiny_project(tmp_path)
    monkeypatch.setattr(run_cycle.pvt, "fluids_by_well",
                        lambda *a, **k: type("F", (), {"note": "тест", "by_well": {}})())
    monkeypatch.setattr(run_cycle.summary, "monthly_rows",
                        lambda *a, **k: [make_row("2007-01-01", "1", wlpr=30.0,
                                                  womr=20.0, liquid=900.0, oil=600.0)])
    seen = {}

    def remember(*args, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(run_cycle.loop, "run_cycle", remember)
    source = tmp_path / "runs" / "RUN"
    source.mkdir(parents=True)
    source.with_suffix(".UNSMRY").write_bytes(b"")
    other = tmp_path / "candidate.inc"
    # дек крошечного проекта держит скважину «1» на 30 м3/сут, здесь она на 45
    other.write_text(
        "DATES\n 01 JAN 2007 /\n/\n\nWCONPROD\n"
        " '1' 'OPEN' 'LRAT' 1* 1* 1* 45.0 1* 50 1* 1* /\n/\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        run_cycle.main(["--from", str(source.parent), "--out", str(tmp_path / "out"),
                        "--base-schedule", str(other), "--config", str(config)])

    assert seen["start_multipliers"] == {("production", "1"): 1.5}
    assert "старт:" in capsys.readouterr().out


def test_search_calibration_can_match_the_mpi_layout_of_the_existing_corpus(tmp_path, capsys):
    """Пары считаются на том же числе процессов, что base и уже собранный корпус.

    `runs/base`, `inj-1.6`, `matched-0995` и `policy` посчитаны на 4 процессах
    MPI, а `[run].processes` в конфиге — 2. Разбиение области на процессы слегка
    меняет результат, поэтому калибровочный прогон обязан уметь повторить среду
    корпуса, не переписывая конфиг под текущую машину.
    """
    from scripts.calibrate_search import main

    code = main(["--model", "artifacts/surrogate.joblib", "--out-root", str(tmp_path),
                 "--processes", "4", "--dry-run"])
    printed = capsys.readouterr().out.splitlines()

    assert code == 0
    runs = [line for line in printed if "run_all.py" in line]
    assert runs, "план пуст"
    assert all("--processes 4" in line for line in runs)


def test_search_calibration_resumes_past_seeds_that_already_have_an_export(tmp_path):
    """Возобновление пропускает посчитанное, но собирает пары по всем каталогам.

    Пакет идёт около восьми часов; падение на девятом прогоне не должно стоить
    восьми часов заново. Готовность seed'а — `rows.csv`: именно его читает
    `collect_pairs.py`. Сбор в конце идёт по всем каталогам, включая пропущенные,
    иначе возобновлённая ночь потеряет уже посчитанные пары.
    """
    from scripts.calibrate_search import commands

    done = tmp_path / "search-seed-11"
    done.mkdir()
    (done / "rows.csv").write_text("DATA,well\n", encoding="utf-8")

    plan = commands(model="artifacts/surrogate.joblib", base=Path("runs/base"),
                    out_root=tmp_path, config="config.toml", seeds=(11, 12),
                    processes=4, flow_args="--parsing-strictness=low",
                    pairs=Path("docs/data/pairs.csv"), family="search", resume=True)

    runs = [command for command in plan if command[1].endswith("run_all.py")]
    assert [command[command.index("--search-seed") + 1] for command in runs] == ["12"]
    assert str(done) in plan[-1], "пропущенный seed обязан попасть в сбор пар"


#: Скрипты, которые ничего не печатают сами, — им `reconfigure` не нужен.
_SILENT_ENTRYPOINTS = frozenset({"__init__.py"})


@pytest.mark.parametrize(
    "name", [n for n in PYTHON_ENTRYPOINTS if n not in _SILENT_ENTRYPOINTS]
)
def test_every_script_prints_in_utf8_regardless_of_locale(name: str):
    """Ни один скрипт пакета не должен зависеть от локали консоли (п. 3.5 ТЗ).

    Найдено 31.08 при подготовке к КТ-2: `compare_reference.py` — команда, которую
    план защиты предписывает запустить **вживую на экране**, — падала
    `UnicodeEncodeError` на символе «Δ» в шапке таблицы, то есть после того, как
    сверка уже посчитана. Под cp1251 (умолчание русской Windows) «Δ» не кодируется
    вовсе. Тест `test_profiles.py` закрывал этот класс поштучно и утверждал, что
    «остальные десять скриптов защищены»; на деле без `reconfigure` их было семь.
    Поэтому проверка стала общей: новый скрипт получает её автоматически.
    """
    source = (SCRIPTS / name).read_text(encoding="utf-8")
    if "print(" not in source:
        pytest.skip(f"{name} ничего не печатает")

    assert 'sys.stdout.reconfigure(encoding="utf-8"' in source, (
        f"scripts/{name} печатает, но не переключает stdout в UTF-8 — "
        "под cp1251 упадёт на выдаче уже посчитанного результата"
    )
    assert 'sys.stderr.reconfigure(encoding="utf-8"' in source, (
        f"scripts/{name} печатает, но не переключает stderr в UTF-8 — "
        "сообщение об ошибке само станет ошибкой"
    )


@pytest.mark.parametrize("arguments", [
    ["--charge-initial-pump", "--brief"],
    ["--charge-initial-pump", "--explain"],
])
def test_defence_comparison_command_survives_a_narrow_locale(arguments: list[str]):
    """Обе команды блока 2 плана защиты запускаются на экране — и не под UTF-8.

    Проверяется именно то, что делает жюри: запуск как есть, без переменных
    окружения. `PYTHONIOENCODING=ascii` — самая узкая локаль; проходит она —
    пройдёт и cp1251, и cp437.
    """
    if not (ROOT / "runs" / "base" / "rows.csv").is_file():
        pytest.skip("нет выгрузки runs/base — сверять нечего")

    environment = dict(os.environ, PYTHONIOENCODING="ascii")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "compare_reference.py"), *arguments],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600,
    )

    assert result.returncode == 0, (
        "команда защиты упала на выдаче уже посчитанного:\n" + result.stderr[-600:]
    )
    assert "UnicodeEncodeError" not in result.stderr
