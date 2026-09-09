"""Шлюз допуска проверяет файлы, а не обещания о них."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_package import _schedule_problem, audit  # noqa: E402


def _package(root: Path) -> tuple[Path, Path]:
    files = {
        "aios/core.py": "",
        "aios/neural.py": "",
        "aios/explain.py": "",
        "aios/web/app.py": "",
        "scripts/run_all.py": "",
        "docs/architecture.md": "",
        "docs/report.md": "",
        "docs/surrogate.md": "",
        "docs/submission.md": "| **ЧДД** | **13 271,8 млн руб.** |\n",
        # инструкция запуска проверяется по смыслу, а не по буквальным заголовкам
        # (02.09): установка, запуск и единая команда прогона
        "README.md": "## Установка\n## Сквозной прогон\npython scripts/run_all.py\n",
        "config.toml": "[doe]\nseed = 20260817\n",
        "docker/Dockerfile": (
            "COPY LICENSE /opt/aios/\n"
            "COPY LICENSES /opt/aios/LICENSES\n"
            "COPY THIRD_PARTY_NOTICES.md MODIFICATIONS.md /opt/aios/\n"
        ),
        "requirements.lock": "",
        "GIT.md": "",
        # с 24.08 пакет обязан нести собственную лицензию (Р-107)
        "LICENSE": "MIT License\n",
        ".git/HEAD": "ref: refs/heads/main\n",
        "THIRD_PARTY_NOTICES.md": "",
        "MODIFICATIONS.md": "",
        "PROVENANCE_AND_RIGHTS.md": "",
        "LICENSES/README.md": "",
    }
    for number in range(5):
        files[f"aios/web/templates/{number}.html"] = ""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    schedule = root / "result" / "wells_schedule.inc"
    model = root / "result" / "surrogate.joblib"
    schedule.parent.mkdir()
    schedule.write_text("DATES\n 1 JAN 2007 /\n/\n", encoding="utf-8")
    model.write_bytes(b"model")
    _trace(schedule, 13271.8)
    return schedule, model


def _trace(schedule: Path, claimed: float, **overrides: object) -> Path:
    """Машинный след расчёта рядом с расписанием — его пишет `verify_declared.py`.

    С 03.09 пункт 6 без следа отказывает: «число вписано в документ» и «число
    посчитано из этой выгрузки» — разные утверждения (Р-172).
    """
    import json

    from aios import economics

    record = {
        "run": schedule.parent.name,
        "chddM": claimed,
        "claimedM": claimed,
        "agreed": True,
        "vendorVersion": economics.VENDOR_VERSION,
        "scheduleMd5": hashlib.md5(schedule.read_bytes()).hexdigest(),
    }
    record.update(overrides)
    path = schedule.parent / "npv.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_complete_package_passes_all_nine_requirements(tmp_path):
    schedule, model = _package(tmp_path)

    checks = audit(
        tmp_path,
        schedule,
        model,
        model_validator=lambda _: None,
        schedule_validator=lambda _: None,
    )

    assert len(checks) == 10  # девять пунктов ТЗ + лицензионный контроль
    assert all(item.ready for item in checks)


def test_results_and_trained_model_cannot_be_replaced_by_documentation(tmp_path):
    schedule, model = _package(tmp_path)
    schedule.unlink()
    model.unlink()

    checks = {item.number: item for item in audit(tmp_path, schedule, model)}

    assert not checks["6"].ready
    assert "нет" in checks["6"].detail
    assert not checks["9"].ready
    assert "нет обученного артефакта" in checks["9"].detail


def test_seed_is_required_for_reproduction(tmp_path):
    schedule, model = _package(tmp_path)
    (tmp_path / "config.toml").write_text("[doe]\nscenarios = 12\n", encoding="utf-8")

    checks = {
        item.number: item
        for item in audit(
            tmp_path,
            schedule,
            model,
            model_validator=lambda _: None,
            schedule_validator=lambda _: None,
        )
    }

    assert not checks["4"].ready
    assert "seed" in checks["4"].detail


def test_existing_model_must_pass_the_quality_gate(tmp_path):
    schedule, model = _package(tmp_path)

    checks = {
        item.number: item
        for item in audit(
            tmp_path,
            schedule,
            model,
            model_validator=lambda _: "MLP не прошёл порог",
            schedule_validator=lambda _: None,
        )
    }

    assert not checks["9"].ready
    assert checks["9"].detail == "MLP не прошёл порог"


def test_another_quality_model_cannot_replace_the_declared_artifact(tmp_path):
    schedule, _model = _package(tmp_path)
    declared = tmp_path / "artifacts" / "surrogate.joblib"
    declared.parent.mkdir()
    declared.write_bytes(b"declared model")
    other = tmp_path / "result" / "other.joblib"
    other.write_bytes(b"other quality model")
    submission = tmp_path / "docs" / "submission.md"
    submission.write_text(
        submission.read_text(encoding="utf-8")
        + "| модель | `artifacts/surrogate.joblib` |\n",
        encoding="utf-8",
    )

    checks = {item.number: item for item in audit(
        tmp_path,
        schedule,
        other,
        model_validator=lambda _: None,
        schedule_validator=lambda _: None,
    )}

    assert not checks["9"].ready
    assert "в пакет входит artifacts/surrogate.joblib" in checks["9"].detail


def test_changed_bytes_under_the_declared_model_name_are_rejected(tmp_path):
    schedule, _model = _package(tmp_path)
    model = tmp_path / "artifacts" / "surrogate.joblib"
    model.parent.mkdir()
    model.write_bytes(b"accepted quality model")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    submission = tmp_path / "docs" / "submission.md"
    submission.write_text(
        submission.read_text(encoding="utf-8")
        + "| модель | `artifacts/surrogate.joblib`, "
        f"`sha256 {expected}` |\n",
        encoding="utf-8",
    )

    ready = {item.number: item for item in audit(
        tmp_path,
        schedule,
        model,
        model_validator=lambda _: None,
        schedule_validator=lambda _: None,
    )}
    assert ready["9"].ready
    assert f"sha256 {expected} совпал с заявкой" in ready["9"].detail

    model.write_bytes(b"another quality model")
    changed = {item.number: item for item in audit(
        tmp_path,
        schedule,
        model,
        model_validator=lambda _: None,
        schedule_validator=lambda _: None,
    )}
    assert not changed["9"].ready
    assert "другой артефакт" in changed["9"].detail


def test_existing_schedule_must_pass_the_schedule_gate(tmp_path):
    schedule, model = _package(tmp_path)

    checks = {
        item.number: item
        for item in audit(
            tmp_path,
            schedule,
            model,
            model_validator=lambda _: None,
            schedule_validator=lambda _: "в файле нет режимов",
        )
    }

    assert not checks["6"].ready
    assert checks["6"].detail == "в файле нет режимов"


def test_schedule_gate_parses_and_validates_controls(tmp_path):
    valid = tmp_path / "valid.inc"
    valid.write_text(
        "DATES\n 1 JAN 2007 /\n/\n"
        "WCONPROD\n 'W1' OPEN LRAT 1* 1* 1* 20 1* 100 1* 1* /\n/\n",
        encoding="utf-8",
    )
    empty = tmp_path / "empty.inc"
    empty.write_text("-- нет режимов\n", encoding="utf-8")

    assert _schedule_problem(valid) is None
    assert _schedule_problem(empty) == "расписание не содержит ни одного шага DATES"


def test_the_audit_refuses_a_schedule_from_another_run(tmp_path):
    """Число и файл обязаны быть из одного прогона — проверка это и проверяет.

    24.08 прогон пакета на машине Б показал: `--schedule` принимал любой файл, и
    проверка отвечала «ГОТОВО … + 13 133,8 млн руб.» на **историческом** базовом
    расписании. То есть благословляла пару «файл одного прогона, число другого»
    — ровно тот механизм, которым расхождение с пересчётом организаторов
    возникает молча.
    """
    root = tmp_path
    _package(root)
    (root / "docs" / "submission.md").write_text(
        "| **ЧДД** | **13 133,8 млн руб.** |\n"
        "| сдаваемый файл | `runs/matched-0995/wells_schedule.inc` |\n",
        encoding="utf-8",
    )
    other = root / "runs" / "base" / "wells_schedule.inc"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("-- расписание другого прогона\n", encoding="utf-8")

    checks = {check.number: check for check in audit(
        root, other, root / "artifacts" / "surrogate.joblib",
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}

    assert not checks["6"].ready
    assert "из одного прогона" in checks["6"].detail
    assert "runs/matched-0995" in checks["6"].detail


def test_the_audit_needs_a_machine_trace_of_the_calculation(tmp_path):
    """«Число в документе» и «число посчитано из этой выгрузки» — разные вещи.

    До 03.09 пункт 6 проверял `md5` файла и наличие числа в `docs/submission.md`,
    но не связь между ними: число вписано руками, машинного следа расчёта в
    каталоге прогона не было вовсе (ревизия 03.09, 2.3 #9).
    """
    schedule, model = _package(tmp_path)
    (schedule.parent / "npv.json").unlink()

    checks = {item.number: item for item in audit(
        tmp_path, schedule, model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}

    assert not checks["6"].ready
    assert "машинного следа" in checks["6"].detail
    assert "verify_declared.py" in checks["6"].detail


def test_a_trace_of_another_file_is_refused(tmp_path):
    """След, посчитанный по другому расписанию, ничего не подтверждает."""
    schedule, model = _package(tmp_path)
    _trace(schedule, 13271.8, scheduleMd5="0" * 32)

    checks = {item.number: item for item in audit(
        tmp_path, schedule, model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}

    assert not checks["6"].ready
    assert "другому файлу расписания" in checks["6"].detail


def test_a_trace_that_says_mismatch_is_refused(tmp_path):
    """След с расхождением — отказ допуска, а не заметка."""
    schedule, model = _package(tmp_path)
    _trace(schedule, 13271.8, agreed=False, chddM=12000.0)

    checks = {item.number: item for item in audit(
        tmp_path, schedule, model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}

    assert not checks["6"].ready
    assert "расхождении" in checks["6"].detail


def test_a_trace_sworn_to_another_number_is_refused(tmp_path):
    """След, сверявшийся с другим числом, не подтверждает заявку."""
    schedule, model = _package(tmp_path)
    _trace(schedule, 12000.0)

    checks = {item.number: item for item in audit(
        tmp_path, schedule, model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}

    assert not checks["6"].ready
    assert "след сверялся" in checks["6"].detail


def test_the_audit_refuses_changed_bytes_under_the_declared_name(tmp_path):
    """Правильный путь не заменяет побайтовую идентичность заявленному файлу."""
    root = tmp_path
    _unused_schedule, model = _package(root)
    schedule = root / "runs" / "matched-0995" / "wells_schedule.inc"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("DATES\n 1 JAN 2007 /\n/\n", encoding="utf-8")
    expected = hashlib.md5(schedule.read_bytes(), usedforsecurity=False).hexdigest()
    (root / "docs" / "submission.md").write_text(
        "| **ЧДД** | **13 133,8 млн руб.** |\n"
        "| сдаваемый файл | `runs/matched-0995/wells_schedule.inc` "
        f"`md5 {expected}` |\n",
        encoding="utf-8",
    )
    _trace(schedule, 13133.8)

    ready = {check.number: check for check in audit(
        root, schedule, model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}
    assert ready["6"].ready
    assert "совпал с заявкой" in ready["6"].detail

    schedule.write_text("DATES\n 1 FEB 2007 /\n/\n", encoding="utf-8")
    changed = {check.number: check for check in audit(
        root, schedule, model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}
    assert not changed["6"].ready
    assert "проверяется другой файл" in changed["6"].detail
    assert expected in changed["6"].detail


def test_the_package_is_incomplete_without_our_own_license(tmp_path):
    """Пакет без `LICENSE` — это код без условий использования.

    Шлюз проверял чужие лицензии (реестр, уведомления, модификации, тексты) и не
    проверял нашу: до 24.08 её не существовало, а с Р-107 она есть. Пакет,
    уехавший без неё, выглядит полным — заметить это некому.
    """
    from check_package import audit

    root = tmp_path
    schedule, model = _package(root)

    checks = {check.number: check for check in audit(
        root, schedule, model,
        schedule_validator=lambda path: None, model_validator=lambda path: None,
    )}
    assert checks["L"].ready, checks["L"].detail

    (root / "LICENSE").unlink()
    checks = {check.number: check for check in audit(
        root, schedule, model,
        schedule_validator=lambda path: None, model_validator=lambda path: None,
    )}
    assert not checks["L"].ready
    assert "LICENSE" in checks["L"].detail


def test_the_package_is_incomplete_when_the_image_omits_our_license(tmp_path):
    schedule, model = _package(tmp_path)
    dockerfile = tmp_path / "docker" / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text(encoding="utf-8").replace("COPY LICENSE /opt/aios/\n", ""),
        encoding="utf-8",
    )

    checks = {check.number: check for check in audit(
        tmp_path,
        schedule,
        model,
        schedule_validator=lambda path: None,
        model_validator=lambda path: None,
    )}

    assert not checks["L"].ready
    assert "Docker-образ не получает" in checks["L"].detail
    assert "COPY LICENSE" in checks["L"].detail


# --- С9: след сверялся не весь -----------------------------------------------
# Проверялись `agreed`, md5 расписания, версия вендора и число заявки — но не
# стартовый год, не нормативы и не выгрузка. Пересимулированный `rows.csv` или
# подменённые нормативы пункт 6 проходили.


def _trace_record(**overrides):
    """Машинный след с заведомо верными полями, поверх которых кладут порчу."""
    import sys as system

    from aios import config as config_module
    from aios import economics

    system.path.insert(0, str(ROOT))
    configuration = config_module.load()
    record = {
        "run": "matched-0995",
        "chddM": 5579.5,
        "claimedM": 5579.5,
        "agreed": True,
        "startYear": configuration.economics.start_date.year,
        "vendorVersion": economics.VENDOR_VERSION,
        "normsSha256": None,
        "rowsMd5": None,
        "scheduleMd5": None,
    }
    record.update(overrides)
    return record


def test_a_trace_from_another_starting_year_is_refused(tmp_path):
    """Окна 1991 и 2007 дают числа, различающиеся в 4,6 раза.

    След, посчитанный в отозванном окне, совпадёт с заявкой того окна и
    промолчит — а проверяющий считает в своём. С 08.09 действующее окно — 2007
    (только период управления, Р-249), поэтому отвергается след от 1991.
    """
    from check_package import _trace_inputs_problem

    problem = _trace_inputs_problem(tmp_path / "wells_schedule.inc",
                                    _trace_record(startYear=1991))

    assert problem is not None
    assert "1991" in problem


def test_a_trace_computed_by_other_norms_is_refused(tmp_path):
    """Нормативы задают экономику целиком: подмена меняет число молча."""
    from check_package import _trace_inputs_problem

    problem = _trace_inputs_problem(tmp_path / "wells_schedule.inc",
                                    _trace_record(normsSha256="0" * 64))

    assert problem is not None
    assert "норматив" in problem


def test_a_trace_computed_by_another_export_is_refused(tmp_path):
    """Пересимулированный `rows.csv` даёт другое число из того же расписания."""
    from check_package import _trace_inputs_problem

    rows = tmp_path / "rows.csv"
    rows.write_text("DATA,well" + chr(10), encoding="utf-8")

    problem = _trace_inputs_problem(tmp_path / "wells_schedule.inc",
                                    _trace_record(rowsMd5="f" * 32))

    assert problem is not None
    assert "выгрузк" in problem


def test_a_missing_export_is_not_treated_as_agreement(tmp_path):
    """Выгрузки рядом нет — сверять нечем, и это не расхождение.

    `rows.csv` намеренно не отслеживается git (5 МБ на прогон), и у проверяющего
    его нет. Отказ здесь объявил бы негодным пакет, который в порядке.
    """
    from check_package import _trace_inputs_problem

    assert _trace_inputs_problem(tmp_path / "wells_schedule.inc",
                                 _trace_record(rowsMd5="f" * 32)) is None


def test_the_shipped_trace_agrees_with_the_files_next_to_it():
    """Настоящий след заявки обязан сходиться с конфигурацией и нормативами."""
    import pytest
    from check_package import _trace_inputs_problem

    schedule = ROOT / "runs" / "matched-0995" / "wells_schedule.inc"
    trace = schedule.parent / "npv.json"
    if not trace.is_file():
        pytest.skip("следа заявки нет на этой машине")

    record = json.loads(trace.read_text(encoding="utf-8"))

    assert _trace_inputs_problem(schedule, record) is None


def test_the_gate_measures_records_even_without_a_scaling_report():
    """Нет карты множителей — мерить всё равно есть что, и это делается.

    Раньше отсутствие `scaling.json` давало строку «не измерена». Честно, но
    слабее возможного: обнулённая уставка видна из пары «эталон и сдаваемый
    файл» без всякого намерения. Сдаваемый кейс собран до появления отчёта,
    поэтому это не гипотетическая ветка, а рабочая.

    Числа привязаны к самому файлу: 21 594 уставочных записи — это ровно та
    величина, которой отчитывался прогон `inj-1.6`, и расхождение означало бы,
    что сдаётся не то расписание.
    """
    import pytest
    from check_package import _audit_without_report

    root = Path(__file__).resolve().parent.parent
    schedule = root / "runs" / "matched-0995" / "wells_schedule.inc"
    if not schedule.is_file():
        pytest.skip("нет сдаваемого расписания")
    try:
        from aios import config as config_module
        from aios.deckfiles import layout

        config = config_module.load(root / "config.toml")
        layout(config.case.model_dir).one("SCHEDULE")
    except Exception:  # noqa: BLE001 — дек в репозиторий не кладём (GIT.md)
        pytest.skip("нет дека Модели Z — сверять не с чем")

    problem, note = _audit_without_report(schedule)

    assert problem is None, note
    assert "21594 уставочных записей" in note, note
    assert "обнулённых нет" in note, note
    assert "producer 1.500–1.500" in note, note


def test_the_gate_refuses_a_schedule_whose_records_lost_their_rate(monkeypatch):
    """Обнуление — отказ, а не примечание: иначе множитель тихо закрыл бы фонд."""
    import check_package
    import pytest

    from aios import optimise

    root = Path(__file__).resolve().parent.parent
    schedule = root / "runs" / "matched-0995" / "wells_schedule.inc"
    if not schedule.is_file():
        pytest.skip("нет сдаваемого расписания")
    if not (root / "vendor" / "Model_Z_OPM" / "Model_Z.data").is_file():
        # Сверка записей идёт через опорное расписание дека: без него
        # `_audit_without_report` честно отвечает «сверка не выполнена», и
        # подменённый аудит до дела не доходит (Р-236).
        pytest.skip("дек Модели Z не распакован")

    dirty = optimise.RecordAudit(records=3, zeroed=(("2014-01-01", "p"),),
                                 off_mode={}, ratios={})
    monkeypatch.setattr(optimise, "audit_records", lambda before, after: dirty)

    problem, _ = check_package._audit_without_report(schedule)

    assert problem is not None
    assert "1 открытых записей без расхода" in problem
    assert "2014-01-01 p" in problem


def test_a_check_that_did_not_run_is_not_reported_as_ready(capsys):
    """«Не выполнена» — это не «претензий нет».

    Сверка сдаваемого файла с деком глушится `except Exception` (нет дека,
    опечатка в `model_dir`, упавший импорт), и пункт всё равно печатался
    «ГОТОВО», а команда отвечала «пакет готов». Паспорт дня отправки принимает
    пакет ровно по этому признаку — «все строки ГОТОВО, последняя строка „пакет
    готов“», — то есть признак достигался и тогда, когда содержательная половина
    пункта не состоялась (Р-65).

    Код возврата при этом остаётся нулевым: ничего не сломано, файлы на месте, а
    проверяющий набирает команду на чистом клоне до распаковки дека — это его
    обычный первый запуск, и красным он быть не должен.
    """
    import check_package

    ready = check_package.Check(number="6", title="результат", ready=True,
                                detail="файл на месте; сверка записей не выполнена: "
                                       "ValueError: нет каталога дека")
    other = check_package.Check(number="7", title="git", ready=True, detail="есть")

    code = check_package.report([ready, other])
    printed = capsys.readouterr().out

    assert code == 0
    assert "НЕ ПРОВ" in printed
    assert "пакет готов" not in printed
    assert "не выполнено" in printed
