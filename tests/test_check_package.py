"""Шлюз допуска проверяет файлы, а не обещания о них."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

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
        "README.md": "## Установка\n## Сквозной прогон\n",
        "config.toml": "[doe]\nseed = 20260817\n",
        "docker/Dockerfile": "",
        "requirements.lock": "",
        "GIT.md": "",
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
    return schedule, model


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
