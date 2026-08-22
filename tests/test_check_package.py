"""Шлюз допуска проверяет файлы, а не обещания о них."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_package import audit  # noqa: E402


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

    checks = audit(tmp_path, schedule, model)

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

    checks = {item.number: item for item in audit(tmp_path, schedule, model)}

    assert not checks["4"].ready
    assert "seed" in checks["4"].detail
