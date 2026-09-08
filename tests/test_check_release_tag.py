"""Тег заявки не должен жить отдельной от манифеста жизнью."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_release_tag import (  # noqa: E402
    ACTIVE_TAG,
    Manifest,
    declared_manifest,
    manifest_problem,
    reproduction_problem,
    tagged_file_problem,
)

# Письмо организаторов 08.09 сменило форму сдачи: и файл, и число — только за
# период управления AIOS (Р-249). Синтетические сообщения тега ниже намеренно
# оставлены на прежней тройке: они проверяют механику сверки манифеста с
# сообщением, а не саму заявку. Совпадать с документом обязан только этот
# MANIFEST.
MANIFEST = Manifest(
    npv="13 133,8",
    base="11 851,8",
    uplift="1 281,9",
    uplift_percent="10,82",
    schedule="runs/matched-0995/submission/wells_schedule.inc",
    md5="a7132dad996a86a30766d06f5ba63c7b",
)


def test_matching_tag_message_passes():
    message = """Заявляемый ЧДД: 13 133,8 млн руб.
База: 11 851,8. Прирост: +1 281,9 (+10,82 %).
Файл: runs/matched-0995/submission/wells_schedule.inc, md5 a7132dad996a86a30766d06f5ba63c7b.
"""

    assert manifest_problem(MANIFEST, message) is None


def test_current_submission_has_the_declared_manifest():
    submission = (Path(__file__).resolve().parent.parent / "docs" / "submission.md")

    assert declared_manifest(submission.read_text(encoding="utf-8")) == MANIFEST


def test_changed_schedule_is_rejected_even_when_numbers_match():
    message = """Заявляемый ЧДД: 13 133,8 млн руб.
База: 11 851,8. Прирост: +1 281,9 (+10,82 %).
Файл: runs/matched-100/submission/wells_schedule.inc, md5 a7132dad996a86a30766d06f5ba63c7b.
"""

    problem = manifest_problem(MANIFEST, message)

    assert problem is not None
    assert "schedule" in problem


def test_historical_correct_number_does_not_hide_wrong_tag_header():
    message = """Заявляемый ЧДД: 13 152,9 млн руб.
База: 11 851,8. Прирост: +1 281,9 (+10,82 %).
Файл: runs/matched-0995/submission/wells_schedule.inc, md5 a7132dad996a86a30766d06f5ba63c7b.

Историческая справка: прежний документ называл 13 133,8 млн руб.
"""

    problem = manifest_problem(MANIFEST, message)

    assert problem is not None
    assert "npv" in problem


def test_tag_snapshot_must_rebuild_the_declared_run():
    wrong = """## 3. Как повторить число
python scripts/build_matched_case.py runs/matched-100
bash scripts/run_flow.sh runs/matched-100/RUN.DATA 4
## 4. Проверки
"""

    problem = reproduction_problem(MANIFEST, wrong)

    assert problem is not None
    assert "runs/matched-0995" in problem


def test_tag_snapshot_with_the_declared_run_passes():
    current = (Path(__file__).resolve().parent.parent / "docs" / "submission.md")

    assert reproduction_problem(
        MANIFEST, current.read_text(encoding="utf-8")
    ) is None


def _repository_with_tag(tmp_path, tracked: dict[str, bytes]) -> Path:
    """Крошечный репозиторий с одним аннотированным тегом `submit-test`."""
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    for name, blob in tracked.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    def run(*args: str) -> None:
        subprocess.run(args, cwd=root, check=True, capture_output=True)

    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "снимок")
    run("git", "tag", "-a", "submit-test", "-m", "заявка")
    return root


def test_a_tag_without_the_declared_file_is_refused(tmp_path):
    """Тег обязан содержать файл, который сам же объявляет.

    08.09 форма сдачи сменилась на хвост за период управления, заявляемым стал
    файл уровнем глубже, а исключение `.gitignore` осталось прежним — и тег
    `submit-2026-09-08` назвал файл, которого в теге нет. Сторож при этом
    отвечал «манифест и инструкция совпадают»: он сверял тексты, а до самого
    файла не доходил (Н-04).
    """
    root = _repository_with_tag(tmp_path, {"README.md": "без заявки\n".encode()})

    problem = tagged_file_problem(root, "submit-test", MANIFEST)

    assert problem is not None
    assert MANIFEST.schedule in problem


def test_a_tag_with_a_different_file_is_refused(tmp_path):
    """Файл в теге есть, но это другой файл — расхождение по `md5`."""
    root = _repository_with_tag(
        tmp_path, {MANIFEST.schedule: b"DATES\n 01 JAN 2007 /\n/\n"}
    )

    problem = tagged_file_problem(root, "submit-test", MANIFEST)

    assert problem is not None
    assert "md5" in problem


def test_a_tag_holding_the_declared_file_passes(tmp_path):
    """Тот же файл с тем же `md5` — возражений нет."""
    import hashlib

    blob = b"DATES\n 01 JAN 2007 /\n/\n"
    manifest = Manifest(
        npv=MANIFEST.npv,
        base=MANIFEST.base,
        uplift=MANIFEST.uplift,
        uplift_percent=MANIFEST.uplift_percent,
        schedule=MANIFEST.schedule,
        md5=hashlib.md5(blob).hexdigest(),
    )
    root = _repository_with_tag(tmp_path, {manifest.schedule: blob})

    assert tagged_file_problem(root, "submit-test", manifest) is None


def test_the_active_tag_is_read_from_the_anchor_not_from_the_first_mention():
    """Действующий тег — тот, что назван якорем, а не тот, что упомянут раньше.

    До 08.09 и скрипт, и сторож слайдов брали **первый** `submit-…` в паспорте.
    Правило держалось на том, что прежние теги в паспорте не упоминаются выше
    действующего, — а они упоминаются по делу: протокол холостого прогона 04.09
    называет тег, действовавший в тот день, и он стоит раньше. Заявка при этом
    сверялась бы с отозванным состоянием и проходила бы молча.
    """
    passport = (Path(__file__).resolve().parent.parent
                / "docs" / "submission-day.md").read_text(encoding="utf-8")

    anchored = ACTIVE_TAG.findall(passport)

    assert len(anchored) == 1, f"якорь действующего тега назван не раз: {anchored}"
    mentioned = re.findall(r"submit-\d{4}-\d{2}-\d{2}(?:\.\d+)?", passport)
    assert mentioned[0] != anchored[0], (
        "в паспорте больше нет упоминания прежнего тега раньше действующего — "
        "тест перестал проверять то, ради чего написан"
    )


def test_a_passport_without_the_anchor_is_refused():
    """Паспорт без якоря — отказ, а не догадка по первому попавшемуся тегу."""
    assert ACTIVE_TAG.search("тег `submit-2026-09-02` упомянут без якоря") is None
