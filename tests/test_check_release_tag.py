"""Тег заявки не должен жить отдельной от манифеста жизнью."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_release_tag import (  # noqa: E402
    Manifest,
    declared_manifest,
    manifest_problem,
    reproduction_problem,
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
