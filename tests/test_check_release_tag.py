"""Тег заявки не должен жить отдельной от манифеста жизнью."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_release_tag import (  # noqa: E402
    Manifest,
    declared_manifest,
    manifest_problem,
)

MANIFEST = Manifest(
    npv="13 133,8",
    base="11 851,8",
    uplift="1 281,9",
    uplift_percent="10,8",
    schedule="runs/matched-0995/wells_schedule.inc",
    md5="41d813345e95f9d484c792ee6312a270",
)


def test_matching_tag_message_passes():
    message = """Заявляемый ЧДД: 13 133,8 млн руб.
База: 11 851,8. Прирост: +1 281,9 (+10,8 %).
Файл: runs/matched-0995/wells_schedule.inc, md5 41d813345e95f9d484c792ee6312a270.
"""

    assert manifest_problem(MANIFEST, message) is None


def test_current_submission_has_the_declared_manifest():
    submission = (Path(__file__).resolve().parent.parent / "docs" / "submission.md")

    assert declared_manifest(submission.read_text(encoding="utf-8")) == MANIFEST


def test_changed_schedule_is_rejected_even_when_numbers_match():
    message = """Заявляемый ЧДД: 13 133,8 млн руб.
База: 11 851,8. Прирост: +1 281,9 (+10,8 %).
Файл: runs/matched-100/wells_schedule.inc, md5 41d813345e95f9d484c792ee6312a270.
"""

    problem = manifest_problem(MANIFEST, message)

    assert problem is not None
    assert MANIFEST.schedule in problem
