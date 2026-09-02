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

# Заявка переведена на базу дисконтирования 1991 ответом трекера 02.09 (Р-158).
# Синтетические сообщения тега ниже намеренно оставлены на прежней тройке: они
# проверяют механику сверки манифеста с сообщением, а не саму заявку. Совпадать с
# документом обязан только этот MANIFEST.
MANIFEST = Manifest(
    npv="5 579,5",
    base="5 300,5",
    uplift="279,0",
    uplift_percent="5,26",
    schedule="runs/matched-0995/wells_schedule.inc",
    md5="41d813345e95f9d484c792ee6312a270",
)


def test_matching_tag_message_passes():
    message = """Заявляемый ЧДД: 5 579,5 млн руб.
База: 5 300,5. Прирост: +279,0 (+5,26 %).
Файл: runs/matched-0995/wells_schedule.inc, md5 41d813345e95f9d484c792ee6312a270.
"""

    assert manifest_problem(MANIFEST, message) is None


def test_current_submission_has_the_declared_manifest():
    submission = (Path(__file__).resolve().parent.parent / "docs" / "submission.md")

    assert declared_manifest(submission.read_text(encoding="utf-8")) == MANIFEST


def test_changed_schedule_is_rejected_even_when_numbers_match():
    message = """Заявляемый ЧДД: 5 579,5 млн руб.
База: 5 300,5. Прирост: +279,0 (+5,26 %).
Файл: runs/matched-100/wells_schedule.inc, md5 41d813345e95f9d484c792ee6312a270.
"""

    problem = manifest_problem(MANIFEST, message)

    assert problem is not None
    assert "schedule" in problem


def test_historical_correct_number_does_not_hide_wrong_tag_header():
    message = """Заявляемый ЧДД: 13 152,9 млн руб.
База: 11 851,8. Прирост: +1 281,9 (+10,8 %).
Файл: runs/matched-0995/wells_schedule.inc, md5 41d813345e95f9d484c792ee6312a270.

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
