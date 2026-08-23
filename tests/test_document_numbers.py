"""Критические числа заявки не расходятся между сдаточными документами."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_declared_npv_is_the_same_in_all_current_documents():
    for name in ("docs/submission.md", "docs/report.md", "docs/defence-final.md"):
        assert "13 271,8" in _read(name), f"{name} потерял заявляемый ЧДД"


def test_submission_pairs_declared_npv_with_same_run_baseline_and_uplift():
    submission = _read("docs/submission.md")

    assert "13 271,8 млн руб." in submission
    assert "11 856,2 млн руб." in submission
    assert "+1 415,6 млн руб. (+11,9 %)" in submission


def test_defence_does_not_present_aggressive_uplift_as_the_declared_result():
    defence = _read("docs/defence-final.md")

    assert "Прирост заявляемого сценария | +1 415,6 млн руб. (+11,9 %)" in defence
    assert "Прирост самого доходного сценария | +1 789,5 млн руб. (+15,1 %)" in defence


def test_the_jury_answers_carry_the_same_declared_triple():
    """Ответы жюри — самый вероятный источник расхождения с заявкой.

    Документ пишется под устное выступление, живёт отдельно от `submission.md` и
    правится чаще: именно так 22.08 доклад открывался приростом сценария, который
    мы не заявляем (`docs/numbers-audit.md`, расхождение 1). Поэтому тройка
    «ЧДД — база — прирост» закреплена и здесь.
    """
    answers = _read("docs/defence-qa.md")

    assert "13 271,8" in answers
    assert "11 856,2" in answers
    assert "+1 415,6 млн руб. (+11,9 %)" in answers


def test_the_jury_answers_keep_the_section_of_open_questions():
    """Раздел «вопросов без ответа» — обязательная часть, а не украшение.

    Он ценен ровно тем, что его неудобно писать: пропадёт он молча — и документ
    превратится в набор уверенных формулировок, которые жюри разваливает вторым
    вопросом.
    """
    answers = _read("docs/defence-qa.md")

    assert "## 6. Вопросы, на которые у нас ответа нет" in answers
    assert answers.count("| Вопрос | Почему нет ответа | Что бы его дало |") == 1


def test_the_presentation_keeps_the_slide_we_agreed_not_to_cut():
    """Слайд «что мы не заявляем» вырезается первым, когда доклад не помещается.

    Он же самый ценный: слабое место, названное докладчиком, стоит дешевле
    найденного жюри. Поэтому его присутствие и его три пункта закреплены тестом,
    а не памятью выступающего.
    """
    presentation = _read("docs/presentation.md")

    assert "## Слайд 7. Что мы не заявляем" in presentation
    assert "13 271,8" in presentation, "презентация потеряла заявляемый ЧДД"
    assert "54–78" in presentation or "54…78" in presentation, "нет оговорки про воду"
    assert "−0,058" in presentation or "минус 0,058" in presentation, "нет порога суррогата"
