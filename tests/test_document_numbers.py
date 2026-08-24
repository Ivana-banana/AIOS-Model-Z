"""Критические числа заявки не расходятся между сдаточными документами."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_declared_npv_is_the_same_in_all_current_documents():
    for name in ("docs/submission.md", "docs/report.md", "docs/defence-final.md",
                 "docs/presentation.md", "docs/defence-qa.md",
                 "docs/submission-day.md"):
        assert "13 133,8" in _read(name), f"{name} потерял заявляемый ЧДД"


def test_submission_pairs_declared_npv_with_same_run_baseline_and_uplift():
    submission = _read("docs/submission.md")

    assert "13 133,8 млн руб." in submission
    assert "11 851,8 млн руб." in submission
    assert "+1 281,9 млн руб. (+10,8 %)" in submission


def test_defence_does_not_present_aggressive_uplift_as_the_declared_result():
    defence = _read("docs/defence-final.md")

    assert "Прирост заявляемого сценария | +1 281,9 млн руб. (+10,8 %)" in defence
    assert "Прирост самого доходного сценария | +1 789,5 млн руб. (+15,1 %)" in defence


def test_the_jury_answers_carry_the_same_declared_triple():
    """Ответы жюри — самый вероятный источник расхождения с заявкой.

    Документ пишется под устное выступление, живёт отдельно от `submission.md` и
    правится чаще: именно так 22.08 доклад открывался приростом сценария, который
    мы не заявляем (`docs/numbers-audit.md`, расхождение 1). Поэтому тройка
    «ЧДД — база — прирост» закреплена и здесь.
    """
    answers = _read("docs/defence-qa.md")

    assert "13 133,8" in answers
    assert "11 851,8" in answers
    assert "+1 281,9 млн руб. (+10,8 %)" in answers


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
    assert "13 133,8" in presentation, "презентация потеряла заявляемый ЧДД"
    assert "27 процентов" in presentation, "нет оговорки про сторонний водоисточник"
    assert "втрое" in presentation, (
        "со слайда пропала главная оговорка про суррогат: он переоценивает отклик "
        "втрое, а прирост цикла втрое меньше, чем у простого правила")


def test_the_declared_pair_names_the_run_it_came_from():
    """Файл и число заявляются из одного прогона — иначе это дисквалификация.

    Пара «файл от одного расчёта, число от другого» и есть механизм, которым
    расхождение с пересчётом организаторов возникает молча. Поэтому каталог
    прогона закреплён здесь же, рядом с числом.
    """
    submission = _read("docs/submission.md")

    assert "runs/matched-0995/wells_schedule.inc" in submission
    assert "13 133,8 млн руб." in submission


def test_the_previous_candidate_survives_only_as_history():
    """13 271,8 остаётся в документах, но нигде не как заявка.

    Число прежнего кандидата ×1.5/×1.6 не выкидывается: им объясняется, за что
    отдано 124,3 млн руб. (Р-93), и оно же печатается промежуточной командой
    воспроизведения. Но строка «Заявляем 13 271,8» обязана исчезнуть отовсюду.
    """
    for name in ("docs/submission.md", "docs/report.md", "docs/defence-final.md",
                 "docs/presentation.md", "docs/defence-qa.md"):
        text = _read(name)
        assert "Заявляем **13 271,8" not in text, f"{name} заявляет прежнего кандидата"
        assert "заявляем** | **13 271,8" not in text, f"{name} заявляет прежнего кандидата"


def test_the_preflight_and_readme_point_at_the_file_we_actually_submit():
    """Сменился сценарий — сменился и файл, и об этом забывают чаще, чем о числе.

    23.08 (Б30) заявка перешла с `runs/inj-1.6` на `runs/matched-0995`, числа во
    всех документах машина А поменяла, а путь к сдаваемому расписанию остался
    прежним в трёх местах: умолчание предполётной проверки, пример в `README.md`
    и шаг демозапуска в презентации. Предполётная проверка при этом молча
    проверяла бы **не тот** файл и говорила «пакет готов».

    Сторож берёт имя файла из самой заявки, а не из литерала здесь: сменится
    сценарий ещё раз — тест переедет вместе с ним.
    """
    import re

    submission = _read("docs/submission.md")
    declared = re.search(r"`(runs/[\w.-]+/wells_schedule\.inc)`", submission)
    assert declared, "в docs/submission.md не найден сдаваемый файл"
    path = declared.group(1)

    assert path in _read("README.md"), "README показывает не тот файл"
    assert path in _read("scripts/check_package.py"), "умолчание предполётной проверки отстало"
    # мест пять, а не три: демонстрация тоже показывает файл, и на защите
    # показывают именно её (Б36)
    # в показе прогон назван каталогом, а не файлом — сравниваем по имени прогона
    name = path.rsplit("/", 2)[-2]
    assert name in _read("docs/presentation.md"), "шаг демозапуска ведёт не туда"
    assert name in _read("docs/demo.md"), "сценарий показа ведёт не туда"
    assert path in _read("docs/submission-day.md"), "паспорт отправки ведёт не туда"

    # И главное: инструкция обязана **собирать** тот прогон, который заявлен.
    # 24.08 заявка сменилась на `runs/matched-0995`, а раздел 3 продолжал строить
    # `runs/matched-100` со старым `--slack 1.0` — то есть проверяющий, идущий по
    # инструкции буквально, получил бы файл, которого мы не заявляем. Это ровно
    # тот механизм расхождения с пересчётом организаторов, о котором предупреждает
    # первый абзац самого документа.
    run = path.rsplit("/", 1)[0]
    commands = submission[submission.index("## 3."):]
    commands = commands[:commands.index("## 4.")]
    assert f"{run}/RUN.DATA" in commands, (
        f"раздел 3 не считает заявляемый прогон {run}: инструкция воспроизводит "
        "не тот файл, который мы сдаём"
    )


def test_submission_day_sheet_keeps_the_release_gate_and_caveat():
    """Паспорт не должен стать уверенной инструкцией без стоп-условий."""
    sheet = _read("docs/submission-day.md")

    for command in (
        "git status --short --branch",
        "python -m pytest -q",
        "python -m ruff check .",
        "python scripts/check_package.py",
        "python scripts/check_submission.py runs/matched-0995/wells_schedule.inc",
    ):
        assert command in sheet
    assert "submit-2026-08-24.1" in sheet
    assert "11 851,8" in sheet
    assert "+1 281,9 млн руб. (+10,8 %)" in sheet
    assert "подменяет секцию SCHEDULE целиком; история включена" in sheet
    assert "В-03 не получен" in sheet


def test_declared_schedule_checksum_is_the_same_in_current_documents():
    """Хэш старого кандидата не должен пережить смену сдаваемого файла."""
    import re

    submission = _read("docs/submission.md")
    match = re.search(r"`md5 ([0-9a-f]{32})`", submission)
    assert match, "в docs/submission.md нет полного md5 сдаваемого файла"
    checksum = match.group(1)

    for name in ("PLAN.md", "docs/numbers-audit.md"):
        assert checksum[:8] in _read(name), f"{name} называет md5 другого файла"
