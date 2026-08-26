"""Критические числа заявки не расходятся между сдаточными документами."""

import csv
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
    assert "| **по достигнутому — заявляем** | **13 133,8** | **+10,8 %** | **20,0 %** |" in submission


def test_audit_uses_the_actual_uplift_percentage():
    audit = _read("docs/numbers-audit.md")
    exact_percent = (13_133.757 - 11_851.811) / 11_851.811 * 100

    assert round(exact_percent, 2) == 10.82
    assert "10,82 % от базы" in audit
    assert "10,98 % от базы" not in audit


def test_declared_conformal_margin_is_recomputed_from_search_pairs():
    from aios.trust import OptimismMargin

    with (ROOT / "docs/data/margin-pairs-search.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 9
    assert all(len(row) == 5 and row[3] == "search" for row in rows)
    assert len({row[2] for row in rows}) == len(rows)

    margin = OptimismMargin.fit(
        [(float(row[0]), float(row[1])) for row in rows], alpha=0.1
    )
    assert margin.calibrated
    assert margin.samples == 9
    assert margin.index == 9
    assert round(margin.value, 1) == 927.8

    threshold = 21.622
    residuals_by_distance = [
        (float(row[4]), float(row[0]) - float(row[1])) for row in rows
    ]
    assert sum(distance <= threshold for distance, _ in residuals_by_distance) == 2
    assert all(
        residual <= 0 if distance <= threshold else residual > 0
        for distance, residual in residuals_by_distance
    )

    for name in (
        "docs/architecture.md",
        "docs/defence-qa.md",
        "docs/explained.md",
        "docs/presentation.md",
        "docs/report.md",
        "docs/surrogate.md",
    ):
        assert "927,8" in _read(name), f"{name} потерял калиброванный запас"


def test_mixed_margin_corpus_size_matches_its_current_description():
    with (ROOT / "docs/data/margin-pairs.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 24
    assert sum(row[3] == "manual" for row in rows) == 21
    assert sum(row[3] == "search" for row in rows) == 3

    for name in ("docs/report.md", "docs/surrogate.md"):
        text = _read(name)
        assert "24" in text
        assert "21 ручн" in text
        assert "3 поисков" in text


def test_beginner_explanation_keeps_pressure_share_and_cost_distinct():
    explained = _read("docs/explained.md")
    comparable_cost_percent = (13_264.6 - 13_133.8) / 13_264.6 * 100

    assert round(comparable_cost_percent, 1) == 1.0
    assert "44,0 % управляемых месяцев закачки" in explained
    assert "15 нагнетательных из 41" in explained
    assert "18 нагнетательных из 41" not in explained
    assert "**1,0 % ЧДД**" in explained
    assert "Цена — 0,7 % ЧДД" not in explained


def test_report_keeps_live_explanation_metrics_and_resolved_water_status():
    report = _read("docs/report.md")

    assert "ASR **0,750** против 1,000 у шаблона" in report
    assert "только на одном принятом пакете" in report
    assert "Качество объяснений измерено на шаблонах, а не на модели" not in report
    assert "Формулировка баланса воды разрешена через компенсацию" in report
    assert "Мощность ППД и формулировка баланса воды** ждут" not in report
    assert "| ASR (невзвешенный) | 1.000 | 28 из 28 |" in report
    assert "| ASR (взвешенный) | 1.000 | 52 из 52 |" in report
    assert "| ASR (невзвешенный) | 1.000 | 20 из 20 |" not in report


def test_report_does_not_reopen_conformal_calibration():
    report = _read("docs/report.md")

    assert "**калиброван 24.08** на девяти парах" in report
    assert "**+927,8 млн руб.**" in report
    assert "двадцать четыре пары распадаются на 21 ручную и 3 кандидата" not in report
    assert "девять пар семейства «поиск», у нас две" not in report
    assert "по-прежнему пишет\n`calibrated: false`" not in report
    assert "две точки с расстоянием 16,11 и 20,29 занижают" in report
    assert "все семь\n   точек дальше порога `τ = 21,622` завышают" in report
    assert "Знак определяется не\n   расстоянием" not in report


def test_current_documents_record_the_completed_surrogate_cycle():
    current = {
        name: _read(name)
        for name in (
            "PLAN.md",
            "docs/defence-qa.md",
            "docs/report.md",
            "docs/surrogate.md",
            "docs/presentation.md",
            "docs/tz-compliance.md",
        )
    }

    for name, text in current.items():
        assert "+436,5" in text, f"{name} потерял первый подтверждённый цикл"
        assert "+460,8" in text, f"{name} потерял второй подтверждённый цикл"

    assert "через полный цикл артефакт не проводился" not in current["PLAN.md"]
    assert "**через полный цикл этот артефакт не проводился**" not in current["docs/report.md"]
    assert "цикл не проводился и полным расчётом не подтверждён" not in current["docs/surrogate.md"]
    assert "но через полный цикл не проводился" not in current["docs/presentation.md"]
    assert "Новый артефакт через цикл **не проводился**" not in current["docs/defence-qa.md"]
    assert "новый артефакт через цикл не проводился" not in current["docs/tz-compliance.md"]


def test_letter_maps_field_injection_and_external_water_to_separate_limits():
    letter = _read("docs/letter-to-organisers.md")

    assert "`max_field_injection_m3_per_day`" in letter
    assert "`max_external_water_m3_per_day`" in letter
    assert "Предел общей закачки" in letter
    assert "доступный сторонний водоисточник" in letter


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
    assert "прогон 24.08, каталог `runs/matched-0995`" in answers
    assert "13 264,6 − 13 133,8 = **130,8 млн руб.**" in answers
    assert "**20,0 %**" in answers


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
    assert "`runs/matched-0995`, прогон 24.08" in presentation
    assert "это 10,8 процента" in presentation
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
        "python scripts/check_release_tag.py",
        "python scripts/check_package.py",
        "python scripts/check_submission.py runs/matched-0995/wells_schedule.inc",
    ):
        assert command in sheet
    assert "submit-2026-08-24.1" in sheet
    assert "11 851,8" in sheet
    assert "+1 281,9 млн руб. (+10,8 %)" in sheet
    assert "подменяет секцию SCHEDULE целиком; история включена" in sheet
    assert "В-03 не получен" in sheet
    assert "всем 103 скважинам с режимом после 01.01.2007" in sheet
    assert "не получен (В-04)" in sheet
    assert "docs/submission-day.md" in _read("README.md")
    assert "все шесть проверок" in _read("README.md")
    assert "в **отдельной чистой копии** `submit-2026-08-24.1`" in sheet
    assert "вернуть в текущий `main`" in sheet


def test_submission_does_not_reopen_resolved_blockers():
    submission = _read("docs/submission.md")

    assert "В-03 и В-04" in submission
    assert "пп. 3, 4 и 6 уже закрыты" in submission
    assert "Открытые вопросы организаторам** — четыре" not in submission


def test_submission_includes_the_required_surrogate_artifact():
    submission = _read("docs/submission.md")

    model_sha = "2941475afb8d13938a66ad46cbac14500f2601a089152b86c6ce2acb37dc5346"
    assert "`artifacts/surrogate.joblib`" in submission
    assert f"`sha256 {model_sha}`" in submission
    assert f"`sha256 {model_sha}`" in _read("docs/submission-day.md")
    assert "допущенный `artifacts/surrogate.joblib`" in submission
    assert "обученные модели `artifacts/` | воспроизводятся" not in submission


def test_explainability_records_the_completed_live_call():
    text = _read("docs/explainability.md")
    architecture = _read("docs/architecture.md")

    assert "Живой вызов\nсостоялся 23.08" in text
    assert "живой вызов\nпланируется один" not in text
    assert "Живой вызов выполнен 23.08" in architecture
    assert "Живой вызов API ещё не выполнялся" not in architecture


def test_declared_schedule_checksum_is_the_same_in_current_documents():
    """Хэш старого кандидата не должен пережить смену сдаваемого файла."""
    import re

    submission = _read("docs/submission.md")
    match = re.search(r"`md5 ([0-9a-f]{32})`", submission)
    assert match, "в docs/submission.md нет полного md5 сдаваемого файла"
    checksum = match.group(1)

    for name in ("PLAN.md", "docs/numbers-audit.md", "docs/submission-day.md"):
        assert checksum[:8] in _read(name), f"{name} называет md5 другого файла"


def test_current_submission_parameters_are_not_mixed_with_matched_100():
    defence = _read("docs/defence-final.md")
    compliance = _read("docs/tz-compliance.md")
    audit = _read("docs/numbers-audit.md")

    assert "(`runs/matched-0995`), прогон **24.08**" in defence
    assert "заявляем **+10,8 %**" in compliance
    assert "13 264,6 − 13 133,8 = 130,8 млн руб." in audit
    assert "111,7 млн — переход" in audit and "ещё 19,1 млн" in audit


def test_current_demo_handoff_points_at_the_submitted_run():
    handoff = _read("HANDOFF.md")
    section = handoff[handoff.index("## Артефакты для демозапуска"):]
    section = section[:section.index("## Машина В")]

    assert "runs/matched-0995/rows.csv" in section
    assert "runs/matched-0995/wells_schedule.inc" in section
    assert "/compare?a=base&b=matched-0995" in section
    assert "runs/matched-100" not in section


def test_handoff_opens_with_the_current_submission_before_historical_snapshots():
    opening = "\n".join(_read("HANDOFF.md").splitlines()[:50])

    assert "## Текущий срез заявки" in opening
    assert "13 133,8 млн руб." in opening
    assert "runs/matched-0995/wells_schedule.inc" in opening
    assert "41d813345e95f9d484c792ee6312a270" in opening
    assert "submit-2026-08-24.1" in opening
