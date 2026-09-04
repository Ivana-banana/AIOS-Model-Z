"""Критические числа заявки не расходятся между сдаточными документами."""

import csv
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_declared_npv_is_the_same_in_all_current_documents():
    for name in ("docs/submission.md", "docs/report.md", "docs/defence-final.md",
                 "docs/presentation.md", "docs/defence-qa.md",
                 "docs/submission-day.md"):
        assert "5 579,5" in _read(name), f"{name} потерял заявляемый ЧДД"


def test_the_old_triple_never_stands_unlabelled_next_to_the_claim():
    """Прежняя тройка при базе 2007 не стоит рядом с заявкой без подписи базы.

    Смена базы дисконтирования на 1991 (Р-158) — не равномерный пересчёт, поэтому
    сравнительные разборы остались при базе 2007, и сами по себе старые числа
    законны. Опасна **смесь в одном абзаце**: ревизия 03.09 нашла в сценарии
    защиты фразу «Заявляем 5 579,5 … при базовом 11 851,8, то есть плюс 1 281,9»
    — докладчик так и прочитает, а жюри услышит противоречие с пакетом.

    Правило: абзац, где заявляемое число стоит вместе с числом прежней тройки,
    обязан называть базу — иначе читатель не может знать, какое из них к чему.
    """
    old = ("11 851,8", "13 133,8", "1 281,9", "13 152,9")
    new = ("5 579,5", "5 300,5", "+279,0")
    excuses = ("2007", "1991", "прежн", "прошл", "не заявля", "отозв")
    offenders: list[str] = []

    for name in ("README.md", "docs/submission.md", "docs/report.md",
                 "docs/defence-final.md", "docs/defence-qa.md", "docs/demo.md",
                 "docs/team-brief.md", "docs/tz-compliance.md",
                 "docs/presentation.md", "docs/submission-day.md"):
        for number, block in enumerate(_read(name).split("\n\n"), start=1):
            if not (any(one in block for one in old) and any(one in block for one in new)):
                continue
            if not any(excuse in block for excuse in excuses):
                offenders.append(f"{name}, абзац {number}: {block.strip()[:100]}")

    assert offenders == [], (
        "заявляемое число стоит в одном абзаце с прежней тройкой и базы не называет:"
        "\n  " + "\n  ".join(offenders)
    )


def test_submission_pairs_declared_npv_with_same_run_baseline_and_uplift():
    submission = _read("docs/submission.md")

    assert "5 579,5 млн руб." in submission
    assert "5 300,5 млн руб." in submission
    assert "+279,0 млн руб. (+5,26 %)" in submission
    assert "| **по достигнутому — наш сценарий** | **13 133,8** | **+10,8 %** | **20,0 %** |" in submission


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
    # колонок может стать больше — 28.08 добавлена шестая, sha256 артефакта;
    # сторож следит за содержанием корпуса, а не за его шириной
    assert all(len(row) >= 5 and row[3] == "search" for row in rows)
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
    distances = [distance for distance, _ in residuals_by_distance]
    residuals = [residual for _, residual in residuals_by_distance]
    mean_distance = statistics.fmean(distances)
    mean_residual = statistics.fmean(residuals)
    pearson = sum(
        (distance - mean_distance) * (residual - mean_residual)
        for distance, residual in residuals_by_distance
    ) / math.sqrt(
        sum((distance - mean_distance) ** 2 for distance in distances)
        * sum((residual - mean_residual) ** 2 for residual in residuals)
    )
    assert round(pearson, 3) == 0.844

    for name in (
        "docs/architecture.md",
        "docs/defence-qa.md",
        "docs/explained.md",
        "docs/presentation.md",
        "docs/report.md",
        "docs/surrogate.md",
    ):
        assert "927,8" in _read(name), f"{name} потерял калиброванный запас"
    assert "Пирсон 0,844" in _read("docs/defence-qa.md")
    assert "по Пирсону 0,844" in _read("docs/explained.md")


def test_conformal_guarantee_scope_is_spelled_out_for_humans():
    """Область действия гарантии обязана жить не только в машинном выводе.

    `GUARANTEE_SCOPE` уходит в паспорт шлюза, но защиту читают люди: уровень
    90 % — маргинальное покрытие, а кандидатов выбирает максимизация прогноза
    той же модели, и на такого кандидата гарантия не распространяется (Р-114).
    """
    from aios.trust import GUARANTEE_SCOPE

    assert "маргинальное покрытие" in GUARANTEE_SCOPE
    for name in ("docs/surrogate.md", "docs/defence-qa.md"):
        text = _read(name)
        assert "аргинальное" in text, f"{name} не говорит, что покрытие маргинальное"
        assert "GUARANTEE_SCOPE" in text, f"{name} не ссылается на константу"

    # формулировки, которые Р-114 запрещает: их отсутствие и есть проверка
    for name in ("docs/report.md", "docs/defence-qa.md", "docs/surrogate.md",
                 "docs/presentation.md", "docs/explained.md"):
        text = _read(name)
        assert "90-й процентиль" not in text, f"{name} выдаёт максимум за процентиль"
        assert "с вероятностью 90 %" not in text, f"{name} обещает вероятность"


def test_cost_of_a_non_degenerate_margin_is_computed_not_guessed():
    """Сколько ещё полных расчётов — считается формулой, а не оценивается.

    Порог равен максимуму выборки, пока `ceil((n+1)(1-alpha)) > n-1`. Числа 19
    и 39 названы в `docs/defence-qa.md`; здесь они выводятся из той же формулы,
    которой считает `OptimismMargin.fit`.
    """
    def first_non_degenerate(coverage: float) -> int:
        return next(n for n in range(2, 200)
                    if math.ceil((n + 1) * coverage) <= n - 1)

    def first_finite(coverage: float) -> int:
        return next(n for n in range(2, 200)
                    if math.ceil((n + 1) * coverage) <= n)

    assert first_finite(0.90) == 9      # ровно наш нынешний корпус
    assert first_non_degenerate(0.90) == 19
    assert first_finite(0.95) == 19
    assert first_non_degenerate(0.95) == 39
    assert math.ceil(20 * 0.90) / 20 == 0.90

    answers = _read("docs/defence-qa.md")
    assert "19 пар" in answers or "с **19 пар**" in answers
    assert "**39**" in answers


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
    assert "~~`requirements.lock` с хэшами~~ — **закрыто 22.08**" in report
    assert "**Открыта проверка\n   финального образа**" in report
    assert "`requirements.lock` с хэшами и SBOM образа" not in report


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
    assert round(1_467.2 / 436.5, 1) == 3.4
    assert round(2_042.7 / 460.8, 1) == 4.4

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

    for name in (
        "docs/defence-qa.md",
        "docs/report.md",
        "docs/surrogate.md",
        "docs/tz-compliance.md",
    ):
        text = current[name]
        assert "+1 467,2" in text, f"{name} потерял прогноз srg-narrow"
        assert "+2 042,7" in text, f"{name} потерял прогноз srg-search"
        assert "3,4" in text and "4,4" in text, f"{name} перепутал кратность ошибки"

    assert "их отклик переоценён в 3,4 и 4,4 раза" in current["PLAN.md"]
    assert "прогнозы переоценили отклик в 3,4 и 4,4 раза" in current["docs/presentation.md"]


def test_cycle_uplift_is_not_mixed_between_different_starting_points():
    cycle_share = 460.8 / 1_281.9 * 100

    assert round(cycle_share, 1) == 35.9
    for name in (
        "docs/defence-qa.md",
        "docs/explained.md",
        "docs/presentation.md",
    ):
        text = _read(name)
        assert "774,9" not in text, f"{name} смешал ЧДД кандидата и прирост цикла"
        assert "60,4 %" not in text, f"{name} считает долю от несопоставимой базы"

    explained = _read("docs/explained.md")
    assert "**+460,8 млн руб.** к историческому сценарию" in explained
    # перенос строки в документе не должен ломать проверку смысла: сверяем текст
    # со схлопнутыми пробелами, иначе сторож держит вёрстку, а не число
    flat = " ".join(explained.split())
    assert "+460,8 у лучшего кандидата цикла, то есть 35,9 %" in flat
    # 03.09: доля 35,9 % снята при базе дисконтирования 2007 и только при ней
    # верна — заявка с 02.09 живёт при базе 1991 (Р-158). Без этой пометки абзац
    # читается как сравнение с действующей заявкой, а это разные точки отсчёта.
    assert "оба числа при прежней базе дисконтирования 2007" in flat


def test_letter_maps_field_injection_and_external_water_to_separate_limits():
    letter = _read("docs/letter-to-organisers.md")

    assert "`max_field_injection_m3_per_day`" in letter
    assert "`max_external_water_m3_per_day`" in letter
    assert "Предел общей закачки" in letter
    assert "доступный сторонний водоисточник" in letter


def test_the_letter_no_longer_asks_the_question_answered_at_the_checkpoint():
    """В-03 закрыт вслух на КТ-2 31.08 — спрашивать его письмом больше нельзя.

    Вопрос, на который ответ уже получен, в письме организаторам стоит дороже
    пустой строки: он показывает, что мы не услышали ответ, и ставит под сомнение
    остальные пять. Нумерация вопросов при этом сдвинулась, и на неё ссылаются
    `BLOCKED.md` и `docs/kt2-plan.md` — расхождение ссылок ловится здесь же.
    """
    letter = _read("docs/letter-to-organisers.md")

    assert "шесть вопросов" in letter
    assert "Как подключается `wells_schedule.inc`" not in letter
    assert "подменяющий секцию целиком, — правильная" not in letter
    assert "получен вслух на КТ-2 31.08" in letter
    assert "## 1. Состав фонда в вашем экспортёре (В-04)" in letter
    assert "## 3. Точные границы коридора баланса воды" in letter
    assert "## 6. Значение `chargeInitialPump` при оценке" in letter
    # Раздел 7 — не седьмой вопрос, а **ответ** на два вопроса трекера от 03.09
    # (где у них стоит 2007 и что с разломами, Д3). Сторож держит прежний смысл:
    # вопросов по-прежнему шесть, восьмого раздела нет, а седьмой обязан начинаться
    # словом «Ответ» — иначе он и вправду станет седьмым вопросом.
    assert "## 7. Ответ на два ваших вопроса" in letter
    assert "## 8." not in letter
    assert "пять вопросов" not in letter, "шапка письма отстала от состава вопросов"

    for name in ("BLOCKED.md", "docs/kt2-plan.md"):
        assert "вопросами 5 и 6" in _read(name), (
            f"{name} ссылается на прежние номера вопросов письма"
        )

    blocked = _read("BLOCKED.md")
    assert "**В-03 СНЯТ 31.08**" in blocked
    assert "## 1. Вопрос организаторам В-04 (В-03 закрыт 31.08)" in blocked

    decisions = _read("docs/decisions.md")
    assert "Р-130" in decisions
    assert "КТ-2, 31.08.2026, трекер, запись 06:39" in decisions


def test_the_llm_rationale_answers_the_question_the_tracker_asked():
    """Требование ТЗ п. 3.1 — обосновать причину использования, а не только вызвать.

    Трекер сформулировала это отдельным пунктом (КТ-2, запись 03:15). Раздел
    обязан отвечать на три вещи сразу: зачем модель здесь, что она даёт сверх
    шаблона **числом**, и почему ей не доверено производить числа и принимать
    решения. Без последней части обоснование превращается в рекламу слоя.
    """
    rationale = _read("docs/llm-rationale.md")

    assert "ASR" in rationale and "0,750" in rationale
    assert "не принимает его" in rationale and "производит чисел" in rationale
    assert "template-rejected" in rationale
    assert "16.08" in rationale, "история про обоснование под неверный результат"
    assert "Границы: чего это обоснование не доказывает" in rationale

    assert "llm-rationale.md" in _read("README.md"), "раздел не найти из README"
    assert "llm-rationale.md" in _read("docs/report.md"), "отчёт на него не ссылается"


def test_the_scalability_concept_keeps_its_honest_boundary():
    """Трекер назвала слабое место сама — концепция обязана его признавать.

    «Если вы обучаете MLP, то переносим на другое месторождение — у вас всё
    сломается» (КТ-2, запись 04:20). Раздел, который обходит это молчанием,
    сильнее не выглядит: на защите первым спросят именно про перенос весов.
    Поэтому тест держит три уровня — что переносится само, что переобучением с
    измеренной ценой, и что не переносится вовсе.
    """
    scalability = _read("docs/scalability.md")

    assert "Переносится архитектурно" in scalability
    assert "Переносится только переобучением" in scalability
    assert "Не переносится вовсе" in scalability
    assert "60 сценариев" in scalability, "цена переноса названа числом"
    assert "825 с" in scalability, "цена переноса переведена во время замером 01.09"
    assert "не проверялось" in scalability, "границы разбора не названы"

    assert "scalability.md" in _read("README.md"), "концепцию не найти из README"


def test_defence_does_not_present_aggressive_uplift_as_the_declared_result():
    defence = _read("docs/defence-final.md")

    assert "плюс 279,0 млн руб., или 5,3 процента" in defence
    assert "Прирост самого доходного сценария | +1 789,5 млн руб. (+15,1 %)" in defence


def test_the_jury_answers_carry_the_same_declared_triple():
    """Ответы жюри — самый вероятный источник расхождения с заявкой.

    Документ пишется под устное выступление, живёт отдельно от `submission.md` и
    правится чаще: именно так 22.08 доклад открывался приростом сценария, который
    мы не заявляем (`docs/numbers-audit.md`, расхождение 1). Поэтому тройка
    «ЧДД — база — прирост» закреплена и здесь.
    """
    answers = _read("docs/defence-qa.md")

    assert "5 579,5" in answers
    assert "5 300,5" in answers
    assert "+279,0 млн руб. (+5,26 %)" in answers
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
    assert "5 579,5" in presentation, "презентация потеряла заявляемый ЧДД"
    assert "`runs/matched-0995`, прогон 24.08" in presentation
    assert "это 5,3 процента" in presentation
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
    assert "5 579,5 млн руб." in submission


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

    assert "## 0. Проверить, какой кейс запрошен" in sheet
    assert "пакет, посчитанный на **Модели Z**" in sheet
    assert "валидационный кейс по п. 4.3 ТЗ" in sheet
    assert "**не\nотправляйте** указанные ниже `matched-0995`, ЧДД и тег" in sheet
    for command in (
        "git status --short --branch",
        "python -m pytest -q",
        "python -m ruff check .",
        "python scripts/check_release_tag.py",
        "python scripts/check_package.py",
        "python scripts/check_submission.py runs/matched-0995/wells_schedule.inc",
    ):
        assert command in sheet
    assert "submit-2026-09-02" in sheet
    assert "5 300,5" in sheet
    assert "+279,0 млн руб. (+5,26 %)" in sheet
    assert "подменяет секцию SCHEDULE целиком; история включена" in sheet
    assert "подтверждённой трекером на КТ-2 31.08.2026 (В-03 закрыт)" in sheet
    assert "В-03 не получен" not in sheet
    assert "всем 103 скважинам с режимом после 01.01.2007" in sheet
    assert "не получен (В-04)" in sheet
    assert "docs/submission-day.md" in _read("README.md")
    assert "все шесть проверок" in _read("README.md")
    assert "в **отдельной чистой копии** `submit-2026-09-02`" in sheet
    assert "вернуть в текущий `main`" in sheet


def test_submission_does_not_reopen_resolved_blockers():
    submission = _read("docs/submission.md")

    assert "Организаторам остаётся В-04" in submission
    assert "В-03 закрыт ответом трекера на" in submission
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
    # 03.09: сторож требовал «заявляем **+10,8 %**» — то есть держал заявку при
    # базе 2007 через неделю после перевода на 1991 (Р-158). Проверка,
    # закреплявшая устаревшее число, и есть та, из-за которой оно дожило.
    assert "заявляем **+5,26 %**" in compliance
    assert "заявляем **+10,8 %**" not in compliance
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
    assert "5 579,5 млн руб." in opening
    assert "runs/matched-0995/wells_schedule.inc" in opening
    assert "41d813345e95f9d484c792ee6312a270" in opening
    assert "submit-2026-09-02" in opening


# --- Числа сверки 30–31.08: документ против пересчёта ------------------------
#
# Найдено 31.08 при подготовке к КТ-2: три величины блока 2 плана защиты стояли
# в документах не теми, какими их печатает расчёт. Доля нефти в расхождении
# (86 % против настоящих 90,7 %), счётчик насосов без флага (99 против 55) и
# заявка на базе 1991 (5 486,5 — отсечённая десятая вместо 5 486,6). Ни одно из
# них не влияет на заявляемый ЧДД, но каждое произносится вслух и проверяется
# делением в уме. Поэтому теперь они не переписываются из документа в документ,
# а пересчитываются здесь из тех же выгрузок.

RUNS = ROOT / "runs"


def _require_rows(*runs: str) -> None:
    """Пропустить проверку, если хоть одной из нужных выгрузок нет.

    Проверять один `base` было мало: фикстура ниже читает и `matched-0995`, и на
    машине, где базовый прогон есть, а заявка не пересчитана, отсутствие второй
    выгрузки роняло сбор **ошибкой**, а не пропуском. `docs/submission-day.md`
    требует в день отправки «0 failed, 0 errors» — то есть паспорт блокировал бы
    пакет по ложной причине.
    """
    missing = [run for run in runs if not (RUNS / run / "rows.csv").is_file()]
    if missing:
        pytest.skip("нет выгрузок runs/: " + ", ".join(missing))


def _npv(run: str, year: int, *, initial_pump: bool):
    import sys

    sys.path.insert(0, str(ROOT))
    from aios import economics
    from scripts.compare_reference import norms_with_initial_pump

    norms = norms_with_initial_pump() if initial_pump else economics.NORMS_FILE
    return economics.compute_npv_from_file(
        RUNS / run / "rows.csv", start_date=f"{year}-01-01", norms_file=norms
    )


@pytest.fixture(scope="module")
def four_reference_systems() -> dict[tuple[str, int, bool], object]:
    """Одно расписание в четырёх системах отсчёта — как в плане защиты."""
    _require_rows("base", "matched-0995")
    return {
        (run, year, pump): _npv(run, year, initial_pump=pump)
        for run in ("base", "matched-0995")
        for year in (2007, 1991)
        for pump in (False, True)
    }


def test_four_reference_systems_in_documents_match_a_fresh_recomputation(
    four_reference_systems,
):
    """Таблица «четыре системы отсчёта» обязана печататься расчётом, а не памятью."""
    values = {key: result.chdd_m for key, result in four_reference_systems.items()}

    def shown(key: tuple[str, int, bool]) -> str:
        return f"{values[key]:,.1f}".replace(",", " ").replace(".", ",")

    rows = {
        "2007 / НЕТ": (shown(("base", 2007, False)), shown(("matched-0995", 2007, False))),
        "2007 / ДА": (shown(("base", 2007, True)), shown(("matched-0995", 2007, True))),
        "1991 / НЕТ": (shown(("base", 1991, False)), shown(("matched-0995", 1991, False))),
        "1991 / ДА": (shown(("base", 1991, True)), shown(("matched-0995", 1991, True))),
    }
    assert rows["2007 / НЕТ"] == ("11 851,8", "13 133,8"), "изменилась сама заявка"
    assert rows["1991 / ДА"] == ("5 207,7", "5 486,6")

    for name in ("docs/kt2-plan.md", "docs/kt2-explained.md", "docs/experiments.md",
                 "docs/decisions.md"):
        text = _read(name)
        for base, case in rows.values():
            assert base in text and case in text, f"{name} потерял строку {base} / {case}"

    # усечённая десятая не должна вернуться туда, где число называют заявкой
    for name in ("docs/kt2-plan.md", "docs/kt2-explained.md"):
        assert "5 486,5" not in _read(name), f"{name} снова округляет вниз"


def test_pump_counter_without_the_flag_is_the_one_documents_name(
    four_reference_systems,
):
    """«145 против 55» — счётчик, которым доказан включённый chargeInitialPump."""
    without = four_reference_systems[("base", 1991, False)].pump_changes
    with_flag = four_reference_systems[("base", 1991, True)].pump_changes

    assert (without, with_flag) == (55.0, 145.0)
    for name in ("docs/kt2-plan.md", "docs/kt2-explained.md", "docs/experiments.md",
                 "docs/decisions.md"):
        text = _read(name)
        assert "145 против 55" in text or "без флага 55" in text or "без флага — 55" in text, (
            f"{name} называет другой счётчик насосов"
        )
        assert "145 против 99" not in text and "без флага 99" not in text


def test_the_fixture_requires_every_export_it_reads():
    """Дефект 01.09 был не в пропуске, а в рассогласовании: читала две, требовала одну.

    Проверять поведение `_require_rows` мало — он и до правки был бы верен. Здесь
    сверяются два места одной функции: имена прогонов, которые фикстура читает в
    своём генераторе, и имена, которые она объявляет обязательными. Разъехались —
    значит на машине без второй выгрузки снова будет ошибка сбора вместо пропуска.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    fixture = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "four_reference_systems")

    required = {
        argument.value
        for call in ast.walk(fixture)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_require_rows"
        for argument in call.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    read = {
        element.value
        for comprehension in ast.walk(fixture)
        if isinstance(comprehension, ast.comprehension) and getattr(
            comprehension.target, "id", "") == "run"
        for element in getattr(comprehension.iter, "elts", [])
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }

    assert read, "в фикстуре не нашлось перечня прогонов — проверка потеряла смысл"
    assert required == read, (
        f"фикстура читает выгрузки {sorted(read)}, а требует {sorted(required)}"
    )


def test_missing_export_skips_the_check_instead_of_erroring():
    """Найдено 01.09: фикстура читала две выгрузки, а проверяла наличие одной.

    Машина с базовым прогоном, но без пересчитанной заявки получала на сборе
    `FileNotFoundError` из вендоренного расчётчика — в паспорте дня отправки это
    выглядит как «есть ошибки, пакет не отправлять».
    """
    with pytest.raises(pytest.skip.Exception) as skipped:
        _require_rows("base", "нет-такого-прогона")
    assert "нет-такого-прогона" in str(skipped.value)


def test_oil_share_of_the_remaining_gap_is_recomputed_not_remembered():
    """86 % было ошибкой деления: 45,98 из 50,71 — это 90,7 %."""
    _require_rows("base")
    import sys

    sys.path.insert(0, str(ROOT))
    from aios import reference

    result = _npv("base", 1991, initial_pump=True)
    table = reference.load(ROOT / "docs" / "reference"
                           / "Расчет ЧДД через OPM Flow Model_Z.xlsx")
    parts = reference.decompose(table, result.annual)
    gap = result.chdd_m - table.cumulative_chdd_m
    # выручка и отчисления — одна и та же тонна с двух сторон, налог 25 %
    oil = (parts["Выручка от основной деятельности"] + parts["Отчисления"]) * 0.75

    assert round(oil, 2) == 45.98
    assert round(gap, 2) == 50.71
    assert round(100 * oil / gap, 1) == 90.7

    for name in ("docs/kt2-plan.md", "docs/kt2-explained.md", "docs/experiments.md",
                 "docs/decisions.md"):
        text = _read(name)
        assert "90,7 %" in text, f"{name} не называет пересчитанную долю нефти"
        assert "то есть 86 %" not in text and "(86 %)" not in text


def test_slide_of_what_we_do_not_claim_speaks_the_confirmed_cycle_uplift():
    """Сторож 26.08 ловил «774,9» и «60,4 %» — но речь слайда писала «775» словами.

    Отозванное число пережило исправление именно потому, что проверялась запись,
    а не величина. Теперь проверяется и произносимый вариант.
    """
    presentation = _read("docs/presentation.md")
    slide = presentation[presentation.index("## Слайд 7."):]
    slide = slide[:slide.index("## Слайд 8.")]

    for forbidden in ("774,9", "775", "60,4 %", "шестьдесят процентов"):
        assert forbidden not in slide, f"слайд 7 снова называет отозванное: {forbidden}"
    assert "460,8 против 1 281,9" in slide
    assert "+461 млн, 36 % от простого правила" in slide


# --- Сверка с эталонами организаторов: документы против самих файлов ---------
#
# Найдено 01.09: числа сверки 30–31.08 стояли только в `kt2-plan.md`,
# `kt2-explained.md`, `experiments.md` и журнале, а отчёт, README, презентация,
# заявка, ответы защиты и матрица ТЗ продолжали называть отозванное «5,0 % по
# накопленной нефти». Раскладки организаторов лежат в репозитории, поэтому эти
# величины пересчитываются здесь, а не переписываются из документа в документ.

REFERENCE_DIR = ROOT / "docs" / "reference"
#: Документы, которые читает жюри и в которых сверка обязана быть названа числом.
JURY_DOCUMENTS = (
    "docs/report.md",
    "README.md",
    "docs/presentation.md",
    "docs/submission.md",
    "docs/defence-qa.md",
    "docs/explained.md",
)

#: Все действующие документы, где отозванное число не имеет права стоять как живое.
#: Шире `JURY_DOCUMENTS`: сценарий защиты чисел сверки не называет — он их
#: проговаривает словами, — но именно в нём 01.09 нашлось «5 процентов при допуске
#: в пару», пережившее правку presentation.md.
LIVE_DOCUMENTS = JURY_DOCUMENTS + (
    "docs/slides.html",
    "docs/defence-final.md",
    "docs/tz-compliance.md",
    "docs/onboarding-team.md",
    "docs/architecture.md",
    "docs/surrogate.md",
    "docs/submission-day.md",
)


def _ru(value: float) -> str:
    """Число в том виде, в каком оно печатается в документах: 5 218,9."""
    return f"{value:,.1f}".replace(",", " ").replace(".", ",")


@pytest.fixture(scope="module")
def published_references():
    """Обе раскладки организаторов по базовому сценарию."""
    import sys

    sys.path.insert(0, str(ROOT))
    from aios import reference

    return (
        reference.load(REFERENCE_DIR / "Расчет ЧДД через OPM Flow Model_Z.xlsx"),
        reference.load(REFERENCE_DIR / "Расчет ЧДД через тНавигатор Model_Z.csv"),
    )


def test_jury_documents_name_the_reference_numbers_the_files_give(published_references):
    """5 157,0, 5 218,9 и разрыв 1,20 % — из файлов, а не из памяти."""
    opm, tnavigator = published_references
    gap_share = tnavigator.cumulative_chdd_m / opm.cumulative_chdd_m - 1

    assert _ru(opm.cumulative_chdd_m) == "5 157,0"
    assert _ru(tnavigator.cumulative_chdd_m) == "5 218,9"
    assert f"{100 * gap_share:.2f}".replace(".", ",") == "1,20"

    for name in JURY_DOCUMENTS:
        document = _read(name)
        assert _ru(opm.cumulative_chdd_m) in document, f"{name} не называет эталон OPM Flow"
        assert _ru(tnavigator.cumulative_chdd_m) in document, f"{name} не называет эталон тНавигатора"
        assert "1,20 %" in document, f"{name} не называет разрыв симуляторов"


def test_withdrawn_oil_gap_is_marked_as_withdrawn_wherever_it_appears():
    """Отозванные «5,0 %» и ориентир 4 771 могут встречаться только как отозванные.

    Число получено сравнением нашей истории с их `OPTIMIZED`-сценарием, то есть с
    другим управлением (Р-120). Запрещать саму строку нельзя — документы обязаны
    объяснять, что именно отменено; проверяется абзац, в котором она стоит.
    """
    # слова, которыми документ признаёт число отменённым. Список намеренно узкий:
    # «прежн» и «не то» отсюда убраны 01.09 — они встречаются в обычном тексте и
    # пропускали бы абзац почти любого содержания
    excuses = ("отозв", "отмен", "OPTIMIZED", "оптимизирован", "прежние редакции",
               "прежнее число", "прежние «5", "некорректн", "не измеряло", "измеряла не то")
    # число живёт и цифрами, и прописью: правка 01.09 вычистила «5,0 %» из
    # презентации, а «5 процентов» в сценарии защиты пережила её
    withdrawn = ("4 771", "5,0 %", "5 процентов", "пять процентов")
    for name in LIVE_DOCUMENTS:
        document = _read(name)
        for paragraph in document.split("\n\n"):
            if not any(one in paragraph for one in withdrawn):
                continue
            assert any(word in paragraph for word in excuses), (
                f"{name}: отозванное число подано как действующее:\n{paragraph[:400]}"
            )
# --- Две базы дисконтирования не смешиваются в документах защиты -------------
#
# 02.09 трекер ответила, что расчёт ведётся с 1991 года, и заявка переехала на
# 5 579,5 / 5 300,5 / +279,0 (Р-158). Смена базы **неравномерна** (Р-159): при
# базе 2007 та же пара давала 13 133,8 / 11 851,8 / +1 281,9, то есть +10,8 %
# вместо +5,26 %. Старая тройка законна там, где сравниваются сценарии между
# собой, — эти замеры при базе 2007 и снимались, — но абзац обязан назвать базу.
# Ревизия 03.09 нашла её без пометки в 26 местах, включая речь `defence-final.md`,
# где в одной фразе стояли «Заявляем 5 579,5» и «при базовом 11 851,8».

#: Числа заявки при прежней базе. `+11,9 %` — прежний кандидат ×1.6 оттуда же.
BASE_2007_NUMBERS = ("13 133,8", "11 851,8", "1 281,9", "+10,8 %",
                     "10,8 процента", "+11,9 %")

#: Чем абзац признаёт, что число снято при прежней базе. Список узкий намеренно:
#: «баз» сюда не годится — «базовый сценарий» содержит его и пропускал бы всё.
BASE_2007_MARKS = ("2007", "Р-158", "Р-159")

#: Документы, которые читают или произносят на защите. `numbers-audit.md` и
#: `defence-qa.md` сюда не входят и это осознанно: первый по устройству — реестр
#: датированных срезов прошлых чисел, второй — разбор сценариев между собой;
#: пометка в каждом их абзаце сделала бы текст нечитаемым, а речь по ним не идёт.
DUAL_BASE_DOCUMENTS = (
    "README.md",
    "docs/defence-final.md",
    "docs/demo.md",
    "docs/team-brief.md",
    "docs/tz-compliance.md",
    "docs/kt2-plan.md",
    "docs/kt2-explained.md",
    "docs/slides.html",
    "docs/presentation.md",
    "docs/submission.md",
    "docs/submission-day.md",
    "docs/report.md",
    "docs/explained.md",
)


def test_numbers_from_the_old_discount_base_always_name_that_base():
    """Число при базе 2007 живёт только в абзаце, который эту базу называет.

    Проверяется абзац, а не документ: запрещать саму тройку нельзя — сравнения
    сценариев между собой при базе 2007 и снимались, и переписывать их значило бы
    выдумывать замеры, которых не было.
    """
    for name in DUAL_BASE_DOCUMENTS:
        document = _read(name)
        for paragraph in document.split('\n\n'):
            if not any(one in paragraph for one in BASE_2007_NUMBERS):
                continue
            assert any(mark in paragraph for mark in BASE_2007_MARKS), (
                f"{name}: число прежней базы подано как действующее: "
                + paragraph[:400]
            )


def test_the_spoken_defence_claims_the_current_triple():
    """То, что произносится вслух, — действующая тройка целиком.

    Ревизия 03.09 нашла в разделе «Что мы заявляем» фразу «Заявляем 5 579,5 …
    при базовом сценарии 11 851,8, то есть плюс 1 281,9 млн руб., плюс 10,8
    процента»: заявка при базе 1991, база и прирост при базе 2007, всё в одном
    предложении. Такое читается вслух и звучит как ошибка в арифметике.
    """
    defence = _read("docs/defence-final.md")
    section = defence[defence.index("## 7. Что мы заявляем"):]
    section = " ".join(section[:section.index("##", 10)].split())

    assert "Заявляем **5 579,5 млн руб.**" in section
    assert "при базовом сценарии 5 300,5" in section
    assert "плюс 279,0 млн руб., плюс 5,3" in section
    for stale in ("11 851,8", "1 281,9", "10,8"):
        assert stale not in section, f"речь заявки называет число прежней базы: {stale}"


# --- Слайды: показываемый файл обязан говорить то же, что сценарий -----------

def test_slides_carry_the_claimed_triple_and_the_reference_check():
    """Слайды собраны из docs/presentation.md — числа обязаны совпасть.

    Слайд живёт отдельным файлом и правится отдельно от сценария; именно так
    26.08 отозванный прирост цикла пережил правку записи и дожил до речи
    прописью.
    """
    slides = _read("docs/slides.html")

    for number in ("5 579,5", "5 300,5", "+279,0", "+5,26 %"):
        assert number in slides, f"слайды не называют {number}"
    for number in ("5 218,9", "5 207,7", "5 157,0", "1,20 %"):
        assert number in slides, f"слайды не называют число сверки {number}"
    for forbidden in ("774,9", "60,4 %", "4 771"):
        assert forbidden not in slides, f"слайды называют отозванное: {forbidden}"


def test_slides_keep_the_nine_speaking_slides_and_mark_the_spares():
    """Девять слайдов доклада и три запасных — ровно так расписан хронометраж."""
    slides = _read("docs/slides.html")

    assert slides.count("<section") == 12
    assert slides.count('class="backup"') == 3
    for mark in ("1 / 9", "5 / 9 · 1:00 · главный", "7 / 9 · 1:00 · не сокращать", "9 / 9"):
        assert mark in slides, f"в слайдах нет отметки «{mark}»"


def test_slides_name_the_limits_count_the_code_actually_has():
    """«Сколько ключей ограничений» слайды берут не из памяти, а из `Limits`.

    Слайды собраны 01.09, когда ключей было 27; за ночь 01→02.09 их стало 33, и
    все документы это подхватили — кроме показываемого файла. Расхождение нашлось
    02.09 сверкой слайдов с кодом, а не тестом: сторожа на это число не было.
    Ставим его здесь, потому что слайд — единственный документ, который жюри
    видит глазами, и правится он отдельно от всех остальных.
    """
    import dataclasses
    import sys

    sys.path.insert(0, str(ROOT))
    from aios.config import Limits

    count = len(dataclasses.fields(Limits))
    slides = _read("docs/slides.html")

    assert count == 33, "число ключей изменилось — обновить слайды и этот тест"
    assert f"{count} ключа" in slides, f"слайды не называют {count} ключей [limits]"
    assert "27 ключ" not in slides, "в слайдах осталось прежнее число ключей"


def _speed_measurement() -> tuple[float, float]:
    """Замер 01.09: секунды полного расчёта ГГДМ и во сколько раз быстрее MLP.

    Источник — запись «Ускорение прогноза измерено» в `docs/experiments.md`: она
    в репозитории и есть на любой машине. Машиночитаемый слепок
    `artifacts/forecast-speed.json` лежит только у той машины, что мерила
    (`artifacts/` не отслеживается), поэтому читать его как **единственный**
    источник значило бы ронять сбор у всех остальных и в чистом клоне, где
    `docs/clean-clone-check.md` требует «0 падений». Слепок не выброшен: когда он
    есть, он сверяется с журналом — расхождение между тем, что померили, и тем,
    что записали, ловится здесь же (Р-161).
    """
    import json
    import re

    section = _read("docs/experiments.md").split("## Ускорение прогноза измерено", 1)[1]
    section = section.split('\n## ', 1)[0]
    ggdm_seconds = float(re.search(r"\| полный расчёт.*?\*\*(\d+) с\*\*", section).group(1))
    speedup = float(
        re.search(r"\| обученный суррогат.*?\*\*в ([\d ]+) раз\*\*", section)
        .group(1).replace(" ", ""))

    artifact = ROOT / "artifacts" / "forecast-speed.json"
    if artifact.is_file():
        speed = json.loads(artifact.read_text(encoding="utf-8"))
        mlp = next(one for one in speed["замеры"] if one["модель"] == "MLP по сценариям DoE")
        assert speed["ггдм_секунд"] == ggdm_seconds, (
            "журнал разошёлся с замером по времени ГГДМ")
        assert round(mlp["быстрее ГГДМ, раз"]) == speedup, (
            "журнал разошёлся с замером по ускорению")

    return ggdm_seconds, speedup


def test_slides_price_of_transfer_follows_the_speed_measurement():
    """Цена переноса в часах пересчитывается из замера, а не переписывается.

    В слайдах стояло «≈ 60 прогонов ≈ 11 часов» при 825 с на расчёт: 11 часов —
    это 60 × 660 с, то есть замер **первой** машины, тогда как ускорение 1 668 раз
    рядом взято из замера второй. Ровно тот след дублирования, о котором
    предупреждает `HANDOFF.md`: числа из двух разных замеров сошлись на одном
    слайде.
    """
    ggdm_seconds, speedup_times = _speed_measurement()

    hours = round(60 * ggdm_seconds / 3600)
    speedup = f"{speedup_times:,.0f}".replace(",", " ")

    slides = _read("docs/slides.html")
    assert hours == 14, "замер изменился — обновить слайды"
    assert f"≈ 60 прогонов ≈ {hours} часов" in slides, "слайды не называют цену переноса замером"
    assert f"{speedup} раз" in slides, "слайды не называют измеренное ускорение"
    assert "11 часов" not in slides, "в слайдах осталась цена переноса из чужого замера"


def test_the_spare_slide_names_the_donor_transfer_with_both_numbers():
    """Запасной слайд про масштабируемость обязан нести перенос донорами.

    Им отвечают на замечание трекера про MLP, и оба числа называются вместе:
    85 % по нефти и 25 % по закачке. Перенос неравноценный, и подавать его без
    второго числа — та же подача, что скорость без точности (Р-193).
    """
    slides = _read("docs/slides.html")
    start = slides.index("Масштабируемость и цена переноса")
    section = slides[start:slides.index("</section>", start)]

    assert "донор" in section, "слайд не называет механизм переноса"
    assert "85 %" in section and "25 %" in section, "названо не оба числа"


def test_no_slide_carries_more_text_than_the_measured_maximum():
    """Вес текста слайда не выше того, что уже мерилось браузером.

    Переполнение сцены 1920×1080 меряется только в браузере, и на машине без
    него добавлять текст вслепую нельзя. Консервативная граница: ни один слайд
    не тяжелее самого тяжёлого из **измеренных** 02.09 — тогда все двенадцать
    занимали 431–796 из 880 пикселей. Слайд, перешагнувший эту границу, обязан
    быть перемерен, а не принят на веру.
    """
    import re

    slides = _read("docs/slides.html")
    measured = 773  # слайд «Заявляемое число»: 796 из 880 пикселей при 773 знаках

    worst = []
    for section in re.findall(r"<section[^>]*>.*?</section>", slides, re.S):
        body = re.search(r'<div class="slide-body">(.*?)</div>\s*<div class="src"',
                         section, re.S)
        if body is None:
            continue
        text = " ".join(re.sub(r"<[^>]+>", " ", body.group(1)).split())
        if len(text) > measured:
            head = re.search(r"<h[12][^>]*>(.*?)</h[12]>", section, re.S)
            name = " ".join(re.sub(r"<[^>]+>", "", head.group(1)).split()) if head else "?"
            worst.append(f"{name}: {len(text)} знаков")

    assert not worst, (
        "слайд тяжелее измеренного максимума — перемерить в браузере: "
        + "; ".join(worst)
    )


def test_the_reference_gap_names_the_flag_it_was_measured_with():
    """Сверка с эталоном обязана называть значение `chargeInitialPump`.

    То же сравнение при двух значениях их же параметра даёт +0,98 % и +2,78 %:
    их публикация посчитана при ДА (счётчик насосов 145 сходится только так,
    Р-119), а наша заявка — при НЕТ, как стоит в поставленных нормативах. Число
    без предпосылки воспроизводится не той командой, которой получено (04.09).
    """
    report = _read("docs/report.md")

    assert "+0,98 %" in report
    assert "chargeInitialPump = ДА" in report, "предпосылка сверки не названа"
    assert "--charge-initial-pump" in report, "команда воспроизведения не названа"
    assert "2,78" in report, "второе значение той же сверки скрыто"

    readme = _read("README.md")
    assert "chargeInitialPump = ДА" in readme, "README называет разрыв без предпосылки"
