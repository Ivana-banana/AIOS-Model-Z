"""Слой интерпретируемости: языковая модель формулирует, но не считает.

Главная проверяемая гарантия — число, которого нет в расчёте, не может попасть в
объяснение. Проект уже получал убедительное физическое обоснование под неверный
результат (форсаж, 16.08), поэтому «звучит правдоподобно» здесь не аргумент.

Сеть в тестах не трогается: клиент подставной. Единственное исключение —
проверка «включён, но не дозвонился»: там клиент строится настоящий, поэтому
учётные данные у неё отбираются, а `base_url` уводится на закрытый порт.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from aios.explain import (
    Evidence,
    Settings,
    explain,
    template_text,
    unverified_numbers,
)


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_Block]


class _FakeMessages:
    def __init__(self, payload: dict[str, str] | Exception):
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload
        return _Response([_Block(json.dumps(self.payload, ensure_ascii=False))])


class _FakeClient:
    def __init__(self, payload: dict[str, str] | Exception):
        self.messages = _FakeMessages(payload)


def evidence() -> Evidence:
    return Evidence(
        subject="увеличение закачки по фонду",
        observations=("нефть перестала расти с уровня отбора 1.08",),
        numbers={
            "ЧДД базового сценария, млн руб.": 11851.8,
            "ЧДД сценария, млн руб.": 13642.4,
            "прирост, млн руб.": 1790.6,
        },
        confirmed=True,
        caveats=("мощность системы ППД стендом не моделируется",),
    )


def answer(**overrides) -> dict[str, str]:
    payload = {
        "what_happened": "Нефть перестала расти при дальнейшем увеличении отбора.",
        "why": "Ограничивала не добыча, а энергия пласта, поэтому усилена закачка.",
        "confirmation": "Подтверждено полным расчётом: прирост 1790.6 млн руб.",
    }
    payload.update(overrides)
    return payload


# --- проверка чисел ---------------------------------------------------------

def test_a_question_is_answered_from_the_same_facts(tmp_path):
    """Вопрос инженера идёт через тот же слой и ту же проверку чисел (Р-184)."""
    from aios.explain import Settings, answer

    client = _FakeClient({"answer": "Прирост 1790.6 млн руб. подтверждён полным расчётом.",
                          "missing": ""})
    settings = Settings(enabled=True, cache_dir=tmp_path)

    reply = answer(evidence(), settings, "чем подтверждён прирост?", client=client)

    assert reply.source == "llm"
    assert "1790.6" in reply.text
    sent = client.messages.calls[0]
    assert "ВОПРОС ИНЖЕНЕРА: чем подтверждён прирост?" in sent["messages"][0]["content"]
    # схема ответа своя: у вопроса одна часть, а не три
    schema = sent["output_config"]["format"]["schema"]
    assert set(schema["required"]) == {"answer", "missing"}


def test_a_question_answer_with_an_invented_number_is_rejected(tmp_path):
    """Барьер проверки чисел работает и на ответе, а не только на объяснении."""
    from aios.explain import Settings, answer

    client = _FakeClient({"answer": "Прирост составил 2500.0 млн руб.", "missing": ""})

    reply = answer(evidence(), Settings(enabled=True, cache_dir=tmp_path),
                   "сколько дал прирост?", client=client)

    assert reply.source == "template-rejected"
    assert any("2500" in one for one in reply.rejected_numbers)


def test_two_different_questions_do_not_share_one_cache_entry(tmp_path):
    """Ответ на один вопрос не выдаётся за ответ на другой."""
    from aios.explain import Settings, answer

    settings = Settings(enabled=True, cache_dir=tmp_path)
    first = _FakeClient({"answer": "Подтверждено полным расчётом.", "missing": ""})
    second = _FakeClient({"answer": "Мощность ППД стендом не моделируется.", "missing": ""})

    one = answer(evidence(), settings, "чем подтверждено?", client=first)
    two = answer(evidence(), settings, "что не доказано?", client=second)

    assert one.fingerprint != two.fingerprint
    assert one.text != two.text
    assert len(second.messages.calls) == 1, "второй вопрос обязан дойти до модели"


def test_an_empty_question_falls_back_to_the_explanation(tmp_path):
    """Пустой вопрос — это отсутствие вопроса, а не пустой ответ."""
    from aios.explain import Settings, answer

    client = _FakeClient({"what_happened": "Отбор вырос.", "why": "Так выгоднее.",
                          "confirmation": "Подтверждено полным расчётом."})

    reply = answer(evidence(), Settings(enabled=True, cache_dir=tmp_path), "   ",
                   client=client)

    assert reply.source == "llm"
    schema = client.messages.calls[0]["output_config"]["format"]["schema"]
    assert set(schema["required"]) == {"what_happened", "why", "confirmation"}


def test_numbers_from_evidence_pass():
    assert unverified_numbers("прирост 1790.6 млн руб.", evidence()) == ()


def test_invented_number_is_caught():
    """Ровно тот случай, ради которого слой вообще проверяется."""
    assert unverified_numbers("прирост 2500.0 млн руб.", evidence()) == ("2500.0",)


def test_recomputed_percentage_is_caught():
    """15,1 % — верная величина, но мы её модели не давали. Значит, она выдумана."""
    assert unverified_numbers("прирост составил 15,1 %", evidence()) == ("15,1",)


def test_integer_part_of_a_known_number_passes():
    assert unverified_numbers("около 1790 млн руб.", evidence()) == ()


def test_thousands_separator_is_understood():
    assert unverified_numbers("ЧДД 13 642,4 млн руб.", evidence()) == ()


def test_dates_are_not_numbers():
    assert unverified_numbers("на шаге 2019-06 и в 09.2025 режим не менялся", evidence()) == ()


def test_all_invented_numbers_are_reported():
    found = unverified_numbers("было 100.0, стало 200.0", evidence())
    assert found == ("100.0", "200.0")


# --- поведение слоя ---------------------------------------------------------

def test_disabled_layer_falls_back_to_template():
    result = explain(evidence(), Settings(enabled=False))
    assert result.source == "template"
    assert not result.from_model
    assert "11851.8" in result.text


def test_enabled_layer_uses_the_model():
    client = _FakeClient(answer())
    result = explain(evidence(), Settings(enabled=True), client=client)
    assert result.source == "llm"
    assert result.from_model
    assert "энергия пласта" in result.text


def test_answer_with_an_invented_number_is_rejected_whole():
    """Отбрасывается весь ответ, а не одно число: ошиблась в одном — доверия нет."""
    client = _FakeClient(answer(confirmation="Прирост 2500.0 млн руб., это 21 %."))
    result = explain(evidence(), Settings(enabled=True), client=client)
    assert result.source == "template-rejected"
    assert result.rejected_numbers == ("2500.0", "21")
    assert result.text == template_text(evidence())


def test_api_failure_does_not_break_the_cycle():
    client = _FakeClient(RuntimeError("нет сети"))
    result = explain(evidence(), Settings(enabled=True), client=client)
    assert result.source == "template-failed"
    assert not result.from_model


def test_the_reason_for_a_failed_call_is_kept():
    """Слой ни разу не работал на живом API: молчаливый откат к шаблону нечинибельный.

    «Шаблон, потому что выключено» и «шаблон, потому что модели с таким именем
    нет» выглядят на странице одинаково, а чинятся по-разному.
    """
    client = _FakeClient(RuntimeError("model: claude-opus-5 not found"))
    result = explain(evidence(), Settings(enabled=True), client=client)

    assert "RuntimeError" in result.failure
    assert "claude-opus-5 not found" in result.failure


def test_a_disabled_layer_reports_no_failure():
    """Выключенный слой — не отказ: причины у него нет и быть не должно."""
    assert explain(evidence(), Settings(enabled=False)).failure == ""


def test_a_long_api_message_is_cut_for_the_page():
    """Ответ API бывает в несколько абзацев, а на странице нужна причина."""
    client = _FakeClient(RuntimeError("подробности " * 100))
    result = explain(evidence(), Settings(enabled=True), client=client)

    assert len(result.failure) <= 300
    assert result.failure.endswith("…")


def test_temperature_is_never_sent():
    """На claude-opus-5 temperature отклоняется запросом; детерминизм даёт кэш."""
    client = _FakeClient(answer())
    explain(evidence(), Settings(enabled=True), client=client)
    sent = client.messages.calls[0]
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_facts_are_passed_to_the_model():
    client = _FakeClient(answer())
    explain(evidence(), Settings(enabled=True), client=client)
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "11851.8" in prompt
    assert "мощность системы ППД" in prompt


# --- воспроизводимость ------------------------------------------------------

def test_cached_explanation_is_reused_without_calling_the_model(tmp_path: Path):
    settings = Settings(enabled=True, cache_dir=tmp_path)
    client = _FakeClient(answer())

    first = explain(evidence(), settings, client=client)
    assert first.source == "llm"

    second = explain(evidence(), settings, client=client)
    assert second.source == "cache"
    assert second.text == first.text
    assert len(client.messages.calls) == 1, "повторный прогон обязан обойтись без вызова"


def test_cache_file_holds_the_explanation_and_nothing_else(tmp_path: Path):
    """В кэше лежит только объяснение: ни настроек, ни клиента, ни учётных данных.

    Кэш — единственное, что слой пишет на диск, и пишет он туда рядом с ключом
    API: `explain()` вызывается уже с построенным клиентом. Дамп «всего контекста
    на всякий случай» положил бы ключ в файл внутри рабочего дерева, а оттуда —
    в сдаваемый репозиторий (`BLOCKED.md`, п. 6: ключ читается только из
    окружения и нигде не сохраняется). Набор полей поэтому закреплён, а не
    подразумевается.
    """
    settings = Settings(enabled=True, cache_dir=tmp_path)
    explain(evidence(), settings, client=_FakeClient(answer()))

    stored = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))

    assert set(stored) == {"text", "model", "fingerprint"}


def test_rejected_answer_is_not_cached(tmp_path: Path):
    settings = Settings(enabled=True, cache_dir=tmp_path)
    client = _FakeClient(answer(confirmation="Прирост 2500.0 млн руб."))
    explain(evidence(), settings, client=client)
    assert list(tmp_path.glob("*.json")) == []


def test_changed_numbers_change_the_fingerprint():
    """Устаревшее объяснение не должно пережить изменение расчёта."""
    other = Evidence(
        subject=evidence().subject,
        observations=evidence().observations,
        numbers={**evidence().numbers, "прирост, млн руб.": 909.8},
        confirmed=True,
        caveats=evidence().caveats,
    )
    assert evidence().fingerprint() != other.fingerprint()


def test_fingerprint_is_stable_across_runs():
    assert evidence().fingerprint() == evidence().fingerprint()


# --- содержание объяснения --------------------------------------------------

@pytest.mark.parametrize("confirmed, expected", [(True, "полный расчёт"), (False, "не подтверждён")])
def test_template_states_the_confirmation_status(confirmed: bool, expected: str):
    facts = Evidence(subject="проверка", confirmed=confirmed)
    assert expected in template_text(facts)


def test_prompt_marks_an_unconfirmed_result_as_a_forecast():
    facts = Evidence(subject="проверка", confirmed=False)
    assert "ПРОГНОЗ" in facts.prompt()


def test_caveats_reach_both_the_prompt_and_the_template():
    facts = evidence()
    assert "мощность системы ППД" in facts.prompt()
    assert "мощность системы ППД" in template_text(facts)


def test_an_enabled_layer_that_cannot_reach_the_model_says_so_instead_of_looking_disabled(monkeypatch):
    """«Включён, но не дозвонился» и «выключен» — разные новости, а вид был один.

    Слой, которому не удалось построить клиента (нет библиотеки `anthropic`, нет
    учётных данных, испорченный профиль), возвращал `source="template"` с пустым
    `failure` — ровно то же, что при `enabled = false`. На странице цикла это
    печатается как «шаблоном — слой выключен в конфигурации либо недоступен», и
    оператор, который **явно** включил слой, отличить одно от другого не мог.

    Цена ошибки здесь конкретная: ключ API — последнее незакрытое требование
    допуска (`BLOCKED.md`, п. 6), и живой прогон планируется один. Молчаливый
    откат к шаблону в нём выглядел бы как успешный запуск. Найдено 20.08 на
    машине, где пакета `anthropic` действительно нет, хотя он есть в
    `requirements.txt`.

    **Учётные данные отбираются у теста намеренно (21.08 → 22.08).** С 22.08
    пакет `anthropic` на машине Б установлен, и `client=None` больше не упирается
    в отсутствующий импорт: клиент строится, а отказ приходит уже из вызова.
    Значит, на машине, где ключ **есть**, этот тест сходил бы к живой модели —
    настоящим запросом, за деньги, при каждом `pytest -q`. Живой вызов в проекте
    планируется ровно один и делается руками (`BLOCKED.md`, п. 6), поэтому здесь
    снимаются все источники учётных данных SDK, а `base_url` уводится на заведомо
    закрытый порт: если ключ всё же придёт из профиля на диске, запрос упрётся в
    отказ соединения, а не в API. Проверяемое поведение от этого не меняется —
    причина обязана быть названа при любой из причин.
    """
    for variable in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")

    result = explain(evidence(), Settings(enabled=True), client=None)

    assert result.source == "template-failed", "включённый слой выглядит как выключенный"
    assert result.failure, "причина не названа"
    assert not result.from_model
    assert "11851.8" in result.text, "шаблонный текст обязан остаться на месте"


def test_a_caveat_does_not_call_every_warning_a_range_violation():
    """Оговорка обязана называть то, что есть, а не одну из причин.

    В `CycleResult.warnings` попадают предупреждения трёх разных родов: выход за
    обучающий диапазон, решение, которого в обучающей выборке нет вовсе (запуск
    из простоя, перевод под закачку), и **нарушение физических проверок в самом
    прогнозе** — нефти больше жидкости. Оговорка называла их все «режимы выходят
    за обучающий диапазон».

    Тяжелее всех третий: физически невозможный прогноз означает, что модели
    нельзя верить вообще, а не что мы вышли за её область. Пряча его за ярлыком
    про диапазон, объяснение занижает то, о чём предупреждает.

    Классификатор причин сюда не переносится: он живёт в
    `scripts/surrogate_warnings.py` и покрыт своим тестом, а второй его экземпляр
    рано или поздно разойдётся с первым. Достаточно не врать: назвать число и
    показать пример настоящим текстом.
    """
    from aios.explain import evidence_from_cycle

    class Result:
        base_chdd_m = 1000.0
        chdd_m = 1100.0
        gain_m = 100.0
        actions: list = []
        rate_choices: list = []
        allocations: list = []
        problems: list = []
        confirmed = False
        warnings = [
            "2014-03: скв. 7: прогноз нарушает физические проверки — "
            "нефти больше жидкости (437.2 > 427.6 т)",
            "2014-01…2014-12: prod:5: множитель 2.10 вне обучающего диапазона 0.8..1.5",
        ]

    caveats = evidence_from_cycle(Result(), "линейный суррогат").caveats
    about_warnings = [item for item in caveats if "предупрежден" in item]

    assert about_warnings, caveats
    assert "выходят за обучающий диапазон" not in about_warnings[0], (
        "все предупреждения названы выходом за диапазон, включая нарушение физики"
    )
    assert "физические проверки" in about_warnings[0], (
        "пример настоящим текстом не показан — читателю не на что опереться"
    )


def test_a_well_number_quoted_from_the_subject_is_not_an_invented_figure():
    """Первый живой вызов 23.08 отвергся ровно на этом.

    Модель написала «Обводнённость скважины 42 составила 96.4 %» — ответ
    безупречный, но `42` не величина из расчёта, и проверка сочла его выдуманным.
    Число, которое мы сами написали словами в фактах, модель не выдумала, а
    прочитала.
    """
    evidence = Evidence(
        subject="остановка скважины 42 с 01.03.2019",
        numbers={"обводнённость, %": 96.4},
    )

    assert unverified_numbers("Обводнённость скважины 42 составила 96.4 %", evidence) == ()


def test_a_figure_quoted_from_an_observation_is_allowed_too():
    evidence = Evidence(
        subject="штуцирование",
        observations=("обводнённость выше 95 % шесть месяцев подряд",),
        numbers={"обводнённость, %": 96.4},
    )

    assert unverified_numbers("обводнённость держалась выше 95 %", evidence) == ()


def test_an_invented_figure_is_still_caught_after_the_relaxation():
    """Смягчение не должно превратиться в отсутствие проверки."""
    evidence = Evidence(
        subject="остановка скважины 42",
        observations=("обводнённость выше 95 %",),
        numbers={"обводнённость, %": 96.4},
    )

    assert unverified_numbers("экономия составила 17,3 млн руб.", evidence) == ("17,3",)


def test_a_digit_inside_a_unit_is_not_a_claimed_quantity():
    """Второй живой вызов 23.08 отвергся на тройке из «м3/сут».

    `Evidence.prompt()` печатает имя показателя вместе с единицей, модель
    повторяет её как есть — и цифра из единицы попадала в проверку как
    выдуманное число.
    """
    evidence = Evidence(
        subject="штуцирование скважины 7",
        numbers={"дебит жидкости, м3/сут": 48.0},
    )

    assert unverified_numbers("дебит жидкости — 48 м3/сут", evidence) == ()


def test_an_invented_number_survives_the_unit_relaxation():
    evidence = Evidence(
        subject="штуцирование скважины 7",
        numbers={"дебит жидкости, м3/сут": 48.0},
    )

    assert unverified_numbers("дебит вырастет до 61 м3/сут", evidence) == ("61",)


def test_a_conversion_is_named_with_its_cost_and_its_unmeasured_gain():
    """Перевод под закачку обязан прийти в объяснение отдельно от остановок.

    Его выигрыш даёт отклик соседей на новую нагнетательную, а суррогат такого
    отклика не предсказывает: в обучающих сценариях у переведённой скважины
    колонки закачки нет вовсе. Положить перевод в один ряд с остановками значит
    выдать за оценённое то, что не оценивалось (Б37).
    """
    from datetime import date as date_type

    from aios.explain import evidence_from_cycle
    from aios.loop import CycleResult
    from aios.policy import Action

    result = CycleResult(
        base_chdd_m=100.0, chdd_m=110.0,
        actions=(
            Action(date_type(2014, 3, 1), "вода", "convert", "обводнённость 99,5 %", 5.0),
            Action(date_type(2014, 5, 1), "убыток", "shut", "убыток 6 мес.", 1.0),
        ),
        rate_choices=(), allocations=(), steps=(), warnings=(), problems=(),
    )

    evidence = evidence_from_cycle(result, "тест")

    assert evidence.numbers["переводов под закачку"] == 1.0
    assert evidence.numbers["стоимость переводов, млн руб."] == 5.0
    assert any("convert — 1" in line and "shut — 1" in line
               for line in evidence.observations)
    assert any("суррогатом не оценивался" in line for line in evidence.caveats)


# --- сторонняя вода в объяснении ---------------------------------------------
# Трекер на КТ-2: «использование докачки попадает в объяснение агента». До 03.09
# слой молчал о ней вовсе — рассказывал про остановки и режимы так, будто воды в
# системе сколько угодно.


def _cycle_with_outside_water(allocations=()):
    from conftest import make_row

    from aios.loop import CycleResult
    from aios.summary import Fluids

    fluids = Fluids()
    rows = []
    for stamp, water_m3, injected_m3 in (("2014-01-01", 500.0, 600.0),
                                         ("2014-07-01", 500.0, 3_000.0)):
        oil_t = 10.0 * fluids.oil_density / 1000
        rows.append(make_row(stamp, "1", wlpr=(10.0 + water_m3) / 30, womr=oil_t / 30,
                             liquid=oil_t + water_m3 * fluids.water_density / 1000,
                             oil=oil_t))
        rows.append(make_row(stamp, "2", wwir=injected_m3 / 30, injection=injected_m3))

    return CycleResult(
        base_chdd_m=100.0, chdd_m=110.0, actions=(), rate_choices=(),
        allocations=tuple(allocations), steps=(), warnings=(), problems=(), rows=rows,
    )


def test_need_for_outside_water_reaches_the_explanation():
    """Потребность в сторонней воде — факт, а не служебная деталь планировщика."""
    from aios.explain import evidence_from_cycle

    evidence = evidence_from_cycle(_cycle_with_outside_water(), "тест")

    assert evidence.numbers["сторонней воды на пике, м³/сут"] > 0
    assert any("2014-07" in line for line in evidence.observations), evidence.observations


def test_unknown_intake_capacity_is_a_caveat_not_a_silence():
    """Мощность водозабора неизвестна — объяснение обязано это сказать.

    Иначе выходит худший из вариантов: система молча приняла режим, обеспеченный
    водой, которой, может быть, нет, и объяснила его не упомянув об этом.
    """
    from aios.explain import evidence_from_cycle

    evidence = evidence_from_cycle(_cycle_with_outside_water(), "тест")

    assert any("мощность водозабора неизвестна" in line for line in evidence.caveats)
    assert "мощность водозабора, м³/сут" not in evidence.numbers


def test_known_capacity_is_compared_and_the_shortfall_is_counted():
    from aios.explain import evidence_from_cycle
    from aios.planner import Limits

    evidence = evidence_from_cycle(_cycle_with_outside_water(), "тест",
                                   limits=Limits(max_external_water_m3_per_day=50.0))

    assert evidence.numbers["мощность водозабора, м³/сут"] == 50.0
    assert evidence.numbers["месяцев выше мощности водозабора"] == 1.0
    assert any("выше его мощности" in line for line in evidence.caveats)
    assert not any("неизвестна" in line for line in evidence.caveats)


def test_capacity_that_covers_the_need_leaves_no_caveat():
    from aios.explain import evidence_from_cycle
    from aios.planner import Limits

    evidence = evidence_from_cycle(_cycle_with_outside_water(), "тест",
                                   limits=Limits(max_external_water_m3_per_day=5_000.0))

    assert evidence.numbers["мощность водозабора, м³/сут"] == 5_000.0
    assert "месяцев выше мощности водозабора" not in evidence.numbers
    assert not any("водозабор" in line for line in evidence.caveats)


def test_injection_cut_by_the_source_is_named_as_such():
    """Срезка по источнику не «мероприятие» — в `actions` её нет, и без этого
    блока о ней не узнать ниоткуда.

    Текст решения берётся у планировщика дословно: разойдётся формулировка —
    упадёт этот тест, а не тихо исчезнет факт из объяснения.
    """
    from aios.explain import evidence_from_cycle
    from aios.planner import Allocation

    cut = Allocation(
        well="7", resource="закачка", before=100.0, after=40.0,
        reason="закачке нужно 812 м³/сут воды со стороны при мощности источника 50 м³/сут",
    )
    evidence = evidence_from_cycle(_cycle_with_outside_water([cut]), "тест")

    assert evidence.numbers["закачка урезана мощностью источника, скважин"] == 1.0
    assert any("мощностью водозабора" in line for line in evidence.observations)


def test_the_planner_really_says_what_the_explanation_looks_for():
    """Связка «планировщик → объяснение» проверяется настоящим прогоном срезки.

    Предыдущий тест берёт формулировку из литерала — то есть проверяет объяснение,
    но не то, что планировщик её произносит. Здесь она приходит из самого
    `planner.enforce`.
    """
    from datetime import date as date_type

    from conftest import make_row

    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.explain import evidence_from_cycle
    from aios.loop import CycleResult
    from aios.planner import Limits, enforce
    from aios.summary import Fluids

    fluids = Fluids()
    oil_t = 10.0 * fluids.oil_density / 1000
    forecast = [
        make_row("2014-07-01", "1", wlpr=(10.0 + 100.0) / 31, womr=oil_t / 31,
                 liquid=oil_t + 100.0 * fluids.water_density / 1000, oil=oil_t),
        make_row("2014-07-01", "2", wwir=3_000.0 / 31, injection=3_000.0),
    ]
    steps = [ScheduleStep(
        date_type(2014, 7, 1),
        (ProducerControl("1", "OPEN", "LRAT", 110.0 / 31, None),),
        (InjectorControl("2", "WATER", "OPEN", "RATE", 3_000.0 / 31, None),),
    )]
    limits = Limits(enforce_water_balance=True, water_balance_rule="external_water",
                    max_external_water_m3_per_day=10.0)

    _, decisions = enforce(steps, forecast, None, limits, fluids)

    assert decisions, "срезка по источнику не сработала — тест проверяет не то"
    result = CycleResult(
        base_chdd_m=100.0, chdd_m=110.0, actions=(), rate_choices=(),
        allocations=tuple(decisions), steps=(), warnings=(), problems=(), rows=forecast,
    )
    evidence = evidence_from_cycle(result, "тест", limits=limits, fluids=fluids)

    assert evidence.numbers["закачка урезана мощностью источника, скважин"] >= 1.0


# --- Е4: кэш демо-объяснений на клоне проверяющего ---------------------------
# ТЗ п. 3.1 требует языковую модель в ядре слоя, а на клоне организаторов нет ни
# ключа, ни сети. `explain()` читает кэш раньше, чем проверяет `enabled`, —
# значит, закоммиченный кэш показывает настоящий ответ модели без вызова.


def test_a_cached_answer_is_shown_even_with_the_layer_switched_off(tmp_path):
    """Кэш читается раньше `enabled` — на этом и стоит вся затея Е4.

    Без этого свойства закоммиченный кэш был бы бесполезен: у проверяющего слой
    выключен по умолчанию (`config.toml`, `[explain].enabled = false`), потому
    что учётных данных у него нет.
    """
    import json

    from aios.explain import Evidence, Settings, explain

    evidence = Evidence(subject="остановка скважины 42", numbers={"убыток, млн руб.": 6.0})
    settings = Settings(enabled=False, cache_dir=str(tmp_path))

    stored = {"text": "Скважина остановлена: убыток 6,0 млн руб.",
              "model": "yandexgpt/latest", "fingerprint": evidence.fingerprint()}
    (tmp_path / f"{evidence.fingerprint()}.json").write_text(
        json.dumps(stored, ensure_ascii=False), encoding="utf-8")

    answer = explain(evidence, settings)

    assert answer.source == "cache", "кэш не прочитан при выключенном слое"
    assert answer.text == stored["text"]
    assert answer.model == "yandexgpt/latest"


def test_a_committed_cache_entry_is_a_real_model_answer():
    """Закоммиченный кэш обязан нести имя модели и свой же отпечаток.

    Смысл кэша — показать проверяющему **ответ модели** без ключа. Файл,
    написанный руками или собранный шаблоном, выдал бы наш текст за ответ модели
    — то есть ровно то, что этот проект запрещает. `explain()` пишет кэш только
    на пути `llm`, уже после проверки чисел; тест стережёт, что в репозиторий
    попадает именно такой файл.

    Пока кэша нет — пропуск: сгенерировать его может только машина с ключом, и
    это записано открытым пунктом Е4.

    **Смотрим то, что в git, а не то, что на диске.** До 06.09 брались файлы из
    рабочего дерева, и на машине разработки тест проходил по **неотслеживаемому**
    файлу: там лежал живой ответ модели от 23.08, который решением Р-226
    запрещено коммитить (он мёртв по адресу). То есть сторож «закоммиченного
    кэша» судил о том, чего в репозитории нет, и его вердикт зависел от
    локального мусора — ровно болезнь, которую Р-161 лечил в другом месте.
    """
    import json
    import subprocess

    import pytest

    root = Path(__file__).resolve().parent.parent
    listed = subprocess.run(
        ["git", "ls-files", "artifacts/explanations"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    cached = [root / name for name in listed.stdout.split()
              if name.endswith(".json")] if listed.returncode == 0 else []
    if not cached:
        pytest.skip("кэша демо-объяснений в репозитории нет — Е4 остаётся открытым")

    for path in cached:
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored.get("text", "").strip(), f"{path.name}: пустой текст"
        assert stored.get("model", "").strip(), (
            f"{path.name}: нет имени модели — такой файл неотличим от шаблона"
        )
        assert stored.get("fingerprint") == path.stem, (
            f"{path.name}: отпечаток не совпадает с именем файла — кэш не от этих фактов"
        )


# --- выбор поставщика: `config.toml` называет рабочим yandex -------------------


def test_the_configured_provider_is_the_one_that_gets_asked(monkeypatch):
    """Слой обращается к тому поставщику, который назван в конфигурации.

    `config.toml` называет рабочим `yandex`, а умолчание кода — `anthropic`.
    Адаптер Яндекса покрыт своими тестами, но **выбор** между поставщиками через
    `explain()` до 05.09 не проверялся: опечатка или подмена умолчания увела бы
    запрос к другому вендору и другому счёту молча.
    """
    from aios import explain as explain_module

    asked = []

    def fake_yandex(folder_id):
        asked.append(("yandex", folder_id))
        return None, "проверка: клиент не создаётся"

    monkeypatch.setattr("aios.yandex.from_environment", fake_yandex)

    result = explain_module.explain(
        evidence(), Settings(enabled=True, provider="yandex", folder_id="каталог"))

    assert asked == [("yandex", "каталог")], (
        f"спрошен не тот поставщик: {asked}"
    )
    assert result.source == "template-failed"
    assert "проверка" in (result.failure or ""), "причина поставщика потеряна"


def test_an_unknown_provider_is_refused_not_silently_replaced():
    """Опечатка в имени поставщика — отказ, а не тихий откат на Anthropic.

    Молчаливая подмена увела бы запрос к другому вендору: другой счёт, другая
    политика данных, другой ответ.
    """
    from aios import explain as explain_module

    result = explain_module.explain(
        evidence(), Settings(enabled=True, provider="yandxe"))

    assert result.source == "template-failed"
    assert "yandxe" in (result.failure or ""), "имя опечатки не названо"
    assert "anthropic" in (result.failure or "").lower(), "не сказано, что допустимо"


def test_a_text_claiming_a_full_run_that_did_not_happen_is_rejected():
    """Статус — не число, и барьер чисел его не видел вовсе.

    Ответ, где прирост подписан «подтверждено полным расчётом ГГДМ» при
    `confirmed = False`, проходил обе проверки: числа в нём могли быть
    процитированы из фактов, а слово «подтверждено» числом не является. Между тем
    это ровно то утверждение, из-за которого расхождение заявленного ЧДД с
    пересчётом организаторов становится дисквалификацией — прогноз суррогата,
    поданный как факт ГГДМ (Р-51, ревизия ночи 08→09.09).
    """
    from aios.explain import Evidence, false_confirmation

    facts = Evidence(subject="остановка скважины 42", observations=(),
                     numbers={"прирост ЧДД, млн руб.": 681.16}, caveats=(),
                     confirmed=False)

    assert false_confirmation("Результат подтверждён полным расчётом ГГДМ.", facts)
    assert false_confirmation("Прирост получен полным расчётом.", facts)
    assert false_confirmation("Это прогноз суррогата, ГГДМ не считалась.", facts) is None

    # когда расчёт действительно был, называть его — не ложь, а обязанность
    confirmed = Evidence(subject="x", observations=(), numbers={}, caveats=(),
                         confirmed=True)
    assert false_confirmation("Подтверждено полным расчётом ГГДМ.", confirmed) is None
