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
