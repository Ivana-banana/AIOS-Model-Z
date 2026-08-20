"""Слой интерпретируемости: языковая модель формулирует, но не считает.

Главная проверяемая гарантия — число, которого нет в расчёте, не может попасть в
объяснение. Проект уже получал убедительное физическое обоснование под неверный
результат (форсаж, 16.08), поэтому «звучит правдоподобно» здесь не аргумент.

Сеть в тестах не трогается: клиент подставной.
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


def test_an_enabled_layer_that_cannot_reach_the_model_says_so_instead_of_looking_disabled():
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
    """
    result = explain(evidence(), Settings(enabled=True), client=None)

    assert result.source == "template-failed", "включённый слой выглядит как выключенный"
    assert result.failure, "причина не названа"
    assert not result.from_model
    assert "11851.8" in result.text, "шаблонный текст обязан остаться на месте"
