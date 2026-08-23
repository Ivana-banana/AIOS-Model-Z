"""Адаптер Yandex Foundation Models: сеть не трогается, проверяется контракт.

Живой вызов — отдельный шаг и отдельная оговорка в отчёте. Здесь проверяется то,
что от адаптера требуется независимо от сети: он выглядит как клиент Anthropic,
собирает верное тело запроса, не даёт ключу утечь в конфиг и отличает отказ от
пустого ответа.
"""

from __future__ import annotations

import json

import pytest

from aios import explain as explain_module
from aios import yandex


class Recorder:
    """Подменяет сетевой вызов и запоминает, что именно ушло бы на сервер."""

    def __init__(self, answer: dict):
        self.answer = answer
        self.payload: dict | None = None

    def __call__(self, payload: dict) -> dict:
        self.payload = payload
        return self.answer


def answer(text: str) -> dict:
    return {"result": {"alternatives": [{"message": {"role": "assistant", "text": text}}],
                       "modelVersion": "23.08.2026"}}


def client_with(recorder: Recorder) -> yandex.YandexClient:
    made = yandex.YandexClient(api_key="ключ", folder_id="каталог")
    object.__setattr__(made.messages, "_post", recorder)
    return made


# ── адрес модели ─────────────────────────────────────────────────────────────

def test_a_short_model_name_becomes_a_full_uri():
    assert yandex.model_uri("yandexgpt/latest", "b1g") == "gpt://b1g/yandexgpt/latest"


def test_a_ready_uri_is_passed_through_untouched():
    """Дообученная модель адресуется `ds://…` — оборачивать её нельзя."""
    assert yandex.model_uri("ds://abc/xyz", "b1g") == "ds://abc/xyz"


# ── тело запроса ─────────────────────────────────────────────────────────────

def test_the_request_carries_the_model_uri_and_the_folder():
    recorder = Recorder(answer('{"ok": 1}'))

    client_with(recorder).messages.create(model="yandexgpt/latest", messages=[
        {"role": "user", "content": "факты"}])

    assert recorder.payload["modelUri"] == "gpt://каталог/yandexgpt/latest"
    assert recorder.payload["completionOptions"]["stream"] is False


def test_the_temperature_is_zero_because_the_explanation_must_repeat():
    recorder = Recorder(answer("{}"))

    client_with(recorder).messages.create(model="m", messages=[{"role": "user", "content": "x"}])

    assert recorder.payload["completionOptions"]["temperature"] == 0.0


def test_max_tokens_goes_as_a_string_because_the_api_wants_it_so():
    recorder = Recorder(answer("{}"))

    client_with(recorder).messages.create(model="m", max_tokens=512,
                                          messages=[{"role": "user", "content": "x"}])

    assert recorder.payload["completionOptions"]["maxTokens"] == "512"


def test_the_json_schema_reaches_the_model_as_a_requirement(monkeypatch):
    """Серверной гарантии формата у этого API нет, поэтому схема идёт текстом."""
    recorder = Recorder(answer('{"why": "потому что"}'))
    schema = {"type": "object", "properties": {"why": {"type": "string"}}}

    client_with(recorder).messages.create(
        model="m", system="ты объясняешь решения",
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": "факты"}],
    )

    system = recorder.payload["messages"][0]
    assert system["role"] == "system"
    assert "ты объясняешь решения" in system["text"]
    # образец объекта, а не сама схема: на схему модель отвечает схемой
    assert '"why": "<текст>"' in system["text"]
    assert "Запрещено: возвращать описание схемы" in system["text"]
    assert '"properties"' not in system["text"], "схему целиком отдавать нельзя"


def test_anthropic_only_options_are_ignored_rather_than_sent():
    """`effort` у Яндекса соответствия не имеет; падать из-за него нельзя."""
    recorder = Recorder(answer("{}"))

    client_with(recorder).messages.create(
        model="m", output_config={"effort": "low"},
        messages=[{"role": "user", "content": "x"}],
    )

    assert "effort" not in json.dumps(recorder.payload)


def test_block_content_is_flattened_the_way_anthropic_sends_it():
    recorder = Recorder(answer("{}"))

    client_with(recorder).messages.create(model="m", messages=[
        {"role": "user", "content": [{"type": "text", "text": "первое"},
                                     {"type": "text", "text": "второе"}]}])

    assert recorder.payload["messages"][-1]["text"] == "первое\nвторое"


# ── ответ ────────────────────────────────────────────────────────────────────

def test_the_answer_looks_like_an_anthropic_response():
    """`aios.explain._first_text` ходит по `.content` и ищет блок с `type == text`."""
    made = client_with(Recorder(answer('{"why": "потому что"}')))

    response = made.messages.create(model="m", messages=[{"role": "user", "content": "x"}])

    assert explain_module._first_text(response) == '{"why": "потому что"}'


def test_a_fenced_answer_is_unwrapped_instead_of_being_lost():
    """Требование «без обёртки» стоит в запросе, но полагаться на него нельзя."""
    made = client_with(Recorder(answer('```json\n{"why": "потому что"}\n```')))

    response = made.messages.create(model="m", messages=[{"role": "user", "content": "x"}])

    assert json.loads(explain_module._first_text(response)) == {"why": "потому что"}


def test_an_empty_answer_is_a_refusal_not_an_empty_explanation():
    made = client_with(Recorder(answer("   ")))

    with pytest.raises(ValueError, match="пустой текст"):
        made.messages.create(model="m", messages=[{"role": "user", "content": "x"}])


def test_no_alternatives_is_a_refusal():
    made = client_with(Recorder({"result": {"alternatives": []}}))

    with pytest.raises(ValueError, match="без альтернатив"):
        made.messages.create(model="m", messages=[{"role": "user", "content": "x"}])


# ── учётные данные ───────────────────────────────────────────────────────────

def test_the_key_is_taken_only_from_the_environment(monkeypatch):
    monkeypatch.delenv(yandex.API_KEY_ENV, raising=False)

    made, why = yandex.from_environment("каталог")

    assert made is None
    assert yandex.API_KEY_ENV in why
    assert "config.toml" in why, "причина обязана называть, где ключу НЕ место"


def test_a_missing_folder_is_named_separately_because_it_is_fixed_differently(monkeypatch):
    monkeypatch.setenv(yandex.API_KEY_ENV, "ключ")
    monkeypatch.delenv(yandex.FOLDER_ENV, raising=False)

    made, why = yandex.from_environment("")

    assert made is None
    assert "каталог" in why


def test_the_environment_folder_wins_over_the_config(monkeypatch):
    monkeypatch.setenv(yandex.API_KEY_ENV, "ключ")
    monkeypatch.setenv(yandex.FOLDER_ENV, "из-окружения")

    made, why = yandex.from_environment("из-конфига")

    assert why == ""
    assert made is not None and made.folder_id == "из-окружения"


# ── выбор поставщика ─────────────────────────────────────────────────────────

def test_an_unknown_provider_is_refused_rather_than_falling_back(monkeypatch):
    """Опечатка в конфиге не должна молча увести запрос к другому вендору."""
    settings = explain_module.Settings(enabled=True, provider="opneai")

    made, why = explain_module._client_for(settings)

    assert made is None
    assert "неизвестный поставщик" in why


def test_the_yandex_provider_reaches_the_yandex_client(monkeypatch):
    monkeypatch.setenv(yandex.API_KEY_ENV, "ключ")
    monkeypatch.setenv(yandex.FOLDER_ENV, "каталог")
    settings = explain_module.Settings(enabled=True, provider="yandex")

    made, why = explain_module._client_for(settings)

    assert why == ""
    assert isinstance(made, yandex.YandexClient)


def test_a_model_echoing_the_schema_is_a_refusal_not_an_empty_explanation():
    """Ловушка первого живого вызова 23.08, закреплённая навсегда.

    YandexGPT вернул саму JSON Schema, вписав содержание в поля `description`.
    `json.loads` такой ответ принимает, нужных ключей в нём нет — и слой выдавал
    пустой текст с источником `llm`, то есть выглядел работающим.
    """
    echoed = json.dumps({
        "type": "object",
        "properties": {"why": {"type": "string", "description": "Потому что обводнённость"}},
        "required": ["why"],
    }, ensure_ascii=False)
    made = client_with(Recorder(answer(echoed)))

    with pytest.raises(ValueError, match="описание схемы, а не объект"):
        made.messages.create(
            model="m",
            output_config={"format": {"schema": {"required": ["why", "what_happened"]}}},
            messages=[{"role": "user", "content": "x"}],
        )


def test_a_proper_object_passes_the_echo_check():
    made = client_with(Recorder(answer('{"why": "потому что", "what_happened": "вода"}')))

    response = made.messages.create(
        model="m",
        output_config={"format": {"schema": {"required": ["why", "what_happened"]}}},
        messages=[{"role": "user", "content": "x"}],
    )

    assert json.loads(explain_module._first_text(response))["why"] == "потому что"
