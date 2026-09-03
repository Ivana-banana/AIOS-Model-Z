"""Клиент Yandex Foundation Models в том же контракте, что Anthropic.

Зачем он вообще. Слой интерпретируемости (`aios/explain.py`) написан под
Anthropic: `client.messages.create(model=…, system=…, output_config=…)`. Ключ у
нас появился не от Anthropic, а от Yandex AI Studio, и переписывать под него весь
слой значило бы привязаться к следующему поставщику так же жёстко. Поэтому здесь
адаптер: снаружи он выглядит как клиент Anthropic, внутри ходит в REST Яндекса.

**Зависимостей не добавляет.** Один POST с JSON делается `urllib.request` из
стандартной библиотеки. Это осознанный выбор после Р-79: мы только что закрепили
39 пакетов хэшами и разобрали лицензии каждого — тянуть ради одного запроса
`openai` или `yandex-cloud-ml-sdk` со своим деревом зависимостей значит заново
открыть то, что закрыли.

**Что знать про API.** Эндпоинт `foundationModels/v1/completion`, авторизация
заголовком `Authorization: Api-Key <ключ>`, модель адресуется URI вида
`gpt://<folder-id>/yandexgpt/latest`. Идентификатор каталога не секрет и живёт в
конфиге; ключ читается **только** из окружения и в конфиг попасть не может.

**Чего адаптер не делает.** Он не изображает структурированный вывод там, где его
нет: `output_config` с JSON-схемой у Anthropic — гарантия формата, у Яндекса на
момент написания в этом API её нет. Поэтому схема уходит в текст запроса
требованием, а ответ разбирается и проверяется на нашей стороне; если модель
вернула не JSON, это отказ с названной причиной, а не «почти подошло».
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
API_KEY_ENV = "YANDEX_API_KEY"
FOLDER_ENV = "YANDEX_FOLDER_ID"
DEFAULT_MODEL = "yandexgpt/latest"
TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class TextBlock:
    """Блок ответа в форме, которую ожидает `aios.explain._first_text`."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class Response:
    """Ответ в форме клиента Anthropic: список блоков в `content`."""

    content: list[TextBlock]
    model_version: str = ""


def model_uri(model: str, folder_id: str) -> str:
    """Собрать `gpt://<folder>/<model>`. Готовый URI пропускается как есть."""
    if model.startswith(("gpt://", "ds://", "emb://")):
        return model
    return f"gpt://{folder_id}/{model}"


@dataclass(frozen=True)
class Messages:
    """Подобъект `client.messages` — ровно один метод, который зовёт слой."""

    api_key: str
    folder_id: str
    endpoint: str = ENDPOINT
    timeout: float = TIMEOUT_SECONDS

    def create(self, *, model: str, max_tokens: int = 1024,
               system: str = "", messages: list[dict[str, Any]] | None = None,
               output_config: dict[str, Any] | None = None,
               **_ignored: Any) -> Response:
        """Один запрос к модели. Подпись — как у Anthropic, чтобы слой не менялся.

        `output_config` принимается и **используется**, но иначе: JSON-схема
        оттуда уходит в системную часть запроса требованием к формату, потому что
        серверной гарантии формата этот API не даёт. Незнакомые поля Anthropic
        (`effort` и прочие) молча игнорируются — это не потеря смысла, у Яндекса
        им нет соответствия.
        """
        schema = ((output_config or {}).get("format") or {}).get("schema")
        instructions = (system or "") + (_schema_instruction(schema) if schema else "")

        payload = {
            "modelUri": model_uri(model, self.folder_id),
            "completionOptions": {
                "stream": False,
                # температура ноль: объяснение обязано быть воспроизводимым, а
                # разнообразие формулировок здесь не ценность
                "temperature": 0.0,
                "maxTokens": str(max_tokens),
            },
            "messages": (
                ([{"role": "system", "text": instructions}] if instructions else [])
                + [
                    {"role": str(one.get("role", "user")), "text": _content_text(one)}
                    for one in (messages or [])
                ]
            ),
        }

        raw = self._post(payload)
        alternatives = (raw.get("result") or {}).get("alternatives") or []
        if not alternatives:
            raise ValueError("Yandex вернул ответ без альтернатив")
        text = ((alternatives[0].get("message") or {}).get("text") or "").strip()
        if not text:
            raise ValueError("Yandex вернул пустой текст ответа")
        body = _unfenced(text)
        if schema:
            _refuse_if_schema_echoed(body, schema)
        return Response(
            content=[TextBlock(body)],
            model_version=str((raw.get("result") or {}).get("modelVersion", "")),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}",
                "x-folder-id": self.folder_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                return json.loads(answer.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            # тело ошибки Яндекса содержит причину; без него «HTTP 400» не чинится
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise ValueError(f"Yandex ответил {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ValueError(f"Yandex недоступен: {error.reason}") from error


@dataclass(frozen=True)
class YandexClient:
    """Клиент в контракте `aios.explain.MessagesClient`."""

    api_key: str
    folder_id: str
    endpoint: str = ENDPOINT
    messages: Messages = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "messages",
            Messages(self.api_key, self.folder_id, self.endpoint),
        )


def from_environment(folder_id: str = "") -> tuple[YandexClient | None, str]:
    """Клиент из окружения — или причина, названная словами.

    Ключ берётся **только** из переменной окружения: в конфиге ему не место, и
    это не стиль, а то, что мешает ему уехать в git вместе с настройками.
    Идентификатор каталога не секрет, поэтому приходит из конфига, а окружение
    его лишь переопределяет.
    """
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        return None, (
            f"нет ключа: переменная {API_KEY_ENV} не задана. Ключ читается только "
            "из окружения — в config.toml его быть не должно"
        )
    folder = os.environ.get(FOLDER_ENV, "").strip() or folder_id.strip()
    if not folder:
        return None, (
            f"не задан каталог Yandex Cloud: ни {FOLDER_ENV} в окружении, "
            "ни folder_id в [explain] конфига"
        )
    return YandexClient(api_key=key, folder_id=folder), ""


def _schema_instruction(schema: dict[str, Any]) -> str:
    """Требование формата — образцом, а не самой схемой.

    Первый живой вызов 23.08 показал ловушку: если отдать модели JSON Schema и
    попросить «ответ по схеме», YandexGPT возвращает **саму схему**, аккуратно
    вписав содержание в поля `description`. Формально это ответ «по схеме», и
    разбор его принимает: `json.loads` проходит, нужных ключей нет — слой
    объяснений выдал пустой текст с источником `llm`, то есть выглядел
    работающим.

    Поэтому здесь показывается **образец объекта** с нужными ключами и прямо
    сказано, чего делать нельзя. Схема остаётся рядом описанием полей, но
    перестаёт быть тем, что модель копирует.
    """
    fields = schema.get("properties", {})
    keys = list(schema.get("required") or fields)
    template = dict.fromkeys(keys, "<текст>")
    described = "\n".join(
        f"- {key}: {fields.get(key, {}).get('description', 'строка')}" for key in keys
    )
    return (
        "\n\nВерни РОВНО один объект JSON вот такого вида, заполнив значения:\n"
        + json.dumps(template, ensure_ascii=False, indent=2)
        + "\n\nЧто означает каждое поле:\n" + described
        + "\n\nЗапрещено: возвращать описание схемы (объект с ключами type, "
          "properties, required), оборачивать ответ в markdown, писать что-либо "
          "до или после объекта. Значения всех полей — строки на русском языке."
    )


def _refuse_if_schema_echoed(body: str, schema: dict[str, Any]) -> None:
    """Отказать, если модель вернула схему вместо объекта по ней.

    Без этой проверки отказ выглядит как успех: `source` остаётся `llm`, а текст
    пустой. Пустое объяснение, поданное как объяснение, хуже честного отказа — на
    объяснимости стоит отдельный критерий регламента.
    """
    try:
        parsed = json.loads(body)
    except ValueError:
        return  # не JSON вовсе — это поймает вызывающий, у него свой разбор
    if not isinstance(parsed, dict):
        return
    required = set(schema.get("required") or schema.get("properties", {}))
    if required and not required & set(parsed):
        raise ValueError(
            "модель вернула описание схемы, а не объект по ней: в ответе нет ни "
            f"одного из полей {sorted(required)}"
        )


def _content_text(message: dict[str, Any]) -> str:
    """Текст сообщения. Anthropic допускает и строку, и список блоков."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type", "text") == "text"
    )


def _unfenced(text: str) -> str:
    """Снять markdown-обёртку, если модель всё же её поставила.

    Требование «без обёртки» в запросе стоит, но полагаться на его исполнение
    нельзя: серверной гарантии формата у этого API нет, а падать из-за трёх
    обратных кавычек — терять ответ, который на самом деле верен.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return body.rsplit("```", 1)[0].strip()
