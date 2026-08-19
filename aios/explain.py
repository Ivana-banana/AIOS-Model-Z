"""Слой интерпретируемости: языковая модель объясняет решение цикла.

Требование ТЗ (п. 3.1): «решение должно содержать слой интерпретируемости/агентов,
имеющих в ядре языковую (мультимодальную) модель». Это условие комплектности
решения, а не критерий качества презентации, поэтому слой обязан работать в
сдаваемом пакете и деградировать предсказуемо, когда ключа API нет.

**Главное ограничение, вокруг которого построен модуль: языковая модель
формулирует объяснение, но не принимает решение и не производит чисел.**

Все величины приходят из расчёта — маржа скважины, прогноз суррогата, ЧДД
эталонного расчётчика — и передаются модели готовым набором фактов. Сгенерированный
текст затем проверяется: каждое число в нём обязано совпасть с числом из набора,
иначе объяснение отвергается целиком и в ход идёт шаблон. Проверка не косметическая
— 16.08 проект уже получил убедительное физическое обоснование под неверный
результат форсажа, и поймал ошибку контрольный прогон, а не рассуждение.

Воспроизводимость. На `claude-opus-5` параметр `temperature` не принимается вовсе
(запрос с ним отклоняется), то есть **сделать вывод модели детерминированным
нечем**. Регламент требует отсутствия неконтролируемых случайных параметров,
поэтому воспроизводится не вызов, а артефакт: объяснение кэшируется на диск по
отпечатку набора фактов, и повторный прогон берёт сохранённый текст. Отпечаток
считается по фактам, а не по тексту, — изменились числа, изменится и объяснение.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

#: Модель по умолчанию. Меняется через конфиг, а не правкой кода.
DEFAULT_MODEL = "claude-opus-5"

#: Ответ модели ограничен схемой: свободный текст сложнее проверять и цитировать.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "what_happened": {
            "type": "string",
            "description": "Что увидела система в состоянии фонда. Одно-два предложения.",
        },
        "why": {
            "type": "string",
            "description": "Почему принято именно такое решение. Одно-два предложения.",
        },
        "confirmation": {
            "type": "string",
            "description": "Чем решение подтверждено и что этим НЕ доказано.",
        },
    },
    "required": ["what_happened", "why", "confirmation"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
Ты — слой интерпретируемости многоагентной системы управления фондом нефтяных
скважин. Твоя задача — объяснить инженеру уже принятое решение на русском языке.

Жёсткие правила:

1. Ты НЕ принимаешь решений и НЕ считаешь. Решение уже принято, числа уже
   посчитаны и переданы тебе в блоке фактов.
2. Используй ТОЛЬКО числа из блока фактов, дословно в том виде, как они там
   записаны. Не округляй, не пересчитывай, не выводи новых чисел (в том числе
   процентов и разностей), даже если это кажется очевидным.
3. Если для объяснения не хватает факта — так и напиши словами, без числа.
4. Разделяй прогноз и подтверждённый результат. Прогноз суррогата — не результат.
5. Пиши сжато и по существу, без вводных оборотов и без похвалы решению.
"""


class MessagesClient(Protocol):
    """Минимум от клиента Anthropic, который нужен модулю (и который легко подменить)."""

    @property
    def messages(self) -> Any: ...


@dataclass(frozen=True)
class Evidence:
    """Набор фактов о решении: всё, на что модели разрешено ссылаться.

    `numbers` — единственный источник чисел в объяснении. Ключ читается человеком
    и попадает в подсказку, значение проверяется в сгенерированном тексте.
    """

    subject: str
    """Что объясняем: «остановка скважины 42», «режимы фонда за 2019 год»."""

    observations: tuple[str, ...] = ()
    """Что система увидела — текстом, без чисел."""

    numbers: dict[str, float] = field(default_factory=dict)
    """Посчитанные величины: имя → значение. Только отсюда берутся числа."""

    confirmed: bool = False
    """Подтверждено ли полным расчётом ГГДМ."""

    caveats: tuple[str, ...] = ()
    """Что этим не доказано. Идёт в объяснение обязательно."""

    def fingerprint(self) -> str:
        """Отпечаток фактов — ключ кэша и признак «объяснение устарело»."""
        payload = json.dumps(
            {
                "subject": self.subject,
                "observations": list(self.observations),
                # округление до 6 знаков: шум последнего бита не должен плодить кэш
                "numbers": {k: round(v, 6) for k, v in sorted(self.numbers.items())},
                "confirmed": self.confirmed,
                "caveats": list(self.caveats),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def prompt(self) -> str:
        """Блок фактов для модели. Числа даются готовыми строками — их и цитируем."""
        lines = [f"РЕШЕНИЕ: {self.subject}", ""]
        if self.observations:
            lines.append("ЧТО УВИДЕЛА СИСТЕМА:")
            lines += [f"- {item}" for item in self.observations]
            lines.append("")
        if self.numbers:
            lines.append("ПОСЧИТАННЫЕ ВЕЛИЧИНЫ (использовать только их):")
            lines += [f"- {name}: {_format(value)}" for name, value in self.numbers.items()]
            lines.append("")
        status = (
            "подтверждено полным расчётом ГГДМ"
            if self.confirmed
            else "ПРОГНОЗ суррогата, полным расчётом не подтверждено"
        )
        lines += [f"СТАТУС: {status}", ""]
        if self.caveats:
            lines.append("ЧТО ЭТИМ НЕ ДОКАЗАНО (упомянуть обязательно):")
            lines += [f"- {item}" for item in self.caveats]
        return "\n".join(lines).strip()


@dataclass(frozen=True)
class Explanation:
    """Объяснение и его происхождение — второе для отчёта важнее первого."""

    text: str
    source: str
    """`llm`, `cache`, `template`, `template-rejected` или `template-failed`."""

    fingerprint: str
    model: str = ""
    rejected_numbers: tuple[str, ...] = ()
    """Числа, которых не было в фактах. Непусто только при отказе от ответа модели."""

    failure: str = ""
    """Почему вызов не состоялся. Непусто только при `template-failed`.

    Слой не роняет цикл, но и молчать о причине не должен: без неё «шаблон»
    из-за выключённой настройки и «шаблон» из-за неверного имени модели или
    просроченного ключа выглядят одинаково, а чинятся по-разному.
    """

    @property
    def from_model(self) -> bool:
        return self.source in {"llm", "cache"}


@dataclass(frozen=True)
class Settings:
    """Настройки слоя. Живут в `config.toml`, а не в коде."""

    enabled: bool = False
    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    effort: str = "low"
    """Объяснение готового решения — не та задача, где нужна глубина рассуждения."""

    cache_dir: Path | None = None


#: Числа в тексте: 1 234,5 / 1234.5 / -12 / +15,1 %
_NUMBER_RE = re.compile(r"[-+]?\d[\d  ]*(?:[.,]\d+)?")

#: Величины, которые не являются «числом из фактов»: годы и номера месяцев в датах.
_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{2}\.\d{4}\b")


def _format(value: float) -> str:
    """Единый формат числа: как передали модели, так и ищем в ответе."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.1f}"


def _parse(token: str) -> float | None:
    cleaned = token.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def unverified_numbers(text: str, evidence: Evidence) -> tuple[str, ...]:
    """Числа из текста, которых нет в фактах.

    Это и есть защита от выдуманной цифры: модель физически не может подставить
    число, которого мы не посчитали, — оно будет замечено и ответ отвергнут.

    Даты не считаются числами: `2019-06` — это ссылка на шаг, а не величина.
    Допуск сравнения — половина последнего знака нашего же формата, потому что
    модель цитирует ровно то, что мы ей дали.
    """
    allowed = {round(value, 1) for value in evidence.numbers.values()}
    # целые части тоже законны: «выросла на 402 тыс. т» при факте 402.0
    allowed |= {float(int(value)) for value in evidence.numbers.values()}

    stripped = _DATE_RE.sub(" ", text)
    unknown: list[str] = []
    for token in _NUMBER_RE.findall(stripped):
        value = _parse(token)
        if value is None:
            continue
        if any(abs(value - candidate) <= 0.05 for candidate in allowed):
            continue
        unknown.append(token.strip())
    return tuple(dict.fromkeys(unknown))


def template_text(evidence: Evidence) -> str:
    """Шаблонное объяснение — то, что было до слоя и осталось запасным вариантом.

    Оно всегда доступно, всегда одинаково и не требует сети. Слой на языковой
    модели обязан быть лучше него, а не заменять его собой.
    """
    lines = [evidence.subject]
    if evidence.observations:
        lines += ["", "Что увидели:"]
        lines += [f"  - {item}" for item in evidence.observations]
    if evidence.numbers:
        lines += ["", "Посчитано:"]
        lines += [f"  - {name}: {_format(value)}" for name, value in evidence.numbers.items()]
    lines += [
        "",
        "Подтверждение: "
        + ("полный расчёт ГГДМ" if evidence.confirmed else "прогноз суррогата, не подтверждён"),
    ]
    if evidence.caveats:
        lines += ["", "Этим не доказано:"]
        lines += [f"  - {item}" for item in evidence.caveats]
    return "\n".join(lines)


def evidence_from_cycle(result: Any, surrogate_name: str = "") -> Evidence:
    """Собрать набор фактов из итога цикла (`aios.loop.CycleResult`).

    Числа берутся ровно те, что посчитал цикл, — модели остаётся их изложить.
    Оговорки собираются здесь же, а не в подсказке: неподтверждённый прирост,
    выход за обучающий диапазон суррогата и нарушения ограничений обязаны попасть
    в объяснение, даже если модель сочтёт их незначительными.
    """
    numbers = {
        "ЧДД базового сценария, млн руб.": result.base_chdd_m,
        "ЧДД после решений, млн руб.": result.chdd_m,
        "прирост, млн руб.": result.gain_m,
        "решений об остановке": float(len(result.actions)),
        "скважин с подобранным режимом": float(len(result.rate_choices)),
    }
    if result.allocations:
        numbers["срезано планировщиком, скважин"] = float(len(result.allocations))

    observations = []
    if surrogate_name:
        observations.append(f"прогноз получен источником: {surrogate_name}")
    if result.actions:
        kinds = sorted({action.kind for action in result.actions})
        observations.append("типы принятых решений: " + ", ".join(kinds))

    caveats = []
    if not result.confirmed:
        caveats.append(
            "прирост посчитан по прогнозу суррогата и полным расчётом ГГДМ не подтверждён"
        )
    if result.warnings:
        caveats.append(
            f"режимы выходят за обучающий диапазон суррогата: предупреждений "
            f"{len(result.warnings)}"
        )
        numbers["предупреждений о выходе за диапазон"] = float(len(result.warnings))
    if result.problems:
        caveats.append(
            f"расписание нарушает ограничения ({len(result.problems)}) и организаторам "
            "не отправляется"
        )
        numbers["нарушений ограничений"] = float(len(result.problems))

    return Evidence(
        subject="цикл управления фондом: подбор режимов и остановка убыточных скважин",
        observations=tuple(observations),
        numbers=numbers,
        confirmed=bool(result.confirmed),
        caveats=tuple(caveats),
    )


def _cache_path(settings: Settings, fingerprint: str) -> Path | None:
    if settings.cache_dir is None:
        return None
    return Path(settings.cache_dir) / f"{fingerprint}.json"


def _compose(payload: dict[str, str]) -> str:
    parts = [payload.get("what_happened", ""), payload.get("why", ""), payload.get("confirmation", "")]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def explain(evidence: Evidence,
            settings: Settings,
            *,
            client: MessagesClient | None = None) -> Explanation:
    """Объяснить решение: кэш → языковая модель → проверка чисел → шаблон.

    Порядок отказов важен и весь наблюдаем по `Explanation.source`:

    * `cache` — объяснение уже генерировалось для этих же фактов, взято с диска.
      Это и есть воспроизводимость: детерминизма у вызова нет, у прогона есть;
    * `llm` — сгенерировано сейчас и прошло проверку чисел;
    * `template-rejected` — модель ввела число, которого нет в фактах. Ответ
      отброшен целиком, а не отредактирован: если модель ошиблась в одном числе,
      доверия к остальному тексту нет;
    * `template` — слой выключен, ключа нет, библиотеки нет или запрос не удался.

    Отказ языковой модели никогда не роняет цикл: объяснение — это отчётность о
    решении, а не само решение.
    """
    fingerprint = evidence.fingerprint()

    cached = _cache_path(settings, fingerprint)
    if cached is not None and cached.is_file():
        stored = json.loads(cached.read_text(encoding="utf-8"))
        return Explanation(
            text=stored["text"],
            source="cache",
            fingerprint=fingerprint,
            model=stored.get("model", ""),
        )

    if not settings.enabled:
        return Explanation(text=template_text(evidence), source="template", fingerprint=fingerprint)

    if client is None:
        client = _default_client()
        if client is None:
            return Explanation(
                text=template_text(evidence), source="template", fingerprint=fingerprint
            )

    try:
        response = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            # temperature на claude-opus-5 не принимается: детерминизм даёт кэш,
            # а не параметр выборки. Глубину рассуждения задаёт effort.
            output_config={
                "effort": settings.effort,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            messages=[{"role": "user", "content": evidence.prompt()}],
        )
        text = _compose(json.loads(_first_text(response)))
    except Exception as error:  # noqa: BLE001 — любая проблема слоя не должна ронять цикл
        return Explanation(
            text=template_text(evidence),
            source="template-failed",
            fingerprint=fingerprint,
            model=settings.model,
            failure=_failure_text(error),
        )

    rejected = unverified_numbers(text, evidence)
    if rejected:
        return Explanation(
            text=template_text(evidence),
            source="template-rejected",
            fingerprint=fingerprint,
            model=settings.model,
            rejected_numbers=rejected,
        )

    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(
            json.dumps(
                {"text": text, "model": settings.model, "fingerprint": fingerprint},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return Explanation(text=text, source="llm", fingerprint=fingerprint, model=settings.model)


def _failure_text(error: Exception) -> str:
    """Причина отказа одной строкой — для интерфейса, а не для трассировки.

    Длина обрезается: сообщения API бывают в несколько абзацев, а на странице
    нужна причина, по которой понятно, что чинить.
    """
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return text if len(text) <= 300 else text[:299] + "…"


def _first_text(response: Any) -> str:
    """Текст первого текстового блока ответа."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("в ответе модели нет текстового блока")


def _default_client() -> MessagesClient | None:
    """Клиент Anthropic, если библиотека установлена и учётные данные найдены.

    Отсутствие ключа — штатная ситуация, а не ошибка: проверяющий запускает пакет
    без наших учётных данных, и объяснения тогда идут по шаблону.
    """
    try:
        import anthropic
    except ImportError:
        return None
    try:
        return anthropic.Anthropic()
    except Exception:  # noqa: BLE001 — нет ключа или профиля
        return None
