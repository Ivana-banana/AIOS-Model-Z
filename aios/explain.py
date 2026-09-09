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

from aios import balance

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

#: Схема ответа на вопрос инженера. Отдельная от `RESPONSE_SCHEMA`, потому что
#: вопрос — это другая задача: объяснение всегда имеет три части («что увидели»,
#: «почему», «чем подтверждено»), а ответ на вопрос имеет одну и обязан честно
#: сказать, когда фактов не хватает. Пустой `answer` при непустом `missing` —
#: законный ответ, и это лучшее, что модель может сделать с вопросом не по фактам.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Ответ на вопрос по переданным фактам. Два-четыре предложения.",
        },
        "missing": {
            "type": "string",
            "description": (
                "Каких фактов не хватило для полного ответа. Пустая строка, "
                "если хватило всех."
            ),
        },
    },
    "required": ["answer", "missing"],
    "additionalProperties": False,
}

#: Добавка к системной подсказке, когда инженер задал вопрос. Правила те же:
#: числа только из фактов, решений не принимать. Разница одна — отвечать на
#: заданное, а не пересказывать решение целиком.
QUESTION_RULES = """\

Инженер задал вопрос по этому решению. Ответь на него, соблюдая правила выше.
Если фактов для ответа не хватает — скажи об этом прямо в поле `missing` и не
подставляй правдоподобное число вместо отсутствующего. Отвечать на вопросы, не
относящиеся к этому решению, не нужно: так и напиши.
"""

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

    period: str = ""
    """Период, к которому относятся величины: «2007-01…2025-09».

    Заведено 22.08 вместе с метрикой NTI (`aios/explain_quality.py`): числовое
    утверждение — это кортеж «объект, показатель, значение, единица, период», и
    до появления этого поля период проверить было не по чему. Пустая строка
    означает «период не объявлен», а не «любой»: непроверяемое поле честнее
    оставить пустым, чем подставить правдоподобное.
    """

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
                "period": self.period,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def prompt(self) -> str:
        """Блок фактов для модели. Числа даются готовыми строками — их и цитируем."""
        lines = [f"РЕШЕНИЕ: {self.subject}", ""]
        if self.period:
            lines += [f"ПЕРИОД (величины относятся к нему): {self.period}", ""]
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
    provider: str = "anthropic"
    """Поставщик модели: `anthropic` или `yandex`.

    Слой написан под контракт Anthropic (`client.messages.create`), но ключ у
    проекта может быть от другого поставщика — у нас так и вышло. Поставщик
    выбирается конфигом, а не кодом, чтобы смена не требовала правок в модуле
    объяснений: адаптер живёт отдельно (`aios/yandex.py`).
    """

    folder_id: str = ""
    """Каталог Yandex Cloud. Не секрет — секрет только ключ, и он из окружения."""

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
    # Числа, которые мы сами написали словами в фактах, — не заявления модели, а
    # цитата. Номер скважины из `subject` («остановка скважины 42») или «выше
    # 95 %» из наблюдения проверке не подлежат: модель их не выдумала, она их
    # прочитала. Первый живой вызов 23.08 отвергся ровно на номере скважины —
    # ответ был безупречен, а слой откатился к шаблону.
    allowed |= {
        value
        for token in _NUMBER_RE.findall(_quoted_facts(evidence))
        if (value := _parse(token)) is not None
    }

    # Порядок обязателен: сначала полные даты, потом голые годы. Наоборот —
    # из «2007-01» уходит год, а остаток «-01» превращается в «неизвестное
    # число −1», и шаблонное объяснение перестаёт проходить собственную же
    # проверку. Поймано замером негативного набора 22.08.
    stripped = _years_inside_period(_DATE_RE.sub(" ", text), evidence)
    unknown: list[str] = []
    for token in _NUMBER_RE.findall(stripped):
        value = _parse(token)
        if value is None:
            continue
        if any(abs(value - candidate) <= 0.05 for candidate in allowed):
            continue
        unknown.append(token.strip())
    return tuple(dict.fromkeys(unknown))


#: Слова, которыми текст объявляет полный расчёт состоявшимся.
CONFIRMED_WORDS = (
    "подтвержд", "полным расчётом", "полного расчёта", "полный расчёт",
    "рассчитано на ггдм", "посчитано на ггдм",
)


def false_confirmation(text: str, evidence: Evidence) -> str | None:
    """Текст объявляет полный расчёт, которого не было. `None` — не объявляет.

    Статус — не число, и барьер чисел его не видел вовсе: ответ, где прирост
    подписан «подтверждено полным расчётом ГГДМ» при `confirmed = False`,
    проходил обе проверки. А это ровно то утверждение, из-за которого расхождение
    заявленного ЧДД с пересчётом организаторов и становится дисквалификацией:
    прогноз суррогата, поданный как факт ГГДМ (Р-51).

    Обратную сторону не проверяем: не назвать подтверждение, когда оно есть, —
    скромность, а не ложь.
    """
    if evidence.confirmed:
        return None
    lowered = text.lower()
    for word in CONFIRMED_WORDS:
        if word in lowered:
            return (f"текст объявляет полный расчёт («{word}»), а факты говорят "
                    "«ПРОГНОЗ суррогата, полным расчётом не подтверждено»")
    return None


def _quoted_facts(evidence: Evidence) -> str:
    """Весь текст, который модель видела и может процитировать дословно.

    Сюда входят и **имена** показателей, а не только `subject`, `observations` и
    `caveats`. Сначала имена были исключены — рассуждение звучало убедительно
    («имя показателя не число»), и оно оказалось неверным: `Evidence.prompt()`
    печатает строку `- дебит жидкости, м3/сут: 48`, модель повторяет единицу как
    есть, а `м3` содержит цифру. Второй живой вызов 23.08 отвергся ровно на этой
    тройке из «м3/сут».

    Значения показателей сюда не нужны — они разобраны отдельно и с допуском.
    """
    return " ".join((
        evidence.subject,
        *evidence.observations,
        *evidence.caveats,
        *evidence.numbers,
    ))


#: Голый год: «в 2019 году». Без границ периода это просто число и остаётся им.
_YEAR_RE = re.compile("(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])")


def _years_inside_period(text: str, evidence: Evidence) -> str:
    """Убрать из текста годы, попадающие в объявленный период фактов.

    Замер 22.08 (`aios/explain_quality.py`, негативный набор): фраза «прирост
    получен к 2019 году» отвергалась целиком — `2019` не число из расчёта, и
    проверка честно этого не знала. Отказ по такой причине — **ложный**: год
    внутри `Evidence.period` не выдуманная величина, а ссылка на период, который
    мы модели сами и сообщили.

    Год **вне** периода не вычёркивается: «прирост за 1998 год» при фактах за
    2007…2025 — настоящая ошибка, и отвергать её надо по-прежнему. Периода нет —
    не вычёркивается ничего: fail-closed остаётся умолчанием.
    """
    if not evidence.period:
        return text
    bounds = [int(match.group()) for match in _YEAR_RE.finditer(evidence.period)]
    if not bounds:
        return text
    first, last = min(bounds), max(bounds)
    return _YEAR_RE.sub(
        lambda match: " " if first <= int(match.group()) <= last else match.group(),
        text,
    )


def template_text(evidence: Evidence) -> str:
    """Шаблонное объяснение — то, что было до слоя и осталось запасным вариантом.

    Оно всегда доступно, всегда одинаково и не требует сети. Слой на языковой
    модели обязан быть лучше него, а не заменять его собой.
    """
    lines = [evidence.subject]
    if evidence.period:
        lines += ["", f"Период: {evidence.period}"]
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


def evidence_from_cycle(result: Any, surrogate_name: str = "", *,
                        limits: Any = None, fluids: Any = None) -> Evidence:
    """Собрать набор фактов из итога цикла (`aios.loop.CycleResult`).

    Числа берутся ровно те, что посчитал цикл, — модели остаётся их изложить.
    Оговорки собираются здесь же, а не в подсказке: неподтверждённый прирост,
    выход за обучающий диапазон суррогата и нарушения ограничений обязаны попасть
    в объяснение, даже если модель сочтёт их незначительными.

    `limits` и `fluids` нужны одному факту — сторонней воде. Оба необязательны:
    без них потребность всё равно считается (плотности по умолчанию — первый
    PVT-регион Модели Z, та же условность, что во всём `aios.balance`), но
    сравнить её с мощностью водозабора нечем, и объяснение говорит об этом
    прямо, а не умалчивает.
    """
    numbers = {
        "ЧДД базового сценария, млн руб.": result.base_chdd_m,
        "ЧДД после решений, млн руб.": result.chdd_m,
        "прирост, млн руб.": result.gain_m,
        "решений по фонду": float(len(result.actions)),
        "скважин с подобранным режимом": float(len(result.rate_choices)),
    }
    if result.allocations:
        numbers["срезано планировщиком, скважин"] = float(len(result.allocations))

    observations = []
    if surrogate_name:
        observations.append(f"прогноз получен источником: {surrogate_name}")
    if result.actions:
        kinds: dict[str, int] = {}
        for action in result.actions:
            kinds[action.kind] = kinds.get(action.kind, 0) + 1
        observations.append(
            "типы принятых решений: "
            + ", ".join(f"{kind} — {count}" for kind, count in sorted(kinds.items()))
        )
    # Перевод под закачку объясняется отдельно: его выигрыш даёт отклик соседей,
    # а суррогат его не предсказывает — в обучающих сценариях у переведённой
    # скважины колонки закачки нет вовсе. Молча положить такое решение в один ряд
    # с остановками значит выдать за оценённое то, что не оценивалось (Б37).
    converted = [action for action in result.actions if action.kind == "convert"]
    if converted:
        numbers["переводов под закачку"] = float(len(converted))
        numbers["стоимость переводов, млн руб."] = float(
            sum(action.cost_m for action in converted)
        )

    # Сторонняя вода. Без этого блока слой интерпретируемости молчал о докачке
    # вовсе: планировщик её учитывает и при нужде режет закачку, а объяснение
    # рассказывало про остановки и режимы так, будто воды в системе сколько
    # угодно. Требование трекера с КТ-2 — «использование докачки попадает в
    # объяснение агента».
    rows = getattr(result, "rows", None) or []
    capacity = getattr(limits, "max_external_water_m3_per_day", None)
    outside: list[str] = []
    if rows:
        months = balance.external_water_by_month(rows, fluids)
        if months:
            stamp, need = max(months, key=lambda item: item[1])
            if need > 0:
                numbers["сторонней воды на пике, м³/сут"] = need
                observations.append(
                    f"закачка не покрывается собственной попутной водой: пик "
                    f"потребности в {stamp}"
                )
                if capacity is None:
                    outside.append(
                        "мощность водозабора неизвестна: потребность в сторонней воде "
                        "ничем не ограничивалась, и по источнику воды режим не подтверждён"
                    )
                else:
                    numbers["мощность водозабора, м³/сут"] = float(capacity)
                    over = balance.external_water_capacity_problems(rows, fluids, capacity)
                    if over:
                        numbers["месяцев выше мощности водозабора"] = float(len(over))
                        outside.append(
                            f"расход водозабора выше его мощности в {len(over)} мес.: "
                            f"закачка обеспечена водой не во всём периоде"
                        )

    # Срезка по источнику — отдельное решение планировщика, и его видно только
    # здесь: в `actions` его нет, это не мероприятие, а изменение уставок.
    limited = [one for one in getattr(result, "allocations", ()) or ()
               if "со стороны" in getattr(one, "reason", "")]
    if limited:
        numbers["закачка урезана мощностью источника, скважин"] = float(len(limited))
        observations.append(
            "закачка урезана не нормой компенсации, а мощностью водозабора: "
            f"{limited[0].reason}"
        )

    caveats = []
    caveats.extend(outside)
    if converted:
        caveats.append(
            "выигрыш перевода под закачку суррогатом не оценивался: отклик соседей "
            "на новую нагнетательную он не предсказывает — проверяется только "
            "полным расчётом ГГДМ"
        )
    if not result.confirmed:
        caveats.append(
            "прирост посчитан по прогнозу суррогата и полным расчётом ГГДМ не подтверждён"
        )
    if result.warnings:
        # Не «выход за диапазон»: сюда попадают предупреждения трёх разных родов —
        # выход за обучающий диапазон, решение, которого в обучающей выборке нет
        # вовсе (запуск из простоя, перевод под закачку), и нарушение физических
        # проверок в самом прогнозе. Последнее тяжелее всех: физически невозможный
        # прогноз означает, что модели нельзя верить вообще, а не что мы вышли за
        # её область. Классификатор причин живёт в scripts/surrogate_warnings.py
        # и второго экземпляра здесь заводить не надо — достаточно не врать:
        # назвать число и показать пример настоящим текстом.
        example = result.warnings[0]
        caveats.append(
            f"суррогат сопроводил прогноз предупреждениями: {len(result.warnings)}, "
            f"например «{example[:160]}{'…' if len(example) > 160 else ''}»"
        )
        numbers["предупреждений суррогата"] = float(len(result.warnings))
    if result.problems:
        caveats.append(
            f"расписание нарушает ограничения ({len(result.problems)}) и организаторам "
            "не отправляется"
        )
        numbers["нарушений ограничений"] = float(len(result.problems))

    # период берётся из шагов самого расписания: «за 2007–2025» в тексте должно
    # быть проверяемым утверждением, а не общим местом (NTI, aios/explain_quality.py)
    steps = getattr(result, "steps", ())
    period = ""
    if steps:
        period = f"{steps[0].date:%Y-%m}…{steps[-1].date:%Y-%m}"

    return Evidence(
        subject="цикл управления фондом: подбор режимов и остановка убыточных скважин",
        observations=tuple(observations),
        numbers=numbers,
        confirmed=bool(result.confirmed),
        caveats=tuple(caveats),
        period=period,
    )


def _cache_path(settings: Settings, fingerprint: str) -> Path | None:
    if settings.cache_dir is None:
        return None
    return Path(settings.cache_dir) / f"{fingerprint}.json"


def _compose_answer(payload: dict[str, str]) -> str:
    """Ответ на вопрос и, если чего-то не хватило, честная приписка об этом."""
    parts = [payload.get("answer", "").strip()]
    missing = payload.get("missing", "").strip()
    if missing:
        parts.append(f"Чего не хватило для полного ответа: {missing}")
    return "\n\n".join(part for part in parts if part)


def _compose(payload: dict[str, str]) -> str:
    parts = [payload.get("what_happened", ""), payload.get("why", ""), payload.get("confirmation", "")]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def answer(evidence: Evidence,
           settings: Settings,
           question: str,
           *,
           client: MessagesClient | None = None) -> Explanation:
    """Ответить на вопрос инженера по тем же фактам, что и объяснение.

    Зачем это отдельно от `explain`. Объяснение отвечает на вопрос, которого
    никто не задавал: «почему так решили». На защите спрашивают другое — «а что
    с этой скважиной», «почему не больше», «чем это подтверждено», — и слой
    интерпретируемости, который умеет только монолог, на такие вопросы не
    отвечает. Требование регламента при этом ровно про диалог: решение обязано
    быть объяснимо человеку, а не изложено ему.

    Чем это **не** является: моделью не спрашивают решения и не дают ей считать.
    Факты те же самые, проверка чисел та же самая (`unverified_numbers` и
    `_broken_tuples`), отказ такой же — ответ с выдуманным числом отбрасывается
    целиком. Вопрос попадает в отпечаток, поэтому кэш у каждого вопроса свой.

    Пустой вопрос — не ошибка, а отсутствие вопроса: возвращается обычное
    объяснение.
    """
    question = question.strip()
    if not question:
        return explain(evidence, settings, client=client)
    return _talk(evidence, settings, question=question, client=client)


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
    return _talk(evidence, settings, question="", client=client)


def _talk(evidence: Evidence,
          settings: Settings,
          *,
          question: str,
          client: MessagesClient | None = None) -> Explanation:
    """Общий путь объяснения и ответа на вопрос: кэш → модель → проверка → шаблон.

    Единственная разница между двумя вызовами — что просят у модели и по какой
    схеме она отвечает. Всё остальное — кэш, отпечаток, барьеры проверки чисел,
    порядок отказов — обязано быть общим, иначе ответ на вопрос окажется слабее
    защищён, чем объяснение, и по коду этого не будет видно.
    """
    fingerprint = evidence.fingerprint()
    if question:
        # вопрос входит в отпечаток: у каждого вопроса свой кэш, и ответ на один
        # вопрос не выдаётся за ответ на другой
        fingerprint = hashlib.sha256(
            f"{fingerprint}|{question}".encode()).hexdigest()[:16]

    cached = _cache_path(settings, fingerprint)
    if cached is not None and cached.is_file():
        stored = json.loads(cached.read_text(encoding="utf-8"))
        # Барьеры проходит и текст из кэша. Инвариант «ответ, не прошедший
        # проверку, не может быть показан принятым — ни на странице цикла, ни в
        # отчёте, ни в кэше» держался только на записи: файл кэша правится
        # руками, приезжает с другой машины и — по плану `BLOCKED.md`, п. 0.2-бис
        # — кладётся в сдаваемый пакет. Проверять его на чтении стоит доли
        # секунды, а не проверять — значит показывать выдуманное число под
        # подписью «проверка чисел пройдена» (Р-52).
        rejected = _rejected(stored["text"], evidence)
        if rejected:
            return Explanation(
                text=template_text(evidence),
                source="cache-rejected",
                fingerprint=fingerprint,
                model=stored.get("model", ""),
                rejected_numbers=rejected,
            )
        return Explanation(
            text=stored["text"],
            source="cache",
            fingerprint=fingerprint,
            model=stored.get("model", ""),
        )

    if not settings.enabled:
        return Explanation(text=template_text(evidence), source="template", fingerprint=fingerprint)

    if client is None:
        client, unreachable = _client_for(settings)
        if client is None:
            # Слой включён явно, но до модели не дошли. Это не то же самое, что
            # `enabled = false`, и выглядеть одинаково не должно: живой прогон
            # планируется один (`BLOCKED.md`, п. 6), и молчаливый откат к шаблону
            # в нём читался бы как успешный запуск.
            return Explanation(
                text=template_text(evidence),
                source="template-failed",
                fingerprint=fingerprint,
                model=settings.model,
                failure=unreachable,
            )

    try:
        response = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT + (QUESTION_RULES if question else ""),
            # temperature на claude-opus-5 не принимается: детерминизм даёт кэш,
            # а не параметр выборки. Глубину рассуждения задаёт effort.
            output_config={
                "effort": settings.effort,
                "format": {"type": "json_schema",
                           "schema": ANSWER_SCHEMA if question else RESPONSE_SCHEMA},
            },
            messages=[{"role": "user", "content": (
                evidence.prompt() if not question
                else f"{evidence.prompt()}\n\nВОПРОС ИНЖЕНЕРА: {question}"
            )}],
        )
        payload = json.loads(_first_text(response))
        text = _compose_answer(payload) if question else _compose(payload)
    except Exception as error:  # noqa: BLE001 — любая проблема слоя не должна ронять цикл
        return Explanation(
            text=template_text(evidence),
            source="template-failed",
            fingerprint=fingerprint,
            model=settings.model,
            failure=_failure_text(error),
        )

    # Два барьера, но одно число — одна строка отказа: выдуманное число ловят оба,
    # и печатать его дважды значит преувеличивать количество ошибок.
    rejected = _rejected(text, evidence)
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


def _rejected(text: str, evidence: Evidence) -> tuple[str, ...]:
    """Всё, из-за чего ответ нельзя показать принятым. Пусто — можно.

    Три барьера, но одно число — одна строка отказа: выдуманное число ловят два
    первых, и печатать его дважды значит преувеличивать количество ошибок.
    Третий про статус: он не число, и первые два его не видят.
    """
    literal = unverified_numbers(text, evidence)
    rejected = literal + tuple(f"{token} ({verdict})"
                               for token, verdict in _broken_tuples(text, evidence)
                               if token not in literal)
    lie = false_confirmation(text, evidence)
    return rejected + (lie,) if lie is not None else rejected


def _broken_tuples(text: str, evidence: Evidence) -> tuple[tuple[str, str], ...]:
    """Числа, у которых сломан кортеж «показатель — значение — единица — период».

    Второй барьер поверх `unverified_numbers`, и он ловит другое. Первый отвечает
    на вопрос «есть ли такое число в расчёте», этот — «о том ли оно». Замер 22.08
    на негативном наборе из восьми подмен: прежняя сверка ловила четыре
    (выдуманное число, знак, потерянный разделитель, свой пересчёт), эта — все
    восемь, добавляя подмену показателя, подмену единицы, округление и чужой
    период. Ложных отказов на верных текстах — ноль из пяти
    (`docs/explainability.md`).

    Инвариант, ради которого барьер стоит именно здесь, а не в интерфейсе: ответ,
    не прошедший проверку, **не может быть показан принятым** — ни на странице
    цикла, ни в отчёте, ни в кэше. Проверка стоит до записи в кэш.

    Импорт локальный: `aios.explain_quality` читает `Evidence` отсюда, и на
    уровне модуля это был бы цикл. Метрика живёт рядом с остальными метриками, а
    не дублируется здесь второй реализацией.
    """
    from aios.explain_quality import integrity

    return tuple((number.token, number.verdict)
                 for number in integrity(text, evidence).failures)


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


def _default_client() -> tuple[MessagesClient | None, str]:
    """Клиент Anthropic и — если его нет — причина, названная словами.

    Отсутствие ключа само по себе штатно: при `enabled = false` проверяющий
    запускает пакет без наших учётных данных, и объяснения идут по шаблону. Но
    когда слой включён явно, «не смогли построить клиента» обязано быть отличимо
    от «слой выключен», а причины между собой — тем более: не установлена
    библиотека и не найден ключ чинятся разными действиями.

    Где какая причина всплывает — проверено на `anthropic==0.116.0` (22.08):
    отсутствие библиотеки ловится здесь, а вот **отсутствие ключа сюда не
    доходит**. Конструктор `Anthropic()` без учётных данных не падает: он
    откладывает разрешение ключа до запроса, и `TypeError: Could not resolve
    authentication method…` прилетает уже из `messages.create` — то есть через
    `except` в `explain()`, тем же `template-failed` и с той же названной
    причиной. Сетевого запроса при этом не происходит. `except` ниже остаётся
    для остальных отказов конструктора (испорченный профиль, неверный
    `base_url`), а не для пустого ключа.
    """
    try:
        import anthropic
    except ImportError:
        return None, ("библиотека anthropic не установлена — она есть в "
                      "requirements.txt, но не в этом окружении")
    try:
        return anthropic.Anthropic(), ""
    except Exception as error:  # noqa: BLE001 — нет ключа или профиля
        return None, f"клиент Anthropic не создан: {_failure_text(error)}"


def _client_for(settings: Settings) -> tuple[MessagesClient | None, str]:
    """Клиент выбранного поставщика — или причина словами.

    Неизвестный поставщик это отказ, а не тихий откат на Anthropic: опечатка в
    конфиге не должна молча уводить запрос к другому вендору и другому счёту.
    """
    provider = (settings.provider or "anthropic").strip().lower()
    if provider == "anthropic":
        return _default_client()
    if provider == "yandex":
        from aios import yandex
        return yandex.from_environment(settings.folder_id)
    return None, (
        f"неизвестный поставщик модели {settings.provider!r}; "
        "допустимы 'anthropic' и 'yandex'"
    )
