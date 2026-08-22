"""Измеримое качество объяснений: ASR, DTC, NTI поверх пакета фактов.

До 22.08 слой интерпретируемости мерялся ровно одним способом:
`explain.unverified_numbers` проверял, что каждое число текста есть в расчёте.
Этого мало для утверждения «решения объяснимы» — оно идёт отдельным критерием
жюри, а без числа это обещание, а не результат.

Протокол взят из внешнего разбора (`manus-answered/Измерение качества объяснений
языковой модели…`, разбор — `docs/manus/round4.md`, 4-9). Главное различение,
ради которого он написан: **faithfulness** — текст соответствует настоящим
основаниям решения — против **plausibility** — текст выглядит убедительно.
Здесь меряется только первое; второе не меряется ничем и не заявляется.

Три метрики:

* **ASR** — доля атомарных утверждений, у которых нашёлся источник в `Evidence`;
* **DTC** — совпадение действия, объекта, направления и активного ограничения
  с тем, что записано в решении;
* **NTI** — целостность числового кортежа: не «такое число есть в расчёте», а
  «это число относится к этому показателю в этих единицах».

**Чем этот модуль не является.** Извлекатель утверждений здесь лексический: он
сравнивает слова текста со словами фактов. Для шаблонного объяснения это точно
(текст собран из тех же строк), для перефразирующей языковой модели — **оценка
снизу**: модель, сказавшая то же самое другими словами, будет засчитана как
неподтверждённая. Поэтому ASR публикуется вместе со списком неподтверждённых
утверждений: разбор прямо требует ручной проверки подвыборки, иначе систематику
вносит сам извлекатель, а не объяснитель.

Чего в пакете фактов нет и что поэтому **не** проверяется: сценарий и — до
появления поля `Evidence.period` — период. Кортеж проверяется в той части, для
которой есть источник: объект, показатель, значение, единица, период.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from aios.explain import _NUMBER_RE, Evidence, _format, _parse

#: Вес утверждения. Разбор требует 2 для действия, правила, ограничения и
#: прогноза и 1 для второстепенного описания — и требует публиковать взвешенный
#: и невзвешенный счёт рядом, иначе весами можно спрятать ошибку.
WEIGHTS = {"решение": 2, "число": 2, "ограничение": 2, "статус": 2, "описание": 1}

#: Доля значимых слов утверждения, которую источник обязан покрыть.
#: 0.5 выбрана до замера и не подбиралась по результату: половина слов —
#: граница, ниже которой «поддержка» превращается в случайное совпадение слов.
SUPPORT_COVERAGE = 0.5

#: Слова, которые не несут привязки к факту: служебные и общепроектные.
_STOPWORDS = frozenset([
    "и", "или", "но", "что", "как", "для", "при", "это", "этом", "эта", "эти", "тот", "та", "то",
    "те", "так", "там", "где", "когда", "если", "чтобы", "либо", "ведь", "уже", "ещё", "же", "бы",
    "был", "была", "были", "быть", "есть", "нет", "не", "ни", "на", "в", "во", "из", "от", "до",
    "по", "за", "над", "под", "перед", "между", "через", "без", "о", "об", "про", "с", "со", "у",
    "к", "ко", "а", "также", "млн", "руб", "тыс", "сут", "шт", "около", "примерно", "всего",
    "итого", "более", "менее", "чем", "его", "их", "её"
])

# Точка кончает утверждение только перед заглавной буквой: «тыс. т» и «млн руб. до»
# — сокращения внутри одного утверждения. Замер 22.08: без этого условия подмена
# единицы («642.4 тыс. т» вместо «млн руб.») проходила проверку, потому что единица
# оставалась в соседнем «предложении», а рядом с числом её уже не было.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?=[А-ЯЁA-Z])|[\r\n]+")
_WORD = re.compile(r"[a-zA-Zа-яА-ЯёЁ][a-zA-Zа-яА-ЯёЁ-]*")

#: Словарь перефразировок действий. Нужен DTC: «остановили» и «вывели из работы» —
#: одно действие, а «остановили» и «запустили» — разные, и путать их дороже всего.
ACTION_WORDS = {
    "остановка": ("останов", "заглуш", "вывед", "выключ", "закрыт"),
    "запуск": ("запуск", "запусти", "запущ", "пусти", "ввод", "введ", "открыт"),
    "режим": ("режим", "уставк", "дебит", "штуцер", "форсаж", "подобран", "подбор"),
    "закачка": ("закачк", "нагнетат", "внс", "заводнен"),
}
# Корни, а не словарные формы: русский текст склоняется, и «запустила» не
# начинается с «запуск». Первый прогон тестов ловил это как «действие не
# сказано» — то есть метрика молчала там, где текст прямо противоречил решению.

#: Слова направления. Направление проверяется отдельно от действия: «прирост» и
#: «падение» при одном и том же действии — разные утверждения о результате.
DIRECTION_WORDS = {
    "рост": ("прирост", "вырос", "увелич", "подъём", "подъем", "выше", "плюс"),
    "снижение": ("снижен", "снизил", "снизи", "паден", "упал", "сократ", "ниже", "минус"),
}
# «убыточная скважина» — про доходность, а не про направление: слово «убыт»
# стояло здесь до первого замера и делало разошедшимся направление у собственного
# шаблонного текста («остановка убыточных скважин» + прирост +642,4). Словарь
# направления обязан состоять из слов, у которых нет второго прочтения.

#: Слова, которыми текст ссылается на активное ограничение или предупреждение.
LIMIT_WORDS = ("ограничен", "нарушен", "предупрежд", "срезал", "срезано", "предел",
               "не подтверж", "не доказан", "вне обучающ")


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def _significant(text: str) -> set[str]:
    """Значимые слова: без служебных и без коротких, которые совпадают случайно."""
    return {word for word in _words(text) if len(word) >= 4 and word not in _STOPWORDS}


def claims(text: str) -> tuple[str, ...]:
    """Разбить объяснение на атомарные утверждения.

    Атом — предложение или строка списка: шаблон печатает факты построчно, а
    языковая модель отвечает тремя абзацами. Пустые строки и заголовки вроде
    «Посчитано:» утверждениями не считаются — они ничего не утверждают.
    """
    pieces: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text):
        piece = raw.strip(" \t-•").strip()
        if not piece:
            continue
        if piece.endswith(":") and not _NUMBER_RE.search(piece):
            continue
        pieces.append(piece)
    return tuple(pieces)


def kind_of(claim: str) -> str:
    """Тип утверждения — он же вес. Классификация лексическая и намеренно грубая."""
    lowered = claim.lower()
    if any(word in lowered for word in LIMIT_WORDS):
        return "ограничение"
    if any(word in lowered for group in ACTION_WORDS.values() for word in group):
        return "решение"
    if _NUMBER_RE.search(_DATE_FREE(claim)):
        return "число"
    if "подтвержд" in lowered or "прогноз" in lowered or "ггдм" in lowered:
        return "статус"
    return "описание"


def _DATE_FREE(text: str) -> str:
    """Текст без дат: `2019-06` — ссылка на шаг, а не величина."""
    return re.sub(r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{2}\.\d{4}\b", " ", text)


# ── источники: пакет фактов как набор именованных полей ──────────────────────

@dataclass(frozen=True)
class Fact:
    """Одно поле пакета фактов — то, на что утверждению разрешено опираться."""

    fact_id: str
    text: str
    value: float | None = None
    unit: str = ""


def facts_of(evidence: Evidence) -> tuple[Fact, ...]:
    """Пакет фактов идентификаторами: `subject`, `observation:N`, `number:имя`…"""
    items = [Fact("subject", evidence.subject)]
    items += [Fact(f"observation:{index}", text)
              for index, text in enumerate(evidence.observations)]
    for name, value in evidence.numbers.items():
        items.append(Fact(f"number:{name}", f"{name}: {_format(value)}", value, _unit_of(name)))
    items.append(Fact(
        "status",
        "подтверждено полным расчётом ГГДМ" if evidence.confirmed
        else "прогноз суррогата, полным расчётом не подтверждено",
    ))
    items += [Fact(f"caveat:{index}", text) for index, text in enumerate(evidence.caveats)]
    if getattr(evidence, "period", ""):
        items.append(Fact("period", f"период {evidence.period}"))
    return tuple(items)


def _unit_of(name: str) -> str:
    """Единица берётся из имени величины: «прирост, млн руб.» → «млн руб.».

    Единица живёт в ключе `numbers`, а не отдельным полем, — значит и читать её
    надо оттуда, а не заводить второй источник истины.
    """
    if "," not in name:
        return ""
    return name.rsplit(",", 1)[1].strip().rstrip(".").strip()


# ── ASR: доля утверждений, у которых нашёлся источник ────────────────────────

@dataclass(frozen=True)
class ClaimSupport:
    claim: str
    kind: str
    weight: int
    fact_ids: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return bool(self.fact_ids)


@dataclass(frozen=True)
class Support:
    """Результат ASR по одному объяснению."""

    checks: tuple[ClaimSupport, ...]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def supported(self) -> int:
        return sum(1 for check in self.checks if check.supported)

    @property
    def weighted_total(self) -> int:
        return sum(check.weight for check in self.checks)

    @property
    def weighted_supported(self) -> int:
        return sum(check.weight for check in self.checks if check.supported)

    @property
    def unsupported(self) -> tuple[ClaimSupport, ...]:
        return tuple(check for check in self.checks if not check.supported)


def support(text: str, evidence: Evidence) -> Support:
    """ASR: каждое утверждение — против пакета фактов.

    Поддержкой считается либо число, совпавшее со значением величины, либо
    покрытие значимых слов утверждения словами факта не ниже `SUPPORT_COVERAGE`.
    Первое точно, второе — оценка снизу для перефразированного текста; об этом
    сказано в шапке модуля, и список неподтверждённого возвращается целиком
    именно поэтому.
    """
    pool = facts_of(evidence)
    checks: list[ClaimSupport] = []
    for claim in claims(text):
        kind = kind_of(claim)
        found: list[str] = []
        for value in _numbers_in(claim):
            found += [fact.fact_id for fact in pool
                      if fact.value is not None and _same_number(value, fact.value)]
        tokens = _significant(claim)
        if tokens:
            for fact in pool:
                covered = len(tokens & _significant(fact.text)) / len(tokens)
                if covered >= SUPPORT_COVERAGE:
                    found.append(fact.fact_id)
        checks.append(ClaimSupport(claim, kind, WEIGHTS[kind], tuple(dict.fromkeys(found))))
    return Support(tuple(checks))


def _numbers_in(text: str) -> list[float]:
    values = []
    for token in _NUMBER_RE.findall(_DATE_FREE(text)):
        value = _parse(token)
        if value is not None:
            values.append(value)
    return values


def _same_number(left: float, right: float) -> bool:
    """Сравнение как у `explain.unverified_numbers`: целая часть тоже законна.

    Правило унаследовано намеренно: ASR отвечает на вопрос «нашёлся ли источник
    у утверждения», и «выросла на 402 тыс. т» при факте 402.4 источник имеет.
    """
    return abs(left - right) <= 0.05 or abs(left - int(right)) <= 0.05


def _exact_number(left: float, right: float) -> bool:
    """Сравнение для NTI: строже, целая часть вместо значения не принимается.

    Здесь вопрос другой — «верен ли кортеж», — и 642.0 вместо 642.4 неверен:
    это округление, то есть другое значение того же показателя. Замер 22.08:
    старое правило пропускало такую подмену, новое ловит. Разница между двумя
    сравнениями и есть то, что NTI добавляет к прежней сверке чисел.
    """
    return abs(left - right) <= 0.05


# ── DTC: совпадает ли текст с самим решением ─────────────────────────────────

@dataclass(frozen=True)
class Consistency:
    """Результат DTC по одному объяснению: по полю на строку матрицы ошибок."""

    fields: dict[str, str] = field(default_factory=dict)
    """Поле → `совпало` / `разошлось` / `не сказано` / `нечего сверять`."""

    @property
    def consistent(self) -> bool:
        """Расхождений нет ни по одному полю. Молчание расхождением не считается.

        Умолчание — не ошибка faithfulness: текст, не назвавший ограничение,
        не соврал о нём. Ошибка — назвать не то, поэтому `разошлось` и
        `не сказано` в матрице стоят раздельно и никогда не складываются.
        """
        return all(state != "разошлось" for state in self.fields.values())


def consistency(text: str, evidence: Evidence) -> Consistency:
    """DTC: действие, объект, направление и активное ограничение — против решения."""
    lowered = text.lower()
    return Consistency({
        "действие": _match_action(lowered, evidence),
        "объект": _match_object(lowered, evidence),
        "направление": _match_direction(lowered, evidence),
        "ограничение": _match_limit(lowered, evidence),
    })


def _decided_actions(evidence: Evidence) -> set[str]:
    """Какие действия записаны в решении: по теме и по наблюдениям цикла."""
    source = " ".join((evidence.subject, *evidence.observations)).lower()
    return {action for action, words in ACTION_WORDS.items()
            if any(word in source for word in words)}


def _mentioned_actions(lowered: str) -> set[str]:
    return {action for action, words in ACTION_WORDS.items()
            if any(word in lowered for word in words)}


def _match_action(lowered: str, evidence: Evidence) -> str:
    decided = _decided_actions(evidence)
    mentioned = _mentioned_actions(lowered)
    if not decided:
        return "нечего сверять"
    if not mentioned:
        return "не сказано"
    return "совпало" if mentioned <= decided else "разошлось"


def _match_object(lowered: str, evidence: Evidence) -> str:
    """Объект решения: конкретная скважина из темы либо фонд целиком.

    Сначала объект сверялся долей совпавших слов темы, и это давало «разошлось»
    на верном тексте: тема пишется словарными формами («остановка убыточных
    скважин»), а текст склоняет («остановил убыточные скважины»). Метрика мерила
    морфологию, а не объект.

    Теперь различаются два случая. Тема называет **скважину по имени** — имя
    обязано быть в тексте, а чужое имя означает расхождение. Тема про **фонд** —
    объект считается совпавшим, если текст говорит о фонде или скважинах вообще;
    «разошлось» здесь не выносится вовсе, потому что различить объекты не по чему,
    а выносить вердикт без основания хуже, чем не выносить.
    """
    named = _well_names(evidence.subject)
    if named:
        mentioned = _well_names(lowered)
        if not mentioned & named:
            return "разошлось" if mentioned else "не сказано"
        return "совпало"
    return "совпало" if ("фонд" in lowered or "скважин" in lowered) else "не сказано"


#: Имя скважины: «42», «Р-66», «B-1H». Именно по нему объект и различается.
_WELL_NAME = re.compile(r"\b[A-ZА-Яa-zа-я]{0,2}-?\d{1,4}[A-ZА-Яa-zа-я]?\b")


def _well_names(text: str) -> set[str]:
    return {match.group().lower().lstrip("-") for match in _WELL_NAME.finditer(text)}


def _match_direction(lowered: str, evidence: Evidence) -> str:
    """Направление берётся из знака прироста, а не из слов решения."""
    gains = [value for name, value in evidence.numbers.items() if "прирост" in name.lower()]
    if not gains:
        return "нечего сверять"
    expected = "рост" if gains[0] > 0 else "снижение" if gains[0] < 0 else ""
    if not expected:
        return "нечего сверять"
    said = {name for name, words in DIRECTION_WORDS.items()
            if any(word in lowered for word in words)}
    if not said:
        return "не сказано"
    return "совпало" if said == {expected} else "разошлось"


def _match_limit(lowered: str, evidence: Evidence) -> str:
    """Ограничение: оговорки решения против того, что текст о них говорит."""
    said = any(word in lowered for word in LIMIT_WORDS)
    if evidence.caveats:
        return "совпало" if said else "не сказано"
    return "разошлось" if said else "нечего сверять"


# ── NTI: число как кортеж, а не как значение ─────────────────────────────────

#: Единицы, которые встречаются в наших величинах и рядом с ними в тексте.
#: Нужны, чтобы отличить «единица не названа» от «названа чужая» — это разные
#: новости: первая неполна, вторая неверна.
UNIT_WORDS = ("млн руб", "тыс. руб", "тыс. т", "тыс т", "тыс. м3", "тыс. м³",
              "м3/сут", "м³/сут", "т/сут", "атм", "%")
# Счётных слов («скважин», «месяцев») здесь нет намеренно: они стоят рядом с
# любым числом фонда, и замер 22.08 поймал на этом ложный отказ — «прирост
# составил 642.4» рядом с фразой «остановлено 3 скважины» объявлялся подменой
# единицы. Чужой единицей считается только физическая или денежная.

#: Вердикты, при которых ответ считается непрошедшим проверку. «Единица не
#: названа» сюда не входит намеренно: замер 22.08 на текстах в стиле ответа
#: модели показал, что жёсткий отказ по ней выбрасывает верные объяснения
#: («прирост составил 642.4 — против базового 11851.8»), то есть даёт ложное
#: отклонение там, где ошибки нет. Названная **чужая** единица — отказ.
HARD_FAILURES = ("неизвестное число", "не тот показатель", "не та единица", "не тот период")


@dataclass(frozen=True)
class NumericClaim:
    token: str
    value: float
    verdict: str
    """`ok`, `неизвестное число`, `не тот показатель`, `не та единица`,
    `не тот период` или мягкое `единица не названа`."""

    matched: str = ""

    @property
    def hard(self) -> bool:
        return self.verdict in HARD_FAILURES


@dataclass(frozen=True)
class Integrity:
    """Результат NTI по одному объяснению."""

    numbers: tuple[NumericClaim, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> tuple[NumericClaim, ...]:
        """Только жёсткие вердикты: непройденная проверка, а не неполнота."""
        return tuple(number for number in self.numbers if number.hard)

    @property
    def unnamed_units(self) -> tuple[NumericClaim, ...]:
        return tuple(number for number in self.numbers if number.verdict == "единица не названа")


def integrity(text: str, evidence: Evidence) -> Integrity:
    """NTI: каждое число текста — как кортеж «показатель, значение, единица».

    Сверка значений (`explain.unverified_numbers`) отвечает на вопрос «есть ли
    такое число в расчёте». Здесь вопрос другой: **о том ли оно**. «100 м³/сут»
    может быть законным числом и при этом относиться к другой скважине или к
    жидкости вместо нефти — разбор 4-9 называет это главной дырой простой сверки.

    Что проверяется: значение, показатель (имя величины рядом с числом) и
    единица. Что **не** проверяется: сценарий — такого поля в пакете фактов нет
    вовсе; период — только если он объявлен (`Evidence.period`).
    """
    pool = [fact for fact in facts_of(evidence) if fact.value is not None]
    results: list[NumericClaim] = list(_period_claims(text, evidence))
    for claim in claims(text):
        results.extend(_numeric_claims(claim, pool))
    return Integrity(tuple(results))


def _numeric_claims(claim: str, pool: list[Fact]) -> list[NumericClaim]:
    """Числа одного утверждения. Окно — само утверждение, а не соседний текст.

    Сначала окном была рамка в ±70 символов, и она читала соседние строки:
    фраза «остановлено 642.4 скважин» проходила проверку, потому что слово
    «прирост» стояло строкой выше, в другом утверждении (замер 22.08). Соседство
    — не связь; кортеж собирается внутри одного утверждения или не собирается
    вовсе.
    """
    stripped = _DATE_FREE(claim)
    window = stripped.lower()
    results: list[NumericClaim] = []
    for match in _NUMBER_RE.finditer(stripped):
        token = match.group().strip()
        value = _parse(token)
        if value is None:
            continue
        candidates = [fact for fact in pool if _exact_number(value, fact.value)]
        if not candidates:
            # Голый год — это ссылка на период, а не величина: его судит
            # `_period_claims`, и объявлять его «неизвестным числом» значило бы
            # считать одну ошибку дважды и по неверной причине.
            if _YEAR.fullmatch(token):
                continue
            results.append(NumericClaim(token, value, "неизвестное число"))
            continue
        named = [fact for fact in candidates if _names_quantity(window, fact)]
        if not named:
            results.append(NumericClaim(token, value, "не тот показатель",
                                        candidates[0].fact_id))
            continue
        fact = named[0]
        results.append(NumericClaim(token, value, _unit_verdict(window, fact), fact.fact_id))
    return results


_YEAR = re.compile("(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])")


def _period_claims(text: str, evidence: Evidence) -> tuple[NumericClaim, ...]:
    """Годы, названные в тексте, — против объявленного периода.

    Период — часть числового кортежа, и до появления `Evidence.period` проверить
    его было не по чему: даты вычёркивались из проверки целиком, поэтому «прирост
    за 2019 год» с фактами за 2007…2025 выглядел безупречно. Если период не
    объявлен, проверка не выполняется вовсе — молча считать её пройденной значит
    заявлять больше, чем сделано.
    """
    if not evidence.period:
        return ()
    bounds = [int(match.group()) for match in _YEAR.finditer(evidence.period)]
    if not bounds:
        return ()
    first, last = min(bounds), max(bounds)
    found = []
    for match in _YEAR.finditer(text):
        year = int(match.group())
        inside = first <= year <= last
        found.append(NumericClaim(match.group(), float(year),
                                  "ok" if inside else "не тот период", "period"))
    return tuple(found)


def _names_quantity(window: str, fact: Fact) -> bool:
    """Названа ли рядом с числом сама величина, а не только её значение.

    Порог длины слова здесь ниже, чем у `_significant`: показатель может
    называться коротким словом — «ЧДД». Замер 22.08: при общем пороге в четыре
    буквы отвергался верный текст «ЧДД вырос с 11851.8 млн руб.», потому что
    якорями оставались «базового» и «сценария», которых модель не повторяет.
    Достаточно **любого** якоря: значение уже выбрало факт, окно лишь
    подтверждает, что речь о нём.
    """
    name = fact.fact_id.split(":", 1)[1] if ":" in fact.fact_id else fact.fact_id
    anchors = {word for word in _words(name.split(",")[0])
               if len(word) >= 3 and word not in _STOPWORDS}
    if not anchors:
        return True
    return any(anchor[:6] in window for anchor in anchors)


def _unit_verdict(window: str, fact: Fact) -> str:
    """Единица рядом с числом: та же, чужая или не названная вовсе."""
    if not fact.unit:
        return "ok"
    expected = fact.unit.lower().rstrip(".")
    if expected in window:
        return "ok"
    others = [unit for unit in UNIT_WORDS
              if unit.rstrip(".") != expected and unit in window]
    return "не та единица" if others else "единица не названа"


# ── свод по набору объяснений ────────────────────────────────────────────────

@dataclass(frozen=True)
class Measurement:
    """Одно измеренное объяснение: и метрики, и то, чем оно порождено."""

    fingerprint: str
    source: str
    stratum: str
    support: Support
    consistency: Consistency
    integrity: Integrity


@dataclass(frozen=True)
class Report:
    """Свод по набору. Числитель и знаменатель хранятся, а не только доля."""

    measurements: tuple[Measurement, ...]

    @property
    def asr(self) -> tuple[int, int]:
        return (sum(m.support.supported for m in self.measurements),
                sum(m.support.total for m in self.measurements))

    @property
    def asr_weighted(self) -> tuple[int, int]:
        return (sum(m.support.weighted_supported for m in self.measurements),
                sum(m.support.weighted_total for m in self.measurements))

    @property
    def dtc(self) -> tuple[int, int]:
        return (sum(1 for m in self.measurements if m.consistency.consistent),
                len(self.measurements))

    @property
    def nti(self) -> tuple[int, int]:
        return (sum(1 for m in self.measurements if m.integrity.passed),
                len(self.measurements))

    def confusion(self) -> dict[str, dict[str, int]]:
        """Матрица ошибок DTC: поле → исход → сколько раз."""
        table: dict[str, dict[str, int]] = {}
        for measurement in self.measurements:
            for name, state in measurement.consistency.fields.items():
                table.setdefault(name, {})[state] = table.setdefault(name, {}).get(state, 0) + 1
        return table

    def strata(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for measurement in self.measurements:
            counts[measurement.stratum] = counts.get(measurement.stratum, 0) + 1
        return counts


def measure(text: str, evidence: Evidence, *, source: str = "", stratum: str = "") -> Measurement:
    return Measurement(
        fingerprint=evidence.fingerprint(),
        source=source,
        stratum=stratum,
        support=support(text, evidence),
        consistency=consistency(text, evidence),
        integrity=integrity(text, evidence),
    )


def share(pair: tuple[int, int]) -> str:
    """Доля печатается вместе с числителем и знаменателем, а не вместо них."""
    numerator, denominator = pair
    if not denominator:
        return "нет данных (n=0)"
    return f"{numerator / denominator:.3f} ({numerator} из {denominator})"
