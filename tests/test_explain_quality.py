"""Метрики объяснимости: ASR, DTC, NTI — и инвариант, ради которого они заведены.

«Решения объяснимы» — это утверждение, а не свойство кода. До 22.08 оно
держалось на одной проверке (каждое число текста есть в расчёте), и всё, что
эта проверка не видит, было невидимо и для нас: чужой показатель, чужая единица,
чужой период, округление. Здесь закреплено, что именно каждая метрика ловит, —
и, отдельно, что непрошедший ответ **не может быть показан принятым**.

Тексты в тестах написаны руками: живого вызова модели в проекте ещё не было
(`BLOCKED.md`, п. 6). Это проверка защиты, а не замер языковой модели.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from aios import explain_quality as quality
from aios.explain import Evidence, Settings, explain, template_text


def evidence() -> Evidence:
    return Evidence(
        subject="цикл управления фондом: подбор режимов и остановка убыточных скважин",
        observations=("прогноз получен источником: тривиальный бейзлайн",
                      "типы принятых решений: остановка"),
        numbers={
            "ЧДД базового сценария, млн руб.": 11851.8,
            "ЧДД после решений, млн руб.": 12494.2,
            "прирост, млн руб.": 642.4,
            "решений об остановке": 3.0,
            "скважин с подобранным режимом": 41.0,
        },
        period="2007-01…2025-09",
        caveats=("прирост посчитан по прогнозу суррогата и полным расчётом ГГДМ "
                 "не подтверждён",),
    )


# ── разбор на утверждения ────────────────────────────────────────────────────

def test_an_abbreviation_does_not_end_a_claim():
    """«тыс. т» и «млн руб. до» — сокращения, а не конец предложения.

    Пока точка кончала утверждение всегда, единица уезжала в соседний «атом», и
    подмена единицы («642.4 тыс. т» вместо «млн руб.») проходила проверку: рядом
    с числом единицы уже не было. Найдено замером негативного набора 22.08.
    """
    assert quality.claims("Прирост составил 642.4 тыс. т.") == ("Прирост составил 642.4 тыс. т.",)
    assert len(quality.claims("Первое утверждение. Второе утверждение.")) == 2


# ── ASR ──────────────────────────────────────────────────────────────────────

def test_a_claim_without_a_source_is_counted_as_unsupported():
    """Утверждение, которого нет в фактах, обязано быть названо, а не растворяться."""
    text = template_text(evidence()) + "\n\nОбводнённость фонда снижается третий год подряд."

    result = quality.support(text, evidence())

    assert result.supported < result.total
    assert any("обводнённость" in check.claim.lower() for check in result.unsupported)


def test_the_template_text_is_fully_supported_by_its_own_facts():
    """Нижняя граница конвейера: шаблон собран из тех же строк, что и факты.

    Это не оценка языковой модели — это проверка, что извлекатель не теряет
    источник там, где он заведомо есть. ASR ниже единицы здесь означал бы
    дефект метрики, а не текста.
    """
    result = quality.support(template_text(evidence()), evidence())

    assert result.supported == result.total
    assert result.weighted_supported == result.weighted_total


# ── DTC ──────────────────────────────────────────────────────────────────────

def test_a_wrong_action_is_a_mismatch_not_a_silence():
    """«Остановили» и «запустили» — разные решения; путать их дороже всего."""
    text = "Система запустила скважины из простоя, прирост 642.4 млн руб."

    fields = quality.consistency(text, evidence()).fields

    assert fields["действие"] == "разошлось"


def test_saying_nothing_about_a_limit_is_not_a_mismatch():
    """Умолчание — неполнота, а не ложь: текст, не назвавший ограничение, не соврал.

    Метрика, складывающая «не сказано» с «разошлось», объявила бы краткий, но
    верный ответ ошибочным — и тем самым толкала бы объяснитель к многословию.
    """
    text = "Цикл остановил три убыточные скважины."

    fields = quality.consistency(text, evidence()).fields

    assert fields["ограничение"] == "не сказано"
    assert quality.consistency(text, evidence()).consistent


def test_a_limit_invented_out_of_nothing_is_a_mismatch():
    """Обратный случай: оговорок в решении нет, а текст ссылается на ограничение."""
    facts = Evidence(subject="цикл управления фондом", numbers={"прирост, млн руб.": 1.0})

    fields = quality.consistency("Решение упирается в ограничение по закачке.", facts).fields

    assert fields["ограничение"] == "разошлось"


def test_direction_is_checked_against_the_sign_not_against_words():
    text = "ЧДД снизился, прирост 642.4 млн руб."

    assert quality.consistency(text, evidence()).fields["направление"] == "разошлось"


# ── NTI ──────────────────────────────────────────────────────────────────────

def _mutations(text: str) -> dict[str, str]:
    """Восемь подмен из разбора 4-9. Каждая обязана быть отвергнута."""
    return {
        "чужое число": f"{text}\n\nПрирост составил 999.9 млн руб.",
        "не тот показатель": f"{text}\n\nОстановлено 642.4 скважин.",
        "подмена единицы": f"{text}\n\nПрирост составил 642.4 тыс. т.",
        "смена знака": f"{text}\n\nПрирост составил -642.4 млн руб.",
        "округление": f"{text}\n\nПрирост составил 642.0 млн руб.",
        "потерян разделитель": f"{text}\n\nПрирост составил 6424 млн руб.",
        "чужой период": f"{text}\n\nПрирост 642.4 млн руб. получен в 1998 году.",
        "свой пересчёт": f"{text}\n\nЭто рост на 5.4 %.",
    }


@pytest.mark.parametrize("case", sorted(_mutations("")))
def test_every_substitution_of_the_numeric_tuple_is_refused(case: str):
    """Замер 22.08: прежняя сверка чисел ловила пять подмен из восьми.

    Невидимыми для неё были подмена показателя, подмена единицы и округление —
    то есть ровно те случаи, где число «есть в расчёте», но относится не к тому. Это и есть разница между «число знакомое» и «кортеж
    верный».
    """
    mutated = _mutations(template_text(evidence()))[case]

    assert not quality.integrity(mutated, evidence()).passed, f"подмена прошла: {case}"


def test_a_correct_text_is_not_rejected():
    """Ложное отклонение так же вредно, как пропущенная ошибка, — и вероятнее.

    Пересказ своими словами с единицами — законный ответ, и барьер обязан его
    пропустить. Проверка стоит здесь потому, что первый живой вызов планируется
    один: отвергнутый верный ответ будет стоить нам этого единственного замера.
    """
    text = ("Цикл остановил 3 убыточные скважины и подобрал режим 41 скважине.\n\n"
            "ЧДД вырос с 11851.8 млн руб. до 12494.2 млн руб., прирост 642.4 млн руб.\n\n"
            "Прирост посчитан по прогнозу суррогата и полным расчётом не подтверждён.")

    assert quality.integrity(text, evidence()).passed


def test_a_number_without_a_unit_is_incompleteness_not_a_failure():
    """«Единица не названа» — мягкий вердикт: значение верное, кортеж неполон."""
    text = "Прирост составил 642.4 — против базового ЧДД 11851.8."

    result = quality.integrity(text, evidence())

    assert result.passed
    assert len(result.unnamed_units) == 2


def test_a_year_inside_the_declared_period_is_not_a_stray_number():
    """Год из объявленного периода — ссылка, а не выдуманная величина."""
    inside = "Прирост 642.4 млн руб. получен к 2019 году."
    outside = "Прирост 642.4 млн руб. получен в 1998 году."

    assert quality.integrity(inside, evidence()).passed
    assert not quality.integrity(outside, evidence()).passed


def test_the_period_is_not_checked_when_it_is_not_declared():
    """Нет поля — нет проверки, а не «проверка пройдена»."""
    facts = Evidence(subject="цикл", numbers={"прирост, млн руб.": 642.4})

    checked = quality.integrity("Прирост 642.4 млн руб. за 1998 год.", facts)

    assert not any(number.matched == "period" for number in checked.numbers)


# ── инвариант: непрошедший ответ не показывается принятым ────────────────────

@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_Block]


class _FakeMessages:
    def __init__(self, payload: dict[str, str]):
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response([_Block(json.dumps(self.payload, ensure_ascii=False))])


class _FakeClient:
    def __init__(self, payload: dict[str, str]):
        self.messages = _FakeMessages(payload)


def test_an_answer_with_a_broken_tuple_is_never_shown_as_accepted(tmp_path: Path):
    """Главный инвариант протокола 4-9, и он обязан держаться на коде.

    Ответ, не прошедший серверную проверку, не показывается принятым **нигде**:
    источник становится `template-rejected`, `from_model` — ложью, текст —
    шаблонным, и в кэш такой ответ не попадает. Иначе завтрашний единственный
    живой вызов запишет в артефакт объяснение, которое мы сами же забраковали.

    Здесь ломается именно кортеж, а не значение: число `642.4` в расчёте есть,
    но оно про деньги, а не про скважины. Прежняя сверка чисел это пропускала.
    """
    client = _FakeClient({
        "what_happened": "Цикл прошёл по фонду.",
        "why": "Остановлено 642.4 скважин.",
        "confirmation": "Прогноз суррогата, не подтверждено.",
    })
    settings = Settings(enabled=True, cache_dir=tmp_path)

    result = explain(evidence(), settings, client=client)

    assert result.source == "template-rejected"
    assert not result.from_model
    assert any("не тот показатель" in item for item in result.rejected_numbers)
    assert result.text == template_text(evidence())
    assert list(tmp_path.glob("*.json")) == [], "отвергнутый ответ не может попасть в кэш"


def test_a_rejection_lists_each_number_once():
    """Барьера два, число одно: выдуманное число не должно считаться дважды."""
    client = _FakeClient({
        "what_happened": "Цикл прошёл по фонду.",
        "why": "Прирост составил 2500.0 млн руб.",
        "confirmation": "Прогноз суррогата, не подтверждено.",
    })

    result = explain(evidence(), Settings(enabled=True), client=client)

    tokens = [item.split(" ")[0] for item in result.rejected_numbers]
    assert tokens.count("2500.0") == 1


# ── свод ─────────────────────────────────────────────────────────────────────

def test_the_report_keeps_numerator_and_denominator_not_only_the_share():
    """«17 из 19» информативнее, чем «0.89» — разбор требует печатать оба."""
    measurement = quality.measure(template_text(evidence()), evidence(),
                                  source="шаблон", stratum="проба")
    report = quality.Report((measurement,))

    numerator, denominator = report.asr
    assert denominator > 0
    assert quality.share(report.asr) == f"{numerator / denominator:.3f} ({numerator} из {denominator})"
    assert quality.share((0, 0)) == "нет данных (n=0)"


def test_the_template_passes_its_own_barrier():
    """Шаблон обязан проходить обе проверки — иначе отвергается вообще всё.

    Случай не выдуманный: правка, вычёркивающая годы внутри объявленного
    периода, сначала выполнялась **до** удаления полных дат. Из «2007-01» уходил
    год, остаток «-01» читался как число −1, и барьер отвергал любой текст,
    в котором есть период, — включая наш собственный шаблон. Проверка «прогони
    шаблон против собственного барьера» ловит такие правки сразу.
    """
    from aios.explain import unverified_numbers

    text = template_text(evidence())

    assert unverified_numbers(text, evidence()) == ()
    assert quality.integrity(text, evidence()).passed
