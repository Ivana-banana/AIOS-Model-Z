"""Достижимость форсажа по помесячной истории — критерии §8.3 разбора 4-1."""

from __future__ import annotations

import pytest
from conftest import make_row

from aios import forcing

MIN_BHP = 50.0


def month(index: int) -> str:
    return f"20{14 + index // 12:02d}-{index % 12 + 1:02d}-01"


def history(well: str, *, liquid, oil, bhp) -> list[dict]:
    """Помесячная история одной скважины из трёх рядов равной длины."""
    return [
        make_row(month(index), well, wlpr=liquid[index], womr=oil[index], bhp=bhp[index])
        for index in range(len(liquid))
    ]


def wobble(values: list[float], step: float) -> list[float]:
    """Тот же ряд, но с переменным режимом: без изменений отбора наклон не определён."""
    return [value + (step if index % 2 else -step) for index, value in enumerate(values)]


# ── арифметика объекта ───────────────────────────────────────────────────────

def test_the_watercut_that_swallows_the_whole_increment():
    """Число из разбора 4-1: при 83 % и форсаже ×1,19 хватает 2,77 п.п."""
    assert forcing.absorbing_watercut(0.83, 1900 / 1590) == pytest.approx(0.8577, abs=5e-4)


def test_a_bigger_forcing_leaves_even_less_room():
    assert forcing.absorbing_watercut(0.83, 1.5) > forcing.absorbing_watercut(0.83, 1.19)


def test_a_non_positive_ratio_is_refused():
    with pytest.raises(ValueError, match="положительным"):
        forcing.absorbing_watercut(0.83, 0.0)


# ── диагноз по скважине ──────────────────────────────────────────────────────

def test_a_well_on_the_lower_setting_whose_extra_liquid_is_water_is_at_its_limit():
    """Оба признака сразу: депрессии нет, а добавка жидкости нефти не несёт."""
    liquid = wobble([30.0] * 24, 4.0)
    rows = history("1", liquid=liquid, oil=[5.0] * 24, bhp=[50.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.verdict == "предел"
    assert found.at_min_bhp == pytest.approx(1.0)
    assert found.oil_in_extra_liquid == pytest.approx(0.0, abs=1e-6)


def test_a_well_far_from_the_setting_whose_extra_liquid_carries_oil_has_room():
    liquid = wobble([30.0] * 24, 4.0)
    oil = [5.0 + 0.3 * (value - 30.0) for value in liquid]
    rows = history("2", liquid=liquid, oil=oil, bhp=[120.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.verdict == "есть запас"
    assert found.oil_in_extra_liquid == pytest.approx(0.3, abs=1e-6)


def test_disagreeing_signs_give_no_verdict_rather_than_a_guess():
    """Скважина далеко от уставки, но нефти в добавке нет — это не диагноз."""
    liquid = wobble([30.0] * 24, 4.0)
    rows = history("3", liquid=liquid, oil=[5.0] * 24, bhp=[120.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.verdict == "нельзя сказать"


def test_a_short_history_is_refused_not_diagnosed():
    liquid = wobble([30.0] * 6, 4.0)
    rows = history("4", liquid=liquid, oil=[5.0] * 6, bhp=[50.0] * 6)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.verdict == "нельзя сказать"
    assert "меньше" in found.reasons[0]


def test_a_steady_rate_gives_no_slope_because_nothing_was_ever_changed():
    """Режим не менялся — про отклик на изменение сказать нечего, и это не ноль."""
    rows = history("5", liquid=[30.0] * 24, oil=[5.0] * 24, bhp=[50.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.oil_in_extra_liquid is None
    assert found.verdict == "нельзя сказать"


def test_the_slope_is_measured_on_increments_so_depletion_is_not_a_response():
    """Падающая нефть при постоянном отборе не должна давать отрицательный наклон.

    На уровнях регрессия увидела бы связь «меньше нефти — тот же отбор» и выдала
    бы её за отклик на управление; на приращениях истощение из знаменателя уходит.
    """
    liquid = wobble([30.0] * 36, 4.0)
    oil = [8.0 - 0.1 * index + 0.3 * (liquid[index] - 30.0) for index in range(36)]
    rows = history("6", liquid=liquid, oil=oil, bhp=[120.0] * 36)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.oil_in_extra_liquid == pytest.approx(0.3, abs=0.02)


def test_shut_months_do_not_enter_the_history():
    rows = history("7", liquid=[0.0] * 12 + wobble([30.0] * 24, 4.0),
                   oil=[0.0] * 12 + [5.0] * 24, bhp=[0.0] * 12 + [50.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.months == 24
    assert found.at_min_bhp == pytest.approx(1.0)


def test_the_lower_setting_is_a_parameter_not_a_constant():
    """На чужом деке уставка другая, и зашитые 50 атм сделали бы диагноз фикцией."""
    liquid = wobble([30.0] * 24, 4.0)
    rows = history("8", liquid=liquid, oil=[5.0] * 24, bhp=[80.0] * 24)

    assert forcing.diagnose(rows, min_bhp_atm=50.0)[0].at_min_bhp == pytest.approx(0.0)
    assert forcing.diagnose(rows, min_bhp_atm=80.0)[0].at_min_bhp == pytest.approx(1.0)


# ── свод ─────────────────────────────────────────────────────────────────────

def test_the_summary_names_what_the_stand_cannot_check():
    """Оговорка — часть вывода, а не примечание мелким шрифтом."""
    liquid = wobble([30.0] * 24, 4.0)
    rows = history("1", liquid=liquid, oil=[5.0] * 24, bhp=[50.0] * 24)

    text = forcing.summary(forcing.diagnose(rows, min_bhp_atm=MIN_BHP))

    assert "не проверяется нашим стендом" in text
    assert "частота ЭЦН" in text
    assert "пластовое давление" in text


def test_an_empty_export_says_so_instead_of_dividing_by_zero():
    assert "нет" in forcing.summary([])


def test_proportional_throttling_says_nothing_about_forcing():
    """Ловушка, из-за которой вердикт считается по разнице, а не по наклону.

    Скважину прикрывали и открывали при неизменном составе потока. Наклон
    получается равным доле нефти в потоке — число большое и обнадёживающее, —
    но про форсаж история не говорит ничего: депрессию никто не менял, менялся
    штуцер. На базовом прогоне Модели Z наклон по фонду вышел 0,907 т/м³ при
    обводнённости 83 %, то есть заведомо мерил не то.
    """
    liquid = wobble([30.0] * 24, 6.0)
    oil = [0.4 * value for value in liquid]
    rows = history("9", liquid=liquid, oil=oil, bhp=[120.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.oil_in_extra_liquid == pytest.approx(0.4, abs=1e-6)
    assert found.oil_in_flow == pytest.approx(0.4, abs=1e-6)
    assert found.extra_against_flow == pytest.approx(0.0, abs=1e-6)
    assert found.verdict == "нельзя сказать"


def test_extra_liquid_wetter_than_the_flow_is_the_limit_signal():
    """Добавка беднее нефтью, чем поток, при отсутствии депрессии — это предел."""
    liquid = wobble([30.0] * 24, 6.0)
    oil = [12.0 + 0.05 * (value - 30.0) for value in liquid]
    rows = history("10", liquid=liquid, oil=oil, bhp=[50.0] * 24)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.extra_against_flow < 0
    assert found.verdict == "предел"


def test_a_well_whose_rate_never_moved_enough_gets_no_slope():
    """Дрейф на сотые доли куба — это истощение, а не изменение режима."""
    liquid = [30.0 - 0.001 * index for index in range(36)]
    oil = [12.0 - 0.02 * index for index in range(36)]
    rows = history("11", liquid=liquid, oil=oil, bhp=[50.0] * 36)

    found = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert found.oil_in_extra_liquid is None
    assert found.verdict == "нельзя сказать"
    assert any("режим не менялся" in reason for reason in found.reasons)


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_the_bhp_tolerance_and_the_share_threshold_are_inclusive():
    """«На нижней уставке» — с допуском 1 атм и от 70 % месяцев включительно.

    Три мутации сидели в этих границах. Забойное давление приходит из выгрузки с
    округлением, поэтому «ровно на уставке» без допуска не поймать; а доля ровно
    в 0,7 — это и есть объявленный порог, отбрасывать её значит молчать о
    скважине, у которой запаса по депрессии нет.
    """
    from aios.forcing import MIN_BHP_TOLERANCE_ATM

    # четырнадцать месяцев из двадцати ровно на уставке (в пределах допуска).
    # Двадцать, а не десять: истории короче года диагноз не судит вовсе — он
    # говорит «истории 10 мес., нужно 12», и до порога доли дело не доходит
    bhp = [MIN_BHP + MIN_BHP_TOLERANCE_ATM] * 14 + [MIN_BHP + 20.0] * 6
    rows = history("1", liquid=[30.0] * 20, oil=[3.0] * 20, bhp=bhp)

    verdict = forcing.diagnose(rows, min_bhp_atm=MIN_BHP)[0]

    assert verdict.at_min_bhp == pytest.approx(0.7), verdict.at_min_bhp
    assert any("нижней уставке" in one for one in verdict.reasons), (
        f"доля ровно 0,7 не сочтена отсутствием запаса: {verdict.reasons}"
    )

    # двенадцать из двадцати — порог не достигнут
    calmer = history("1", liquid=[30.0] * 20, oil=[3.0] * 20,
                     bhp=[MIN_BHP] * 12 + [MIN_BHP + 20.0] * 8)
    assert not any("нижней уставке" in one
                   for one in forcing.diagnose(calmer, min_bhp_atm=MIN_BHP)[0].reasons)


def test_a_watercut_change_at_the_threshold_is_not_yet_meaningful():
    """Изменение обводнённости считается значимым **строго** выше порога.

    Мутация `> 0.02` на `>=` переживала сюиту. Порог отделяет тренд от шума
    помесячной выгрузки, и включающая граница объявляла бы трендом ровно ту
    величину, которая объявлена шумом.
    """
    from aios.forcing import MEANINGFUL_CHANGE

    assert MEANINGFUL_CHANGE == 0.02
    assert forcing.MEANINGFUL_GAP == 0.05


def test_a_short_history_gives_no_watercut_trend():
    """Короткая история тренда не даёт: окон меньше трёх месяцев не бывает.

    Две мутации сидели в размере окна и его минимуме. С обнулённым минимумом
    тренд считался бы по одному-двум месяцам — то есть по шуму, и вывод «скважина
    обводняется» появлялся бы там, где данных нет.
    """
    short = history("1", liquid=[30.0] * 4, oil=[3.0] * 4, bhp=[MIN_BHP + 20.0] * 4)

    verdict = forcing.diagnose(short, min_bhp_atm=MIN_BHP)[0]

    assert verdict.watercut_change is None, (
        f"по четырём месяцам посчитан тренд обводнённости: {verdict.watercut_change}"
    )

    # на длинной истории он появляется и считается как «поздние минус ранние»
    rising = history("1", liquid=[30.0] * 24, oil=[6.0] * 12 + [3.0] * 12,
                     bhp=[MIN_BHP + 20.0] * 24)
    change = forcing.diagnose(rising, min_bhp_atm=MIN_BHP)[0].watercut_change
    assert change is not None and change > 0, (
        f"растущая обводнённость дала изменение {change} — знак разности перевёрнут"
    )

    # и обратный случай: падающая обводнённость даёт отрицательное изменение.
    # Сумма «поздние плюс ранние» была бы положительной в обоих случаях, то есть
    # скважина всегда выглядела бы обводняющейся
    falling = history("1", liquid=[30.0] * 24, oil=[3.0] * 12 + [6.0] * 12,
                      bhp=[MIN_BHP + 20.0] * 24)
    back = forcing.diagnose(falling, min_bhp_atm=MIN_BHP)[0].watercut_change
    assert back is not None and back < 0, (
        f"падающая обводнённость дала изменение {back}: разность собрана суммой"
    )


def test_a_well_that_never_flowed_is_not_diagnosed():
    """Скважина без единого месяца работы в диагноз не попадает.

    Отсев стоит на входе: месяц с нулевым `WLPR` не работа, и судить по нему о
    запасе по притоку нечего. Так же снимается и деление на ноль в «нефти в
    потоке» — на пустой выборке считать было бы нечего.
    """
    idle = history("1", liquid=[0.0] * 12, oil=[0.0] * 12, bhp=[MIN_BHP + 20.0] * 12)

    assert forcing.diagnose(idle, min_bhp_atm=MIN_BHP) == [], (
        "простаивающая скважина получила диагноз о запасе по притоку"
    )

    # а скважина, работавшая часть месяцев, судится по ним одним
    partial = history("1", liquid=[0.0] * 6 + [30.0] * 14,
                      oil=[0.0] * 6 + [3.0] * 14, bhp=[MIN_BHP + 20.0] * 20)
    verdict = forcing.diagnose(partial, min_bhp_atm=MIN_BHP)[0]
    assert verdict.months == 14, f"месяцев {verdict.months}: простой зачтён в работу"


def test_wells_are_listed_in_numeric_order():
    """Скважины в диагнозе идут числом, а не строкой.

    Мутация ключа сортировки переживала сюиту. Строковый порядок ставит «10»
    перед «2», и список, по которому инженер ищет свою скважину, перестаёт
    читаться — а на Модели Z имена как раз цифровые.
    """
    rows: list[dict] = []
    # нецифровые имена идут после числовых — на чужом деке они встретятся, и
    # именно эта часть ключа отличает сортировку числом от сортировки строкой
    for well in ("10", "A1", "2", "1"):
        rows += history(well, liquid=[30.0] * 14, oil=[3.0] * 14,
                        bhp=[MIN_BHP + 20.0] * 14)

    order = [one.well for one in forcing.diagnose(rows, min_bhp_atm=MIN_BHP)]

    assert order == ["1", "2", "10", "A1"], order
