"""Экономический предел скважины.

Главный тест здесь — сверка с эталонным расчётчиком: маржа обязана складываться в
ту же EBITDA, иначе бейзлайн отбирает скважины не по тем деньгам.
"""

from __future__ import annotations

import pytest
from conftest import make_row, months

from aios import economics
from aios.margin import (
    Norms,
    breakeven_oil_t,
    breakeven_watercut,
    counted_by_methodology,
    is_active,
    month_margin,
    screen,
    shutdown_candidates,
)


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def test_norms_are_read_from_the_organisers_file(norms):
    assert norms.oil_price_m_per_t == pytest.approx(0.028)
    assert norms.deductions_m_per_t == pytest.approx(0.0196)
    assert norms.oil_netback_m_per_t == pytest.approx(0.028 - 0.0196 - 0.00004)
    assert norms.fund_m_per_well_month == pytest.approx(1.0 / 12)


def test_month_margin_matches_the_reference_ebitda(norms):
    """Сумма месячных марж = EBITDA года у расчётчика (событий в сценарии нет)."""
    rows = [
        make_row(data, "A", wlpr=30, womr=5, liquid=900, oil=150)
        for data in months(2014, 12)
    ]
    result = economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )

    ours = sum(month_margin(row, norms) for row in rows)
    assert ours == pytest.approx(result.raw["annual"][0]["ebitdaM"], abs=1e-9)


def test_month_margin_matches_the_reference_for_an_injector(norms):
    rows = [make_row(data, "B", wwir=50, injection=1500) for data in months(2014, 12)]
    result = economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )

    ours = sum(month_margin(row, norms) for row in rows)
    assert ours == pytest.approx(result.raw["annual"][0]["ebitdaM"], abs=1e-9)
    assert ours < 0  # закачка сама по себе денег не приносит, она их стоит


def test_stopped_month_costs_nothing(norms):
    """Остановленная скважина выпадает из фонда — её месяц не стоит 1/12 млн."""
    stopped = make_row("2014-01-01", "A")
    assert not is_active(stopped)
    assert month_margin(stopped, norms) == pytest.approx(0.0)


def test_breakeven_watercut_is_the_point_where_margin_turns_zero(norms):
    liquid_t = 1000.0
    watercut = breakeven_watercut(liquid_t, norms)
    oil_t = liquid_t * (1 - watercut)

    row = make_row("2014-01-01", "A", wlpr=30, womr=1, liquid=liquid_t, oil=oil_t)
    assert month_margin(row, norms) == pytest.approx(0.0, abs=1e-9)
    assert oil_t == pytest.approx(breakeven_oil_t(liquid_t, norms))


def test_tiny_well_is_unprofitable_at_any_watercut(norms):
    """Если отбор мал, содержание фонда не отбивается даже на чистой нефти."""
    assert breakeven_watercut(5.0, norms) == 0.0


def test_screen_puts_the_worst_well_first(norms):
    good = [make_row(data, "good", wlpr=30, womr=5, liquid=900, oil=150) for data in months(2014, 6)]
    bad = [make_row(data, "bad", wlpr=30, womr=0.1, liquid=900, oil=3) for data in months(2014, 6)]

    verdicts = screen(good + bad, norms)
    assert [verdict.well for verdict in verdicts] == ["bad", "good"]
    assert verdicts[0].total_margin_m < 0 < verdicts[1].total_margin_m
    assert verdicts[0].negative_months == 6
    assert verdicts[0].trailing_negative_months == 6


def test_shutdown_candidates_need_a_sustained_loss(norms):
    """Один плохой месяц — не повод дёргать фонд."""
    rows = [make_row(data, "A", wlpr=30, womr=5, liquid=900, oil=150) for data in months(2014, 11)]
    rows.append(make_row("2014-12-01", "A", wlpr=30, womr=0.1, liquid=900, oil=3))

    assert shutdown_candidates(rows, norms) == []


def test_shutdown_candidates_need_the_loss_to_cover_the_stop_cost(norms):
    """Остановка стоит 1,0 млн руб. — убыток обязан её перекрыть."""
    barely = [
        make_row(data, "A", wlpr=1, womr=0.3, liquid=30, oil=9.5) for data in months(2014, 6)
    ]
    verdict = screen(barely, norms)[0]
    assert verdict.trailing_negative_months == 6
    assert -verdict.trailing_margin_m < norms.stop_start_m
    assert shutdown_candidates(barely, norms) == []


def test_shutdown_candidate_is_found_when_both_conditions_hold(norms):
    rows = [make_row(data, "A", wlpr=30, womr=0.1, liquid=900, oil=3) for data in months(2014, 12)]
    candidates = shutdown_candidates(rows, norms)

    assert [verdict.well for verdict in candidates] == ["A"]
    assert candidates[0].trailing_negative_months == 12
    assert -candidates[0].trailing_margin_m > norms.stop_start_m
    assert candidates[0].watercut > 0.99


def test_shutting_a_screened_well_actually_raises_the_reference_npv(norms):
    """Проверка бейзлайна тем же числом, каким нас будут оценивать.

    Скважину не «убираем из данных», а останавливаем: платим 1,0 млн за остановку
    и дальше отдаём нулевые месяцы, как это и выглядит в выгрузке симулятора.
    """
    good = [make_row(data, "good", wlpr=30, womr=5, liquid=900, oil=150)
            for data in months(2014, 12)]
    bad = [make_row(data, "bad", wlpr=30, womr=0.1, liquid=900, oil=3)
           for data in months(2014, 12)]

    as_is = economics.compute_npv(
        good + bad, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )

    assert [verdict.well for verdict in shutdown_candidates(good + bad, norms)] == ["bad"]

    stopped = good + bad[:1] + [make_row(data, "bad") for data in months(2014, 11, first=2)]
    after = economics.compute_npv(
        stopped, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )

    assert after.start_stop_count == 1          # одна остановка, пуска не было
    assert after.chdd_m > as_is.chdd_m


def test_the_screen_matches_the_reference_ebitda_even_with_an_excluded_row(norms):
    """Равенство «сумма марж = EBITDA» обязано держаться и на отрицательных строках.

    Оно проверялось только на чистой выгрузке — и там держалось. Методика же
    исключает строку с отрицательным приращением **целиком**, а `screen`
    суммировал её как есть. Замер 21.08 на двенадцати месяцах с одной такой
    строкой: наша сумма 10,460 против EBITDA расчётчика 11,887 — расхождение
    1,427 млн руб. на одну строку. После фильтрации совпадение точное.

    Цена не только в числе: страница прогона утверждает, что скважины
    отбираются «по тем же деньгам, по каким нас будут оценивать», и ссылается
    на этот самый тест. Утверждение обязано быть верным на том материале, который
    бывает: 102 отрицательные строки за 2025-09 в выгрузке организаторов.
    """
    rows = [make_row(f"2007-{month:02d}-01", "1", wlpr=30, womr=5, liquid=900, oil=150)
            for month in range(1, 13)]
    rows[5]["WOMT_Diff"] = -150.0

    ours = sum(verdict.total_margin_m for verdict in screen(rows, norms))
    theirs = economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date="2007-01-01"
    ).raw["annual"][0]["ebitdaM"]

    assert ours == pytest.approx(theirs, abs=1e-9)


def test_a_verdict_counts_money_and_tonnes_over_the_same_rows(norms):
    """Внутри одной строки таблицы всё обязано быть посчитано по одному набору.

    После фильтрации исключаемых строк маржа считалась по одному набору, а
    тонны, месяцы в работе и `average_margin_m` — по другому, полному. На
    странице прогона это стоит рядом в одной строке: «маржа 10,46 / нефть
    1 950 т / мес. в работе 12», где деньги уже без исключённой строки, а тонны
    и месяцы всё ещё с ней.

    Правило то же, что вывело сегодняшнюю серию находок: величины, стоящие рядом
    как объяснение одного решения, обязаны быть посчитаны по одному материалу.
    """
    rows = [make_row(f"2007-{month:02d}-01", "1", wlpr=30, womr=5, liquid=900, oil=150)
            for month in range(1, 13)]
    rows[5]["WOMT_Diff"] = -150.0
    rows[5]["WLPT_Diff"] = -900.0

    verdict = screen(rows, norms)[0]
    kept = [row for row in rows if counted_by_methodology(row)]

    assert verdict.months == len(kept), "месяцы считаются по всем строкам, а деньги — нет"
    assert verdict.oil_t == pytest.approx(sum(row["WOMT_Diff"] for row in kept))
    assert verdict.liquid_t == pytest.approx(sum(row["WLPT_Diff"] for row in kept))
    assert 0.0 <= verdict.watercut <= 1.0, (
        f"обводнённость {verdict.watercut:.3f} вне физического диапазона"
    )
    assert verdict.average_margin_m == pytest.approx(verdict.total_margin_m / len(kept))


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_the_wacc_is_read_as_a_fraction_not_a_percent(norms):
    """WACC в нормативах — проценты, в расчёте — доля.

    Мутация делила на сто вместо умножения... вернее, наоборот: подмена `/ 100`
    на `* 100` даёт ставку 1000 вместо 0,1. Дисконт после этого не «немного
    другой», а другой на порядки, и любое сравнение сценариев теряет смысл.
    Сюита этого не видела: ставка проверялась косвенно, через готовые числа.
    """
    assert norms.wacc_rate == pytest.approx(0.1), (
        f"ставка дисконтирования {norms.wacc_rate} — проценты приняты за долю "
        "или наоборот"
    )
    assert 0.0 < norms.wacc_rate < 1.0


def test_the_norms_cannot_be_edited_after_loading():
    """Нормативы неизменяемы: они приходят из файла организаторов."""
    import dataclasses

    from aios import economics

    assumptions, _ = economics.load_norms()
    loaded = Norms.from_assumptions(assumptions)

    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.oil_price_m_per_t = 0.0


def test_a_zero_increment_row_is_counted_not_excluded():
    """Методика исключает **отрицательные** приросты, а не нулевые.

    Две мутации меняли `< 0` на `<= 0`, и обе переживали сюиту: строки с ровно
    нулевым приростом в фикстурах не было. Между тем месяц простоя даёт ровно
    её, и исключать такие строки целиком значило бы вычёркивать из экономики
    остановленные скважины — вместе с их постоянными затратами.
    """
    zero = make_row("2014-01-01", "1", wlpr=0.0, womr=0.0,
                    liquid=0.0, oil=0.0, injection=0.0)

    assert counted_by_methodology(zero), (
        "строка с нулевым приростом исключена из экономики — методика исключает "
        "только отрицательные"
    )

    negative = make_row("2014-02-01", "1", wlpr=0.0, womr=0.0, liquid=-1.0, oil=0.0)
    assert not counted_by_methodology(negative)


def test_a_well_without_liquid_has_no_watercut_and_no_breakeven():
    """Нулевой отбор: обводнённость и предельная обводнённость — нули, не деления.

    Две мутации меняли границу `> 0` и `<= 0` на нестрогие. Разница проявляется
    ровно на нуле, а нуль здесь обычное дело: остановленная скважина даёт его
    каждый месяц простоя.
    """
    assert breakeven_watercut(0.0, norms_for_test()) == 0.0
    assert breakeven_watercut(-1.0, norms_for_test()) == 0.0
    assert breakeven_watercut(1000.0, norms_for_test()) > 0.0


def norms_for_test() -> Norms:
    """Нормативы из файла — в тестах граничных случаев они те же, что в расчёте."""
    from aios import economics

    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def test_the_verdict_counts_active_and_losing_months(norms):
    """Вердикт считает действующие и убыточные месяцы, а не подставляет числа.

    Две мутации сидели в счётчиках: число действующих месяцев и число месяцев с
    отрицательной маржой. По ним отбираются кандидаты на остановку — то есть
    решения на 1,0 млн руб. каждое.
    """
    stamps = months(2014, 4)
    rows = [
        make_row(stamps[0], "w", wlpr=30, womr=5, liquid=900, oil=150),   # прибыль
        make_row(stamps[1], "w", wlpr=30, womr=0.1, liquid=900, oil=3),   # убыток
        make_row(stamps[2], "w", wlpr=0.0, womr=0.0, liquid=0.0, oil=0.0),  # простой
        make_row(stamps[3], "w", wlpr=30, womr=0.1, liquid=900, oil=3),   # убыток
    ]

    verdict = screen(rows, norms)[0]

    assert verdict.months == 4, "нулевая строка выпала из счёта месяцев"
    assert verdict.active_months == 3, (
        f"действующих месяцев {verdict.active_months}: простой посчитан работой "
        "или наоборот"
    )
    # месяц простоя даёт маржу **ровно ноль**, а не убыток: постоянные затраты
    # начисляются иначе. Поэтому убыточных ровно два — и это же отличает
    # `margin < 0` от `<= 0`, где нулевой месяц зачёлся бы третьим
    assert verdict.negative_months == 2, (
        f"убыточных месяцев {verdict.negative_months}: месяц с нулевой маржой "
        "зачтён убыточным"
    )
    assert verdict.trailing_negative_months == 1, (
        "серия убытков считается с конца и обрывается на первом неубыточном"
    )


def test_a_shutdown_needs_the_loss_to_cover_its_own_cost(norms):
    """Остановка предлагается, только когда убыток перекрывает её стоимость.

    Мутация `< stop_start_m` на `<=` переживала сюиту: случая, где накопленный
    убыток **ровно** равен стоимости остановки, в фикстурах не было. Граница
    здесь содержательная: остановка и пуск — два платных события, и «окупается
    ровно в ноль» это не выигрыш, а размен.
    """
    stamps = months(2014, 8)
    rows = [make_row(stamp, "w", wlpr=30, womr=0.1, liquid=900, oil=3)
            for stamp in stamps]

    candidates = shutdown_candidates(rows, norms, min_trailing_months=6)

    if candidates:
        assert -candidates[0].trailing_margin_m >= norms.stop_start_m, (
            "кандидат предложен при убытке меньше стоимости остановки"
        )
    # и наоборот: двух убыточных месяцев не хватает ни по серии, ни по деньгам
    assert shutdown_candidates(rows[:2], norms, min_trailing_months=6) == []


def test_a_verdict_without_liquid_reports_zero_watercut(norms):
    """Обводнённость вердикта при нулевой жидкости — ноль, а не деление.

    Прежний тест проверял `breakeven_watercut`, а мутация сидела в свойстве
    самого вердикта: две разные функции с одинаковой на вид защитой, и закрыта
    была одна.
    """
    stamps = months(2014, 3)
    idle = [make_row(stamp, "w", wlpr=0.0, womr=0.0, liquid=0.0, oil=0.0)
            for stamp in stamps]

    verdict = screen(idle, norms)[0]

    assert verdict.liquid_t == 0.0
    assert verdict.watercut == 0.0, "обводнённость простоя ненулевая"
    assert verdict.average_margin_m == pytest.approx(0.0)


def test_a_loss_exactly_equal_to_the_shutdown_cost_is_already_enough(norms):
    """Убыток ровно в стоимость остановки — уже основание её предложить.

    Мутация `< stop_start_m` на `<=` переживала сюиту: случай точного равенства
    в фикстурах не встречался, а построить его можно только подбором. Объём
    2500/3 т жидкости даёт маржу −1/6 млн руб. в месяц, шесть месяцев — ровно
    1,0 млн, то есть ровно стоимость остановки (замер 06.09).

    Граница включающая, и это осознанно: отсеиваются те, чей убыток **меньше**
    стоимости. Равенство означает, что остановка окупается в точности, и
    отбрасывать такую скважину не за что. С подменой на `<=` она молча выпадала
    бы из кандидатов — вместе со всеми, кто стоит ровно на границе.
    """
    stamps = months(2014, 6)
    on_the_edge = [make_row(stamp, "w", wlpr=30, womr=0.1,
                            liquid=2500.0 / 3, oil=0.0) for stamp in stamps]

    verdict = screen(on_the_edge, norms)[0]
    assert verdict.trailing_negative_months == 6
    assert -verdict.trailing_margin_m == pytest.approx(norms.stop_start_m, abs=1e-9), (
        f"проверка вырождена: накоплено {-verdict.trailing_margin_m}, "
        f"а нужно ровно {norms.stop_start_m}"
    )

    assert shutdown_candidates(on_the_edge, norms, min_trailing_months=6), (
        "скважина с убытком ровно в стоимость остановки выпала из кандидатов: "
        "граница стала строгой с обеих сторон"
    )

    # а вот убыток меньше стоимости кандидата не даёт — граница работает
    below = [make_row(stamp, "w", wlpr=30, womr=0.1, liquid=800.0, oil=0.0)
             for stamp in stamps]
    verdict_below = screen(below, norms)[0]
    assert -verdict_below.trailing_margin_m < norms.stop_start_m
    assert shutdown_candidates(below, norms, min_trailing_months=6) == [], (
        "остановка предложена при убытке меньше её стоимости"
    )

