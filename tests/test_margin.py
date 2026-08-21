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
