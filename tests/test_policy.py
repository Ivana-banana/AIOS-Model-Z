"""Бейзлайн-политика: от диагноза к расписанию."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row, months

from aios import economics, policy
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms
from aios.schedule import validate

CONTROL_START = date(2014, 1, 1)


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def losing(data: str, well: str = "bad"):
    """Скважина с почти чистой водой: убыток около 0,17 млн руб. в месяц."""
    return make_row(data, well, wlpr=30, womr=0.1, liquid=900, oil=3)


def earning(data: str, well: str = "good"):
    return make_row(data, well, wlpr=30, womr=5, liquid=900, oil=150)


def test_no_actions_while_the_loss_is_short(norms):
    rows = [losing(data) for data in months(2014, 4)]
    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_no_actions_for_a_profitable_well(norms):
    rows = [earning(data) for data in months(2014, 24)[:12]]
    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_shutdown_takes_effect_the_month_after_the_decision(norms):
    """Решение принимается по данным до даты, а действует со следующего месяца."""
    rows = [losing(data) for data in months(2014, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=CONTROL_START)

    assert len(actions) == 1
    action = actions[0]
    assert action.well == "bad"
    assert action.kind == policy.SHUT
    assert action.cost_m == pytest.approx(1.0)
    # шесть убыточных месяцев (янв–июн) дают убыток > 1 млн только к седьмому
    assert action.date > date(2014, 6, 1)


def test_decision_uses_only_the_past(norms):
    """Обрезав данные после даты решения, получаем то же самое действие.

    Это и есть проверка причинности: правило, подглядывающее в будущее, здесь
    дало бы другой ответ.
    """
    full = [losing(data) for data in months(2014, 12)]
    action = policy.plan_shutdowns(full, norms, control_start=CONTROL_START)[0]

    cutoff = action.date.isoformat()
    truncated = [row for row in full if row["DATA"] <= cutoff]
    again = policy.plan_shutdowns(truncated, norms, control_start=CONTROL_START)

    assert [a.date for a in again] == [action.date]
    assert again[0].reason == action.reason


def test_a_recovered_month_resets_the_streak(norms):
    rows = [losing(data) for data in months(2014, 5)]
    rows.append(earning(months(2014, 6)[5], well="bad"))  # шестой месяц прибыльный
    rows.extend(losing(data) for data in months(2014, 5, first=7))
    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_actions_never_precede_the_control_start(norms):
    rows = [losing(data) for data in months(2006, 12)] + [losing(data) for data in months(2007, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=date(2007, 1, 1))
    assert all(action.date >= date(2007, 1, 1) for action in actions)


def test_a_well_is_shut_only_once(norms):
    rows = [losing(data) for data in months(2014, 12)] + [losing(data) for data in months(2015, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=CONTROL_START)
    assert [action.well for action in actions] == ["bad"]


def _steps(dates, wells=("1", "2")):
    return [
        ScheduleStep(
            day,
            tuple(ProducerControl(well, "OPEN", "LRAT", 45.0, 50.0) for well in wells),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, 300.0),),
        )
        for day in dates
    ]


def test_apply_writes_shut_once_and_then_leaves_the_well_alone():
    """Режим в Eclipse действует до следующего изменения — повторять SHUT незачем."""
    steps = _steps([date(2014, month, 1) for month in (1, 2, 3, 4)])
    actions = [policy.Action(date(2014, 2, 1), "1", policy.SHUT, "убыточна", 1.0)]

    result = policy.apply(steps, actions)

    assert [c.well for c in result[0].producers] == ["1", "2"]
    assert result[0].producers[0].status == "OPEN"

    shut = [c for c in result[1].producers if c.well == "1"]
    assert len(shut) == 1 and shut[0].status == "SHUT" and shut[0].liquid_rate is None

    for step in result[2:]:
        assert [c.well for c in step.producers] == ["2"]


def test_apply_keeps_injectors_untouched():
    steps = _steps([date(2014, 1, 1)])
    result = policy.apply(steps, [policy.Action(date(2014, 1, 1), "1", policy.SHUT, "x", 1.0)])
    assert result[0].injectors == steps[0].injectors


def test_result_passes_the_schedule_validator():
    steps = _steps([date(2014, month, 1) for month in (1, 2, 3)])
    result = policy.apply(steps, [policy.Action(date(2014, 2, 1), "1", policy.SHUT, "x", 1.0)])
    assert validate(result) == []


def test_apply_rejects_unknown_actions():
    with pytest.raises(ValueError, match="неизвестное действие"):
        policy.apply(_steps([date(2014, 1, 1)]), [policy.Action(date(2014, 1, 1), "1", "fly", "", 0)])


def test_explain_is_readable(norms):
    rows = [losing(data) for data in months(2014, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=CONTROL_START)
    text = policy.explain(actions)

    assert "Действий: 1" in text
    assert "скв. bad" in text
    assert "обводнённость" in text
    assert "1.0 млн руб" in text


def test_explain_says_so_when_there_is_nothing_to_do():
    assert "Действий нет" in policy.explain([])


def injecting(data: str, well: str = "нагн"):
    """Нагнетательная: закачка есть, добычи нет — выручки нет по построению."""
    return make_row(data, well, wlpr=0.0, womr=0.0, liquid=0.0, oil=0.0,
                    wwir=300.0, injection=9000.0)


def test_an_injector_is_not_judged_by_a_rule_written_for_producers(norms):
    """Политика не должна выносить решение, которого расписание не исполняет.

    У нагнетательной скважины выручки нет по построению, поэтому её месячная
    маржа отрицательна всегда: закачка — чистый расход. Правило «убыточна
    `lookback` месяцев подряд» срабатывало на ней автоматически, и политика
    выносила остановку. Расписание её при этом не исполняло: `apply` выдаёт
    только `ProducerControl SHUT` и нагнетательные пропускает. Замер 20.08:
    решение вынесено (`shut`, 1,0 млн руб., «убыток 6 мес. подряд, накоплено
    2,12 млн»), реально остановленных месяцев — ноль.

    Цена не в деньгах, а в доверии к слою интерпретируемости: решение попадает
    в таблицу «Принятые решения», объясняется языковой моделью и стоит миллион
    на бумаге — а его не происходит. В базовом расписании Модели Z 41
    нагнетательная скважина (11 никогда не добывали, 30 переведены под закачку).

    Остановка закачки — отдельное решение со своими последствиями для энергии
    пласта, и принимать его как побочный эффект правила про добывающие нельзя:
    замер того же дня показал, что срез закачки обрушивает нефть вчетверо.
    """
    rows = [injecting(data) for data in months(2014, 12)]

    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_a_well_converted_to_injection_stops_accruing_a_producer_loss(norms):
    """Перевод под закачку обрывает счётчик убытка, а не продолжает его.

    После перевода скважина не добывает: судить её правилом про добывающие
    нечем, и накопленный до перевода убыток продолжать нельзя.
    """
    rows = [losing(data, "внс") for data in months(2014, 3)]
    rows += [injecting(data, "внс") for data in months(2014, 12)[3:]]

    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_a_row_the_methodology_drops_does_not_count_as_a_loss(norms):
    """Строка, исключённая методикой, не должна закрывать прибыльную скважину.

    Методика исключает строку с отрицательным `WLPT_Diff`, `WOMT_Diff` или
    `WWIT_Diff` из экономики **целиком** — не обнуляет и не клампит. Скрининг
    считал её как есть, то есть приписывал скважине убыток, которого в деньгах
    не существует.

    Замер 20.08: месяц с `WOMT_Diff = −150` даёт маржу −1,427 млн руб. против
    +1,081 у того же месяца с нормальным приращением. Прибыльная скважина, у
    которой выгрузка раз за разом откатывает накопленное, останавливалась с
    обоснованием «убыток 6 мес. подряд, накоплено 8,56 млн руб.; обводнённость
    **116,7 %**» — обводнённость выше ста процентов физически невозможна и сама
    выдаёт, что решение принято по несуществующим деньгам.

    Отрицательные строки на этом объекте не гипотетические: 102 штуки за
    2025-09 в выгрузке организаторов (`JOURNAL.md`, принцип 1).
    """
    rows = []
    for month in months(2014, 12):
        row = earning(month, "П1")
        if month >= "2014-04-01":
            row["WOMT_Diff"] = -150.0
        rows.append(row)

    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def _watery(well: str, months: int, *, watercut: float, liquid: float = 900.0,
            start_year: int = 2014):
    """Скважина, поднимающая почти одну воду и потому убыточная."""
    rows = []
    for index in range(months):
        year = start_year + index // 12
        month = index % 12 + 1
        rows.append(make_row(f"{year}-{month:02d}-01", well, wlpr=30.0,
                             womr=liquid * (1 - watercut) / 30,
                             liquid=liquid, oil=liquid * (1 - watercut)))
    return rows


def test_a_water_producer_is_offered_for_conversion_not_only_for_shutdown(norms):
    """Скважина, поднимающая одну воду, — кандидат на перевод под закачку.

    Остановка убирает и воду, и энергию пласта; перевод оставляет объём в
    системе, но разворачивает его — вода идёт обратно и поддерживает давление у
    соседей. Методика оценивает событие отдельно: 5,0 млн руб. ВНС без стоимости
    ЭЦН (Р-30), то есть планка впятеро выше, чем у остановки.
    """
    from aios.policy import CONVERT, plan_conversions

    # сорок восемь месяцев не для красоты: накопленный убыток обязан перекрыть
    # 5,0 млн руб. ВНС, а это впятеро выше планки остановки — на двух годах
    # такая скважина её не набирает
    rows = _watery("вода", 48, watercut=0.995)

    actions = plan_conversions(rows, norms, control_start=date(2014, 1, 1))

    assert actions, "кандидат на перевод не найден"
    assert actions[0].kind == CONVERT
    assert actions[0].cost_m == norms.conversion_m == 5.0
    assert "обводнённость" in actions[0].reason
    assert "полным расчётом" in actions[0].reason, (
        "предложение обязано говорить, что выигрыш по соседям суррогатом не оценён"
    )


def test_a_well_with_oil_is_not_offered_for_conversion(norms):
    """С заметной нефтью перевод отнимает добычу, а размен оценить нечем."""
    from aios.policy import plan_conversions

    rows = _watery("нефтяная", 24, watercut=0.80)

    assert plan_conversions(rows, norms, control_start=date(2014, 1, 1)) == []


def test_conversion_is_not_offered_without_headroom_under_the_water_ceiling(norms):
    """Без запаса под потолком компенсации перевод — обещание, которое срежут."""
    from aios.policy import plan_conversions

    rows = _watery("вода", 48, watercut=0.995)

    assert plan_conversions(rows, norms, control_start=date(2014, 1, 1),
                            headroom=False) == []


def test_after_conversion_the_well_leaves_the_producing_fund(norms):
    """Роль меняется: добыча закрыта, появилась запись закачки.

    Это третье условие задачи: планировщик и баланс воды обязаны увидеть
    скважину в новой роли, а не в старой.
    """
    from aios.deck import ProducerControl, ScheduleStep
    from aios.policy import CONVERT, Action, apply

    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("вода", "OPEN", "LRAT", 30.0, 50.0),), ())
             for month in range(1, 7)]
    action = Action(date=date(2014, 3, 1), well="вода", kind=CONVERT,
                    reason="обводнённость 99,5 %", cost_m=norms.conversion_m)

    result = apply(steps, [action])

    before = result[0]
    assert before.producers[0].status == "OPEN" and not before.injectors

    # запись закачки ставится один раз: в Eclipse режим действует до следующего
    # изменения, и повторять её на каждом шаге не нужно
    after = result[2]
    assert [control.status for control in after.producers] == ["SHUT"]
    assert len(after.injectors) == 1, "скважина обязана появиться как нагнетательная"
    assert after.injectors[0].well == "вода"
    assert after.injectors[0].rate == 30.0, "закачивает столько, сколько поднимала"
    # а добыча остаётся закрытой и дальше: дек переоткрывает её на каждом шаге
    assert all(control.status == "SHUT"
               for step in result[2:] for control in step.producers)


# ── что нашёл мутационный прогон 05.09 (Р-221) ───────────────────────────────


def test_the_streak_needs_exactly_the_lookback_months(norms):
    """Остановка приходит ровно на `lookback`-м убыточном месяце, не раньше.

    Мутации границы серии (`<` на `<=`) и умолчания `lookback = 6` переживали
    сюиту: проверялись «мало» и «много», но не сама граница. Между тем это цена
    решения — каждая остановка стоит 1,0 млн руб., и лишний месяц ожидания или
    лишняя ранняя остановка меняют ЧДД напрямую.
    """
    # убыток крупный намеренно: у `losing` порог окупаемости (1,0 млн руб.)
    # набирается позже шестого месяца, и граница серии измерялась бы вместе с
    # ним. Замер 05.09: на `losing` остановка приходит с восьмого месяца, на
    # крупном убытке — с седьмого, то есть шесть месяцев серии плюс месяц
    # вступления решения в силу
    def big(data: str):
        return make_row(data, "bad", wlpr=300, womr=0.1, liquid=9000, oil=3)

    six = [big(data) for data in months(2014, 6)]
    seven = [big(data) for data in months(2014, 7)]

    assert policy.plan_shutdowns(six, norms, control_start=CONTROL_START) == [], (
        "остановка раньше шестого месяца серии — порог не соблюдается"
    )
    assert policy.plan_shutdowns(seven, norms, control_start=CONTROL_START), (
        "шести месяцев серии и месяца на вступление не хватило для остановки"
    )


def test_the_default_lookback_is_six_months(norms):
    """Умолчание окна — шесть месяцев, а не ноль.

    Мутация `6 -> 0` переживала сюиту: вызовы всегда передавали `lookback`
    явно. Ноль означал бы остановку по первому же убыточному месяцу — то есть
    реакцию на шум вместо устойчивого убытка.
    """
    import inspect

    for name in ("plan_shutdowns", "plan_conversions"):
        default = inspect.signature(getattr(policy, name)).parameters["lookback"].default
        assert default == 6, f"{name}: окно по умолчанию {default}"

    one_month = [losing(data) for data in months(2014, 1)]
    assert policy.plan_shutdowns(one_month, norms, control_start=CONTROL_START) == []


def test_a_profitable_month_breaks_the_streak(norms):
    """Серия убытков прерывается прибыльным месяцем, а не только простоем.

    Мутация `and` на `or` в условии накопления переживала сюиту: с ней серия
    росла бы и на прибыльных месяцах, если скважина просто действующая, — и
    остановка приходила бы к работающей в плюс скважине.
    """
    stamps = months(2014, 12)
    big = [make_row(data, "bad", wlpr=300, womr=0.1, liquid=9000, oil=3)
           for data in stamps]
    mixed = (big[:5] + [earning(stamps[5], well="bad")] + big[6:11])

    assert policy.plan_shutdowns(mixed, norms, control_start=CONTROL_START) == [], (
        "прибыльный месяц не прервал серию убытков"
    )


def wet_and_losing(data: str, well: str = "wet"):
    """Сильно обводнённая и крупно убыточная: 0,99 воды, −1,07 млн руб. в месяц.

    Числа подобраны замером 05.09 так, чтобы **оба** порога перевода набирались
    к одному сроку: шесть месяцев серии и 5,0 млн руб. накопленного убытка.
    Иначе граница серии измеряется вместе с порогом окупаемости, и мутация в
    одной из них переживает тест.
    """
    return make_row(data, well, wlpr=500, womr=0.2, liquid=60000, oil=600)


def test_a_conversion_needs_the_watercut_as_well_as_the_loss(norms):
    """Перевод под закачку требует и убытка, и обводнённости — обоих сразу.

    Мутация `and` на `or` переживала сюиту. Перевод стоит 5,0 млн руб. в OPEX;
    с `or` его получала бы убыточная, но не обводнённая скважина — то есть та,
    которую надо чинить, а не переводить.
    """
    # тот же убыток, но обводнённость 0,90 против порога 0,98
    dry = [make_row(data, "dry", wlpr=500, womr=0.2, liquid=60000, oil=6000)
           for data in months(2014, 12)]
    assert policy._watercut(dry[0]) < policy.CONVERSION_WATERCUT

    actions = policy.plan_conversions(dry, norms, control_start=CONTROL_START)

    assert actions == [], (
        f"перевод назначен недостаточно обводнённой скважине: "
        f"{[one.reason for one in actions]}"
    )


def test_a_conversion_needs_exactly_the_streak(norms):
    """Перевод приходит на седьмом месяце: шесть в серии плюс месяц вступления.

    Мутация границы серии (`<` на `<=`) переживала прежний тест — он проверял
    «много месяцев», а не саму границу.
    """
    six = [wet_and_losing(data) for data in months(2014, 6)]
    seven = [wet_and_losing(data) for data in months(2014, 7)]

    assert policy.plan_conversions(six, norms, control_start=CONTROL_START) == []
    assert policy.plan_conversions(seven, norms, control_start=CONTROL_START)


def test_a_month_below_the_watercut_breaks_the_conversion_streak(norms):
    """Один месяц ниже порога обводнённости обнуляет серию перевода.

    Мутации `margin < 0` на `<= 0` и `and` на `or` жили ровно здесь: серия
    должна расти только когда выполнены все три условия сразу.
    """
    stamps = months(2014, 12)
    mixed = ([wet_and_losing(data) for data in stamps[:5]]
             + [make_row(stamps[5], "wet", wlpr=500, womr=0.2, liquid=60000, oil=6000)]
             + [wet_and_losing(data) for data in stamps[6:11]])

    assert policy.plan_conversions(mixed, norms, control_start=CONTROL_START) == [], (
        "месяц ниже порога обводнённости не прервал серию"
    )


def test_an_injector_row_resets_the_conversion_streak(norms):
    """Скважина, уже работающая на закачку, из перевода выпадает — и обнуляет серию.

    Две мутации строк 206–207 переживали сюиту: нагнетательной строки в
    фикстурах не было. Переводить под закачку то, что уже закачивает, — двойной
    расход 5,0 млн руб. на несделанное мероприятие.
    """
    stamps = months(2014, 12)
    with_injection = ([wet_and_losing(data) for data in stamps[:5]]
                      + [make_row(stamps[5], "wet", wlpr=0.0, wwir=100.0, injection=3000.0)]
                      + [wet_and_losing(data) for data in stamps[6:11]])

    assert policy.plan_conversions(with_injection, norms,
                                   control_start=CONTROL_START) == [], (
        "месяц работы на закачку не прервал серию перевода"
    )


def test_watercut_of_a_well_without_liquid_is_zero_not_a_crash():
    """Обводнённость при нулевой жидкости — ноль, а не деление на ноль.

    Мутация `> 0` на `>= 0` переживала сюиту: строки с нулевой жидкостью в
    фикстурах не было. На выгрузке она появится сама — остановленная скважина
    даёт ровно такую строку.
    """
    dry = make_row("2014-01-01", "1", wlpr=0, womr=0, liquid=0.0, oil=0.0)

    assert policy._watercut(dry) == 0.0


def test_the_explanation_counts_every_action():
    """Сводка решений считает все действия, а не на одно меньше.

    Две мутации счётчика в `explain` переживали сюиту: текст объяснения никто
    не сверял с числом действий. Слой интерпретируемости обязателен по
    регламенту, и «остановка × 2» вместо трёх — это неверный отчёт о том, что
    система сделала.
    """
    actions = [
        policy.Action(well=str(index), kind="shut", date=date(2014, 1, 1),
                      reason="убыток", cost_m=1.0)
        for index in range(3)
    ]

    text = policy.explain(actions)

    assert "shut × 3" in text, text
    assert "Действий: 3" in text, text


def test_an_idle_well_is_not_converted_however_watered_out(norms):
    """Простаивающая скважина под закачку не переводится, даже если обводнена.

    Строка с `WLPR = 0` и `WWIR = 0` — простой, но приращения за месяц в ней
    есть, обводнённость 0,99 и маржа −0,98 млн руб. То есть два условия из трёх
    выполнены, а третье — нет. Мутация `and` на `or` в строке 213 переживала
    все прежние тесты именно потому, что такой строки в них не было: везде
    условия совпадали по знаку.

    Смысл прямой: переводить под закачку то, что и так стоит, — 5,0 млн руб. за
    мероприятие, которого никто не делал.
    """
    idle = [make_row(data, "idle", wlpr=0.0, womr=0.0, liquid=60000, oil=600)
            for data in months(2014, 12)]
    assert not policy.is_active(idle[0])
    assert policy._watercut(idle[0]) >= policy.CONVERSION_WATERCUT

    assert policy.plan_conversions(idle, norms, control_start=CONTROL_START) == [], (
        "простаивающая скважина отправлена под закачку"
    )


def test_a_zero_margin_month_does_not_feed_the_conversion_streak(norms):
    """Ноль маржи — не убыток: серия перевода растёт строго на отрицательной.

    Мутация `margin < 0` на `<= 0` переживала сюиту: месяца с ровно нулевой
    маржой в фикстурах не было. Разница смысловая — «не заработала» и «принесла
    убыток» разные основания для мероприятия за 5,0 млн руб.
    """
    stamps = months(2014, 12)
    zero = make_row(stamps[5], "wet", wlpr=500, womr=0.2, liquid=0.0, oil=0.0)
    mixed = ([wet_and_losing(data) for data in stamps[:5]] + [zero]
             + [wet_and_losing(data) for data in stamps[6:11]])

    assert policy.plan_conversions(mixed, norms, control_start=CONTROL_START) == [], (
        "месяц с нулевой маржой зачтён в серию убытков"
    )


def test_the_conversion_takes_effect_the_month_after_the_decision(norms):
    """Решение вступает в силу **следующим** месяцем, а не тем же.

    Две мутации строки 225 (`index + 1` на `index + 0` и на `index - 1`)
    переживали сюиту: дату мероприятия никто не сверял с датой последнего
    убыточного месяца. Причинность здесь не формальность — решение, принятое по
    данным месяца, не может действовать в том же месяце.
    """
    stamps = months(2014, 7)
    rows = [wet_and_losing(data) for data in stamps]

    actions = policy.plan_conversions(rows, norms, control_start=CONTROL_START)

    assert len(actions) == 1, [one.reason for one in actions]
    assert actions[0].date == date(2014, 7, 1), (
        f"мероприятие назначено на {actions[0].date}, а шесть убыточных месяцев "
        f"кончились 2014-06: решение обязано действовать со следующего месяца"
    )

