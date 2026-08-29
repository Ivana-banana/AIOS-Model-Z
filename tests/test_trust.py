"""Второй контур шлюза доверия: сценарное расстояние и запас против оптимизма.

Главное, что здесь закреплено, — свойство, из-за которого модуль и появился:
кандидат, проходящий покоординатную проверку целиком, обязан быть отброшен по
совместному режиму. Разбор 4-7 называет это ровно тем классом ошибок, который
прямоугольник не видит по построению.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aios import trust

ROOT = Path(__file__).resolve().parent.parent


def flat(values: dict[str, float], months: int = 4) -> dict[date, dict[str, float]]:
    """Расписание с постоянным режимом: множители не меняются по месяцам."""
    return {date(2007, month, 1): dict(values) for month in range(1, months + 1)}


def diagonal_scenarios(count: int = 12) -> dict[str, dict[str, float]]:
    """DoE «по диагонали»: обе скважины двигаются вместе, вразнобой — никогда.

    Ровно тот двумерный пример, которым разбор 4-7 показывает дыру
    покоординатной проверки: точки лежат около диагонали от (0.8, 0.8) до
    (1.2, 1.2), поэтому (0.8, 1.2) проходит min–max по каждой координате, но
    далеко от всех наблюдённых сочетаний.
    """
    return {
        f"s{index:02d}": trust.profile(
            flat({"prod:1": value, "prod:2": value})
        )
        for index, value in enumerate(
            0.8 + 0.4 * step / (count - 1) for step in range(count)
        )
    }


# ── сценарное представление ──────────────────────────────────────────────────

def test_profile_keeps_both_the_level_and_the_swing():
    """Два сценария с одним средним, но разным поведением, не должны совпасть."""
    steady = trust.profile(flat({"prod:1": 1.0}))
    swinging = trust.profile({
        date(2007, 1, 1): {"prod:1": 0.5},
        date(2007, 2, 1): {"prod:1": 1.5},
        date(2007, 3, 1): {"prod:1": 0.5},
        date(2007, 4, 1): {"prod:1": 1.5},
    })

    assert steady["prod:1|уровень"] == pytest.approx(swinging["prod:1|уровень"])
    assert steady["prod:1|размах"] == pytest.approx(0.0)
    assert swinging["prod:1|размах"] > 0.4


def test_an_empty_schedule_is_refused_not_silently_averaged():
    with pytest.raises(ValueError, match="ни одного месяца"):
        trust.profile({})


# ── доверенная область ───────────────────────────────────────────────────────

def test_joint_mode_outside_the_doe_is_rejected_although_each_axis_is_inside():
    """То, ради чего модуль написан.

    (0.8, 1.2) внутри диапазона обеих координат по отдельности, но такого
    сочетания DoE не содержал. Покоординатная проверка его пропускает; сценарное
    расстояние обязано его пометить.
    """
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)

    on_diagonal = region.distance(trust.profile(flat({"prod:1": 1.0, "prod:2": 1.0})))
    off_diagonal = region.distance(trust.profile(flat({"prod:1": 0.8, "prod:2": 1.2})))

    assert not on_diagonal.outside
    assert off_diagonal.outside
    assert off_diagonal.value > on_diagonal.value


def test_the_threshold_comes_from_the_sample_not_from_a_chosen_number():
    """Порог — насколько одинока самая одинокая точка самой выборки."""
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)

    inside = [
        region.distance(profile).value
        for profile in diagonal_scenarios().values()
    ]

    assert region.threshold > 0
    # каждый обучающий сценарий сам себе ближайший, поэтому его d_k строго меньше
    # порога, посчитанного без него самого
    assert max(inside) <= region.threshold


def test_neighbours_name_the_training_scenarios_for_the_explanation():
    """«Покажите пять ближайших режимов DoE» — требование слоя интерпретируемости."""
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)

    found = region.distance(trust.profile(flat({"prod:1": 0.8, "prod:2": 0.8})))

    assert len(found.neighbours) == 3
    assert found.neighbours[0].scenario == "s00"
    assert [n.distance for n in found.neighbours] == sorted(n.distance for n in found.neighbours)


def test_a_control_the_doe_never_moved_is_the_hardest_boundary():
    """Нулевая дисперсия признака не выбрасывается: сдвинуть его — уйти далеко."""
    profiles = {
        name: one | trust.profile(flat({"inj:9": 1.0}))
        for name, one in diagonal_scenarios().items()
    }
    region = trust.TrustRegion.fit(profiles, k=3)

    moved = region.distance(
        trust.profile(flat({"prod:1": 1.0, "prod:2": 1.0, "inj:9": 2.0}))
    )

    assert moved.outside


def test_fewer_scenarios_than_neighbours_is_refused():
    with pytest.raises(ValueError, match="не определено"):
        trust.TrustRegion.fit(diagonal_scenarios(4), k=5)


# ── конформный запас ─────────────────────────────────────────────────────────

def test_too_few_pairs_means_not_calibrated_not_zero():
    """Мало пар — «не измерено», а не «риска нет». Разница в том, что заявляем."""
    margin = trust.OptimismMargin.fit([(100.0, 90.0), (100.0, 95.0)], alpha=0.1)

    assert not margin.calibrated
    assert margin.value is None
    assert "не существует" in margin.reason
    assert margin.index == 3 and margin.samples == 2


def test_the_margin_is_the_order_statistic_the_theory_names():
    """`⌈(n+1)(1−α)⌉`-й по величине остаток, а не среднее и не максимум."""
    pairs = [(100.0 + gap, 100.0) for gap in (1.0, 2.0, 3.0, 4.0, 5.0,
                                              6.0, 7.0, 8.0, 9.0, 10.0)]
    margin = trust.OptimismMargin.fit(pairs, alpha=0.2)

    assert margin.samples == 10
    assert margin.index == 9
    assert margin.value == pytest.approx(9.0)


def test_a_surrogate_that_understates_gets_a_negative_margin():
    """Занижающий суррогат не наказывается: запас против оптимизма односторонний."""
    margin = trust.OptimismMargin.fit([(90.0, 100.0)] * 20, alpha=0.1)

    assert margin.calibrated
    assert margin.value == pytest.approx(-10.0)


# ── цель поиска ──────────────────────────────────────────────────────────────

def test_safe_value_penalises_distance_and_optimism():
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)
    far = region.distance(trust.profile(flat({"prod:1": 0.8, "prod:2": 1.2})))
    margin = trust.OptimismMargin.fit([(100.0 + gap, 100.0) for gap in range(1, 21)],
                                      alpha=0.1)

    assert trust.safe_value(1000.0, None, None) == pytest.approx(1000.0)
    assert trust.safe_value(1000.0, far, margin) < 1000.0 - margin.value


def test_an_uncalibrated_margin_is_not_silently_treated_as_zero_risk():
    """Не вычитаем выдуманное — но и не молчим: кандидат уходит на полный расчёт."""
    margin = trust.OptimismMargin.fit([(100.0, 90.0)], alpha=0.1)

    assert trust.safe_value(1000.0, None, margin) == pytest.approx(1000.0)
    reasons = trust.routed_to_full_run(None, margin)
    assert any("не калибрована" in reason for reason in reasons)


def test_a_covered_candidate_with_calibration_is_not_routed_to_a_full_run():
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)
    near = region.distance(trust.profile(flat({"prod:1": 1.0, "prod:2": 1.0})))
    margin = trust.OptimismMargin.fit([(100.0 + gap, 100.0) for gap in range(1, 21)],
                                      alpha=0.1)

    assert trust.routed_to_full_run(near, margin) == []


def test_hard_rules_are_carried_through_untouched():
    """Первый контур не подменяется вторым — оба идут в одно решение."""
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)
    near = region.distance(trust.profile(flat({"prod:1": 1.0, "prod:2": 1.0})))
    margin = trust.OptimismMargin.fit([(100.0 + gap, 100.0) for gap in range(1, 21)],
                                      alpha=0.1)

    reasons = trust.routed_to_full_run(near, margin, ["скв. 66: режим вне диапазона"])
    assert reasons == ["скв. 66: режим вне диапазона"]


# ── контракт артефакта ───────────────────────────────────────────────────────

def test_the_artifact_states_what_the_guarantee_does_not_cover():
    """Область действия гарантии — обязательное поле, а не примечание в отчёте."""
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)
    margin = trust.OptimismMargin.fit([(100.0 + gap, 100.0) for gap in range(1, 21)],
                                      alpha=0.1)

    written = trust.contract(region, margin, dataset_hash="abc", model_hash="def")

    assert "не гарантия для адаптивно выбранного оптимизатором кандидата" in written["guarantee_scope"]
    assert written["scenario_unit"].startswith("целый сценарий")
    assert written["calibrated"] is True
    assert written["k"] == 3
    assert written["training_scenario_ids"] == sorted(diagonal_scenarios())


def test_a_missing_block_makes_the_artifact_uncalibrated():
    """Нет калибровки — интерфейс обязан сказать это, а не показать зелёный статус."""
    region = trust.TrustRegion.fit(diagonal_scenarios(), k=3)

    assert trust.contract(region, None)["calibrated"] is False
    assert trust.contract(None, trust.OptimismMargin.fit(
        [(100.0 + gap, 100.0) for gap in range(1, 21)], alpha=0.1))["calibrated"] is False


def test_the_gate_judges_a_candidate_by_the_region_inside_the_artifact(tmp_path, capsys):
    """Шлюз обязан работать там, где датасетов нет, — на машине без прогонов.

    Область доверия обучается вместе с суррогатом и едет внутри артефакта
    (Р-72). До 24.08 `scripts/trust_gate.py` умел только строить её заново по
    `--dataset`, то есть на машине Б, где датасетов нет по построению, был
    неисполним вовсе — а задача Б31 требует вердикт по каждому кандидату.

    Здесь же закреплено и то, что скрипт вообще импортируется: он не добавлял
    корень проекта в `sys.path` и падал `ModuleNotFoundError: No module named
    'aios'` — при том что Б27 проверила «PYTHONPATH не нужен ни одной команде».
    """
    import sys

    import joblib
    import pandas as pd
    from test_dataset import _tiny_project

    sys.path.insert(0, str(ROOT / "scripts"))
    import trust_gate

    from aios.neural import NeuralBundle, TargetMetric, TrainingReport
    from aios.surrogate import NumericRange, TrainingEnvelope

    metrics = {target: TargetMetric(3.0, 1.0) for target in ("WLPT_Diff", "WOMT_Diff",
                                                            "WWIT_Diff", "THP", "BHP")}
    region = trust.TrustRegion.fit(
        {"s1": {"prod:1": 1.0, "prod:1:swing": 0.0},
         "s2": {"prod:1": 0.8, "prod:1:swing": 0.1},
         "s3": {"prod:1": 1.2, "prod:1:swing": 0.2}},
        k=1,
    )
    bundle = NeuralBundle(
        ("prod:1",), {}, pd.Timestamp("2007-01-01"), 12,
        TrainingEnvelope({"prod:1": NumericRange(0.5, 1.5)},
                         NumericRange(0.5, 1.5), NumericRange(0.0, 0.0)),
        TrainingReport(("a",), ("b",), 1, 2, metrics),
        trust_region=region,
    )
    model = tmp_path / "surrogate.joblib"
    joblib.dump(bundle, model)
    config = _tiny_project(tmp_path)
    schedule = tmp_path / "wells_schedule.inc"
    schedule.write_text(
        "DATES\n 01 JAN 2007 /\n/\n\nWCONPROD\n '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /\n/\n",
        encoding="utf-8",
    )

    code = trust_gate.main(["--model", str(model), "--schedule", str(schedule),
                            "--config", str(config)])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "расстояние:" in printed
    assert "порог" in printed
    # запас у артефакта не калиброван — и это обязано быть названо причиной
    assert "полный расчёт" in printed

    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "".join(f"{100 + gap},100\n" for gap in range(1, 10)),
        encoding="utf-8",
    )
    code = trust_gate.main([
        "--model", str(model),
        "--schedule", str(schedule),
        "--config", str(config),
        "--pairs", str(pairs),
    ])
    printed = capsys.readouterr().out

    assert code == 0, printed
    assert "конформный запас: 9.0" in printed
    assert "причин идти на полный расчёт: 1" in printed
    assert "мера доверия не калибрована" not in printed
    assert "совместный режим далёк" in printed


def test_the_gate_refuses_half_of_the_candidate_mode(tmp_path, capsys):
    """Один ключ из двух — отказ строкой, а не молчаливая сборка области."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import trust_gate

    assert trust_gate.main(["--model", str(tmp_path / "нет.joblib")]) == 1
    assert "оба ключа" in capsys.readouterr().err


def test_the_gate_refuses_a_pairs_file_that_is_not_there(tmp_path, capsys):
    """Опечатка в `--pairs` не имеет права выглядеть как «пар не собирали».

    `load_pairs` отдавал `[]` на несуществующий путь молча, и вердикт выходил
    тот же самый, что и без ключа вовсе: «нет ни одной пары». Оператор, который
    путь **задал**, читает такой ответ как ожидаемый, а не как промах мимо файла.
    Направление ошибки безопасное — лишний полный расчёт, — но `--pairs` стал
    штатным ключом режима кандидата, и холостой ход теперь на главном пути.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import trust_gate

    code = trust_gate.main(["--dataset", str(tmp_path / "нет.csv"),
                            "--pairs", str(tmp_path / "опечатка.csv")])
    assert code == 1
    assert "нет файла калибровочных пар" in capsys.readouterr().err


def test_the_surrogate_passport_documents_calibrated_candidate_mode():
    passport = (ROOT / "docs/surrogate.md").read_text(encoding="utf-8")

    assert "python scripts/trust_gate.py" in passport
    assert "--model artifacts/surrogate.joblib" in passport
    assert "--schedule runs/matched-0995/wells_schedule.inc" in passport
    assert "--pairs docs/data/margin-pairs-search.csv" in passport
    assert "финальный результат всё равно считается\nГГДМ" in passport



def test_the_margin_is_negative_when_the_model_never_overpromises():
    """Калибровка по сценариям, которых никто не оптимизировал, защиты не даёт.

    Замер 24.08 (`docs/data/margin-pairs.csv`): на 21 ручном сценарии суррогат
    занижает ЧДД **всегда** — остатки от −340 до −1 211 млн руб., и односторонний
    квантиль выходит отрицательным (−351,8). На двух кандидатах оптимизатора та
    же модель завышает: +376,6 и +927,8. Знак остатка разделяет семейства
    начисто, и это ровно тот механизм, ради которого запас и существует: применять
    к кандидатам поиска запас, посчитанный по ручным прогонам, значит не иметь
    защиты вовсе.
    """
    from aios.trust import OptimismMargin

    manual = [(11197.7, 11851.8), (12255.4, 12783.2), (12730.2, 13264.6),
              (12099.7, 13152.9), (11659.4, 12812.0), (12181.6, 13183.5),
              (12379.8, 13224.6), (11744.9, 12956.4), (13305.3, 13645.7),
              (11698.3, 12428.8), (11516.0, 12396.0), (11847.7, 12761.6)]
    search = [(12664.9, 12288.3), (13240.4, 12312.6)]

    on_manual = OptimismMargin.fit(manual)
    on_search = OptimismMargin.fit(search)

    assert on_manual.calibrated
    assert on_manual.value < 0.0, "запас по ручным сценариям отрицателен — защиты нет"
    assert not on_search.calibrated, "двух пар не хватает на уровень 90 %"
    assert "нужен" in on_search.reason
    assert all(predicted > actual for predicted, actual in search), (
        "на кандидатах оптимизатора суррогат завышает — знак остатка разделяет семейства")
