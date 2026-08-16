"""Веб-платформа — обязательный по регламенту артефакт, значит и тестируемый."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest
from conftest import make_row
from fastapi.testclient import TestClient

from aios import economics, summary
from aios.web.app import create_app, read_rows_csv

CONFIG = """
[case]
model_dir = "vendor/Model_Z_OPM"
run_dir = "runs/base"
case_name = "RUN"
control_start = 2014-01-01

[economics]
start_date = 2014-01-01
norms_file = "{norms}"

[fluids]
oil_density = 913.0765883224138
water_density = 1172.0

[limits]
max_field_liquid_m3_per_day = 0
enforce_water_balance = true
water_tolerance = 1.0

[policy]
lookback_months = 6

[run]
seed = 20260817
processes = 2
"""

SCHEDULE = """\
DATES
 01 JAN 2014 /
/

WCONPROD
 'good' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
 'bad' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""


def rows_for(well: str, watercut_start: float, watercut_step: float, months: int = 18):
    fluids = summary.Fluids()
    rows = []
    for index in range(months):
        year = 2013 + index // 12
        month = index % 12 + 1
        liquid_m3 = 30.0 * 30
        watercut = min(watercut_start + watercut_step * index, 0.999)
        oil_m3 = liquid_m3 * (1 - watercut)
        oil_t = oil_m3 * fluids.oil_density / 1000
        liquid_t = oil_t + (liquid_m3 - oil_m3) * fluids.water_density / 1000
        rows.append(
            make_row(date(year, month, 1).isoformat(), well, wlpr=30.0,
                     womr=oil_t / 30, liquid=liquid_t, oil=oil_t)
        )
    return rows


@pytest.fixture
def project(tmp_path: Path):
    """Проект с одним прогоном, у которого есть выгрузка и расписание."""
    norms = Path(economics.NORMS_FILE).resolve().as_posix()
    (tmp_path / "config.toml").write_text(CONFIG.format(norms=norms), encoding="utf-8")

    run_dir = tmp_path / "runs" / "base"
    run_dir.mkdir(parents=True)
    (run_dir / "wells_schedule.inc").write_text(SCHEDULE, encoding="utf-8")

    rows = rows_for("good", 0.50, 0.005) + rows_for("bad", 0.985, 0.0005)
    with (run_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=economics.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    (tmp_path / "runs" / "empty").mkdir()
    return tmp_path


@pytest.fixture
def client(project):
    return TestClient(create_app(project))


def test_index_lists_runs_and_configuration(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "base" in response.text
    assert "2014-01-01" in response.text          # дата начала управления из конфига
    assert "20260817" in response.text            # seed — регламент про воспроизводимость


def test_index_marks_a_run_without_results(client):
    assert "результатов нет" in client.get("/").text


def test_index_reports_missing_surrogate(client):
    assert "Модель ещё не обучена" in client.get("/").text


def test_index_shows_surrogate_metrics(project):
    report = {
        "train_scenarios": ["doe:1", "doe:2", "doe:3"],
        "test_scenarios": ["doe:4"],
        "wells": 103,
        "rows": 914640,
        "metrics": {
            "WLPT_Diff": {
                "neural_mae": 1.5, "linear_mae": 2.5,
                "neural_rmse": 1.8, "linear_rmse": 2.9,
                "neural_r2": 0.91, "linear_r2": 0.72,
            },
            "WOMT_Diff": {"neural_mae": 0.8, "linear_mae": 1.1},
        },
        "training": {
            "seed": 20260817,
            "sklearn_version": "1.9.0",
            "pandas_version": "3.0.1",
            "datasets": [{"path": "data/dataset.csv", "sha256": "abcdef1234567890"}],
        },
    }
    artifacts = project / "artifacts"
    artifacts.mkdir()
    (artifacts / "surrogate.joblib").write_bytes(b"model placeholder")
    (artifacts / "surrogate-metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    text = TestClient(create_app(project)).get("/").text

    assert "3 / 1" in text
    assert "914640" in text
    assert "порог качества" in text.lower()
    assert "пройден" in text
    assert "WLPT_Diff" in text
    assert "0.91" in text
    assert "20260817" in text
    assert "abcdef123456" in text
    assert "Запустить MLP" in TestClient(create_app(project)).get("/run/base").text


def test_index_survives_invalid_surrogate_report(project):
    artifacts = project / "artifacts"
    artifacts.mkdir()
    (artifacts / "surrogate-metrics.json").write_text("{nope", encoding="utf-8")

    response = TestClient(create_app(project)).get("/")

    assert response.status_code == 200
    assert "повреждён" in response.text


def test_index_rejects_non_finite_surrogate_metrics(project):
    artifacts = project / "artifacts"
    artifacts.mkdir()
    (artifacts / "surrogate-metrics.json").write_text(
        '{"train_scenarios": ["a"], "test_scenarios": ["b"], "wells": 1, '
        '"rows": 2, "metrics": {"WLPT_Diff": {"neural_mae": NaN, '
        '"linear_mae": 1}, "WOMT_Diff": {"neural_mae": 1, "linear_mae": 2}}}',
        encoding="utf-8",
    )

    text = TestClient(create_app(project)).get("/").text

    assert "повреждён" in text


def test_run_page_shows_the_reference_calculator_number(client, project):
    response = client.get("/run/base")

    assert response.status_code == 200
    assert "ЧДД" in response.text
    assert economics.VENDOR_VERSION in response.text
    assert "Своей формулы ЧДД в проекте нет" in response.text


def test_run_page_ranks_the_worst_well_first(client):
    text = client.get("/run/base").text
    assert text.index("bad") < text.index("good")


def test_pages_have_no_external_resources(client):
    """Регламент: никаких зависимостей от ресурсов, не поставляемых с решением."""
    for path in ("/", "/run/base"):
        text = client.get(path).text
        assert "http://" not in text
        assert "https://" not in text
        assert "<script" not in text


def test_charts_are_inline_svg(client):
    text = client.get("/run/base").text
    assert "<svg" in text and "polyline" in text and "<rect" in text


def test_missing_run_is_404(client):
    assert client.get("/run/nope").status_code == 404


def test_run_without_results_is_404(client):
    assert client.get("/run/empty").status_code == 404


def test_cycle_runs_and_reports_a_forecast(client):
    response = client.post("/run/base/cycle")

    assert response.status_code == 200
    assert "ЧДД базового сценария" in response.text
    assert "прогноз" in response.text
    assert "не подтверждено" in response.text or "не подтверждён" in response.text
    assert "тривиальный бейзлайн" in response.text


def test_invalid_neural_artifact_is_a_controlled_error(project):
    artifacts = project / "artifacts"
    artifacts.mkdir()
    (artifacts / "surrogate.joblib").write_bytes(b"not a joblib model")

    response = TestClient(create_app(project)).post("/run/base/cycle?model=neural")

    assert response.status_code == 409
    assert "не удалось загрузить" in response.text


def test_cycle_explains_each_decision(client):
    text = client.post("/run/base/cycle").text
    assert "Принятые решения" in text
    assert "обводнённость" in text or "Действий нет" in text


def test_rows_csv_is_read_as_numbers(project):
    rows = read_rows_csv(project / "runs" / "base" / "rows.csv")

    assert isinstance(rows[0]["WLPR"], float)
    assert isinstance(rows[0]["WOMT_Diff"], float)
    assert rows[0]["well"] in {"good", "bad"}


def test_rows_csv_round_trips_through_the_writer(tmp_path):
    """Выгрузка, записанная нами, читается нами же — без этого обмен файлами не работает."""
    rows = rows_for("1", 0.5, 0.01, months=3)
    path = summary.write_csv(rows, tmp_path / "rows.csv")

    restored = read_rows_csv(path)

    assert len(restored) == len(rows)
    assert restored[0]["WOMT_Diff"] == pytest.approx(rows[0]["WOMT_Diff"])


@pytest.fixture
def two_runs(project):
    """Второй прогон: тот же фонд, но обводнённая скважина остановлена."""
    run_dir = project / "runs" / "tuned"
    run_dir.mkdir(parents=True)
    (run_dir / "wells_schedule.inc").write_text(SCHEDULE, encoding="utf-8")

    rows = rows_for("good", 0.50, 0.005) + [
        make_row(row["DATA"], "bad") for row in rows_for("bad", 0.985, 0.0005)
    ]
    with (run_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=economics.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return project


def test_compare_shows_both_numbers_and_the_difference(two_runs):
    client = TestClient(create_app(two_runs))
    response = client.get("/compare", params={"a": "base", "b": "tuned"})

    assert response.status_code == 200
    assert "base" in response.text and "tuned" in response.text
    assert "Разница, млн руб." in response.text
    assert "Кто дал разницу" in response.text


def test_compare_attributes_the_difference_to_wells(two_runs):
    """Разница, собранная на двух-трёх скважинах, — признак хрупкого решения."""
    client = TestClient(create_app(two_runs))
    text = client.get("/compare", params={"a": "base", "b": "tuned"}).text

    assert "bad" in text
    assert "решение хрупкое" in text


def test_compare_needs_results_on_both_sides(two_runs):
    client = TestClient(create_app(two_runs))
    assert client.get("/compare", params={"a": "base", "b": "empty"}).status_code == 404


def test_index_offers_comparison_when_there_are_two_runs(two_runs):
    client = TestClient(create_app(two_runs))
    assert 'action="/compare"' in client.get("/").text


def test_index_hides_comparison_when_there_is_nothing_to_compare(client):
    assert 'action="/compare"' not in client.get("/").text
