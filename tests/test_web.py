"""Веб-платформа — обязательный по регламенту артефакт, значит и тестируемый."""

from __future__ import annotations

import csv
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
