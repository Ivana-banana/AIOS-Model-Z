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
max_field_injection_m3_per_day = 0
enforce_water_balance = true
max_external_water_m3_per_day = 0
water_balance_rule = "compensation"
water_tolerance = 1.0

[explain]
enabled = false
model = "claude-opus-5"
max_tokens = 1024
effort = "low"
cache_dir = "artifacts/explanations"

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


def test_a_refusal_is_a_page_not_json(client):
    """Отказ показывается страницей — в том числе на несуществующем маршруте.

    До 03.09 любой 4xx отдавал `{"detail": …}`, включая `/favicon.ico`, который
    браузер просит сам. На демо это выглядит как сломанный сервис, хотя отказ
    законный (ревизия 03.09, 2.4 #4).
    """
    for url in ("/run/нет-такого", "/favicon.ico"):
        answer = client.get(url)
        assert answer.status_code == 404
        assert answer.headers["content-type"].startswith("text/html"), url
        assert "к списку прогонов" in answer.text


def test_the_forecast_page_survives_a_marker_without_a_number(project):
    """Маркер в той же системе отсчёта, но без числа, не роняет страницу.

    Найдено критиком 03.09: `"%.1f"|format(None)` бросает `TypeError`, а это не
    `HTTPException` — новый обработчик отказов его не ловит, и страница
    отдавала голый `Internal Server Error`, то есть ровно то, что обработчик и
    должен был устранить.
    """
    (project / "runs" / "base" / "cycle.json").write_text(
        json.dumps({"start_date": "2014-01-01", "source_run": "base"}), encoding="utf-8")

    answer = TestClient(create_app(project)).get("/forecast")

    assert answer.status_code == 200
    assert "не записан" in answer.text


def test_the_forecast_page_does_not_call_a_run_its_own_base(project):
    """Прогон, названный источником самого себя, прироста не измеряет.

    Разность выйдет нулём при любом решении, и печатать «получено 0» значило бы
    выдать вырожденный случай за измерение (найдено критиком 03.09).
    """
    (project / "runs" / "base" / "cycle.json").write_text(
        json.dumps({"forecast_chdd_m": 100.0, "forecast_gain_m": 50.0,
                    "start_date": "2014-01-01", "source_run": "base"}),
        encoding="utf-8")

    answer = TestClient(create_app(project)).get("/forecast")

    assert "источник — он сам" in answer.text
    assert "+0.0" not in answer.text


def test_a_broken_error_template_still_shows_the_refusal(project, monkeypatch):
    """Сломанный шаблон отказа не превращает 404 в «сервис лежит».

    Найдено критиком 03.09: обработчик рендерил страницу без запасного пути, и
    любая его поломка молча роняла **все** отказы сайта в голый 500.
    """
    from aios.web import app as app_module

    def broken(*args, **kwargs):
        raise RuntimeError("шаблон недоступен")

    monkeypatch.setattr(app_module.TEMPLATES, "TemplateResponse", broken)
    client = TestClient(create_app(project), raise_server_exceptions=False)

    answer = client.get("/run/нет-такого")

    assert answer.status_code == 404
    assert "404" in answer.text
    assert "страница отказа не отрисовалась" in answer.text


def test_the_forecast_page_refuses_to_compare_across_frames(project):
    """Прогноз из другой системы отсчёта не сравнивается, а помечается.

    Маркер, записанный при базе дисконтирования 2007, и факт при 1991 — числа
    из разных систем, и по их величине этого не видно: разница доходит до разов
    (Р-187). Маркеры до 03.09 базу не записывали вовсе.
    """
    (project / "runs" / "base" / "cycle.json").write_text(
        json.dumps({"forecast_chdd_m": 100.0, "forecast_gain_m": 50.0,
                    "start_date": "2007-01-01", "source_run": "base"}),
        encoding="utf-8")

    answer = TestClient(create_app(project)).get("/forecast")

    assert answer.status_code == 200
    assert "другой системе отсчёта" in answer.text
    assert "база 2007" in answer.text


def test_the_forecast_page_compares_within_one_frame(project):
    """В одной системе отсчёта страница показывает обещанное и полученное."""
    (project / "runs" / "base" / "cycle.json").write_text(
        json.dumps({"forecast_chdd_m": 100.0, "forecast_gain_m": 50.0,
                    "start_date": "2014-01-01", "source_run": "base",
                    "surrogate_identity": {"surrogate": "тривиальный бейзлайн"}}),
        encoding="utf-8")

    answer = TestClient(create_app(project)).get("/forecast")

    assert answer.status_code == 200
    assert "другой системе отсчёта" not in answer.text
    assert "тривиальный бейзлайн" in answer.text
    # прогон сравнивается сам с собой, поэтому полученный прирост равен нулю —
    # проверяется, что сравнение вообще состоялось, а не его величина
    assert "+50.0" in answer.text


def test_index_lists_runs_and_configuration(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "base" in response.text
    assert "2014-01-01" in response.text          # дата начала управления из конфига
    assert "20260817" in response.text            # seed — регламент про воспроизводимость
    assert "Мощность ППД" in response.text
    assert "не задана" in response.text


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
    # обученная модель предлагается в форме запуска МАС только когда она есть
    assert 'value="neural"' in TestClient(create_app(project)).get("/run/base").text


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
    response = client.get("/run/base/cycle")

    assert response.status_code == 200
    # проверяется подпись плитки, а не текст отчёта ниже: до 24.08 та же строка
    # находилась в «Отчёте целиком», и тест проходил, ничего не проверяя
    assert "ЧДД базового расписания (прогноз модели)" in response.text
    assert "прогноз" in response.text
    assert "не подтверждено" in response.text or "не подтверждён" in response.text
    assert "тривиальный бейзлайн" in response.text


def test_invalid_neural_artifact_is_a_controlled_error(project):
    artifacts = project / "artifacts"
    artifacts.mkdir()
    (artifacts / "surrogate.joblib").write_bytes(b"not a joblib model")

    response = TestClient(create_app(project)).get("/run/base/cycle?model=neural")

    assert response.status_code == 409
    assert "не удалось загрузить" in response.text


def test_cycle_explains_each_decision(client):
    text = client.get("/run/base/cycle").text
    assert "Принятые решения" in text
    assert "обводнённость" in text or "Действий нет" in text


def test_cycle_page_shows_the_interpretability_layer(client):
    """ТЗ п. 3.1: интерфейс должен быть достаточен для интерпретации результатов."""
    text = client.get("/run/base/cycle").text
    assert "Почему система решила так" in text


def test_the_origin_of_the_explanation_is_visible(client):
    """Читателю обязано быть видно, объяснение от модели или шаблонное."""
    text = client.get("/run/base/cycle").text
    assert "Чем получено" in text
    assert "Отпечаток фактов" in text


def test_template_explanation_is_labelled_as_such(client):
    """Слой выключен в конфигурации тестов — значит, шаблон, и так и написано."""
    text = client.get("/run/base/cycle").text
    assert "слой выключен в конфигурации" in text


def test_a_rejected_model_answer_is_shown_not_hidden(project, monkeypatch):
    """Отвергнутый ответ с выдуманным числом — находка, а не мусор: он виден."""
    from aios import explain as explain_module

    real = explain_module.explain

    def rejected(evidence, settings, **kwargs):
        base = real(evidence, explain_module.Settings(enabled=False))
        return explain_module.Explanation(
            text=base.text,
            source="template-rejected",
            fingerprint=base.fingerprint,
            model="claude-opus-5",
            rejected_numbers=("2500.0",),
        )

    monkeypatch.setattr("aios.web.app.explain_module.explain", rejected)
    text = TestClient(create_app(project)).get("/run/base/cycle").text

    assert "ответ модели отвергнут" in text
    assert "2500.0" in text


def test_a_failed_model_call_names_the_reason(project, monkeypatch):
    """Слой на живом API не работал ни разу: молчаливый шаблон нечем чинить."""
    from aios import explain as explain_module

    def failing(evidence, settings, **kwargs):
        base = real_explain(evidence, explain_module.Settings(enabled=False))
        return explain_module.Explanation(
            text=base.text,
            source="template-failed",
            fingerprint=base.fingerprint,
            model="claude-opus-5",
            failure="NotFoundError: model claude-opus-5 not found",
        )

    real_explain = explain_module.explain
    monkeypatch.setattr("aios.web.app.explain_module.explain", failing)
    text = TestClient(create_app(project)).get("/run/base/cycle").text

    assert "вызов модели не состоялся" in text
    assert "model claude-opus-5 not found" in text


def test_the_explanation_never_breaks_the_page(project, monkeypatch):
    """Отказ слоя — это отчётность о решении, а не само решение. Страница живёт.

    Результат цикла — числа ЧДД и расписание — не должен пропадать оттого, что
    сломалось его словесное изложение.
    """
    def broken(evidence, settings, **kwargs):
        raise RuntimeError("слой сломался")

    monkeypatch.setattr("aios.web.app.explain_module.explain", broken)
    response = TestClient(create_app(project)).get("/run/base/cycle")

    assert response.status_code == 200
    # проверяется подпись плитки, а не текст отчёта ниже: до 24.08 та же строка
    # находилась в «Отчёте целиком», и тест проходил, ничего не проверяя
    assert "ЧДД базового расписания (прогноз модели)" in response.text
    assert "Почему система решила так" not in response.text


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


def test_rows_csv_missing_required_column_is_rejected(project):
    """Неполная выгрузка не должна выглядеть как нулевой корректный прогон."""
    path = project / "runs" / "base" / "rows.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    columns = [column for column in economics.REQUIRED_COLUMNS if column != "WOMT_Diff"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(source)

    with pytest.raises(ValueError, match="WOMT_Diff"):
        read_rows_csv(path)

    response = TestClient(create_app(project)).get("/run/base")
    assert response.status_code == 422
    assert "WOMT_Diff" in response.text


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


def test_cycle_can_start_from_any_ready_scenario(two_runs):
    """МАС запускается от выбранного прогона, а не только от каталога `base`."""
    client = TestClient(create_app(two_runs))

    run_page = client.get("/run/tuned")
    cycle_page = client.get("/run/tuned/cycle")

    assert run_page.status_code == 200
    assert 'action="/run/tuned/cycle"' in run_page.text
    assert cycle_page.status_code == 200
    assert "Цикл управления · прогон «tuned»" in cycle_page.text
    assert "/run/tuned/cycle?model=decline" in cycle_page.text


def test_index_hides_comparison_when_there_is_nothing_to_compare(client):
    assert 'action="/compare"' not in client.get("/").text


def test_run_page_says_where_the_densities_came_from(client, project):
    """Плотность — множитель в выручке, и на чём посчитано, обязано быть видно.

    В этом проекте дека нет, значит фонд считается по общей плотности из
    конфигурации. Без пометки такой прогон на странице выглядит ровно так же,
    как посчитанный по PVT-регионам дека, — а на чужом пласте (ТЗ п. 4.3) это
    разные числа.
    """
    text = client.get("/run/base").text

    assert "плотност" in text.lower()
    assert "конфигурации" in text
    # и подано как оговорка, а не как обычная строка: подеквенных плотностей нет
    assert "множитель в выручке" in text


def test_pages_pull_nothing_from_the_network():
    """Страницы обязаны открываться без сети — их смотрят на чужой машине.

    Подключить CDN с библиотекой графиков легко и незаметно, а проверяющий может
    открыть интерфейс в закрытом контуре. Поэтому графики рисуются inline-SVG, и
    это свойство закрепляется тестом, а не обещанием в README.
    """
    templates = Path(__file__).resolve().parent.parent / "aios" / "web" / "templates"
    offenders = []
    for page in sorted(templates.glob("*.html")):
        text = page.read_text(encoding="utf-8")
        for marker in ("http://", "https://", "//cdn", "<script src", "<link rel=\"stylesheet\""):
            if marker in text:
                offenders.append(f"{page.name}: {marker}")
    assert not offenders, "страницы тянут внешние ресурсы:\n" + "\n".join(offenders)


# --- запуск МАС из интерфейса: режимы и сборка кейса -------------------------

DECK_DATA = """\
RUNSPEC
METRIC

GRID
INCLUDE
'Model_Z_grid.inc' /

SUMMARY
INCLUDE
'Model_Z_summary.inc' /

SCHEDULE
INCLUDE
'Model_Z_sch.inc' /

END
"""

DECK_SCHEDULE = """\
WELSPECS
 'good' 'GROUP' 23 17 1* 'OIL' /
 'bad' 'GROUP' 24 18 1* 'OIL' /
/

DATES
 01 NOV 2013 /
/

COMPDATMD
 'good' 1* 1326.0 1329.3 MD OPEN 2* 0.146 1* 2 1* 1 /
 'bad' 1* 1301.0 1305.0 MD OPEN 2* 0.146 1* 2 1* 1 /
/

WCONPROD
 'good' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
 'bad' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""


@pytest.fixture
def project_with_deck(project):
    """Тот же проект, но с деком организаторов: без него кейс не собрать."""
    deck = project / "vendor" / "Model_Z_OPM"
    deck.mkdir(parents=True)
    (deck / "Model_Z.data").write_text(DECK_DATA, encoding="utf-8")
    (deck / "Model_Z_sch.inc").write_text(DECK_SCHEDULE, encoding="utf-8")
    (deck / "Model_Z_summary.inc").write_text("", encoding="utf-8")
    (deck / "Model_Z_grid.inc").write_text("-- сетка\n", encoding="utf-8")
    return project


def test_cycle_page_names_the_launch_parameters(client):
    """Числа подписаны запуском: по странице видно, чем именно её посчитали."""
    text = client.get("/run/base/cycle?model=decline&mode=stops&lookback=9").text

    assert "только остановка убыточных" in text
    assert "окно диагноза 9 мес." in text


def test_stops_only_mode_skips_rate_tuning(client):
    """Режим «только остановки» обязан выключать подбор режимов, а не подписывать его."""
    full = client.get("/run/base/cycle?mode=full").text
    stops = client.get("/run/base/cycle?mode=stops").text

    assert "Подбор режимов" in full
    assert "Подбор режимов" not in stops


def test_the_address_repeats_the_launch(client):
    """Ссылка на страницу цикла несёт все параметры: прогон повторяется по адресу."""
    text = client.get("/run/base/cycle?model=decline&mode=stops&lookback=9").text

    # ссылка пересчёта несёт все три параметра, а не одно имя прогона
    assert "model=decline&amp;mode=full&amp;lookback=9" in text


def test_unknown_launch_parameters_are_rejected(client):
    """Молча подставить умолчание — значит показать не тот прогон, что просили."""
    assert client.get("/run/base/cycle?mode=назад").status_code == 400
    assert client.get("/run/base/cycle?model=gpt").status_code == 400
    assert client.get("/run/base/cycle?lookback=0").status_code == 400
    assert client.get("/run/base/cycle?lookback=600").status_code == 400


def test_saving_the_cycle_builds_a_case(project_with_deck):
    """МАС из интерфейса обязана давать артефакт, а не только картинку."""
    client = TestClient(create_app(project_with_deck))

    response = client.post("/run/base/cycle/save?model=decline&mode=full&lookback=6")

    assert response.status_code == 200
    case_dir = project_with_deck / "runs" / "base-mas-decline"
    assert (case_dir / "wells_schedule.inc").is_file()
    assert (case_dir / "RUN.DATA").is_file()
    # история дека до начала управления сохраняется дословно — в ней вскрытия
    assert "01 NOV 2013" in (case_dir / "wells_schedule.inc").read_text(encoding="utf-8")


def test_the_saved_case_says_on_which_densities_it_was_chosen(project_with_deck):
    """Кейс уезжает считаться на другую машину — подпись обязана ехать с ним.

    Хвост Р-47: `rows.csv` в каталоге кейса не появляется, его пишет
    `export_rows.py` уже по результату полного расчёта. Без этой строки то, каким
    числом цикл выбирал решения, оставалось только в прокрутке терминала.
    """
    client = TestClient(create_app(project_with_deck))

    client.post("/run/base/cycle/save?model=decline&mode=full&lookback=6")

    note = project_with_deck / "runs" / "base-mas-decline" / "fluids.source.txt"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "кейс: base-mas-decline" in text
    assert "прогноз цикла считался одной плотностью" in text


def test_the_saved_case_is_called_a_forecast(project_with_deck):
    """Прирост собранного кейса — прогноз суррогата, пока нет полного расчёта."""
    client = TestClient(create_app(project_with_deck))

    text = client.post("/run/base/cycle/save").text

    assert "прогноз суррогата" in text
    assert "run_flow.sh" in text
    assert "--confirm runs/base-mas-decline" in text


def test_two_launch_modes_do_not_share_a_directory(project_with_deck):
    """Разные режимы запуска — разные кейсы: иначе один тихо затирает другой."""
    client = TestClient(create_app(project_with_deck))

    client.post("/run/base/cycle/save?mode=full")
    client.post("/run/base/cycle/save?mode=stops")

    assert (project_with_deck / "runs" / "base-mas-decline").is_dir()
    assert (project_with_deck / "runs" / "base-mas-decline-stops").is_dir()


def test_an_existing_run_is_never_overwritten(project_with_deck):
    """Прогон стоит часов расчёта: сборка кейса не смеет его затирать."""
    client = TestClient(create_app(project_with_deck))
    assert client.post("/run/base/cycle/save").status_code == 200

    response = client.post("/run/base/cycle/save")

    assert response.status_code == 409
    assert "уже есть" in response.text


def test_without_the_deck_the_case_is_refused_with_a_reason(client):
    """Кейс собирается поверх дека организаторов; без него — отказ, а не пустышка.

    Код 503, а не 409: дека нет в окружении, и повторять тот же запрос
    бессмысленно, пока его не положат на место.
    """
    response = client.post("/run/base/cycle/save")

    assert response.status_code == 503
    assert "дек модели недоступен" in response.text


def test_a_broken_schedule_never_reaches_the_disk(project_with_deck, monkeypatch):
    """Расписание с нарушениями не должно доходить до файла, который мы отправляем."""
    from dataclasses import replace

    from aios import loop as loop_module

    real = loop_module.run_cycle

    def with_problems(*args, **kwargs):
        return replace(real(*args, **kwargs), problems=("дебит 700 м3/сут выше предела",))

    monkeypatch.setattr("aios.web.app.loop.run_cycle", with_problems)
    response = TestClient(create_app(project_with_deck)).post("/run/base/cycle/save")

    assert response.status_code == 422
    assert "нарушает ограничения" in response.text
    assert not (project_with_deck / "runs" / "base-mas-decline").exists()


def test_a_failed_build_is_not_reported_as_a_conflict(project_with_deck, monkeypatch):
    """Сборка сорвалась на записи — это наш сбой, а не занятый каталог.

    Три отказа сборки должны различаться кодом, а не только текстом: страницу на
    защите трогают руками. 503 — дека нет, 422 — расписание непригодно,
    409 — каталог занят, 500 — сборка сорвалась.
    """
    def fails(*args, **kwargs):
        raise OSError("диск переполнен")

    monkeypatch.setattr("aios.web.app.build_case", fails)
    response = TestClient(create_app(project_with_deck)).post("/run/base/cycle/save")

    assert response.status_code == 500
    assert "диск переполнен" in response.text


def test_the_cycle_page_offers_the_case_when_the_deck_is_there(project_with_deck):
    """Кейс собирается с той же страницы, где показаны решения цикла."""
    text = TestClient(create_app(project_with_deck)).get("/run/base/cycle").text

    assert "Собрать кейс в runs/base-mas-decline" in text
    assert "/run/base/cycle/save?model=decline&amp;mode=full" in text


def test_without_the_deck_the_page_says_why_there_is_no_button(client):
    """Кнопка сборки не предлагается там, где сборка заведомо не выйдет."""
    text = client.get("/run/base/cycle").text

    assert "Собрать кейс в runs/" not in text
    assert "Дек модели недоступен" in text


def test_comparison_covers_both_funds_not_only_the_left_one():
    """«Где разошлись деньги» обязано смотреть в обе стороны.

    Дельты считались перебором фонда левого прогона, а отсутствующей справа
    скважине подставлялся ноль. Замер 21.08: скважина, которой нет справа,
    показывалась как «справа +0.00, дельта −1.08» — будто потеряла все деньги, —
    а скважина, которой нет слева, не показывалась вовсе.

    На двух прогонах одного дека фонд совпадает и разницы не видно. Она видна
    там, где сравнение и нужно: наш прогон против прогона на кейсе организаторов,
    где фонд другой.
    """
    from conftest import make_row

    from aios import economics
    from aios.margin import Norms
    from aios.web.app import margin_deltas

    assumptions, _ = economics.load_norms()
    norms = Norms.from_assumptions(assumptions)
    left = [make_row("2007-01-01", "общая", wlpr=30, womr=5, liquid=900, oil=150),
            make_row("2007-01-01", "только слева", wlpr=30, womr=5, liquid=900, oil=150)]
    right = [make_row("2007-01-01", "общая", wlpr=30, womr=4, liquid=900, oil=120),
             make_row("2007-01-01", "только справа", wlpr=30, womr=9, liquid=900, oil=270)]

    deltas = margin_deltas(left, right, norms)
    by_well = {well: (before, after) for well, before, after in deltas}

    assert "только справа" in by_well, "скважина, которой нет слева, выпала из сравнения"
    assert by_well["только слева"][1] is None, (
        "отсутствующая скважина показана нулём — это «работала и не заработала», "
        "а не «её здесь нет»"
    )
    assert by_well["только справа"][0] is None
    assert by_well["общая"][0] is not None and by_well["общая"][1] is not None


def _report_with(metrics: dict) -> dict:
    return {
        "train_scenarios": ["doe:1", "doe:2", "doe:3"],
        "test_scenarios": ["doe:4"],
        "wells": 103,
        "rows": 914640,
        "metrics": metrics,
        "training": {"seed": 1, "sklearn_version": "1.9.0", "pandas_version": "3.0.1",
                     "datasets": []},
    }


def _render(project, report: dict) -> str:
    artifacts = project / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "surrogate.joblib").write_bytes(b"model placeholder")
    (artifacts / "surrogate-metrics.json").write_text(json.dumps(report), encoding="utf-8")
    return TestClient(create_app(project)).get("/").text


def test_the_interface_separates_the_level_from_the_response(project):
    """Одно `R²` показывало отличную модель там, где управление не предсказано.

    Замер 22.08 по нефти: `R²` цели 0,9504, из них уровень даёт 0,9142. Интерфейс
    обязан показывать обе величины рядом, иначе он вводит в заблуждение ровно в
    том месте, где принимается решение.
    """
    text = _render(project, _report_with({
        "WLPT_Diff": {
            "neural_mae": 1.5, "linear_mae": 2.5,
            "neural_r2": 0.966, "level_r2": 0.9104,
            "neural_response_r2": 0.6203, "neural_skill_median": 0.3647,
        },
        "WOMT_Diff": {
            "neural_mae": 0.8, "linear_mae": 1.1,
            "neural_r2": 0.9504, "level_r2": 0.9142,
            "neural_response_r2": 0.4215, "slope_response_r2": 0.2793,
            "neural_skill": 0.0444, "neural_skill_median": -0.0581,
            "neural_skill_iqr": 0.3495,
        },
    }))

    assert "0.9504" in text and "0.9142" in text and "0.4215" in text
    assert "R² уровня" in text and "R² отклика" in text


def test_a_model_worse_than_doing_nothing_is_shown_as_unfit(project):
    """Отрицательная медиана `Skill` обязана читаться словами, а не числом в углу."""
    text = _render(project, _report_with({
        "WLPT_Diff": {
            "neural_mae": 1.5, "linear_mae": 2.5,
            "neural_r2": 0.966, "neural_response_r2": 0.62,
            "neural_skill_median": 0.3647,
        },
        "WOMT_Diff": {
            "neural_mae": 0.8, "linear_mae": 1.1,
            "neural_r2": 0.9504, "neural_response_r2": 0.4215,
            "neural_skill_median": -0.0581,
        },
    }))

    assert "хуже бездействия" in text
    assert "порог качества" in text.lower() and "не пройден" in text


def test_a_model_that_predicts_the_response_is_shown_as_fit(project):
    text = _render(project, _report_with({
        target: {
            "neural_mae": 0.8, "linear_mae": 1.1,
            "neural_r2": 0.96, "neural_response_r2": 0.62,
            "neural_skill_median": 0.3647,
        }
        for target in ("WLPT_Diff", "WOMT_Diff")
    }))

    assert "хуже бездействия" not in text
    assert "не пройден" not in text


def test_metrics_missing_from_an_older_artifact_read_as_not_measured(project):
    """`нет` и `не измерено` — не то же самое, что «плохо»."""
    text = _render(project, _report_with({
        target: {"neural_mae": 0.8, "linear_mae": 1.1}
        for target in ("WLPT_Diff", "WOMT_Diff")
    }))

    assert "не измерено" in text
    assert "хуже бездействия" not in text


def test_a_missing_trust_artifact_says_the_second_gate_is_absent(project):
    text = TestClient(create_app(project)).get("/").text

    assert "Доверенная область не собрана" in text


def test_an_uncalibrated_trust_gate_is_never_shown_as_neutral(project):
    """Требование разбора 4-7 дословно: не зелёный статус, а прямая фраза."""
    artifacts = project / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "trust-gate.json").write_text(json.dumps({
        "calibrated": False,
        "training_scenario_ids": [f"s{i}" for i in range(48)],
        "threshold_tau": 20.587,
        "conformal_samples": 0,
        "optimism_margin": None,
        "margin_reason": "нет ни одной пары «прогноз и полный расчёт»",
        "guarantee_scope": "маргинальное покрытие при сценарной обменяемости",
        "shift_audit": {"candidates": 200, "passed_coordinatewise": 200,
                        "passed_scenario_distance": 89,
                        "caught_by_the_second_gate_only": 111},
    }), encoding="utf-8")

    text = TestClient(create_app(project)).get("/").text

    assert "не калибрована; кандидат обязателен к полному расчёту" in text
    assert "48" in text and "20.6" in text
    assert "111" in text, "аудит сдвинутых режимов — то, чем защита доказывает себя"


def test_a_calibrated_trust_gate_shows_the_margin_and_still_names_the_scope(project):
    artifacts = project / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "trust-gate.json").write_text(json.dumps({
        "calibrated": True,
        "training_scenario_ids": ["s1", "s2"],
        "threshold_tau": 3.5,
        "conformal_samples": 12,
        "optimism_margin": 421.7,
        "guarantee_scope": "не гарантия для адаптивно выбранного кандидата",
    }), encoding="utf-8")

    text = TestClient(create_app(project)).get("/").text

    assert "421.7" in text
    assert "не калибрована" not in text
    assert "не гарантия для адаптивно выбранного кандидата" in text


def test_the_hour_long_search_is_refused_instead_of_hanging_the_request(project,
                                                                       monkeypatch):
    """Веб-запрос не имеет права считать час — он обязан отказать и назвать команду.

    Ссылка «обученным суррогатом» ведёт на `model=neural&mode=full`, то есть на
    совместный подбор режимов: 24.08 тот же расчёт из командной строки шёл
    **1 ч 20 мин** (`docs/experiments.md`). На защите это зависший браузер
    посреди доклада, а не «медленная страница».

    Подменять режим на «только остановки» нельзя — пользователь просил одно,
    система сделала бы другое и не сказала (тот же класс, что Р-100).
    """
    from aios.web import app as app_module

    class Chosen:
        name = "обученный суррогат"
        joint_rate_search = True
        model = object()

    monkeypatch.setattr(app_module, "for_cycle", lambda **kwargs: Chosen())

    response = TestClient(create_app(project)).get(
        "/run/base/cycle?model=neural&mode=full"
    )

    assert response.status_code == 409
    assert "больше часа" in response.text
    assert "run_cycle.py" in response.text, "отказ обязан называть команду"
    assert "остановка убыточных" in response.text, "и рабочую замену на странице"


def test_the_cycle_page_folds_warnings_instead_of_printing_all_of_them(client,
                                                                      monkeypatch):
    """Страница показывает причины, а не по строке на скважино-месяц.

    Замер 24.08 (Б32): на тривиальном бейзлайне цикл даёт 7 990 пометок, и
    страница весила **1 948 КБ** против 36 КБ у обученной модели. Свёртка по
    причинам (`aios.surrogate.fold_warnings`) — тот же приём, которым уже
    сворачивались списки в командной строке.
    """
    from aios.loop import CycleResult
    from aios.web import app as app_module

    warnings = tuple(
        f"скв. {index}: режим 1.9 вне обучающего диапазона 0.5…1.5"
        for index in range(4000)
    ) + tuple(
        f"скв. {index}: прогноз нарушает физические проверки, строк 3"
        for index in range(1000)
    )
    monkeypatch.setattr(app_module.loop, "run_cycle", lambda *a, **k: CycleResult(
        base_chdd_m=100.0, chdd_m=120.0, actions=(), rate_choices=(), allocations=(),
        steps=(), warnings=warnings, problems=(),
    ))

    response = client.get("/run/base/cycle")
    text = response.text

    assert "Пометок 5000" in text and "причин 2" in text
    assert "например:" in text, "у каждой причины обязан остаться пример"
    # в блоке границ применимости — ровно по одному примеру на причину; текстовый
    # «Отчёт целиком» ниже печатает свои первые десять строк и в счёт не идёт
    assert text.count("например:") == 2
    assert len(text.encode("utf-8")) < 100_000, "страница обязана весить десятки килобайт"


# --- С7: имя прогона из адреса попадало в HTML без экранирования -------------


def test_the_fallback_refusal_escapes_the_name_from_the_url(project, monkeypatch):
    """Запасной путь отказа обязан экранировать текст причины.

    Причина несёт имя прогона **из адреса** (`_find`: «Прогон «{name}» не
    найден»), то есть текст, который задаёт открывающий ссылку. Основной путь
    идёт через шаблон Jinja и экранирует сам; дыра была в запасном, который
    собирает HTML строкой. Воспроизведено 04.09: `/run/<img src=x onerror=…>`
    возвращал разметку как есть.
    """
    import aios.web.app as app_module

    def broken(*args, **kwargs):
        raise RuntimeError("шаблон сломан")

    monkeypatch.setattr(app_module.TEMPLATES, "TemplateResponse", broken)
    client = TestClient(create_app(project), raise_server_exceptions=False)

    payload = "<img src=x onerror=alert(1)>"
    answer = client.get("/run/" + payload)

    assert answer.status_code == 404
    assert payload not in answer.text, "разметка из адреса дошла до страницы как есть"
    assert "&lt;img" in answer.text, "текст причины пропал вместе с экранированием"


# --- С8: /forecast падал в голый 500 на трёх формах cycle.json ---------------


def _write_marker(project, payload):
    import json

    run = project / "runs" / "probe"
    if not run.is_dir():
        run.mkdir(parents=True)
        import shutil
        shutil.copy(project / "runs" / "base" / "rows.csv", run / "rows.csv")
    (run / "cycle.json").write_text(json.dumps(payload), encoding="utf-8")
    return TestClient(create_app(project), raise_server_exceptions=False)


def test_forecast_survives_a_marker_that_is_not_an_object(project):
    """Один испорченный `cycle.json` не должен уносить всю таблицу.

    `list.get` роняло страницу в голый 500 — то есть сравнение «прогноз против
    факта», главный ответ трека 2, пропадало целиком из-за одного файла.
    """
    answer = _write_marker(project, [1, 2, 3]).get("/forecast")

    assert answer.status_code == 200


def test_forecast_survives_a_number_written_as_a_string(project):
    """`cycle.json` пишем мы, но читаем у чужого прогона: файл мог быть правлен."""
    answer = _write_marker(project, {
        "start_date": "2014-01-01", "source_run": "base",
        "forecast_gain_m": "12.5", "forecast_chdd_m": "1.0",
    }).get("/forecast")

    assert answer.status_code == 200


def test_forecast_survives_a_marker_without_the_promised_gain(project):
    """Обещания нет, а полученное есть — разность роняла страницу.

    Столбец диаграммы требует **обоих** чисел; до правки `None - 12.5` уходило в
    500 ровно на этом сочетании.
    """
    answer = _write_marker(project, {
        "start_date": "2014-01-01", "source_run": "base",
    }).get("/forecast")

    assert answer.status_code == 200


def test_a_number_from_a_marker_is_read_or_refused_but_never_guessed():
    """Помощник разбора: строка с числом читается, мусор даёт None.

    `True` отдельно: он `int` по наследству, и без явной проверки прогноз
    «единица» появился бы из флага.
    """
    from aios.web.app import _as_number

    assert _as_number(12.5) == 12.5
    assert _as_number("12.5") == 12.5
    assert _as_number("не число") is None
    assert _as_number(None) is None
    assert _as_number([1]) is None
    assert _as_number(True) is None
    assert _as_number("nan") is None
    assert _as_number("inf") is None


# --- страница цикла не должна распухать от решений планировщика ---------------
# Замер 04.09: 570 строк таблицы и 222 КБ против задокументированных 30 КБ.
# Причина — коридор по годам (Р-189): одна скважина получает решение на каждый
# год горизонта, и страница печатала их все.


def test_planner_decisions_are_folded_not_listed_one_by_one():
    """Свёрнуто по ресурсу, а не усечено: ни одно решение не пропадает бесследно.

    Счётчик группы показывает, сколько решений в неё вошло, и полный список
    остаётся в `cycle.json` и в выводе `run_cycle.py`.
    """
    from aios.planner import Allocation
    from aios.web.app import fold_allocations

    made = [
        Allocation(well="I1", resource="закачка", before=200.0, after=float(rate),
                   reason=f"{2007 + n}: компенсация выше потолка")
        for n, rate in enumerate((150.0, 40.0, 120.0))
    ] + [
        Allocation(well="I2", resource="закачка", before=100.0, after=90.0,
                   reason="2009: компенсация выше потолка"),
        Allocation(well="P1", resource="жидкость", before=50.0, after=25.0,
                   reason="предел объекта по жидкости"),
    ]
    groups = fold_allocations(made)

    assert [one["resource"] for one in groups] == ["закачка", "жидкость"]
    assert groups[0]["count"] == 4, "счётчик решений потерян"
    assert groups[0]["wells"] == 2, "скважины не посчитаны"
    # пример — наибольшая срезка: 200 → 40 у I1, а не первая попавшаяся
    assert groups[0]["example"].after == 40.0, "пример не самый показательный"


def test_the_cycle_page_stays_small_enough_for_the_demo(client):
    """Демо длится три минуты; страница на сотни килобайт его съедает.

    Порог с запасом к замеру 04.09 после свёртки — 21 КБ на обученном суррогате
    и 69 КБ на тривиальном бейзлайне, у которого решений политики больше.
    """
    answer = client.get("/run/base/cycle?model=decline&mode=stops&lookback=6")

    assert answer.status_code == 200

    # Порог сверху — от замера 04.09 после свёртки (21 КБ на обученном, 69 на
    # тривиальном), с запасом вдвое. Порог **снизу** тоже нужен: без него тест
    # проходит на пустой странице, а мутация «убрать свёртку из шаблона» его не
    # роняет — на фикстуре решений мало и полный список тоже помещается.
    assert 2_000 < len(answer.content) < 150_000, (
        f"страница цикла {len(answer.content) // 1024} КБ — проверить, не "
        f"печатается ли список решений целиком и не пуста ли страница"
    )

    # Таблица согласования обязана быть свёрнутой по ресурсу, а не перечислять
    # решения по одному. Раздел появляется, только когда планировщик что-то
    # срезал; на фикстуре без ограничений его нет вовсе — и это законно.
    text = answer.text
    if "Согласование планировщиком" in text:
        assert "Наибольшая срезка" in text, (
            "таблица согласования печатает решения по одному — свёртка снята"
        )
        assert "ресурсов" in text, "страница не называет число ресурсов"


def test_the_allocation_table_is_folded_when_the_planner_actually_cuts(project):
    """Свёрнутый вид проверяется там, где планировщик действительно режет.

    Предыдущий тест сторожит вес страницы, но на фикстуре без ограничений
    раздела согласования нет — и мутация «печатать решения по одному» его не
    роняла. Здесь ограничения заданы конфигурацией, раздел появляется, и
    проверяется именно его форма.
    """
    config = (project / "config.toml").read_text(encoding="utf-8")
    config = config.replace("max_field_liquid_m3_per_day = 0",
                            "max_field_liquid_m3_per_day = 10")
    (project / "config.toml").write_text(config, encoding="utf-8")

    answer = TestClient(create_app(project), raise_server_exceptions=False).get(
        "/run/base/cycle?model=decline&mode=stops&lookback=6")

    assert answer.status_code == 200
    text = answer.text
    if "Согласование планировщиком" not in text:
        import pytest

        pytest.skip("планировщик ничего не срезал — проверять форму не на чем")

    assert "Наибольшая срезка" in text, (
        "решения печатаются по одному: коридор по годам даст сотни строк"
    )
    assert "Решений" in text and "ресурсов" in text


def test_the_form_default_does_not_lead_straight_into_a_refusal(client):
    """Умолчание формы не должно вести в 409 на первом же нажатии.

    Совместный подбор режимов на обученной модели идёт больше часа — веб-запрос
    столько не живёт, и страница честно отвечает отказом с командой для
    терминала. Но пока «подбор режимов» стоял первым в списке, именно он
    выбирался по умолчанию: проверяющий, выбравший обученную модель и нажавший
    «Запустить», получал отказ, ничего не сделав неправильно (2.4 #1).
    """
    import re

    page = client.get("/run/base").text
    start = page.index('<select name="mode">')
    modes = re.findall(r'<option value="(\w+)"', page[start:start + 500])

    assert modes, "в форме не осталось вариантов режима"
    assert modes[0] == "stops", (
        f"по умолчанию выбран {modes[0]!r} — на обученной модели это сразу 409"
    )
    assert "full" in modes, "вариант с подбором режимов пропал вовсе"
