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
    assert "ЧДД базового сценария" in response.text
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
    assert "ЧДД базового сценария" in response.text
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
