"""Обвязка ЧДД обязана давать ровно то же, что эталонный расчётчик организаторов.

Расхождение заявленного ЧДД с их пересчётом = дисквалификация, поэтому паритет
проверяется на каждом прогоне, а контрольные числа заморожены.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import EXAMPLE_INPUT

from aios import economics

#: ЧДД примера организаторов при старте по умолчанию (минимальная дата данных).
GOLDEN_CHDD_DEFAULT_START = 52.05489669421487

#: Тот же пример, но год дисконтирования задан явно.
GOLDEN_CHDD_START_2014 = 54.10613636363636


#: Каталог сдаваемого прогона. В git его нет (`.gitignore`), поэтому паритет на
#: нём пропускается на чистом клоне — но там, где выгрузка есть, он обязателен:
#: маленький пример организаторов не проверяет ни 38 110 строк, ни 103 скважины,
#: ни отрицательные приращения на реальном горизонте (ревизия 03.09, раздел 3).
SUBMITTED_RUN = economics.VENDOR_DIR.parent.parent / "runs" / "matched-0995"


def test_the_declared_run_matches_the_organisers_entry_point():
    """Паритет на **той** выгрузке, которую сдаём, а не только на примере.

    Проверяются обе стороны сразу: наша обвязка и прямой запуск
    `РАСЧЕТ_ЧДД.py` — той самой командой, которой считают нас, — обязаны дать
    одно число до последнего знака, и это число обязано совпасть с машинным
    следом `npv.json`, который оставил `scripts/verify_declared.py`.
    """
    rows_file = SUBMITTED_RUN / "rows.csv"
    trace_file = SUBMITTED_RUN / "npv.json"
    if not rows_file.is_file() or not trace_file.is_file():
        pytest.skip(f"нет выгрузки или следа расчёта в {SUBMITTED_RUN}")

    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    start_year = trace["startYear"]

    ours = economics.compute_npv_from_file(
        rows_file, start_date=f"{start_year}-01-01", norms_file=economics.NORMS_FILE
    ).chdd_m

    assert abs(ours - trace["chddM"]) <= 1e-9, (
        "наша обвязка расходится со следом, оставленным точкой входа организаторов"
    )
    assert trace["agreed"], (
        f"след говорит о расхождении с заявкой: {trace['chddM']} против {trace['claimedM']}"
    )
    assert trace["vendorVersion"] == economics.VENDOR_VERSION


def test_vendor_version_is_pinned():
    """Обновился архив организаторов — контрольные числа надо перепроверить, а не подогнать."""
    assert economics.VENDOR_VERSION == "7.0.2-negative-row-filter"


def test_required_columns_are_the_organisers_contract():
    assert economics.REQUIRED_COLUMNS == [
        "DATA", "well", "WLPT", "WLPR", "WOMT", "WOMR", "WWIR", "WWIT",
        "THP", "BHP", "WEFF", "WLPT_Diff", "WOMT_Diff", "WWIT_Diff",
    ]


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_example_npv_matches_frozen_value():
    result = economics.compute_npv_from_file(EXAMPLE_INPUT)
    assert result.chdd_m == pytest.approx(GOLDEN_CHDD_DEFAULT_START, abs=1e-9)


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_start_date_shifts_discounting_base():
    result = economics.compute_npv_from_file(EXAMPLE_INPUT, start_date="2014-01-01")
    assert result.chdd_m == pytest.approx(GOLDEN_CHDD_START_2014, abs=1e-9)
    assert result.chdd_m != pytest.approx(GOLDEN_CHDD_DEFAULT_START, abs=1e-9)


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_wrapper_is_bit_identical_to_direct_vendor_call():
    """Наша обвязка ничего не добавляет от себя — сверяем с прямым вызовом эталона."""
    import chdd_model  # вендор попадает в sys.path при импорте aios.economics

    headers, records = economics.load_records(EXAMPLE_INPUT)
    assumptions, pumps = economics.load_norms()
    direct = chdd_model.compute_calculation(
        records, headers=headers, assumptions=assumptions, pumps=pumps, start_date=None
    )
    wrapped = economics.compute_npv(
        records, headers=headers, assumptions=assumptions, pumps=pumps, start_date=None
    )
    assert wrapped.chdd_m == direct["summary"]["totalChddM"]
    assert wrapped.raw["annual"] == direct["annual"]
    assert wrapped.raw["fieldMonthly"] == direct["fieldMonthly"]


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
@pytest.mark.parametrize(
    ("start_year", "golden"),
    [(None, GOLDEN_CHDD_DEFAULT_START), (2014, GOLDEN_CHDD_START_2014)],
)
def test_organisers_own_script_gives_the_same_number(tmp_path, start_year, golden):
    """Прямой запуск `РАСЧЕТ_ЧДД.py` — то, чем считают нас.

    Сверка выше идёт с `chdd_model` внутри процесса, то есть проверяет, что мы
    не добавили ничего поверх модели. Но организаторы запускают не модель, а свою
    точку входа: она сама читает исходный файл и сама строит `start_date` из
    года. Именно там расхождение и осталось бы незамеченным, а расхождение
    заявленного ЧДД с их пересчётом — дисквалификация без второй попытки.
    """
    json_file = tmp_path / "result.json"
    command = [
        sys.executable,
        str(economics.VENDOR_DIR / "РАСЧЕТ_ЧДД.py"),
        "--input", str(EXAMPLE_INPUT),
        "--output", str(tmp_path / "report.xlsx"),
        "--json", str(json_file),
    ]
    if start_year is not None:
        command += ["--start-year", str(start_year)]

    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert completed.returncode == 0, completed.stderr

    theirs = json.loads(json_file.read_text(encoding="utf-8"))["summary"]["totalChddM"]
    start_date = f"{start_year}-01-01" if start_year is not None else None
    ours = economics.compute_npv_from_file(EXAMPLE_INPUT, start_date=start_date).chdd_m

    assert theirs == pytest.approx(golden, abs=1e-9)
    # сравнение по модулю разности, а не по round: округление скрыло бы дрейф
    assert abs(theirs - ours) <= 1e-9


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_norms_come_from_the_organisers_file_not_from_literals():
    assumptions, pumps = economics.load_norms()
    assert assumptions["oilPriceRubT"] == 28_000.0
    assert assumptions["waccRate"] == 10.0
    assert assumptions["profitTaxRate"] == 25.0
    assert assumptions["propertyTaxRate"] == 2.2
    assert len(pumps) == 12
    assert pumps[-1]["nominal"] == 400 and pumps[-1]["max"] == 500


def test_start_date_has_no_default_because_forgetting_it_changes_the_declared_number():
    """Базовый год дисконтирования нельзя забыть молча.

    У эталонного расчётчика умолчание есть — минимальная дата данных. Оно не
    ошибка, но тихо меняет базовый год: выгрузка с историей от 1994 и она же с
    2007 дисконтируются по-разному и обе считаются без единой жалобы. Заявляемое
    число при этом другое, а расхождение с пересчётом организаторов — это
    дисквалификация с одной попытки.

    Поэтому у нашей обвязки умолчания нет: аргумент обязателен, и передать `None`
    осознанно по-прежнему можно — но это решение будет видно в коде вызова.
    Проверено 21.08: все десять вызовов в `aios/` и `scripts/` и все семнадцать
    в тестах дату передают, так что требование ничего не ломает.
    """
    import inspect

    signature = inspect.signature(economics.compute_npv)
    start_date = signature.parameters["start_date"]

    assert start_date.default is inspect.Parameter.empty, (
        "у start_date появилось умолчание — забытый аргумент снова станет тихим"
    )
    assert start_date.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        economics.compute_npv([], headers=economics.REQUIRED_COLUMNS)


# ── что нашёл мутационный прогон 05.09 (Р-221) ───────────────────────────────


def test_a_computed_npv_cannot_be_edited_afterwards():
    """Итог расчёта неизменяем: число заявки не правится после получения.

    Мутация `frozen=True -> False` переживала сюиту. Между тем главное правило
    проекта — заявляемое число приходит из эталонного расчёта и ниоткуда больше;
    изменяемый результат позволяет поправить его в коде между расчётом и
    записью, и сделать это молча.
    """
    import dataclasses

    assert dataclasses.is_dataclass(economics.NpvResult)
    fields = {one.name for one in dataclasses.fields(economics.NpvResult)}
    assert "chdd_m" in fields, fields

    result = economics.NpvResult(**dict.fromkeys(fields))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.chdd_m = 999.9


def test_half_given_norms_are_completed_before_the_vendor_is_called(monkeypatch):
    """Передан один норматив из двух — второй дочитывается, а не уходит `None`.

    Мутация `or -> and` в условии дочитывания переживала сюиту и **числом её не
    отличить**: эталон при `None` подставляет свои `DEFAULT_PUMPS` и
    `DEFAULT_ASSUMPTIONS`, а они сегодня совпадают с `Нормативы_ЧДД.xlsx`.
    Сегодня — да; в валидационном кейсе нормативы придут свои
    (`docs/task.md`), и тогда частичная передача молча посчитает объект по
    умолчаниям эталона вместо нормативов кейса. Разницы в числе не будет видно
    ни здесь, ни там — увидят её организаторы на своём расчёте.

    Поэтому проверяется контракт, а не число: в эталон обязаны уходить оба
    набора явно, при любой комбинации переданного.
    """
    # экономика импортирует вендора как модуль верхнего уровня (правит sys.path),
    # поэтому подменять надо ровно тот объект, который она и зовёт
    import chdd_model

    assumptions, pumps = economics.load_norms(economics.NORMS_FILE)
    seen: dict[str, object] = {}

    def spy(records, **keywords):
        seen.update(keywords)
        return {"chddM": 0.0, "years": [], "rows": []}

    monkeypatch.setattr(chdd_model, "compute_calculation", spy)
    monkeypatch.setattr(economics, "_to_result", lambda raw: raw)

    for given in ({}, {"assumptions": assumptions}, {"pumps": pumps},
                  {"assumptions": assumptions, "pumps": pumps}):
        seen.clear()
        economics.compute_npv(_one_month_of_rows(),
                              headers=economics.REQUIRED_COLUMNS,
                              start_date="2014-01-01",
                              norms_file=economics.NORMS_FILE, **given)
        assert seen.get("assumptions"), (
            f"при {sorted(given)} в эталон ушли пустые нормативы — он молча "
            "возьмёт свои умолчания вместо нормативов кейса"
        )
        assert seen.get("pumps"), (
            f"при {sorted(given)} в эталон ушли пустые типоразмеры ЭЦН"
        )


def _one_month_of_rows() -> list[dict[str, object]]:
    """Минимальная выгрузка: одна скважина, один месяц, все 14 колонок."""
    return [
        {"DATA": "2014-01-01", "well": "1", "WLPT": 0.0, "WLPR": 30.0,
          "WOMT": 0.0, "WOMR": 20.0, "WWIR": 0.0, "WWIT": 0.0,
          "THP": 20.0, "BHP": 90.0, "WEFF": 1.0,
          "WLPT_Diff": 900.0, "WOMT_Diff": 600.0, "WWIT_Diff": 0.0},
        {"DATA": "2014-02-01", "well": "1", "WLPT": 900.0, "WLPR": 30.0,
          "WOMT": 600.0, "WOMR": 20.0, "WWIR": 0.0, "WWIT": 0.0,
          "THP": 20.0, "BHP": 90.0, "WEFF": 1.0,
          "WLPT_Diff": 900.0, "WOMT_Diff": 600.0, "WWIT_Diff": 0.0},
    ]



def test_the_committed_trace_records_a_command_anyone_can_run():
    """Команда в машинном следе заявки обязана запускаться не только у автора записи.

    `runs/matched-0995/npv.json` отслеживается git и уходит в сдаваемый пакет:
    это доказательство того, что заявленное число получено из этой выгрузки
    эталонной точкой входа. До 05.09 поле `command` хранило `sys.executable` и
    абсолютные пути той машины, которая считала последней, — то есть команду,
    которую жюри повторить не может, а `claimedFrom` под Windows писался через
    обратную косую. Побочный эффект был виден в дне отправки: пятый пункт
    паспорта менял отслеживаемый файл, и первый пункт («`git status` чист»)
    после него переставал держаться (Р-227).

    Проверяется закоммиченный файл, а не свежий пересчёт: выгрузки на чистом
    клоне нет, а след есть — и портится он именно в git.
    """
    trace = json.loads(
        (SUBMITTED_RUN / "npv.json").read_text(encoding="utf-8"))
    command = trace["command"]

    assert command.startswith("python "), (
        "команда следа начинается не с переносимого «python»: " + command[:60]
    )
    assert "vendor/CHDD_PYTHON/" in command, (
        "команда следа не называет точку входа организаторов"
    )
    for marker in ("\\", ":/", ".venv", "/home/", "/Users/"):
        assert marker not in command, (
            f"в команде следа остался машинный путь ({marker!r}): {command[:120]}"
        )
    assert "\\" not in trace["claimedFrom"], (
        "источник заявленного числа записан через обратную косую: " + trace["claimedFrom"]
    )


def test_a_refusal_of_the_reference_calculator_carries_its_reason(tmp_path):
    """Отказ эталона обязан доехать до человека вместе с причиной.

    Вендорная точка входа печатает причину в **stdout** (`Ошибка: …`,
    `РАСЧЕТ_ЧДД.py`), а обе наши обвязки клали в исключение только `stderr`,
    который при этом пуст. Оператор видел «эталонный расчётчик отказал
    (код 1):» и пустоту — ни дебита свыше 500, ни дублей скважино-месяцев, ни
    битой шапки. Читают эту строку на валидационном кейсе, когда эталон
    отвергает выгрузку и разбираться надо за минуты.
    """
    sys.path.insert(0, str(economics.VENDOR_DIR.parent.parent / "scripts"))
    try:
        from verify_declared import recompute
    finally:
        sys.path.pop(0)

    rows = tmp_path / "rows.csv"
    rows.write_text(
        "DATA,well,WLPT,WLPR,WOMT,WOMR,WWIR,WWIT,THP,BHP,WEFF,"
        "WLPT_Diff,WOMT_Diff,WWIT_Diff\n"
        "2007-01-01,A,0,550,0,5,0,0,100,80,1,10,5,0\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as refusal:
        recompute(rows, 2007, economics.NORMS_FILE, tmp_path)

    assert "WLPR свыше 500" in str(refusal.value), (
        "причина отказа эталона потеряна — осталась только строка «отказал»: "
        + str(refusal.value)
    )


def test_the_declared_run_is_read_from_the_document_without_the_tail_directory():
    """`declared_run()` называет каталог прогона, а не каталог сдаваемого хвоста.

    Паспорт дня отправки велит прогнать `verify_declared` по обоим прогонам —
    заявленному и базовому. На базовом сверять нечего: у него своё число, и до
    09.09 скрипт отвечал на это «РАСХОЖДЕНИЕ … отправлять пакет нельзя» —
    то есть пугал провалом заявки там, где сверялся другой сценарий.

    С 08.09 документ называет файл в подкаталоге `submission/`, а прогоном
    остаётся каталог уровнем выше: `--run` принимает именно его.
    """
    sys.path.insert(0, str(economics.VENDOR_DIR.parent.parent / "scripts"))
    try:
        from verify_declared import declared_run
    finally:
        sys.path.pop(0)

    tail = "| сдаваемый файл | `runs/matched-0995/submission/wells_schedule.inc`, `md5 a7` |"
    flat = "| сдаваемый файл | `runs/matched-100/wells_schedule.inc`, `md5 a7` |"

    assert declared_run(tail) == "runs/matched-0995"
    assert declared_run(flat) == "runs/matched-100"
    assert declared_run("документ без заявленного файла") is None


def test_the_current_submission_names_the_run_that_verify_declared_defaults_to():
    """Умолчание `--run` и заявленный прогон — одно и то же.

    Иначе команда из README сверяет не тот прогон и объявляет расхождение
    заявки на ровном месте.
    """
    sys.path.insert(0, str(economics.VENDOR_DIR.parent.parent / "scripts"))
    try:
        from verify_declared import ROOT, build_parser, declared_run
    finally:
        sys.path.pop(0)

    document = (ROOT / "docs" / "submission.md").read_text(encoding="utf-8")
    default = build_parser().get_default("run")

    assert declared_run(document) == default.as_posix()


def test_paths_inside_the_repository_are_recorded_relative_to_it():
    """`portable()` укорачивает свои пути и не трогает чужие.

    Путь вне репозитория обязан остаться абсолютным: относительный увёл бы
    записанную команду к другому файлу молча.
    """
    sys.path.insert(0, str(economics.VENDOR_DIR.parent.parent / "scripts"))
    try:
        from verify_declared import ROOT, portable
    finally:
        sys.path.pop(0)

    assert portable(ROOT / "runs" / "matched-0995" / "rows.csv") == \
        "runs/matched-0995/rows.csv"
    assert portable(ROOT / "vendor" / "CHDD_PYTHON") == "vendor/CHDD_PYTHON"

    outside = ROOT.parent / "чужой-каталог" / "rows.csv"
    assert portable(outside) == str(outside), (
        "путь вне репозитория подменён относительным — команда возьмёт другой файл"
    )
