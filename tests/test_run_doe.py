"""Возобновляемая очередь расчётов DoE."""

from pathlib import Path

from scripts.run_doe import discover_cases, main


def package(tmp_path: Path) -> Path:
    root = tmp_path / "doe"
    root.mkdir()
    (root / "doe.json").write_text("{}", encoding="utf-8")
    for name in ("doe-002", "doe-001"):
        case = root / name
        case.mkdir()
        (case / "RUN.DATA").write_text("", encoding="utf-8")
    return root


def test_discovery_is_sorted_and_requires_the_manifest(tmp_path):
    root = package(tmp_path)

    assert [path.parent.name for path in discover_cases(root)] == ["doe-001", "doe-002"]


def finished(case: Path) -> None:
    """Досчитанный сценарий: результат на месте и стоит отметка о завершении."""
    (case / "RUN.UNSMRY").write_text("done", encoding="utf-8")
    (case / ".doe-done").write_text("2026-08-18T00:00:00+00:00", encoding="utf-8")


def test_dry_run_skips_a_finished_case(tmp_path, capsys):
    root = package(tmp_path)
    finished(root / "doe-001")

    status = main([str(root), "2", "--dry-run"])
    output = capsys.readouterr().out

    assert status == 0
    assert "готово: 1, в очереди: 1" in output
    assert "doe-002" in output
    assert "doe-001" not in output.splitlines()[-1]


def test_an_interrupted_case_is_recomputed_not_accepted(tmp_path, capsys):
    """Оборванный прогон оставляет `.UNSMRY`, но это не готовый сценарий.

    Настоящий случай 18.08: пакет прервали на середине, от сценария остался
    результат вдвое меньше нормального (723 КБ против 1,64 МБ). По наличию файла
    возобновление засчитало бы его за готовый, и в датасет суррогата уехал бы
    обрезок истории — молча, потому что формально файл есть и читается.
    """
    root = package(tmp_path)
    (root / "doe-001" / "RUN.UNSMRY").write_text("обрывок", encoding="utf-8")

    status = main([str(root), "2", "--dry-run"])
    output = capsys.readouterr().out

    assert status == 0
    assert "готово: 0, в очереди: 2" in output, "сценарий без отметки обязан пересчитаться"
    assert "doe-001" in output


def test_a_marker_without_a_result_does_not_count_either(tmp_path, capsys):
    """Обратный случай: отметка есть, а результата нет — тоже не готово."""
    root = package(tmp_path)
    (root / "doe-001" / ".doe-done").write_text("2026-08-18", encoding="utf-8")

    main([str(root), "2", "--dry-run"])

    assert "готово: 0, в очереди: 2" in capsys.readouterr().out


def test_missing_manifest_is_a_controlled_error(tmp_path, capsys):
    status = main([str(tmp_path), "--dry-run"])

    assert status == 1
    assert "сначала запустите build_doe.py" in capsys.readouterr().err
