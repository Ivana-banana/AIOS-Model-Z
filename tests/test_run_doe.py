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


def test_dry_run_skips_an_existing_result(tmp_path, capsys):
    root = package(tmp_path)
    (root / "doe-001" / "RUN.UNSMRY").write_text("done", encoding="utf-8")

    status = main([str(root), "2", "--dry-run"])
    output = capsys.readouterr().out

    assert status == 0
    assert "готово: 1, в очереди: 1" in output
    assert "doe-002" in output
    assert "doe-001" not in output.splitlines()[-1]


def test_missing_manifest_is_a_controlled_error(tmp_path, capsys):
    status = main([str(tmp_path), "--dry-run"])

    assert status == 1
    assert "сначала запустите build_doe.py" in capsys.readouterr().err
