"""Сборка `requirements.lock`: без хэшей файл бесполезен, и это должно быть отказом."""

from __future__ import annotations

import pytest

from scripts.make_lock import entries, render


def report(*packages) -> dict:
    return {
        "install": [
            {
                "metadata": {"name": name, "version": version},
                "download_info": {"archive_info": {"hashes": hashes}},
            }
            for name, version, hashes in packages
        ]
    }


def test_packages_are_sorted_so_the_file_diffs_cleanly():
    """Порядок resolver'а не воспроизводим, порядок файла обязан быть."""
    found = entries(report(
        ("zope", "1.0", {"sha256": "b" * 64}),
        ("Alpha", "2.0", {"sha256": "a" * 64}),
    ))

    assert [name for name, _v, _h in found] == ["Alpha", "zope"]


def test_every_hash_of_a_package_lands_in_the_file():
    """У пакета бывает несколько колёс; потерять хэш значит сломать установку."""
    text = render(report(("numpy", "2.5.2", {"sha256": "a" * 64, "sha512": "b" * 128})), "r.json")

    assert "numpy==2.5.2 \\" in text
    assert f"--hash=sha256:{'a' * 64} \\" in text
    assert f"--hash=sha512:{'b' * 128}" in text


def test_a_package_without_hashes_is_refused_instead_of_written_out():
    """Молча пропустить пакет — выдать за гарантию то, чего нет."""
    with pytest.raises(ValueError, match="нет хэшей"):
        render(report(("resdata", "6.3.3", {})), "r.json")


def test_the_header_names_the_platform_because_wheels_are_platform_specific():
    text = render(report(("numpy", "2.5.2", {"sha256": "a" * 64})), "r.json")

    assert "платформа:" in text
    assert "python:" in text
    assert "описывает ЭТУ платформу" in text
    assert "--require-hashes" in text


def test_the_real_lock_is_installable_in_form(tmp_path):
    """Каждая строка файла — либо комментарий, либо пин, либо хэш. Ничего иного."""
    from pathlib import Path

    lock = Path(__file__).resolve().parent.parent / "requirements.lock"
    if not lock.exists():
        pytest.skip("lock не собран на этой машине")

    pinned = 0
    for line in lock.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().rstrip("\\").strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--hash="):
            continue
        assert "==" in stripped, f"незапиненная строка: {line!r}"
        pinned += 1
    assert pinned > 0
