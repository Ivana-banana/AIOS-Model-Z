"""Проверить девять обязательных частей пакета допуска AIOS.

    python scripts/check_package.py
    python scripts/check_package.py --schedule runs/inj-1.6/wells_schedule.inc \
        --model artifacts/surrogate.joblib

Команда ничего не собирает и не исправляет. Её задача — не дать таблице в
`docs/submission.md` превратиться в обещание: отсутствующий результат или
артефакт суррогата остаётся отказом до появления файла на диске.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECLARED_NPV = re.compile(r"\| \*\*ЧДД\*\* \| \*\*([^*]+ млн руб\.)\*\* \|")


@dataclass(frozen=True)
class Check:
    number: str
    title: str
    ready: bool
    detail: str


def audit(
    root: Path,
    schedule: Path,
    model: Path,
    *,
    model_validator: Callable[[Path], str | None] | None = None,
    schedule_validator: Callable[[Path], str | None] | None = None,
) -> list[Check]:
    """Состояние обязательного пакета; пути могут быть вне корня проекта."""
    source = _files(root / "aios", "*.py")
    scripts = _files(root / "scripts", "*.py")
    documentation = (
        root / "README.md",
        root / "docs" / "architecture.md",
        root / "docs" / "report.md",
        root / "docs" / "surrogate.md",
    )
    launch = (root / "README.md", root / "scripts" / "run_all.py")
    config = root / "config.toml"
    docker = (root / "docker" / "Dockerfile", root / "requirements.lock")
    submission = root / "docs" / "submission.md"
    declared = _declared_npv(submission)
    config_error = _config_problem(config)
    schedule_path = _resolve(root, schedule)
    model_path = _resolve(root, model)
    schedule_error = None
    scaling = None
    if schedule_path.is_file():
        schedule_error = (schedule_validator or _schedule_problem)(schedule_path)
        scaling_error, scaling = _scaling_problem(schedule_path)
        schedule_error = schedule_error or scaling_error
    model_error = None
    if model_path.is_file():
        model_error = (model_validator or _model_problem)(model_path)
    ui = (root / "aios" / "web" / "app.py", root / "aios" / "explain.py")
    templates = _files(root / "aios" / "web" / "templates", "*.html")
    licenses = (
        root / "THIRD_PARTY_NOTICES.md",
        root / "MODIFICATIONS.md",
        root / "PROVENANCE_AND_RIGHTS.md",
        root / "LICENSES" / "README.md",
    )

    return [
        Check("1", "исходный код", bool(source and scripts),
              f"aios: {len(source)}, scripts: {len(scripts)} Python-файлов"),
        Check("2", "документация", _all_files(documentation),
              _missing(documentation) or "назначение, архитектура, отчёт и суррогат"),
        Check("3", "инструкция запуска", _all_files(launch) and _readme_has_launch(launch[0]),
              _missing(launch) or "README + единый вход scripts/run_all.py"),
        Check("4", "конфигурация воспроизведения", config_error is None,
              config_error or "config.toml читается, seed задан"),
        Check("5", "материалы сборки Docker", _all_files(docker),
              _missing(docker) or "Dockerfile + lock с хэшами"),
        Check("6", "результат и заявленный ЧДД",
              schedule_path.is_file() and schedule_error is None and declared is not None,
              (
                  schedule_error if schedule_error is not None
                  else f"{schedule_path} + {declared}; {scaling}"
                  if schedule_path.is_file() and declared
                  else f"нет {schedule_path}" if not schedule_path.is_file()
                  else "в docs/submission.md не найдено заявленное ЧДД"
              )),
        Check("7", "git-репозиторий", (root / ".git").is_dir() and (root / "GIT.md").is_file(),
              ".git + GIT.md" if (root / ".git").is_dir() else "нет каталога .git"),
        Check("8", "веб-платформа и LLM", _all_files(ui) and len(templates) >= 5,
              _missing(ui) or f"FastAPI, explain.py, HTML-шаблонов: {len(templates)}"),
        Check("9", "суррогат и описание",
              model_path.is_file() and model_error is None
              and (root / "aios" / "neural.py").is_file()
              and (root / "docs" / "surrogate.md").is_file(),
              (
                  model_error if model_error is not None
                  else f"{model_path} + код + описание" if model_path.is_file()
                  else f"нет обученного артефакта {model_path}"
              )),
        Check("L", "лицензии и происхождение", _all_files(licenses),
              _missing(licenses) or "реестр, уведомления, модификации и тексты лицензий"),
    ]


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _files(directory: Path, pattern: str) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(path for path in directory.glob(pattern) if path.is_file())


def _all_files(paths: tuple[Path, ...]) -> bool:
    return all(path.is_file() for path in paths)


def _missing(paths: tuple[Path, ...]) -> str:
    names = [str(path) for path in paths if not path.is_file()]
    return "нет: " + ", ".join(names) if names else ""


def _readme_has_launch(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return "## Установка" in text and "## Сквозной прогон" in text


def _config_problem(path: Path) -> str | None:
    if not path.is_file():
        return f"нет {path}"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return f"config.toml не читается: {error}"
    if not _contains_seed(raw):
        return "в config.toml не задан seed"
    return None


def _contains_seed(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("seed"), int):
        return True
    return any(_contains_seed(child) for child in value.values())


def _declared_npv(path: Path) -> str | None:
    if not path.is_file():
        return None
    match = DECLARED_NPV.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _model_problem(path: Path) -> str | None:
    """Проверить, что файл является суррогатом, который можно пустить в цикл."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import joblib

        from aios.neural import NeuralBundle, assert_quality

        bundle = joblib.load(path)
        if not isinstance(bundle, NeuralBundle):
            return f"артефакт не NeuralBundle: {type(bundle).__name__}"
        assert_quality(bundle.report)
    except Exception as error:  # битый pickle тоже должен стать отказом допуска
        return f"модель не прошла допуск: {error}"
    return None


def _scaling_problem(path: Path) -> tuple[str | None, str]:
    """Заявленный множитель обязан описывать сдаваемый файл, а не намерение.

    `optimise.scale` умножает поле расхода независимо от режима. Под `LRAT` и
    `RATE` это уставка, и «×1.5» правда; под `BHP`, `THP`, `GRUP`, `ORAT`,
    `WRAT`, `RESV` то же поле — верхний предел, и умножение ослабляет
    ограничение, ничего не форсируя. На Модели Z все записи уставочные, на чужом
    деке валидационного кейса (ТЗ п. 4.3) — не обязаны быть.

    Поэтому доля буквально применённых записей считается при сборке кейса
    (`scripts/build_scaled_case.py` → `scaling.json`) и проверяется здесь
    машиной. Отчёт есть и доля меньше единицы — отказ: «×1.5» в отчёте при
    40 % применённых записей недопустимо. Отчёта нет — не отказ (кейс мог быть
    собран до 23.08), но и не молчание: это сказано в строке проверки.
    """
    report = path.parent / "scaling.json"
    if not report.is_file():
        return None, "доля буквально применённых записей не измерена (нет scaling.json)"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        share = float(data["share"])
        records, literal = int(data["records"]), int(data["literal"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"{report.name} не читается: {error}", "отчёт масштабирования повреждён"

    if records == 0:
        return None, "масштабирование не применялось (множители единичные)"
    if share < 1.0 - 1e-9:
        return (
            f"заявленный множитель применён буквально к {literal} записям из "
            f"{records} ({share:.1%}): ярлык не описывает сдаваемый файл",
            "",
        )
    return None, f"множитель применён буквально ко всем {records} записям"


def _schedule_problem(path: Path) -> str | None:
    """Разобрать сдаваемое расписание и проверить физические ограничения."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from aios.deck import parse_schedule
        from aios.schedule import assert_valid

        parsed = parse_schedule(path)
        if not parsed.steps:
            return "расписание не содержит ни одного шага DATES"
        if not any(step.producers or step.injectors for step in parsed.steps):
            return "расписание не содержит режимов скважин"
        if parsed.unsupported_controls or parsed.preamble_controls:
            names = sorted({*parsed.unsupported_controls, *parsed.preamble_controls})
            return "неподдержанные или недатированные режимы: " + ", ".join(names)
        assert_valid(parsed.steps)
    except Exception as error:
        return f"расписание не прошло проверку: {error}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--schedule", type=Path,
                        default=Path("runs/inj-1.6/wells_schedule.inc"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/surrogate.joblib"))
    arguments = parser.parse_args(argv)

    checks = audit(arguments.root, arguments.schedule, arguments.model)
    for item in checks:
        status = "ГОТОВО" if item.ready else "НЕТ"
        print(f"[{status:6}] {item.number}. {item.title}: {item.detail}")
    missing = [item for item in checks if not item.ready]
    print()
    print("пакет готов" if not missing else f"пакет неполон: {len(missing)} пункт(а)")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
