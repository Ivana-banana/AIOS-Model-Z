"""Проверить девять обязательных частей пакета допуска AIOS.

    python scripts/check_package.py
    python scripts/check_package.py --schedule runs/matched-0995/wells_schedule.inc \
        --model artifacts/surrogate.joblib

Команда ничего не собирает и не исправляет. Её задача — не дать таблице в
`docs/submission.md` превратиться в обещание: отсутствующий результат или
артефакт суррогата остаётся отказом до появления файла на диске.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECLARED_NPV = re.compile(r"\| \*\*ЧДД\*\* \| \*\*([^*]+ млн руб\.)\*\* \|")
#: Сдаваемый файл, как он назван в самой заявке. Число и файл обязаны быть из
#: одного прогона — пара «файл от одного расчёта, число от другого» и есть тот
#: механизм, которым расхождение с пересчётом организаторов возникает молча.
DECLARED_SCHEDULE = re.compile(r"`(runs/[\w.-]+/wells_schedule\.inc)`")
DECLARED_MD5 = re.compile(r"`md5 ([0-9a-fA-F]{32})`")
DECLARED_MODEL = re.compile(r"`(artifacts/[\w.-]+\.joblib)`")
DECLARED_MODEL_SHA256 = re.compile(
    r"`artifacts/[\w.-]+\.joblib`[^\n]*`sha256 ([0-9a-fA-F]{64})`"
)


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
        checksum_error, checksum = _checksum_problem(submission, schedule_path)
        schedule_error = (
            schedule_error
            or scaling_error
            or _wrong_run(submission, schedule_path)
            or checksum_error
            or _trace_problem(schedule_path, declared)
        )
        scaling = f"{scaling}; {checksum}"
    model_error = None
    model_checksum = "sha256 модели в заявке не указан"
    if model_path.is_file():
        model_error = (model_validator or _model_problem)(model_path) or _wrong_model(
            submission, model_path
        )
        if model_error is None:
            model_error, model_checksum = _model_checksum_problem(submission, model_path)
    ui = (root / "aios" / "web" / "app.py", root / "aios" / "explain.py")
    templates = _files(root / "aios" / "web" / "templates", "*.html")
    licenses = (
        # сама лицензия нашего кода — первым файлом: без неё пакет уезжает без
        # условий, на которых его можно использовать, и заметить это некому
        # (Р-107 объявил MIT 24.08, шлюз до этого проверял только чужие)
        root / "LICENSE",
        root / "THIRD_PARTY_NOTICES.md",
        root / "MODIFICATIONS.md",
        root / "PROVENANCE_AND_RIGHTS.md",
        root / "LICENSES" / "README.md",
    )
    licence_error = _missing(licenses) or _docker_licence_problem(docker[0])

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
                  else f"{model_path} + код + описание; {model_checksum}" if model_path.is_file()
                  else f"нет обученного артефакта {model_path}"
              )),
        Check("L", "лицензии и происхождение", licence_error == "",
              licence_error
              or "LICENSE и сторонние лицензии входят в репозиторий и образ"),
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
    """Есть ли в README установка, запуск и единая команда прогона.

    Ищется смысл, а не буквальные заголовки: 02.09 README переоформили под защиту
    (разделы стали нумерованными), и проверка объявила пакет неполным при
    полностью сохранившейся инструкции. Сторож, срабатывающий на смену заголовка,
    отвлекает ровно тогда, когда его читают внимательнее всего — в день отправки.
    """
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8").lower()
    has_install = "установка" in text
    has_launch = "запуск" in text or "сквозной прогон" in text
    return has_install and has_launch and "run_all.py" in text


def _docker_licence_problem(path: Path) -> str:
    """Лицензии должны доехать до получателя образа вместе с кодом."""
    if not path.is_file():
        return f"нет {path}"
    copies = "\n".join(re.findall(r"^COPY .*", path.read_text(encoding="utf-8"), re.M))
    required = ("COPY LICENSE ", "COPY LICENSES ", "THIRD_PARTY_NOTICES.md",
                "MODIFICATIONS.md")
    missing = [item.strip() for item in required if item not in copies]
    if missing:
        return "Docker-образ не получает: " + ", ".join(missing)
    if "PROVENANCE_AND_RIGHTS.md" in copies:
        return "внутренний PROVENANCE_AND_RIGHTS.md не должен входить в образ"
    return ""


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


def _trace_problem(schedule: Path, declared: str | None) -> str | None:
    """Есть ли машинный след того, что заявленное число посчитано из этой выгрузки?

    До 03.09 этот пункт проверял `md5` файла и наличие числа в
    `docs/submission.md`, но не связь между ними: число вписано в документ
    руками, а в каталоге прогона не было ничего, что подтверждало бы расчёт.
    След пишет `scripts/verify_declared.py`, запуская **точку входа
    организаторов** на `rows.csv` прогона; здесь он только читается — иначе
    проверка допуска считала бы ЧДД полминуты (Р-172).

    Отсутствие следа — отказ, а не заметка: «число не проверено машиной» и
    «число проверено» на дне отправки различаются ровно на дисквалификацию.
    """
    trace = schedule.parent / "npv.json"
    if not trace.is_file():
        return (f"нет машинного следа {trace}: запустите "
                f"python scripts/verify_declared.py --run {schedule.parent.name}")
    try:
        record = json.loads(trace.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"след {trace} не читается: {error}"

    if not record.get("agreed"):
        return (f"след говорит о расхождении: пересчёт {record.get('chddM')} против "
                f"заявленного {record.get('claimedM')}")
    if record.get("scheduleMd5") != _md5(schedule):
        return (f"след посчитан по другому файлу расписания: в нём md5 "
                f"{record.get('scheduleMd5')}, у проверяемого {_md5(schedule)}")
    # версия вендора читается лениво: проверка допуска должна работать и там,
    # где зависимости расчётчика не установлены (та же причина, что у `_model_problem`)
    expected_version = None
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from aios import economics

        expected_version = economics.VENDOR_VERSION
    except Exception:  # noqa: BLE001 — отсутствие зависимостей не делает след негодным
        expected_version = None
    if expected_version and record.get("vendorVersion") != expected_version:
        return (f"след посчитан другой версией расчётчика: {record.get('vendorVersion')} "
                f"против {expected_version}")
    # Стартовый год, нормативы и выгрузка — три величины, которые меняют число
    # молча. Прежде след их **писал**, но никто не сверял: пересимулированный
    # `rows.csv` или подменённые нормативы пункт проходили (С9).
    problem = _trace_inputs_problem(schedule, record)
    if problem is not None:
        return problem

    if declared is not None:
        digits = declared.split("млн")[0].replace("\u00a0", "").replace(" ", "").replace(",", ".")
        try:
            if abs(float(digits) - float(record["claimedM"])) > 0.05:
                return (f"в заявке {declared}, а след сверялся с "
                        f"{record['claimedM']} млн руб.")
        except (ValueError, KeyError, TypeError):
            return f"в следе нет сверяемого числа: {record.get('claimedM')!r}"
    return None


def _trace_inputs_problem(schedule: Path, record: dict) -> str | None:
    """Сверить входы расчёта: стартовый год, нормативы и выгрузку.

    Что сверяется и почему каждое:

    * **стартовый год** — одно расписание при базе 1991 и 2007 даёт числа,
      различающиеся в 4,6 раза (Р-158, Р-159). След, посчитанный при прежней
      базе, совпадёт с заявкой прежней базы и промолчит;
    * **нормативы** — цена нефти, ставки и таблица ЭЦН задают экономику целиком;
      подменённый файл меняет число, не трогая ни расписание, ни выгрузку;
    * **выгрузка** — пересимулированный `rows.csv` даёт другое число из того же
      расписания.

    Отсутствие `rows.csv` рядом со следом — **не** расхождение: выгрузка
    намеренно не отслеживается git (5 МБ на прогон), и у проверяющего её нет.
    Тогда сверять нечем, и молчание не выдаётся за совпадение.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from aios import config as config_module
    except Exception:  # noqa: BLE001 — без зависимостей сверяем то, что можем
        return None

    try:
        configuration = config_module.load()
    except Exception:  # noqa: BLE001 — конфигурацию проверяет отдельный пункт
        return None

    expected_year = configuration.economics.start_date.year
    written_year = record.get("startYear")
    if written_year is not None and int(written_year) != expected_year:
        return (f"след посчитан от {written_year} года, а конфигурация называет "
                f"{expected_year}: одно расписание в двух системах отсчёта даёт "
                f"числа, различающиеся в разы")

    norms = Path(configuration.economics.norms_file)
    written_norms = record.get("normsSha256")
    if written_norms is not None and norms.is_file():
        actual = _sha256(norms)
        if actual != written_norms:
            return (f"след посчитан по другим нормативам: в нём sha256 "
                    f"{written_norms[:12]}, у {norms.name} — {actual[:12]}")

    rows = schedule.parent / "rows.csv"
    written_rows = record.get("rowsMd5")
    if written_rows is not None and rows.is_file():
        actual = _md5(rows)
        if actual != written_rows:
            return (f"след посчитан по другой выгрузке: в нём md5 {written_rows}, "
                    f"у {rows.name} — {actual}")
    return None


def _sha256(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            accumulator.update(block)
    return accumulator.hexdigest()


def _md5(path: Path) -> str:
    accumulator = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            accumulator.update(block)
    return accumulator.hexdigest()


def _wrong_run(submission: Path, schedule: Path) -> str | None:
    """Проверяем ли мы тот файл, который заявлен?

    Проверка печатает заявленное ЧДД рядом с любым расписанием, которое ей
    подсунули: 24.08 она отвечала «ГОТОВО … + 13 147,5 млн руб.» на историческом
    базовом расписании — то есть благословляла пару «файл одного прогона, число
    другого». Именно так расхождение с пересчётом организаторов и возникает
    молча, и `docs/submission.md` про это говорит первым абзацем.

    Сравнение идёт с именем файла из самой заявки, а не с литералом здесь:
    сменится сценарий — проверка переедет вместе с ним. Если заявка файла не
    называет, сравнивать не с чем и проверка молчит.
    """
    if not submission.is_file():
        return None
    match = DECLARED_SCHEDULE.search(submission.read_text(encoding="utf-8"))
    if match is None:
        return None
    declared = match.group(1)
    if schedule.as_posix().endswith(declared):
        return None
    return (
        f"проверяется {schedule.as_posix()}, а заявлен {declared}: число и файл "
        "обязаны быть из одного прогона"
    )


def _wrong_model(submission: Path, model: Path) -> str | None:
    """Не принять другую качественную модель вместо артефакта из пакета."""
    if not submission.is_file():
        return None
    match = DECLARED_MODEL.search(submission.read_text(encoding="utf-8"))
    if match is None:
        return None
    declared = match.group(1)
    if model.as_posix().endswith(declared):
        return None
    return f"проверяется {model.as_posix()}, а в пакет входит {declared}"


def _model_checksum_problem(submission: Path, model: Path) -> tuple[str | None, str]:
    """Сверить байты суррогата с артефактом, прошедшим пакетный допуск."""
    if not submission.is_file():
        return None, "sha256 модели в заявке не указан"
    match = DECLARED_MODEL_SHA256.search(submission.read_text(encoding="utf-8"))
    if match is None:
        return None, "sha256 модели в заявке не указан"
    expected = match.group(1).lower()
    digest = hashlib.sha256()
    try:
        with model.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        return f"sha256 модели не посчитан: {error}", "sha256 модели не посчитан"
    actual = digest.hexdigest()
    if actual != expected:
        return (
            f"sha256 модели {actual}, а в заявке {expected}: другой артефакт",
            f"sha256 {actual}",
        )
    return None, f"sha256 {actual} совпал с заявкой"


def _checksum_problem(submission: Path, schedule: Path) -> tuple[str | None, str]:
    """Сверить байты файла с отпечатком, рядом с которым заявлен ЧДД."""
    if not submission.is_file():
        return None, "md5 в заявке не указан"
    match = DECLARED_MD5.search(submission.read_text(encoding="utf-8"))
    if match is None:
        return None, "md5 в заявке не указан"

    expected = match.group(1).lower()
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with schedule.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        return f"md5 не посчитан: {error}", "md5 не посчитан"
    actual = digest.hexdigest()
    if actual != expected:
        return (
            f"md5 файла {actual}, а в заявке {expected}: проверяется другой файл",
            f"md5 {actual}",
        )
    return None, f"md5 {actual} совпал с заявкой"


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
        from aios import config as config_module
        from aios import economics
        from aios.deck import parse_schedule
        from aios.schedule import assert_valid, case_limits

        parsed = parse_schedule(path)
        if not parsed.steps:
            return "расписание не содержит ни одного шага DATES"
        if not any(step.producers or step.injectors for step in parsed.steps):
            return "расписание не содержит режимов скважин"
        if parsed.unsupported_controls or parsed.preamble_controls:
            names = sorted({*parsed.unsupported_controls, *parsed.preamble_controls})
            return "неподдержанные или недатированные режимы: " + ", ".join(names)
        # Оба предела берутся из кейса, а не из умолчаний Модели Z: иначе допуск
        # и предполётная проверка держат разные числа, и файл, прошедший один
        # шлюз, отвергается другим (2.3 #8, С12). Нормативы — те, что названы в
        # конфигурации: на валидационном кейсе они приходят от организаторов.
        configuration = config_module.load()
        assert_valid(parsed.steps, **case_limits(
            economics.load_norms(configuration.economics.norms_file)[1],
            configuration.limits.max_injector_bhp_atm,
        ))
    except Exception as error:
        return f"расписание не прошло проверку: {error}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--schedule", type=Path,
                        default=Path("runs/matched-0995/wells_schedule.inc"))
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
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
