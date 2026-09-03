"""Сквозной прогон одной командой: кейс → ГГДМ → решения → ГГДМ → ЧДД.

Требование ТЗ п. 3.2.3: решение «не требует ручного вмешательства в процесс
расчета после запуска». До 17.08 сквозной прогон был тремя командами с запуском
симулятора между ними — проверяющему пришлось бы вести его за руку.

    python scripts/run_all.py --flow-args=--parsing-strictness=low

Пять шагов, каждый — отдельный процесс, чтобы команда была видна в логе и
воспроизводилась отдельно:

1. собрать базовый кейс из дека         `scripts/build_case.py`
2. посчитать его на ГГДМ                `scripts/run_flow.sh`
3. пройти цикл и собрать расписание     `scripts/run_cycle.py --out`

Шаг 3 идёт на тривиальном бейзлайне, если не передан `--surrogate-model`:
обученный суррогат подставляется тем же ключом, что и в отдельной команде, и
проверяется **до** первого расчёта — иначе отказ порога качества приходит после
одиннадцати минут счёта базового прогона.
4. посчитать получившийся сценарий      `scripts/run_flow.sh`
5. заменить прогноз полным расчётом     `scripts/run_cycle.py --confirm`

Флаги симулятора не зашиты в скрипт намеренно: `--parsing-strictness=low`
меняет физику расчёта (дек содержит неподдерживаемый `PINCHREG`) и обязан быть
виден в команде, которой прогон воспроизводится. Это же требование п. 3.5 —
скрытых зависимостей быть не должно.

Заявляемым числом считается только то, что напечатал шаг 5: до него ЧДД — это
прогноз суррогата, а не результат.

Где запускать: шаги 2 и 4 вызывают `flow`, поэтому команда выполняется там, где
установлен симулятор — в Docker-образе проекта или на Linux (`docs/setup-simulator.md`).
Шаги 1, 3 и 5 — чистый Python и идут где угодно; `--dry-run` показывает план на
любой машине, включая Windows без симулятора.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402

#: Как выполнять шаг. Подменяется в тестах, чтобы проверить состав команд без ГГДМ.
Runner = Callable[[Sequence[str]], int]


def plan(base_dir: Path,
         policy_dir: Path,
         *,
         case_name: str,
         processes: int,
         flow_args: Sequence[str],
         config_path: str,
         skip_base: bool,
         surrogate_model: str | None = None,
         search_seed: int | None = None) -> list[tuple[str, list[str]]]:
    """Список шагов «название → команда». Отдельной функцией — чтобы проверять.

    `--skip-base` нужен не для удобства: базовый прогон стоит 11 минут и не
    зависит от наших решений, поэтому при переборе сценариев его считают один раз.
    """
    python = sys.executable
    flow = str(ROOT / "scripts" / "run_flow.sh")
    steps: list[tuple[str, list[str]]] = []

    if not skip_base:
        steps.append((
            "сборка базового кейса",
            [python, str(ROOT / "scripts" / "build_case.py"), str(base_dir),
             "--config", config_path],
        ))
        steps.append((
            "расчёт базового сценария на ГГДМ",
            ["bash", flow, str(base_dir / f"{case_name}.DATA"), str(processes), *flow_args],
        ))

    cycle = [python, str(ROOT / "scripts" / "run_cycle.py"),
             "--from", str(base_dir), "--out", str(policy_dir), "--config", config_path]
    if surrogate_model:
        cycle += ["--surrogate-model", surrogate_model]
    if search_seed is not None:
        cycle += ["--search-seed", str(search_seed)]
    steps.append(("цикл управления: состояние → решения → расписание", cycle))
    steps.append((
        "расчёт получившегося сценария на ГГДМ",
        ["bash", flow, str(policy_dir / f"{case_name}.DATA"), str(processes), *flow_args],
    ))
    # модель передаётся и сюда: шаг печатает «прогноз суррогата обещал» и ошибку
    # прогноза, а считает их, заново пройдя цикл. Без ключа он прошёл бы его на
    # тривиальном бейзлайне — и ошибка прогноза сравнивала бы полный расчёт с
    # обещанием другой модели. Найдено прогоном 20.08: шаг 3 шёл на линейном
    # суррогате, шаг 5 печатал обещание бейзлайна.
    confirm = [python, str(ROOT / "scripts" / "run_cycle.py"),
               "--confirm", str(policy_dir), "--from", str(base_dir), "--config", config_path]
    if surrogate_model:
        confirm += ["--surrogate-model", surrogate_model]
    if search_seed is not None:
        confirm += ["--search-seed", str(search_seed)]
    steps.append(("подтверждение прироста полным расчётом", confirm))
    return steps


def execute(steps: Sequence[tuple[str, list[str]]], runner: Runner) -> int:
    """Выполнить шаги по порядку. Первый же отказ останавливает прогон.

    Продолжать после неудачи нельзя: шаг 4 на непосчитанном шаге 3 даст не
    ошибку, а тихо неверное число — ровно тот класс дефектов, который проект уже
    ловил (пустая выгрузка, потерянные вскрытия).
    """
    total = len(steps)
    for number, (title, command) in enumerate(steps, start=1):
        print(f"\n=== [{number}/{total}] {title}", flush=True)
        print("    " + " ".join(command), flush=True)
        code = runner(command)
        if code != 0:
            print(f"\nшаг {number} ({title}) завершился с кодом {code}; прогон остановлен",
                  file=sys.stderr)
            return code
    return 0


def refuse_surrogate(path: Path) -> str | None:
    """Почему этим артефактом вести цикл нельзя; `None` — можно.

    Проверяется **до** симулятора намеренно. Шаг 3 читает модель, но идёт после
    двух шагов расчёта: базовый прогон Модели Z стоит 11 минут, и отказ «модель
    не проходит порог качества» после него — это потерянный прогон вместо
    мгновенного ответа. Проверка та же, что и в цикле (`aios.model.for_cycle`
    вызывает `NeuralSurrogate.create` с порогом), поэтому расхождения между
    ранним отказом и поздним быть не может.
    """
    if not path.is_file():
        return f"модель не найдена: {path}"

    import joblib

    from aios.neural import NeuralBundle, assert_quality

    try:
        bundle = joblib.load(path)
    except Exception as error:
        return f"артефакт не читается: {error}"
    if not isinstance(bundle, NeuralBundle):
        return f"это не обученный суррогат: {type(bundle).__name__}"
    try:
        assert_quality(bundle.report)
    except ValueError as error:
        return f"артефакт не годится вести цикл: {error}"
    return None


def _run(command: Sequence[str]) -> int:
    return subprocess.run(list(command), cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=None,
                        help="каталог базового прогона (по умолчанию из конфига)")
    parser.add_argument("--out", default="runs/policy",
                        help="куда собрать сценарий с решениями цикла")
    parser.add_argument("--processes", type=int, default=None,
                        help="процессов MPI (по умолчанию из конфига)")
    parser.add_argument("--flow-args", default="",
                        help="флаги симулятора через пробел, например "
                             "--flow-args=--parsing-strictness=low")
    parser.add_argument("--skip-base", action="store_true",
                        help="базовый прогон уже посчитан — начать с цикла")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать команды и выйти, ничего не запуская")
    parser.add_argument("--surrogate-model", default=None,
                        help="обученный суррогат для шага 3; без него цикл идёт "
                        "на тривиальном бейзлайне")
    parser.add_argument("--search-seed", type=int, default=None,
                        help="seed порядка координат поиска")
    parser.add_argument("--config", default=str(config_module.DEFAULT_PATH))
    arguments = parser.parse_args(argv)

    if arguments.surrogate_model:
        refusal = refuse_surrogate(Path(arguments.surrogate_model))
        if refusal:
            print(refusal, file=sys.stderr)
            return 1

    config = config_module.load(arguments.config)
    base_dir = Path(arguments.base) if arguments.base else config.case.run_dir
    steps = plan(
        base_dir,
        Path(arguments.out),
        case_name=config.case.case_name,
        processes=arguments.processes or config.run.processes,
        flow_args=arguments.flow_args.split() if arguments.flow_args else (),
        config_path=arguments.config,
        skip_base=arguments.skip_base,
        surrogate_model=arguments.surrogate_model,
        search_seed=arguments.search_seed,
    )

    if arguments.dry_run:
        for number, (title, command) in enumerate(steps, start=1):
            print(f"[{number}/{len(steps)}] {title}\n    " + " ".join(command))
        return 0

    code = execute(steps, _run)
    if code == 0:
        print(f"\nготово. Расписание: {Path(arguments.out) / 'wells_schedule.inc'}")
        print("Число, напечатанное последним шагом, посчитано эталонным "
              "расчётчиком по полному расчёту, а не суррогатом.")
        print()
        print("Но заявляемым оно от этого не становится. Эта команда показывает, "
              "что контур проходит целиком; заявка — отдельный, выбранный и "
              "проверенный сценарий, и воспроизводится она командой")
        print("    python scripts/verify_declared.py --run runs/matched-0995")
        print("(docs/submission.md, docs/submission-day.md).")
    return code


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
