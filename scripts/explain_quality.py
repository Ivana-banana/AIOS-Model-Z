"""Замер качества объяснений: ASR, DTC, NTI одной командой.

    python scripts/explain_quality.py --rows runs/base/rows.csv \
        --schedule runs/base/wells_schedule.inc

Что считается и зачем — `aios/explain_quality.py` и `docs/explainability.md`.
Коротко: слой интерпретируемости обязателен по регламенту, а качество его до
22.08 не измерялось ничем, кроме сверки чисел. Здесь три метрики протокола 4-9,
плюс негативный набор, на котором видно, что именно ловит числовая проверка.

Две части замера независимы:

* **корпус** — настоящие пакеты фактов, которые породил цикл на переданной
  выгрузке. Нет выгрузки — нет корпуса, и скрипт об этом говорит, а не
  подставляет выдуманные пакеты;
* **негативный набор** — подмены в тексте объяснения (чужое число, чужой
  показатель, чужая единица, чужой период, округление, потерянный разделитель,
  собственный пересчёт). Он не требует данных вовсе и считается всегда.

Объяснения здесь **шаблонные**: живого вызова языковой модели в проекте ещё не
было (`BLOCKED.md`, п. 6). Метрики на шаблоне меряют не модель, а нижнюю
границу — сам конвейер «факты → текст → проверка». Это прямо написано в выводе.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import explain as explain_module  # noqa: E402
from aios import explain_quality as quality  # noqa: E402
from aios import loop, margin  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.model import for_cycle  # noqa: E402
from aios.web.app import read_rows_csv  # noqa: E402

#: Страты выборки из протокола 4-9. Пустая страта показывается нулём, а не
#: заполняется придуманным примером — разбор требует именно этого.
STRATA = (
    "значимое изменение режима или статуса",
    "решение с активным общим ограничением",
    "пограничный или конфликтный случай",
    "без действия, отказ или неполный пакет фактов",
)


def stratum_of(evidence: explain_module.Evidence) -> str:
    """Отнести пакет фактов к страте по тому, что в нём есть."""
    if any("нарушен" in caveat or "срезано" in caveat for caveat in evidence.caveats):
        return STRATA[1]
    if evidence.numbers.get("срезано планировщиком, скважин"):
        return STRATA[1]
    decisions = (evidence.numbers.get("решений об остановке", 0.0)
                 + evidence.numbers.get("скважин с подобранным режимом", 0.0))
    if not decisions:
        return STRATA[3]
    if evidence.numbers.get("предупреждений суррогата"):
        return STRATA[2]
    return STRATA[0]


def corpus(rows_path: Path, schedule_path: Path, config) -> list[tuple[str, explain_module.Evidence]]:
    """Пакеты фактов, которые породил цикл на этой выгрузке.

    Вариантов запуска несколько, и это не размножение одного и того же: подбор
    режимов включён и выключен — это два разных решения с разными фактами, и
    единицей анализа остаётся уникальный пакет фактов, а не число обращений.
    """
    # чтение выгрузки берётся у веб-платформы: там числовые колонки уже
    # приводятся к float, и второй разбор того же файла разошёлся бы с первым
    history = read_rows_csv(rows_path)
    schedule = parse_schedule(schedule_path)
    base_steps = [step for step in schedule.steps if step.date >= config.case.control_start]
    if not base_steps:
        base_steps = list(schedule.steps)
    norms = margin.Norms.from_file()
    selected = for_cycle(neural_path=None, history=history,
                         reference_steps=base_steps, fluids=config.fluids)

    packets: list[tuple[str, explain_module.Evidence]] = []
    for tune_rates in (True, False):
        result = loop.run_cycle(
            selected.model, base_steps, norms,
            control_start=config.case.control_start,
            start_date=config.economics.start_date,
            lookback_months=config.policy.lookback_months,
            known_wells=frozenset(schedule.wells),
            tune_rates=tune_rates,
            joint_rate_search=selected.joint_rate_search,
            limits=config.limits,
            fluids=config.fluids,
        )
        packets.append((selected.name, explain_module.evidence_from_cycle(result, selected.name)))
    return packets


def negatives(evidence: explain_module.Evidence, text: str) -> dict[str, str]:
    """Подмены, каждая из которых обязана быть отвергнута.

    Список взят из разбора: подмена объекта, показателя, единицы, периода, знака,
    округление, десятичный разделитель и число, вычисленное моделью. Значения
    берутся из самого пакета фактов, поэтому набор не выдуман и не устареет.
    """
    money = next((name for name in evidence.numbers if "прирост" in name.lower()), None)
    if money is None:
        return {}
    value = evidence.numbers[money]
    shown = explain_module._format(value)
    return {
        "чужое число": f"{text}\n\nПрирост составил {explain_module._format(value + 357.5)} млн руб.",
        "не тот показатель": f"{text}\n\nОстановлено {shown} скважин.",
        "подмена единицы": f"{text}\n\nПрирост составил {shown} тыс. т.",
        "смена знака": f"{text}\n\nПрирост составил -{shown} млн руб.",
        "округление": f"{text}\n\nПрирост составил {explain_module._format(round(value))} млн руб.",
        "потерян разделитель": f"{text}\n\nПрирост составил {shown.replace('.', '')} млн руб.",
        "чужой период": f"{text}\n\nПрирост {shown} млн руб. получен в 1998 году.",
        "свой пересчёт": f"{text}\n\nЭто рост на 5.4 %.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, default=None,
                        help="выгрузка в контракте расчётчика (rows.csv) — источник корпуса")
    parser.add_argument("--schedule", type=Path, default=None,
                        help="расписание того же прогона (wells_schedule.inc)")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument(
        "--live", action="store_true",
        help="мерить на ответах языковой модели, а не на шаблонах; тратит по "
             "одному вызову на пакет фактов и требует ключа в окружении",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="сколько пакетов брать при --live; 0 — все")
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    rows_path = arguments.rows or (config.case.run_dir / "rows.csv")
    schedule_path = arguments.schedule or (config.case.run_dir / "wells_schedule.inc")

    packets: list[tuple[str, explain_module.Evidence]] = []
    if rows_path.is_file() and schedule_path.is_file():
        packets = corpus(rows_path, schedule_path, config)
    else:
        print(f"корпуса нет: не найден {rows_path} или {schedule_path}.")
        print("Метрики корпуса не считаются — выдумывать пакеты фактов нельзя.")
        print("Негативный набор ниже данных не требует и считается всегда.\n")

    if arguments.live and arguments.limit > 0:
        packets = packets[: arguments.limit]

    texts, fallbacks = _explanations(packets, config, live=arguments.live)
    if arguments.live:
        # печатается ДО корпуса намеренно: если до модели не дошёл ни один пакет,
        # ниже случится ранний выход, и причина — самое важное в таком прогоне —
        # осталась бы ненапечатанной. Ровно так и вышло в первом живом замере.
        print(f"=== Живой вызов ({config.explain.provider}, {config.explain.model}) ===")
        print(f"ответов модели принято: {len(texts)}, не дошло: {len(fallbacks)}")
        for name, why in fallbacks:
            print(f"  • {name}: {why}")
        print("Не дошедшие пакеты в метрики НЕ входят: мерить шаблон вперемешку "
              "с ответами модели значит мерить смесь.")
        print()
    report = quality.Report(tuple(
        quality.measure(text, evidence,
                        source="модель" if arguments.live else "шаблон",
                        stratum=stratum_of(evidence))
        for (_, evidence), text in zip(packets, texts, strict=True)
    ))

    print("=== Корпус ===")
    print(f"уникальных пакетов фактов: {len(report.measurements)}")
    for name in STRATA:
        print(f"  {name}: {report.strata().get(name, 0)}")

    if report.measurements:
        print("\n=== Метрика / значение / n / что означает ===")
        print(f"ASR (невзвешенный)  {quality.share(report.asr):>22}  "
              "утверждений с источником в пакете фактов")
        print(f"ASR (взвешенный)    {quality.share(report.asr_weighted):>22}  "
              "то же с весом 2 у решения, числа, ограничения и статуса")
        print(f"DTC                 {quality.share(report.dtc):>22}  "
              "объяснений без расхождения по действию, объекту, направлению, ограничению")
        print(f"NTI                 {quality.share(report.nti):>22}  "
              "объяснений, где все числовые кортежи целы")

        print("\nМатрица DTC (поле → исход → сколько):")
        for name, states in report.confusion().items():
            print(f"  {name:14} " + ", ".join(f"{state}: {count}"
                                              for state, count in sorted(states.items())))

        unsupported = [check.claim for measurement in report.measurements
                       for check in measurement.support.unsupported]
        print(f"\nУтверждений без источника: {len(unsupported)}")
        for claim in unsupported[:10]:
            print(f"  • {claim}")

    print("\n=== Негативный набор NTI (обязан быть отвергнут) ===")
    if not packets:
        print("нужен хотя бы один пакет фактов — пропущено")
        return 0
    _, evidence = packets[0]
    text = explain_module.template_text(evidence)
    caught = 0
    for name, mutated in negatives(evidence, text).items():
        failures = quality.integrity(mutated, evidence).failures
        caught += bool(failures)
        reason = ", ".join(f"{item.token}: {item.verdict}" for item in failures) or "ПРОПУЩЕНО"
        print(f"  {name:22} {reason}")
    total = len(negatives(evidence, text))
    print(f"  поймано {caught} из {total}")

    false_rejects = sum(1 for measurement in report.measurements
                        if not measurement.integrity.passed)
    print(f"\nЛожные отказы на корпусе: {false_rejects} из {len(report.measurements)}")
    if arguments.live:
        print("\nОбъяснения корпуса — ответы языковой модели, а не шаблоны.")
    else:
        print("\nОбъяснения корпуса — шаблонные.")
        print("Метрики на шаблоне — нижняя граница конвейера, а не оценка языковой модели.")
        print("Живой замер: тот же скрипт с --live (нужен ключ в окружении).")
    return 0


def _explanations(packets, config, *, live: bool):
    """Тексты объяснений и список пакетов, по которым модель не ответила.

    Шаблон и ответ модели не смешиваются в одном замере намеренно: метрика,
    посчитанная на смеси, не относится ни к конвейеру, ни к модели. Пакет, по
    которому вызов не состоялся или ответ отвергнут проверкой, из корпуса
    **выпадает** и называется отдельно — это честнее, чем подставить на его
    место шаблон и получить более красивое число.
    """
    if not live:
        return [explain_module.template_text(evidence) for _, evidence in packets], []

    settings = replace(config.explain, enabled=True)
    texts: list[str] = []
    fallbacks: list[tuple[str, str]] = []
    kept: list = []
    for name, evidence in packets:
        answer = explain_module.explain(evidence, settings)
        if answer.source in ("llm", "cache"):
            texts.append(answer.text)
            kept.append((name, evidence))
        else:
            detail = answer.failure or ", ".join(answer.rejected_numbers)
            fallbacks.append((name, f"{answer.source}: {detail}"))
    packets[:] = kept
    return texts, fallbacks


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
