#!/usr/bin/env python3
"""Принять историю разработки организаторов и собрать из неё дек кейса.

    python scripts/adopt_history.py --history путь/к/их/SCHEDULE.inc
    python scripts/adopt_history.py --history ... --out vendor/case_deck

Вторая часть валидационного кейса — **история разработки** — приходит 11.09 в
17:00, а сдача трека 2 в 21:00. Четыре часа, из них расчёт на ГГДМ занимает своё.
Значит, шаг «подставить их историю в дек» обязан быть командой, а не набором
действий по памяти в чужом каталоге под таймером.

**Почему подменяется дек, а не заводится ключ конфигурации.** Файл SCHEDULE
находится по `INCLUDE` в корневом `.DATA`, и на него смотрят 23 места кода. Ключ
«взять расписание отсюда» пришлось бы протаскивать через все; каталог дека уже
настраивается (`[case].model_dir`) и уже проверен подстановками валидационного
кейса. Поэтому команда делает копию поставки, кладёт в неё их историю **под тем
же именем файла** — и дальше весь контур работает без единой правки.

Что проверяется до копирования (каждое — отказ, а не заметка):

* файл разбирается нашим парсером и в нём есть хотя бы один шаг `DATES`;
* в нём нет режимов до первого `DATES`: дата такого режима — `START` из RUNSPEC,
  мы читаем только SCHEDULE, и скважина, чей единственный режим стоит там, выпала
  бы из фонда молча;
* в нём нет `WCONHIST`/`WCONINJH`: исторические режимы заменяют то, что
  генерируем мы, и в файл попало бы и то и другое;
* даты возрастают.

Что печатается и зачем: горизонт истории, состав фонда, шаг по времени и
**предлагаемая дата начала управления** — первый месяц за последним шагом
истории. Конфигурацию команда не правит: две строки вносит человек, и он же
отвечает за них перед единственным расчётом организаторов.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios.deck import parse_schedule_text  # noqa: E402
from aios.deckfiles import layout, missing_deck_refusal  # noqa: E402
from aios.run import REPLACING_CONTROL_KEYWORDS, control_tail  # noqa: E402


def portable(path: Path) -> str:
    r"""Путь для строки TOML: от корня репозитория, если он внутри; иначе полный.

    **Всегда через `/`, и это не про красоту.** Строку печатают, чтобы её
    скопировали в `config.toml`, а в TOML обратный слэш — начало escape-
    последовательности: `"C:\tmp\case"` даёт табуляцию вместо `\t`, а
    `\c` — ошибку разбора. То есть напечатанное по-виндовому вклеивается в
    конфигурацию и ломает её ровно в тот вечер, когда чинить уже нельзя.
    Прямые слэши `pathlib` на Windows понимает.
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def next_month(day: date) -> date:
    """Первое число следующего месяца — кандидат в `control_start`."""
    return date(day.year + day.month // 12, day.month % 12 + 1, 1)


def history_problems(text: str) -> list[str]:
    """Причины, по которым эту историю нельзя взять как есть.

    Те же три причины, по которым отказывает `aios.run.build_case`, — но здесь
    они называются **до** копирования дека, пока ещё есть время спросить.
    """
    problems: list[str] = []
    parsed = parse_schedule_text(text)

    if not parsed.steps:
        problems.append(
            "в файле нет ни одного шага DATES: мы ведём время только по ним, "
            "деки на TSTEP не поддерживаются"
        )
        return problems

    if parsed.preamble_controls:
        problems.append(
            "режим задан до первого DATES ("
            + ", ".join(parsed.preamble_controls)
            + "): дата такого режима — START из RUNSPEC, а мы читаем только "
            "SCHEDULE. Скважина, чей единственный режим стоит там, выпадет из "
            "фонда и из выгрузки молча"
        )

    replacing = [name for name in parsed.unsupported_controls
                 if name in REPLACING_CONTROL_KEYWORDS]
    if replacing:
        problems.append(
            "исторические режимы, заменяющие генерируемые нами ("
            + ", ".join(replacing)
            + "): в файл попало бы и они, и наш WCONPROD"
        )

    previous: date | None = None
    for step in parsed.steps:
        if previous is not None and step.date <= previous:
            problems.append(
                f"дата не возрастает: {step.date} после {previous}"
            )
            break
        previous = step.date

    return problems


def cadence(steps) -> tuple[str, list[str]]:
    """Модальный шаг по времени и перечень выбивающихся промежутков.

    Строка выгрузки у нас и у эталона — скважино-**месяц**. Немесячный шаг это
    не «другая частота», а другая экономика: OPEX за период считается по числу
    строк, и квартальный шаг удешевил бы его втрое молча.
    """
    if len(steps) < 2:
        return "шагов меньше двух — судить не о чем", []
    gaps = Counter()
    odd: list[str] = []
    for before, after in zip(steps, steps[1:], strict=False):
        months = (after.date.year - before.date.year) * 12 + after.date.month - before.date.month
        gaps[months] += 1
    common, _ = gaps.most_common(1)[0]
    for before, after in zip(steps, steps[1:], strict=False):
        months = (after.date.year - before.date.year) * 12 + after.date.month - before.date.month
        if months != common:
            odd.append(f"{before.date} → {after.date} ({months} мес.)")
    label = {1: "месячный"}.get(common, f"{common} мес.")
    return label, odd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history", type=Path, required=True,
                        help="файл истории разработки от организаторов (SCHEDULE, Eclipse)")
    parser.add_argument("--deck", type=Path, default=None,
                        help="каталог действующей поставки; по умолчанию [case].model_dir")
    parser.add_argument("--out", type=Path, default=Path("vendor/case_deck"),
                        help="куда собрать дек кейса")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--force", action="store_true",
                        help="перезаписать каталог кейса, если он уже есть")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    history_file = arguments.history
    if not history_file.is_file():
        print(f"нет файла истории {history_file}", file=sys.stderr)
        return 1

    config = config_module.load(arguments.config)
    deck_dir = arguments.deck or config.case.model_dir
    out_dir = arguments.out if arguments.out.is_absolute() else ROOT / arguments.out

    # --force разрешает заменить результат, но не уничтожить входы или
    # рекурсивно скопировать дек в самого себя. Проверяем реальные пути,
    # включая алиасы через символьные ссылки и Windows junction.
    deck_dir = deck_dir.resolve()
    out_dir = out_dir.resolve()
    protected = (deck_dir, history_file.resolve(), arguments.config.resolve(), ROOT.resolve())
    if (out_dir.is_relative_to(deck_dir)
            or any(path.is_relative_to(out_dir) for path in protected)):
        print(f"ОТКАЗ: каталог результата {out_dir} пересекается с исходным деком, "
              "историей, конфигурацией или корнем проекта", file=sys.stderr)
        return 1

    try:
        deck = layout(deck_dir)
        schedule_file = deck.one("SCHEDULE")
        schedule_relative = schedule_file.relative_to(deck_dir)
    except ValueError as error:
        print(missing_deck_refusal(deck_dir, error), file=sys.stderr)
        return 1

    text = history_file.read_text(encoding="utf-8", errors="replace")
    problems = history_problems(text)
    for problem in problems:
        print(f"ОТКАЗ: {problem}", file=sys.stderr)
    if problems:
        print("история не принята: контур собрал бы кейс на материале, "
              "который читает не так, как написан", file=sys.stderr)
        return 1

    parsed = parse_schedule_text(text)
    first, last = parsed.steps[0].date, parsed.steps[-1].date
    control_start = next_month(last)

    # Их история заменяет **историческую часть**, а не весь файл. Собственный
    # горизонт дека за концом их истории — это те самые даты, которыми нам
    # предстоит управлять: наши шаги вешаются на существующие `DATES`, своих мы
    # не порождаем. Затерев файл целиком, команда оставляла бы кейс, в котором
    # управлять нечем, — и обнаружилось бы это на шаге цикла, уже под таймером.
    deck_text = schedule_file.read_text(encoding="utf-8", errors="replace")
    horizon = control_tail(deck_text, control_start)
    adopted_text = text.rstrip() + "\n\n" + horizon if horizon.strip() else text
    if out_dir.exists():
        if not arguments.force:
            print(f"каталог {out_dir} уже есть; повторите с --force", file=sys.stderr)
            return 1
        shutil.rmtree(out_dir)
    shutil.copytree(deck_dir, out_dir)
    (out_dir / schedule_relative).write_text(adopted_text, encoding="utf-8", newline="\n")

    # Сборка обязана читаться нашим же разбором дека: подменённый файл мог
    # оказаться не тем, на который смотрит INCLUDE, и узнать об этом надо здесь,
    # а не на первом запуске симулятора.
    try:
        rebuilt = layout(out_dir)
        adopted = rebuilt.one("SCHEDULE")
    except ValueError as error:
        print(f"собранный дек не читается: {error}", file=sys.stderr)
        return 1
    if adopted.read_text(encoding="utf-8", errors="replace") != adopted_text:
        print(f"INCLUDE в {rebuilt.data_file.name} указывает не на {adopted.name}: "
              "историю подменили не в том файле", file=sys.stderr)
        return 1

    # Шов проверяется разбором, а не доверием к разрезу: повторяющийся или
    # убывающий `DATES` симулятор считает ошибкой, и узнать об этом надо здесь.
    spliced = parse_schedule_text(adopted_text)
    dates = [item.date for item in spliced.steps]
    if any(after <= before for before, after in zip(dates, dates[1:], strict=False)):
        print("шов истории и горизонта дека дал неубывающие даты — кейс не собран",
              file=sys.stderr)
        shutil.rmtree(out_dir, ignore_errors=True)
        return 1
    managed = [item for item in spliced.steps if item.date >= control_start]

    step, odd = cadence(parsed.steps)
    producers = len(parsed.producer_wells)
    injectors = len(parsed.injector_wells)

    print(f"история:     {history_file}")
    print(f"дек кейса:   {out_dir}")
    print(f"горизонт:    {first} — {last}, шагов {len(parsed.steps)}")
    if managed:
        print(f"управление:  {managed[0].date} — {managed[-1].date}, "
              f"шагов {len(managed)} — взяты из горизонта поставленного дека")
    print(f"шаг:         {step}")
    print(f"фонд:        {len(parsed.wells)} скважин "
          f"(добыча {producers}, закачка {injectors})")
    if parsed.unsupported_controls:
        print("  замечание: неразобранные ключевые слова управления: "
              + ", ".join(parsed.unsupported_controls))
    for gap in odd[:5]:
        print(f"  замечание: шаг выбивается — {gap}")
    if len(odd) > 5:
        print(f"  замечание: и ещё {len(odd) - 5} таких промежутков")

    print()
    print("Внесите в config.toml:")
    print(f'    [case] model_dir = "{portable(out_dir)}"')
    print(f"    [case] control_start = {control_start}")
    print(f"    [economics] start_date = {control_start}   # окно ЧДД = период управления")
    print()
    print("Дальше — горизонт управления. Их история кончается "
          f"{last}, значит управление идёт с {control_start}.")

    # Управлять нечем, если история покрывает весь горизонт дека. Это не наша
    # поломка и не их ошибка — это вопрос, который надо задать сразу, а не
    # обнаружить, когда сборка кейса откажет «нет ни одного шага с даты».
    #
    # Условием было `control_start > last`, а это **тождественно истинно**:
    # `control_start` и есть следующий месяц за `last`. Предупреждение звучало
    # всегда, в том числе когда горизонт был, — то есть не отличало опасный
    # случай от обычного и потому не значило ничего. Спрашивается настоящее:
    # остались ли за концом их истории шаги, которыми можно управлять.
    if not managed:
        print()
        print(f"ВНИМАНИЕ: за {last} шагов нет ни в присланной истории, ни в "
              "горизонте поставленного дека. Управлять нечем: наши режимы "
              "вешаются на существующие `DATES`, своих мы не порождаем. "
              "Спросите у трекера, до какой даты идёт период управления и чем "
              "заданы его шаги, — иначе `build_case` откажет, и правильно "
              "сделает.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
