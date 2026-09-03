"""Предполётная проверка файла, который уходит организаторам.

    python scripts/check_submission.py runs/base/wells_schedule.inc

Расчёт по нашему `wells_schedule.inc` организаторы делают **один раз**, повторной
попытки нет (`CLAUDE.md`). Поэтому проверяется не структура в памяти, из которой
файл когда-то собрали, а **сам файл с диска**: он мог быть собран прежней версией
кода, скопирован между машинами, отредактирован руками или обрезан на копировании.
`aios.run.build_case` валидирует расписание в момент записи — здесь оно
валидируется в момент отправки, и это разные проверки.

Что считается отказом (код возврата 1):

* файл не разбирается нашим же парсером — значит мы не знаем, что в нём написано;
* нарушение `aios.schedule.validate` на управляемой части: дебит выше предела,
  неизвестный режим, скважина, которой нет в деке, дата не возрастает.

Остальное — предупреждения: они не делают файл непригодным, но каждое из них
однажды стоило нам прогона. Скважина без вскрытий даёт ноль весь горизонт, а
симулятор сообщает об этом одной строкой в логе; горизонт короче дека — это
недосчитанные годы выручки; неразобранное ключевое слово означает, что часть
режима мы не видели вовсе.

Денег этот скрипт не считает: ЧДД считает только эталонный расчётчик
организаторов по выгрузке, а не расписание.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import deck as deck_module  # noqa: E402
from aios import economics  # noqa: E402
from aios import run as run_module  # noqa: E402
from aios import schedule as schedule_module  # noqa: E402
from aios.deckfiles import layout  # noqa: E402


def check(text: str, deck_text: str, control_start: date,
          max_liquid_rate: float = schedule_module.MAX_LIQUID_RATE_M3_PER_DAY,
          max_injection_bhp: float = schedule_module.MAX_INJECTION_BHP_ATM,
          ) -> tuple[list[str], list[str]]:
    """Разобрать сдаваемый файл и сверить его с деком.

    Возвращает `(отказы, предупреждения)`. Дек нужен целиком: только он знает,
    какие скважины вообще существуют и до какой даты идёт горизонт.
    """
    failures: list[str] = []
    notes: list[str] = []

    submitted = deck_module.parse_schedule_text(text)
    reference = deck_module.parse_schedule_text(deck_text)
    known = frozenset(reference.wells) | _welspecs(reference) | _welspecs(submitted)

    if not submitted.steps:
        failures.append("в файле нет ни одного шага DATES — расписание пустое")
        return failures, notes

    controlled = [step for step in submitted.steps if step.date >= control_start]
    if not controlled:
        failures.append(
            f"в файле нет ни одного шага начиная с {control_start} — управлять нечем"
        )
    # проверяется управляемая часть: история задана организаторами, и отвергать
    # их собственный дек по нашим правилам значило бы забраковать исправный файл
    for problem in schedule_module.validate(controlled, known,
                                            max_liquid_rate=max_liquid_rate,
                                            max_injection_bhp=max_injection_bhp):
        failures.append(str(problem))

    for problem in _preamble_problems(text, submitted, known, control_start,
                                      max_liquid_rate, max_injection_bhp):
        failures.append(f"до первого DATES: {problem}")

    if submitted.unsupported_controls:
        notes.append(
            "режим задан словами, которые мы не разбираем "
            f"({', '.join(submitted.unsupported_controls)}): эта часть управления "
            "нашими проверками не видна вовсе"
        )

    first, last = submitted.steps[0].date, submitted.steps[-1].date
    notes.extend(_horizon_notes(first, last, reference))
    notes.extend(_completion_notes(text, submitted))
    notes.extend(_declaration_notes(submitted))
    notes.extend(_non_ascii_notes(text))

    silent = [well for well in reference.wells if well not in set(submitted.wells)]
    if silent:
        notes.append(
            f"скважин дека без единого режима в файле: {len(silent)} "
            f"({', '.join(silent[:5])}{' …' if len(silent) > 5 else ''}) — по семантике "
            "Eclipse они сохраняют режим, назначенный до этого файла"
        )
    return failures, notes


def _preamble_problems(text: str, submitted: deck_module.Schedule,
                       known: frozenset[str], control_start: date,
                       max_liquid_rate: float = schedule_module.MAX_LIQUID_RATE_M3_PER_DAY,
                       max_injection_bhp: float = schedule_module.MAX_INJECTION_BHP_ATM,
                       ) -> list[str]:
    """Нарушения в режимах, записанных до первого DATES.

    `aios.deck` кладёт такое ключевое слово в преамбулу, а не в шаг: даты у него
    ещё нет, и приписывать её нельзя — режим действует с той даты, на которой дек
    стоял к моменту подключения файла, а она известна только в контексте всего
    дека. Пропускать его нельзя тем более: дебит 900 м³/сут, записанный до
    первого DATES, проходил проверку с нулём отказов, и файл объявлялся пригодным
    к отправке (найдено прогоном 20.08).

    Проверяются нарушения, от даты не зависящие: предел дебита, неизвестный режим,
    скважина не из дека. Судится преамбула ровно тогда, когда весь файл наш, то
    есть первый его шаг лежит в управляемой части; файл, несущий историю, остаётся
    за организаторами целиком — по тому же правилу, что и его шаги.

    Записи разбираются подстановкой `DATES` перед текстом, а не своим разбором:
    парсер у нас один, и второй его вариант рано или поздно разойдётся с первым.
    Дата подставляется тем же генератором, которым мы пишем расписание, — иначе
    её пришлось бы собирать руками, и в сообщении об ошибке стоял бы не тот месяц.
    """
    if not submitted.steps or submitted.steps[0].date < control_start:
        return []
    first = submitted.steps[0].date
    probe = deck_module.parse_schedule_text(
        schedule_module.render([deck_module.ScheduleStep(first, (), ())]) + text
    )
    if not probe.steps:
        return []
    head = probe.steps[0]
    if not head.producers and not head.injectors:
        return []
    return [str(problem) for problem in schedule_module.validate(
        [head], known, max_liquid_rate=max_liquid_rate,
        max_injection_bhp=max_injection_bhp)]


def _non_ascii_notes(text: str) -> list[str]:
    """Не-ASCII байты в сдаваемом файле — заметка, а не отказ.

    Комментарий сам по себе безвреден: парсер обязан выбросить всё от `--` до
    конца строки. Но файл читает чужой симулятор в чужой локали, и под cp1251
    кириллица превращается в мусор ровно там, куда проверяющий смотрит первым
    делом. Заголовок генератора переведён на ASCII (Р-179) — а файл, собранный до
    этой правки, по-прежнему несёт две кириллические строки, и узнать об этом
    надо **до** отправки, а не от организаторов.

    Почему заметка. Перегенерация меняет `md5` сдаваемого файла, а с ним тег и
    числа в семи документах: это решение владельца, а не побочный эффект
    проверки. Отказывать значило бы объявить непригодным файл, которым посчитано
    заявляемое число.
    """
    lines = [number for number, line in enumerate(text.splitlines(), 1)
             if not line.isascii()]
    if not lines:
        return []
    listed = ", ".join(str(number) for number in lines[:5])
    if len(lines) > 5:
        listed += f" и ещё {len(lines) - 5}"
    return [
        f"не-ASCII байты в строках {listed} ({len(lines)} всего): под чужой "
        f"локалью они превратятся в мусор. Генератор с Р-179 пишет заголовок "
        f"только ASCII — значит, файл собран до этой правки и перегенерация "
        f"уберёт их (но изменит md5, тег и числа в документах)"
    ]


def _horizon_notes(first: date, last: date, reference: deck_module.Schedule) -> list[str]:
    notes: list[str] = []
    if reference.steps and last < reference.steps[-1].date:
        months = ((reference.steps[-1].date.year - last.year) * 12
                  + reference.steps[-1].date.month - last.month)
        notes.append(
            f"горизонт короче дека на {months} мес.: файл кончается {last}, "
            f"дек — {reference.steps[-1].date}"
        )
    if reference.steps and first > reference.steps[0].date:
        notes.append(
            f"файл начинается {first}, дек — {reference.steps[0].date}: форма «хвост», "
            "история остаётся за организаторами (вопрос В-03)"
        )
    return notes


def _completion_notes(text: str, submitted: deck_module.Schedule) -> list[str]:
    """Скважины, получающие режим, но не соединённые с пластом.

    Вскрытий в файле может не быть вовсе — тогда они остались в деке, и судить
    отсюда не о чем. Проверять есть смысл ровно тогда, когда файл вскрытия несёт.
    """
    if "COMPDAT" not in text:
        return ["вскрытий (COMPDAT) в файле нет — они остаются в деке организаторов"]
    uncompleted = run_module.wells_without_completions(text, submitted.wells)
    if not uncompleted:
        return []
    return [
        f"скважин с режимом, но без вскрытия: {len(uncompleted)} "
        f"({', '.join(uncompleted[:5])}{' …' if len(uncompleted) > 5 else ''}) — "
        "симулятор оставит их закрытыми и сообщит об этом одной строкой лога"
    ]


def _declaration_notes(submitted: deck_module.Schedule) -> list[str]:
    """Скважины, получающие режим, но не объявленные `WELSPECS` в самом файле.

    Требование из разбора семантики SCHEDULE (`docs/manus/schedule-semantics.md`,
    п. 4 чек-листа): `WELSPECS` обязан стоять до первого `COMPDAT`, `WCONPROD`,
    `WCONINJE` и `WEFAC` для каждой скважины, на которую файл ссылается. Оно
    действует в прочтении «файл подменяет всю секцию SCHEDULE»: наследовать
    объявление там неоткуда, и симулятор отвергнет дек. В прочтении «файл
    дописывается хвостом» объявление приходит из истории, и его отсутствие
    нормально.

    Какое из двух прочтений верно, мы не знаем (В-03), поэтому это предупреждение,
    а не отказ. Наш сдаваемый файл объявляет все 103 скважины и проверку проходит
    молча — но собран он может быть и иначе.
    """
    declared = _welspecs(submitted)
    missing = [well for well in submitted.wells if well not in declared]
    if not missing:
        return []
    return [
        f"скважин с режимом, но без WELSPECS в самом файле: {len(missing)} "
        f"({', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}) — так файл "
        "читается только как дописывание к истории; как замена всей секции "
        "SCHEDULE он неполон (вопрос В-03)"
    ]


def _welspecs(parsed: deck_module.Schedule) -> frozenset[str]:
    """Имена скважин из WELSPECS преамбулы — скважина может быть объявлена и молчать."""
    return frozenset(
        record[0].strip("'\"")
        for keyword in parsed.preamble if keyword.name == "WELSPECS"
        for record in keyword.records if record and record[0]
    )


def events(steps: tuple[deck_module.ScheduleStep, ...]) -> dict[str, int]:
    """Сколько раз расписание останавливает, пускает и переводит скважины.

    Это ориентир, а не деньги: события методики считаются по выгрузке расчёта,
    а не по расписанию, и складывает их эталонный расчётчик организаторов.
    Здесь видно другое — насколько подвижное расписание мы отправляем.

    Сравнивается состояние скважины **на конец шага**, а не каждая запись режима.
    Перевод под закачку записывается двумя строками одного шага — закрыть на
    добычу и открыть на закачку, — и посчитанный по записям он даёт три события
    вместо одного: остановку, пуск и перевод. На историческом расписании Модели Z
    это давало ровно 30 / 30 / 30 (найдено прогоном 20.08).
    """
    counts = {"остановок": 0, "пусков": 0, "переводов под закачку": 0}
    state: dict[str, tuple[str, str]] = {}  # скважина → (роль, статус) на конец шага
    for step in steps:
        now = {control.well: ("добыча", control.status) for control in step.producers}
        now.update({control.well: ("закачка", control.status) for control in step.injectors})
        for well, (role, status) in now.items():
            was = state.get(well)
            if was is None:
                pass
            elif was[0] != role:
                if role == "закачка":
                    counts["переводов под закачку"] += 1
            elif was[1] == "OPEN" and status != "OPEN":
                counts["остановок"] += 1
            elif was[1] != "OPEN" and status == "OPEN":
                counts["пусков"] += 1
        state.update(now)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="путь к wells_schedule.inc")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    if not arguments.path.is_file():
        print(f"нет файла: {arguments.path}", file=sys.stderr)
        return 1

    config = config_module.load(arguments.config)
    text = arguments.path.read_text(encoding="utf-8", errors="replace")
    deck_text = layout(config.case.model_dir).one("SCHEDULE").read_text(
        encoding="utf-8", errors="replace"
    )

    # предел дебита жидкости берётся из таблицы ЭЦН нормативов кейса: на
    # валидационном кейсе (ТЗ п. 4.3) старший насос может быть другим (Б28)
    failures, notes = check(
        text, deck_text, config.case.control_start,
        schedule_module.liquid_limit(economics.load_norms(config.economics.norms_file)[1]),
        # то же и про давление нагнетания: 300 атм — умолчание Модели Z, а кейс
        # называет своё число в `[limits].max_injector_bhp_atm` (ревизия 2.3 #5)
        (schedule_module.MAX_INJECTION_BHP_ATM
         if config.limits.max_injector_bhp_atm is None
         else config.limits.max_injector_bhp_atm),
    )
    submitted = deck_module.parse_schedule_text(text)

    print(f"файл:      {arguments.path} ({arguments.path.stat().st_size / 1e6:.2f} МБ)")
    print(f"дек:       {config.case.model_dir}")
    if submitted.steps:
        print(f"шагов:     {len(submitted.steps)} "
              f"({submitted.steps[0].date} — {submitted.steps[-1].date})")
        print(f"скважин:   {len(submitted.wells)} "
              f"(добыча {len(submitted.producer_wells)}, закачка {len(submitted.injector_wells)})")
        counts = events(submitted.steps)
        print("события:   " + ", ".join(f"{name} {count}" for name, count in counts.items())
              + "  (ориентир: деньги считает эталонный расчётчик по выгрузке)")
    print()

    print(f"--- предупреждения: {len(notes) or 'нет'} ---")
    for note in notes:
        print(f"  {note}")
    print()

    print(f"--- отказы: {len(failures) or 'нет'} ---")
    for failure in failures[:25]:
        print(f"  {failure}")
    if len(failures) > 25:
        print(f"  … ещё {len(failures) - 25}")

    print()
    print("файл пригоден к отправке" if not failures else
          "файл отправлять нельзя: перечисленное выше сломает расчёт или изменит фонд")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
