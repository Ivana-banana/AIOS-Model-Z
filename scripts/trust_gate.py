"""Собрать доверенную область по датасетам DoE и замерить, что она ловит.

Замер, который эта команда обязана дать, задан разбором 4-7: **новый шлюз должен
отбраковывать кандидатов, которых покоординатный min–max пропускает**. Поэтому
скрипт не только строит область, но и строит нарочно сдвинутые режимы — законные
по каждой координате в отдельности, но совместно нетипичные, — и печатает, сколько
из них проходит одну проверку и сколько другую.

    python scripts/trust_gate.py --dataset data/dataset-all-wide.csv \
        --dataset data/dataset-periods.csv --out artifacts/trust-gate.json

Пары для конформной калибровки берутся отдельным файлом: строки
`прогноз_суррогата,значение_полного_расчёта` в млн руб. Если файла нет или пар
недостаточно, ответ «не калибровано» является правильным, а не нулевым риском.

Коды возврата режима кандидата: `0` — причин идти на полный расчёт нет, `2` —
есть, `1` — команда не выполнилась.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import trust  # noqa: E402  (импорт после правки sys.path)
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402


def load_profiles(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Сценарные представления из датасетов DoE.

    Единица — сценарий пакета, а не строка. Имя сценария дополняется именем файла:
    пакеты нумеруют сценарии независимо, и без префикса `s01` из двух разных
    пакетов слились бы в один — то есть выборка выглядела бы вдвое гуще, чем есть.
    """
    profiles: dict[str, dict[str, float]] = {}
    for path in paths:
        # Раздельные множители по фазам появились не сразу: пакеты старого вида
        # (`data/dataset-levels.csv` — тот самый, что стоит умолчанием в
        # `train_surrogate.py`) несут только общий `multiplier`. Требовать их
        # безусловно значило падать трассировкой `Usecols do not match columns`
        # на собственных датасетах — притом что ветка «взять общий множитель»
        # ниже написана ровно под этот случай и была недостижима (Н-30).
        present = set(pd.read_csv(path, nrows=0).columns)
        needed = ["scenario", "DATA", "well", "multiplier"]
        absent = [name for name in needed if name not in present]
        if absent:
            raise ValueError(
                f"в пакете {path} нет обязательных колонок: {', '.join(absent)}. "
                "Ожидается выгрузка DoE от scripts/build_dataset.py"
            )
        optional = [name for name in ("production_multiplier", "injection_multiplier")
                    if name in present]
        frame = pd.read_csv(path, usecols=needed + optional)
        for name in ("production_multiplier", "injection_multiplier"):
            if name not in optional:
                frame[name] = float("nan")
        frame["DATA"] = pd.to_datetime(frame["DATA"])
        for name, block in frame.groupby("scenario"):
            months: dict = {}
            for stamp, month in block.groupby("DATA"):
                controls: dict[str, float] = {}
                for row in month.itertuples():
                    production = row.production_multiplier
                    injection = row.injection_multiplier
                    if pd.isna(production) and pd.isna(injection):
                        production = row.multiplier
                    if not pd.isna(production):
                        controls[f"prod:{row.well}"] = float(production)
                    if not pd.isna(injection):
                        controls[f"inj:{row.well}"] = float(injection)
                months[stamp.date()] = controls
            profiles[f"{path.stem}:{name}"] = trust.profile(months)
    return profiles


def shifted(profiles: dict[str, dict[str, float]], *, seed: int,
            count: int, share: float) -> list[dict[str, float]]:
    """Законные по каждой координате, но совместно нетипичные режимы.

    Рецепт из разбора 4-7, §8, этап 4: **не выходя за min–max**, согласованно
    подтянуть большую часть управлений к одному краю. Прямоугольник такое
    пропускает по построению — каждая координата внутри своего отрезка, — а
    сценарное расстояние обязано это увидеть.
    """
    rng = random.Random(seed)
    # объединение по всем пакетам: управление, которого нет в одном пакете, в нём
    # просто не двигалось, и его нейтральная единица — такое же наблюдение
    every = sorted({key for one in profiles.values() for key in one})
    levels = [key for key in every if key.endswith("|уровень")]
    bounds = {
        key: (min(one.get(key, 1.0) for one in profiles.values()),
              max(one.get(key, 1.0) for one in profiles.values()))
        for key in levels
    }
    baseline = list(profiles.values())

    result = []
    for index in range(count):
        source = {key: baseline[index % len(baseline)].get(key, 1.0) for key in every}
        chosen = rng.sample(levels, k=max(1, int(len(levels) * share)))
        for key in chosen:
            low, high = bounds[key]
            source[key] = high if index % 2 == 0 else low
        result.append(source)
    return result


def coordinatewise(one: dict[str, float],
                   profiles: dict[str, dict[str, float]]) -> bool:
    """Прошёл бы кандидат покоординатную проверку min–max — та самая база сравнения."""
    for key, value in one.items():
        values = [other.get(key, 1.0) for other in profiles.values()]
        if value < min(values) - 1e-9 or value > max(values) + 1e-9:
            return False
    return True


class Pair(NamedTuple):
    """Строка калибровочного файла целиком: числа, прогон и его семейство.

    Имя прогона и семейство `collect_pairs.py` пишет рядом с числами не для
    красоты — по ним решается, обмениваема ли пара с подсудимым кандидатом.
    """

    predicted: float
    actual: float
    run: str
    family: str
    model: str = ""
    """sha256 артефакта, которым посчитан прогноз. Пусто у файлов до 28.08."""


def load_pairs(path: Path | None) -> list[Pair]:
    """Пары «прогноз, полный расчёт, семейство» из CSV `collect_pairs.py`.

    Семейство читается вместе с числами, а не выбрасывается: калибровать запас
    можно только тем семейством, к которому его применяют (Р-106). Нечитаемая
    строка называется по номеру и содержимому, а не роняет команду голым
    `ValueError`: свой файл `collect_pairs.py` пишет без шапки, но ключ
    `--pairs` задокументирован как ручной, а таблица шапку добавляет.
    """
    if path is None or not path.exists():
        return []
    pairs: list[Pair] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            try:
                pairs.append(Pair(float(row[0]), float(row[1]),
                                  row[2].strip() if len(row) > 2 else "",
                                  row[3].strip() if len(row) > 3 else "",
                                  row[5].strip() if len(row) > 5 else ""))
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"{path}, строка {number}: ожидались два числа "
                    f"«прогноз, полный расчёт», получено {row} — {error}"
                ) from error
    return pairs


def by_family(rows: list[Pair], family: str) -> tuple[list[Pair], str]:
    """Оставить то семейство, к которому запас и применяется.

    Зачем. Конформная гарантия держится на обмениваемости, а знак остатка
    разделяет наши семейства начисто: на ручных сценариях суррогат занижает ЧДД,
    на кандидатах оптимизатора завышает. Смешанная выборка даёт меньший запас —
    то есть **разрешает больше**, — и `collect_pairs.py` пишет именно смешанный
    файл по умолчанию (`docs/data/margin-pairs.csv`, 21 ручная пара и 3
    поисковые). Отбор молчаливым быть не может: строка отбора печатается.

    Файл без колонки семейства принимается целиком. Он собран руками, сказать о
    его происхождении нечего, и выбросить его молча значит решить за оператора.
    """
    counts = Counter(pair.family for pair in rows)
    if not any(name for name in counts):
        return (rows, f"пар {len(rows)}: семейство в файле не проставлено, взяты все")
    kept = [pair for pair in rows if pair.family == family]
    shown = ", ".join(f"{name or 'без семейства'} {number}"
                      for name, number in sorted(counts.items()))
    return kept, (f"пар {len(rows)} ({shown}): взяты {len(kept)} "
                  f"семейства «{family}»")


def without_run(rows: list[Pair], run: str) -> tuple[list[Pair], str]:
    """Убрать из калибровки пару самого подсудимого кандидата.

    Односторонний конформный квантиль покрывает обменяемый с выборкой сценарий.
    Кандидат, чей собственный полный расчёт лежит в этой же выборке, обменяемым
    с ней не является ни в каком смысле: его остаток одновременно и калибрует
    порог, и проверяется этим порогом. `docs/data/margin-pairs-search.csv`
    собран ровно по кандидатам поиска, так что совпадение имени — не редкость,
    а обычный ход дела при повторной проверке уже посчитанного расписания.

    Возвращается вторая строка — что именно выброшено. Пустая означает «пары
    этого прогона в файле не было», и молчать об изъятии нельзя: оно меняет `n`,
    а вместе с ним и само существование квантиля.
    """
    kept = [pair for pair in rows if pair.run != run]
    if len(kept) == len(rows):
        return rows, ""
    return kept, (f"пара прогона «{run}» изъята из калибровки: кандидат не "
                  f"обменяем с выборкой, в которой лежит его же расчёт")


def checksum(path: Path) -> str:
    """sha256 файла — тот же отпечаток, которым пакетный допуск сверяет суррогат."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def foreign_model(rows: list[Pair], model: str) -> str:
    """Посчитаны ли пары тем самым артефактом, который сейчас судят.

    Остаток «прогноз минус факт» — свойство пары «модель и расписание», а не
    одного расписания. Переобучили суррогат, файл пар оставили на месте — и
    шлюз отвечает «калиброван» по остаткам модели, которой больше нет.
    Артефакт живёт в `artifacts/` вне git и переобучается под тем же именем,
    так что расхождение здесь — обычный ход дела, а не редкий случай.

    Возвращается описание расхождения или пустая строка. Файлы, собранные до
    появления колонки, отпечатка не несут: о них говорится вслух, но в отказ
    это не превращается — задним числом их не пересобрать.
    """
    stamped = {pair.model for pair in rows if pair.model}
    if not rows or not stamped:
        return "отпечаток модели в парах не записан: чем они посчитаны — неизвестно"
    if stamped == {model}:
        return ""
    names = ", ".join(sorted(one[:12] for one in stamped))
    return (f"пары посчитаны другим артефактом (sha256 {names}), "
            f"а судится {model[:12]}: остатки чужой модели эту не калибруют")


def show_margin(margin: trust.OptimismMargin) -> str:
    """Строка запаса для человека — в том же виде, в каком число стоит в отчёте.

    Единица у запаса та же, что у цели, — млн руб., и приводится он всюду с
    одним знаком (927,8). Сырой `repr` давал `927.8179970000001`, то есть
    команда, которой это число подтверждают, его же и не показывала.
    """
    if not margin.calibrated:
        return f"конформный запас: не калиброван — {margin.reason}"
    return f"конформный запас: {margin.value:.1f} млн руб."


def judge(model: Path, schedule: Path, config_path: Path, *,
          pairs: list[Pair] | None = None,
          alpha: float = trust.DEFAULT_ALPHA) -> int:
    """Вердикт по одному кандидату — по области, лежащей в самом артефакте.

    Область доверия обучается вместе с суррогатом и едет внутри него (Р-72),
    поэтому судить кандидата можно там, где датасетов нет вовсе: на машине без
    прогонов, сразу после того как цикл собрал расписание. Датасеты нужны только
    режиму, который строит область заново и меряет, что она ловит.

    Печатается расстояние до обучающих сценариев, порог, ближайшие соседи и
    список причин идти на полный расчёт. Пустой список означает «суррогату можно
    доверить **ранжирование**» — и только его: заявляемое число подтверждается
    полным расчётом всегда.

    Код возврата несёт вердикт: `0` — причин нет, `2` — есть. Отказ запуска
    остаётся единицей, и смешивать его с вердиктом нельзя: «шлюз сказал нет» и
    «шлюз не запустился» требуют от оператора разных действий. Шлюз стоит шагом
    покандидатного протокола, то есть его читает не только человек.
    """
    import joblib

    from aios.neural import NeuralBundle, NeuralSurrogate

    refusal = config_module.refusal(config_path)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(config_path)
    if not model.is_file():
        # артефакт лежит в `artifacts/`, а этот каталог в git не ведётся: промах
        # мимо него на чистом клоне — обычный первый запуск, а не случайность
        print(f"нет обученного суррогата: {model}", file=sys.stderr)
        return 1
    bundle = joblib.load(model)
    if not isinstance(bundle, NeuralBundle):
        print(f"это не обученный суррогат: {type(bundle).__name__}", file=sys.stderr)
        return 1
    if not schedule.is_file():
        print(f"нет расписания кандидата: {schedule}", file=sys.stderr)
        return 1

    try:
        reference = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    except ValueError as error:
        # опорное расписание берётся из дека, а дек в git не ведётся: на чистом
        # клоне это обычный первый запуск. Трассировка здесь читалась бы как
        # поломка шлюза, хотя шлюз исправен и просто не получил объект (Р-238)
        print(f"Дек не читается: {error}\n"
              f"Ожидается каталог поставки с корневым .DATA: "
              f"{config.case.model_dir}\n"
              f"Как распаковать поставку — README.md, раздел «Установка».",
              file=sys.stderr)
        return 1
    controlled = [step for step in reference.steps
                  if step.date >= config.case.control_start]
    surrogate = NeuralSurrogate.create(bundle, controlled, config.fluids,
                                       require_quality=False)
    candidate = parse_schedule(schedule)
    steps = [step for step in candidate.steps if step.date >= config.case.control_start]
    if not steps:
        print(f"в {schedule} нет шагов с {config.case.control_start}", file=sys.stderr)
        return 1

    region = bundle.trust_region
    distance = region.distance(surrogate.control_profile(steps)) if region else None
    rows = pairs or []
    margin = trust.OptimismMargin.fit([(one.predicted, one.actual) for one in rows],
                                      alpha=alpha)
    # чужой отпечаток — причина не хуже расстояния: остатки модели, которой уже
    # нет, эту модель не калибруют. Отсутствие отпечатка причиной не делаем:
    # файлы, собранные до его появления, задним числом не пересобрать
    binding = foreign_model(rows, checksum(model)) if rows else ""
    stamped = any(one.model for one in rows)
    reasons = trust.routed_to_full_run(distance, margin,
                                       [binding] if binding and stamped else None,
                                       region=region)

    print(f"кандидат:  {schedule}")
    print(f"модель:    {model} ({bundle.report.kind})")
    print(f"шагов:     {len(steps)} с {config.case.control_start}")
    if region is None:
        print("область доверия: в артефакте её нет — судить нечем")
    else:
        print(f"сценариев в области: {len(region.scenarios)}, признаков {len(region.features)}")
        print(f"расстояние: {distance.value:.3f} при пороге {distance.threshold:.3f} — "
              f"{'ВНЕ области' if distance.outside else 'внутри области'}")
        for neighbour in distance.neighbours[:3]:
            print(f"  ближайший обучающий сценарий {neighbour.scenario}: {neighbour.distance:.3f}")
    print(show_margin(margin))
    if binding:
        print(binding)
    print()
    print(f"причин идти на полный расчёт: {len(reasons)}")
    for reason in reasons:
        print(f"  - {reason}")
    return 2 if reasons else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append",
                        help="датасеты DoE: режим сборки области заново")
    parser.add_argument("--model", type=Path, default=None,
                        help="обученный суррогат: судить кандидата областью из него")
    parser.add_argument("--schedule", type=Path, default=None,
                        help="wells_schedule.inc кандидата")
    parser.add_argument("--config", type=Path, default=config_module.DEFAULT_PATH)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="CSV «прогноз, полный расчёт» в млн руб. для калибровки")
    parser.add_argument("--pairs-family", default="search",
                        help="каким семейством калибровать: шлюз судит кандидатов "
                             "оптимизатора, и обмениваемы с ними только они (Р-106)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/trust-gate.json"))
    parser.add_argument("--k", type=int, default=trust.DEFAULT_K)
    parser.add_argument("--alpha", type=float, default=trust.DEFAULT_ALPHA)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--shifted", type=int, default=200)
    parser.add_argument("--share", type=float, default=0.6,
                        help="какую долю управлений сдвигать к краю")
    args = parser.parse_args(argv)

    # путь, заданный руками, обязан существовать: молчаливый `[]` даёт ровно тот
    # же вердикт «нет ни одной пары», что и запуск без ключа вовсе, и опечатка
    # в пути читается как ожидаемый ответ
    if args.pairs is not None and not args.pairs.is_file():
        print(f"нет файла калибровочных пар: {args.pairs}", file=sys.stderr)
        return 1
    try:
        rows = load_pairs(args.pairs)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.schedule is not None:
        rows, dropped = without_run(rows, args.schedule.parent.name)
        if dropped:
            print(dropped)
    pairs, note = by_family(rows, args.pairs_family)
    if args.pairs is not None:
        print(note)

    if args.model or args.schedule:
        if not (args.model and args.schedule):
            print("режиму кандидата нужны оба ключа: --model и --schedule",
                  file=sys.stderr)
            return 1
        return judge(args.model, args.schedule, args.config,
                     pairs=pairs, alpha=args.alpha)

    if not args.dataset:
        print("нечего делать: задайте --dataset (собрать область заново) либо "
              "--model и --schedule (судить кандидата областью из артефакта)",
              file=sys.stderr)
        return 1

    profiles = load_profiles(args.dataset)
    if len(profiles) <= args.k:
        # Пустой или короткий пакет — обычное состояние на чистом клоне: каталог
        # `data/` под .gitignore и собирается из `runs/` командой build_dataset.
        # До 09.09 это отвечало трассировкой из глубины `TrustRegion.fit`
        # («сценариев 0, а нужно больше k = 5»), и читать её приходилось тому,
        # кто просто повторил команду из README (Н-30).
        print(
            f"в пакетах {', '.join(str(one) for one in args.dataset)} "
            f"сценариев {len(profiles)}, а области нужно больше k = {args.k}: "
            "расстояние до k-го соседа при таком наборе не определено.\n"
            "Датасеты DoE не ведутся в git (каталог data/ под .gitignore) — "
            "соберите их командой scripts/build_dataset.py по каталогам прогонов "
            "или укажите --dataset на готовый пакет.",
            file=sys.stderr)
        return 1
    region = trust.TrustRegion.fit(profiles, k=args.k, quantile=args.quantile)
    margin = trust.OptimismMargin.fit(
        [(pair.predicted, pair.actual) for pair in pairs], alpha=args.alpha)

    print(f"сценариев {len(profiles)}, признаков {len(region.features)}, "
          f"замороженных {len(region.frozen)}")
    print(f"порог t = {region.threshold:.3f} "
          f"(квантиль {args.quantile:.2f} leave-one-out расстояний)")
    print(show_margin(margin))

    candidates = shifted(profiles, seed=args.seed, count=args.shifted, share=args.share)
    passed_box = sum(coordinatewise(one, profiles) for one in candidates)
    passed_gate = sum(not region.distance(one).outside for one in candidates)
    caught = sum(coordinatewise(one, profiles) and region.distance(one).outside
                 for one in candidates)

    print()
    print(f"сдвинутых кандидатов {len(candidates)}")
    print(f"  прошли покоординатный min–max: {passed_box}")
    print(f"  прошли сценарное расстояние:   {passed_gate}")
    print(f"  пропущены прямоугольником и отброшены новым шлюзом: {caught}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = trust.contract(region, margin,
                             dataset_hash=";".join(sorted(path.name for path in args.dataset)))
    written["shift_audit"] = {
        "candidates": len(candidates),
        "passed_coordinatewise": passed_box,
        "passed_scenario_distance": passed_gate,
        "caught_by_the_second_gate_only": caught,
        "share_shifted": args.share,
        "seed": args.seed,
    }
    args.out.write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nартефакт доверия: {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
