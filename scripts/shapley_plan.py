"""План коалиций для точного вклада групп в ЧДД — собрать, посчитать, объяснить.

    # 1. собрать 2^m кейсов (по умолчанию четыре группы Р-64 → 16 кейсов)
    python scripts/shapley_plan.py build runs/shapley --production 1.5 --injection 1.6

    # 2. посчитать каждый (16 × ~19 мин ≈ 5 часов)
    for case in runs/shapley/*/; do
        scripts/run_flow.sh "$case"/RUN.DATA 8 && python scripts/export_rows.py "$case"
    done

    # 3. собрать вклады
    python scripts/shapley_plan.py collect runs/shapley

Зачем именно так, а не приближённо. Разбор 4-8 показывает арифметикой: при `m`
группах **точные** значения Шепли стоят `2^m` полных расчётов, и Kernel SHAP при
малом `m` не экономит бюджет, а тратит его. Те же `2^m` ячеек дают заодно все
главные эффекты и все условные парные синергии — то есть ответ на вопрос «мешают
ли группы друг другу», ради которого мультиагентность и затевалась.

Стоимость честная: 16 прогонов по ~19 минут — около пяти часов последовательно.
Свернуть до трёх групп (добывающие одной группой) — 8 прогонов, ~2,5 часа;
`--merge-producers` делает ровно это.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, optimise, shapley  # noqa: E402
from aios import groups as groups_module  # noqa: E402
from aios import schedule as schedule_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402
from aios.run import build_case  # noqa: E402
from aios.schedule import assert_valid  # noqa: E402


def case_name(active: frozenset[str], names: list[str]) -> str:
    """Имя ячейки — маска по группам в фиксированном порядке: `c0101`.

    Не имена групп: они кириллические и с пробелами, а имя каталога уходит в
    путь дека. Расшифровка кладётся рядом в `plan.json`.
    """
    return "c" + "".join("1" if name in active else "0" for name in names)


def active_groups(groups: list[groups_module.Group]) -> list[groups_module.Group]:
    """Оставить только группы, чей множитель меняет хотя бы одну скважину.

    `aios.groups.build_groups` специально возвращает четыре именованные группы
    даже для пустого окна. Это удобно для диагностики, но в плане Шепли пустая
    группа удваивает число дорогих ГГДМ-прогонов и не меняет ни одной коалиции.
    """

    return [group for group in groups if group.weights]


def build(arguments) -> int:
    config = config_module.load(arguments.config)
    model_dir = config.case.model_dir
    schedule = parse_schedule(layout(model_dir).one("SCHEDULE"))
    steps = [step for step in schedule.steps if step.date >= config.case.control_start]
    if not steps:
        print(f"в деке нет шагов с {config.case.control_start}", file=sys.stderr)
        return 1

    rows = pd.read_csv(arguments.rows)
    # окно берётся из конфигурации, а не из литерала в умолчании ключа: на
    # валидационном кейсе (ТЗ п. 4.3) управление начинается с другой даты, и
    # группы, посчитанные «с 2007», описывали бы не тот период (Б28)
    since = arguments.since or config.case.control_start.isoformat()
    found = active_groups(groups_module.build_groups(rows, since=since))
    if not found:
        print(f"с {since} нет управляемых скважин — план коалиций не построен", file=sys.stderr)
        return 1
    if arguments.merge_producers:
        found = _merge_producers(found)

    names = [group.name for group in found]
    plan = shapley.plan(names)
    root = Path(arguments.out)
    root.mkdir(parents=True, exist_ok=True)

    print(f"групп: {len(names)} — {names}")
    print(f"коалиций: {len(plan)}; кандидатная политика — отбор ×{arguments.production}, "
          f"закачка ×{arguments.injection}")

    written = []
    # пределы кейса, а не умолчания Модели Z (С12)
    limits = schedule_module.case_limits(
        economics.load_norms(config.economics.norms_file)[1],
        config.limits.max_injector_bhp_atm,
    )

    for active in plan:
        scaled = list(steps)
        for group in found:
            if group.name not in active:
                continue
            wells = set(group.wells)
            if group.kind == "producer":
                scaled = optimise.scale(scaled, arguments.production, wells)
            else:
                scaled = optimise.scale_injection(scaled, arguments.injection, wells)
        assert_valid(scaled, **limits)
        name = case_name(active, names)
        case = build_case(str(root / name), model_dir, steps=scaled,
                          control_start=config.case.control_start,
                          wells=tuple(schedule.wells), case_name=config.case.case_name,
                          limits=limits)
        written.append({"case": name, "active": sorted(active)})
        print(f"  {name}: {sorted(active) or ['все на базовом режиме']} → {case.data_file}")

    (root / "plan.json").write_text(
        json.dumps({
            "groups": names,
            "group_wells": {group.name: list(group.wells) for group in found},
            "production_multiplier": arguments.production,
            "injection_multiplier": arguments.injection,
            "control_start": config.case.control_start.isoformat(),
            "cases": written,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nплан: {root / 'plan.json'}")
    print(f"дальше — посчитать {len(plan)} кейсов и вызвать "
          f"`python scripts/shapley_plan.py collect {root}`")
    return 0


def collect(arguments) -> int:
    root = Path(arguments.out)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    config = config_module.load(arguments.config)

    known: list[shapley.Coalition] = []
    pending: list[str] = []
    for entry in plan["cases"]:
        rows_file = root / entry["case"] / "rows.csv"
        if not rows_file.exists():
            pending.append(entry["case"])
            continue
        # тем же путём, что и сдаваемое число: эталонный расчётчик организаторов,
        # своей формулы ЧДД здесь нет и быть не может
        result = economics.compute_npv(
            pd.read_csv(rows_file).to_dict("records"),
            headers=economics.REQUIRED_COLUMNS,
            start_date=config.economics.start_date.isoformat(),
            norms_file=config.economics.norms_file,
            name=entry["case"],
        )
        known.append(
            shapley.Coalition(frozenset(entry["active"]), result.chdd_m, shapley.FULL_RUN)
        )

    if pending:
        print(f"не посчитано {len(pending)} кейсов: {pending}")
        print("вкладов не существует, пока план не закрыт целиком")
        return 1

    print(shapley.explain(plan["groups"], known))
    result = root / "shapley.json"
    result.write_text(
        json.dumps(shapley.as_dict(plan["groups"], known), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nрезультат: {result}")
    return 0


def _merge_producers(found):
    """Свернуть добывающие в одну группу: 16 прогонов становятся 8.

    Разбор 4-8 разрешает и три группы, и четыре, и разница только в цене. Свёртка
    именно добывающих, а не нагнетательных, — потому что Р-66 уже показал: рычаг
    на этом объекте у закачки, а множитель отбора при историческом пластовом
    давлении почти не двигает расчёт. Терять разрешение там, где эффекта нет,
    дешевле.
    """
    producers = [group for group in found if group.kind == "producer"]
    others = [group for group in found if group.kind != "producer"]
    if len(producers) < 2:
        return found
    total = sum(sum(group.weights.values()) for group in producers)
    weights = {
        well: weight / total
        for group in producers
        for well, weight in group.weights.items()
    }
    return [groups_module.Group("добывающие", "producer", weights), *others]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("build", "collect"))
    parser.add_argument("out", help="каталог плана коалиций")
    parser.add_argument("--rows", default="runs/base/rows.csv",
                        help="выгрузка, по которой строятся группы")
    parser.add_argument("--since", default=None,
                        help="с какой даты считать связность и обводнённость групп "
                             "(по умолчанию — [case].control_start из конфигурации)")
    parser.add_argument("--production", type=float, default=1.5)
    parser.add_argument("--injection", type=float, default=1.6)
    parser.add_argument("--merge-producers", action="store_true",
                        help="свернуть добывающие в одну группу: 2^4 прогонов станут 2^3")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    arguments = parser.parse_args(argv)

    return build(arguments) if arguments.action == "build" else collect(arguments)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
