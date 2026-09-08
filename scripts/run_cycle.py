"""Прогнать цикл управления поверх результата ГГДМ и собрать кейс под проверку.

Два шага, между ними — полный расчёт на симуляторе:

    python scripts/run_cycle.py --from runs/base --out runs/policy
    docker run --rm -v "$PWD:$PWD" -w "$PWD" aios \
        scripts/run_flow.sh runs/policy/RUN.DATA 4 --parsing-strictness=low
    python scripts/run_cycle.py --confirm runs/policy --from runs/base

Первый шаг обучает суррогат на выгрузке базового прогона, проходит цикл
«состояние → диагноз → решения → расписание → прогноз ЧДД» и пишет кейс с
получившимся расписанием. Число, которое он печатает, — **прогноз суррогата**.

Третий шаг заменяет прогноз результатом полного расчёта: только после него
прирост можно называть результатом (`aios.loop.confirm`, `CycleResult.confirmed`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import (  # noqa: E402
    economics,
    loop,
    optimise,
    pvt,  # noqa: E402
    summary,
    trust,
)
from aios import explain as explain_module  # noqa: E402
from aios.deck import parse_schedule  # noqa: E402
from aios.deckfiles import layout  # noqa: E402
from aios.margin import Norms  # noqa: E402
from aios.model import SurrogateLoadError, for_cycle  # noqa: E402
from aios.run import build_case  # noqa: E402
from aios.schedule import liquid_limit  # noqa: E402
from aios.schedule import render as render_schedule  # noqa: E402


def surrogate_identity(path: str | None, name: str) -> dict[str, str]:
    """Чем именно считался прогноз: имя модели и отпечаток файла артефакта.

    Пишется рядом с кейсом и проверяется при подтверждении. Без этого
    `--confirm` молча берёт **другой** суррогат — тот, что назван в его
    собственной команде, — и печатает «прогноз обещал» от одной модели рядом с
    фактом, полученным расписанием другой. Найдено 24.08 на первом же кандидате:
    кейс собран обученным артефактом, подтверждение шло тривиальным бейзлайном.
    """
    identity = {"surrogate": name}
    if path:
        artefact = Path(path)
        identity["model"] = str(artefact)
        if artefact.is_file():
            identity["sha256"] = hashlib.sha256(artefact.read_bytes()).hexdigest()
    return identity


def written_identity(case_dir: Path) -> dict[str, str] | None:
    """Что записано рядом с уже собранным кейсом. `None` — кейс собран до Р-100."""
    marker = case_dir / "cycle.json"
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("surrogate_identity")
    except (OSError, ValueError):
        return None


def file_identity(path: str | Path | None) -> dict[str, str] | None:
    """Путь и отпечаток файла, от которого зависит найденный план.

    Путь сохраняем разрешённым, чтобы относительный путь первого запуска и
    абсолютный путь команды подтверждения не считались разными входами.
    """
    if path is None:
        return None
    item = Path(path)
    identity = {"path": str(item.resolve())}
    if item.is_file():
        identity["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
    return identity


def decision_identity(*, source: Path, config_path: str | Path, tune_rates: bool,
                      multipliers: tuple[float, ...] | None, search_seed: int,
                      base_schedule: str | Path | None) -> dict[str, object]:
    """Входы, от которых зависят решения уже собранного цикла.

    `--confirm` заново выполняет поиск, чтобы восстановить его прогноз. Поэтому
    одного совпадения суррогата недостаточно: иной seed, набор множителей или
    опорное расписание дают другой план, хотя факт относится к старому кейсу.
    """
    return {
        "source": str(source.resolve()),
        "config": file_identity(config_path),
        "tune_rates": tune_rates,
        "multipliers": list(multipliers) if multipliers is not None else None,
        "search_seed": search_seed,
        "base_schedule": file_identity(base_schedule),
    }


def plan_digest(steps) -> str:
    """`md5` самого плана — того расписания, которое цикл только что решил.

    `--confirm` заново проходит весь поиск, чтобы восстановить прогноз, и
    сверяет входы: суррогат, конфигурацию, seed, набор множителей, опорное
    расписание. Версии **кода** среди входов нет, а решают именно они: любая
    правка `optimise`, `planner`, `policy` или `loop` между `--out` и
    `--confirm` даёт пару «расписание A + отклик ГГДМ на расписание B». Подпись
    «подтверждено полным расчётом» при этом стоит на обоих (Н-20).

    Считается по тексту, а не по объектам: тот же рендер с обеих сторон, значит
    округление одинаковое, и сравниваются решения, а не представление чисел.
    """
    return hashlib.md5(render_schedule(steps).encode("utf-8")).hexdigest()


def written_decision_identity(case_dir: Path) -> dict[str, object] | None:
    """Входы, записанные новым маркером; старый маркер остаётся допустимым."""
    marker = case_dir / "cycle.json"
    if not marker.is_file():
        return None
    try:
        identity = json.loads(marker.read_text(encoding="utf-8")).get("decision_identity")
    except (OSError, ValueError):
        return None
    return identity if isinstance(identity, dict) else None


def written_plan_digest(case_dir: Path) -> str | None:
    """`md5` плана, записанный маркером кейса. `None` — маркер старый."""
    marker = case_dir / "cycle.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8")).get("plan_md5")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, str) else None


def confirmation_command(*, case_dir: str | Path, source: Path,
                         config_path: str | Path, no_tune_rates: bool,
                         surrogate_model: str | None,
                         base_schedule: str | None,
                         multipliers: tuple[float, ...] | None,
                         search_seed: int) -> str:
    """Команда, которая повторяет все значимые параметры первого запуска."""
    command = [
        "python", "scripts/run_cycle.py", "--confirm", str(case_dir),
        "--from", str(source), "--config", str(config_path),
    ]
    if no_tune_rates:
        command.append("--no-tune-rates")
    if surrogate_model:
        command.extend(("--surrogate-model", surrogate_model))
    if base_schedule:
        command.extend(("--base-schedule", base_schedule))
    if multipliers is not None:
        command.extend(("--multipliers", ",".join(str(value) for value in multipliers)))
    command.extend(("--search-seed", str(search_seed)))
    return shlex.join(command)


def _write_explanation(out_dir: Path, result, surrogate_name: str, config) -> Path:
    """Написать объяснение решения рядом с кейсом и вернуть путь к файлу.

    Отказ слоя не роняет прогон: объяснение — отчётность о решении, а не само
    решение. Но и молчать нельзя, поэтому источник (`llm`, `cache`, `template`,
    `template-rejected`, `template-failed`) пишется в файл рядом с текстом: по
    нему видно, объясняла модель или подставился шаблон.
    """
    path = out_dir / "explanation.json"
    try:
        explanation = explain_module.explain(
            explain_module.evidence_from_cycle(
                result, surrogate_name,
                limits=config.limits, fluids=config.fluids),
            config.explain,
        )
        record = {
            "text": explanation.text,
            "source": explanation.source,
            "fingerprint": explanation.fingerprint,
            "model": explanation.model,
            "rejected_numbers": list(explanation.rejected_numbers),
            "failure": explanation.failure,
        }
    except Exception as error:  # noqa: BLE001 — слой не должен уносить результат цикла
        record = {"text": "", "source": "failed", "failure": f"{type(error).__name__}: {error}"}
    try:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except OSError as error:
        # Запись тоже под защитой, а не только вызов модели: кейс к этому моменту
        # уже собран и годен, и ронять прогон из-за недоступного файла отчётности
        # — ровно то, чего обещает не делать докстринг выше (найдено критиком).
        print(f"объяснение не записано: {error}", file=sys.stderr)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", default=None,
                        help="каталог прогона, на котором учится суррогат (по умолчанию из конфига)")
    parser.add_argument("--out", default="runs/policy", help="куда собрать кейс с решениями цикла")
    parser.add_argument("--confirm", default=None,
                        help="каталог посчитанного кейса: заменить прогноз полным расчётом")
    parser.add_argument("--no-tune-rates", action="store_true",
                        help="только остановка убыточных, без подбора режимов")
    parser.add_argument("--surrogate-model", default=None,
                        help="путь к surrogate.joblib; без параметра используется бейзлайн")
    parser.add_argument("--base-schedule", default=None,
                        help="расписание, поверх которого работает цикл; по умолчанию "
                             "SCHEDULE дека. Нужен, чтобы спросить «даёт ли цикл "
                             "что-то сверх уже найденного сценария»")
    parser.add_argument("--multipliers", default=None,
                        help="набор множителей поиска через запятую, например "
                             "0.75,1.0,1.25; по умолчанию — набор из aios.optimise")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument("--search-seed", type=int, default=None,
                        help="seed порядка координат совместного поиска; по умолчанию [run].seed")
    arguments = parser.parse_args(argv)

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(arguments.config)
    multipliers = None
    if arguments.multipliers:
        try:
            multipliers = tuple(float(part) for part in arguments.multipliers.split(","))
        except ValueError:
            print(f"набор множителей не разобран: {arguments.multipliers!r}", file=sys.stderr)
            return 1
        if 1.0 not in multipliers:
            print("в наборе множителей обязан быть 1.0 — режим «как есть» это база",
                  file=sys.stderr)
            return 1
    source = Path(arguments.source) if arguments.source else config.case.run_dir
    search_seed = arguments.search_seed if arguments.search_seed is not None else config.run.seed
    if arguments.confirm:
        confirmed_dir = Path(arguments.confirm)
        written_plan = written_decision_identity(confirmed_dir)
        current_plan = decision_identity(
            source=source,
            config_path=arguments.config,
            tune_rates=not arguments.no_tune_rates,
            multipliers=multipliers,
            search_seed=search_seed,
            base_schedule=arguments.base_schedule,
        )
        if written_plan is not None and written_plan != current_plan:
            print("кейс собран другими параметрами поиска, чем идёт подтверждение:",
                  file=sys.stderr)
            print(f"  в кейсе:        {written_plan}", file=sys.stderr)
            print(f"  в этой команде: {current_plan}", file=sys.stderr)
            print("Передайте ту же команду, которой кейс собирался: параметры поиска, "
                  "конфиг и опорное расписание должны совпасть.", file=sys.stderr)
            return 1
    # плотности по PVT-регионам скважин; пометка печатается всегда — молча
    # посчитать чужой пласт по нашей плотности нельзя (ТЗ п. 4.3)
    deck_fluids = pvt.fluids_by_well(config.case.model_dir)
    print(f"плотности: {deck_fluids.note}")
    case_name = config.case.case_name

    # Проверка до чтения, а не после: без .UNSMRY resdata бросает OSError
    # «Failed to create summary instance», и команда из инструкции падает
    # трассировкой вместо отказа (замер 24.08, Б31 — та же болезнь, что чинили в
    # Б27 и Б30). Отсутствие расчёта — законный исход на машине без симулятора,
    # но узнавать о нём надо строкой.
    if not (source / case_name).with_suffix(".UNSMRY").is_file():
        print(f"нет результата расчёта: {source / case_name}.UNSMRY. Цикл учится на "
              "выгрузке ГГДМ — сначала посчитайте базовый прогон:\n"
              f"  bash scripts/run_flow.sh {source / case_name}.DATA 4 --parsing-strictness=low\n"
              f"  python scripts/export_rows.py {source.as_posix()}", file=sys.stderr)
        return 1

    history = summary.monthly_rows(source / case_name, fluids=config.fluids, well_fluids=deck_fluids.by_well)
    if not history:
        print(f"нет выгрузки в {source}: сначала посчитайте базовый прогон", file=sys.stderr)
        return 1

    assumptions, _pumps = economics.load_norms(config.economics.norms_file)
    norms = Norms.from_assumptions(assumptions)
    schedule = parse_schedule(layout(config.case.model_dir).one("SCHEDULE"))
    # Фонд сверяется с деком всегда: расписание со скважиной, которой в модели
    # нет, симулятор не примет, чей бы файл мы ни взяли за основу.
    #
    # А вот **что** оптимизировать, задаётся отдельно. До 24.08 цикл всегда
    # начинал с исторического расписания дека, и `--from` давал только историю
    # для обучения суррогата. Из-за этого вопрос «даёт ли цикл что-то сверх уже
    # найденного сценария» нельзя было даже задать: поверх заявляемого
    # расписания цикл не запускался ни разу (Б34).
    base_steps = [step for step in schedule.steps
                  if step.date >= config.case.control_start]

    # Расписание, с которого цикл стартует. Опорное при этом **не** подменяется:
    # обученная модель принимает режим множителями к опорному расписанию дека, и
    # подстановка другого опорного меняла бы смысл её входа, а не точку старта
    # (замер 24.08: один и тот же файл — 12 664,9 как кандидат и 11 201,4 как
    # опорное). Поэтому сценарий выражается множителями к деку, и поиск
    # стартует из них — в тех же координатах, в которых модель училась.
    start_multipliers = None
    if arguments.base_schedule:
        base_file = Path(arguments.base_schedule)
        if not base_file.is_file():
            print(f"нет расписания {base_file}", file=sys.stderr)
            return 1
        candidate = [step for step in parse_schedule(base_file).steps
                     if step.date >= config.case.control_start]
        if not candidate:
            print(f"в {base_file} нет шагов с {config.case.control_start}", file=sys.stderr)
            return 1
        start_multipliers = optimise.multipliers_of(base_steps, candidate)
        if not start_multipliers:
            print(f"{base_file} не выражается множителями к расписанию дека: "
                  "общих открытых записей нет", file=sys.stderr)
            return 1
        moved = sum(1 for value in start_multipliers.values() if abs(value - 1.0) > 1e-9)
        print(f"старт:     {base_file} — {len(start_multipliers)} управлений, "
              f"из них отличаются от дека {moved}")

    try:
        selected = for_cycle(
            neural_path=Path(arguments.surrogate_model) if arguments.surrogate_model else None,
            history=history,
            reference_steps=base_steps,
            # плотности по скважинам сюда не передаются намеренно: ни
            # тривиальный суррогат, ни обученный их не принимают — они
            # переводят объёмы в тонны одной плотностью. На вход ЧДД
            # подеквенные плотности идут выше, в summary.monthly_rows, и
            # заявляемое число считается по ним. Ограничение прогноза
            # записано в docs/experiments.md; передавать сюда параметр,
            # который некому применить, — значит лгать сигнатурой.
            fluids=config.fluids,
            # предел дебита кейса — тот же, которым цикл проверяет расписание;
            # без него диагностика прогноза молчала бы про 480 м³/сут на кейсе
            # с верхним насосом до 320 (Н-25)
            max_liquid_rate=liquid_limit(
                economics.load_norms(config.economics.norms_file)[1]),
        )
    except SurrogateLoadError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"суррогат: {selected.name}")
    margin = trust.margin_from_pairs(config.trust.pairs_file,
                                     family=config.trust.pairs_family)
    if margin.calibrated:
        print(f"конформный запас: {margin.value:+.1f} млн руб. "
              f"({margin.samples} пар семейства {config.trust.pairs_family})")
    else:
        print(f"конформный запас: НЕ КАЛИБРОВАН — {margin.reason}")
    started = time.monotonic()

    def note(stage: str) -> None:
        # Отметки этапов с отсчётом от начала: цикл на обученном суррогате идёт
        # десятки минут (замер 05.09 — свыше сорока на этой машине при полном
        # наборе множителей), и без них оператор не отличит работу от зависания.
        print(f"  [{time.monotonic() - started:6.1f} с] {stage}", flush=True)

    result = loop.run_cycle(
        selected.model,
        base_steps,
        norms,
        progress=note,
        control_start=config.case.control_start,
        start_date=config.economics.start_date,
        lookback_months=config.policy.lookback_months,
        known_wells=frozenset(schedule.wells),
        tune_rates=not arguments.no_tune_rates,
        joint_rate_search=selected.joint_rate_search,
        limits=config.limits,
        multipliers=multipliers,
        start_multipliers=start_multipliers,
        margin=margin,
        search_seed=search_seed,
        norms_file=config.economics.norms_file,
        # см. пояснение выше: цикл считает прогноз одной плотностью, а
        # подеквенные идут в выгрузку, из которой берётся заявляемое число
        fluids=config.fluids,
        # Полная выгрузка источника: коридор баланса воды судится тем же
        # окном, что и подтверждение, — по всей истории, а не с начала
        # управления (решение владельца 04.09, Р-204).
        history_rows=history,
    )

    if arguments.confirm:
        confirmed_dir = Path(arguments.confirm)
        written = written_identity(confirmed_dir)
        current = surrogate_identity(arguments.surrogate_model, selected.name)
        if written is not None and written != current:
            print("кейс собран другим суррогатом, чем идёт подтверждение:",
                  file=sys.stderr)
            print(f"  в кейсе:        {written}", file=sys.stderr)
            print(f"  в этой команде: {current}", file=sys.stderr)
            print("прогноз одной модели рядом с фактом другой — та же ловушка, "
                  "что пара «файл от одного прогона, число от другого». "
                  "Передайте тот же --surrogate-model, которым кейс собирался.",
                  file=sys.stderr)
            return 1
        recorded_plan = written_plan_digest(confirmed_dir)
        current_plan = plan_digest(result.steps)
        if recorded_plan is not None and recorded_plan != current_plan:
            print("подтверждается не то расписание, которое считалось:",
                  file=sys.stderr)
            print(f"  в кейсе:        план {recorded_plan}", file=sys.stderr)
            print(f"  в этой команде: план {current_plan}", file=sys.stderr)
            print("между сборкой кейса и подтверждением изменились решения — "
                  "чаще всего это правка optimise/planner/policy/loop, которой "
                  "нет среди сверяемых входов. Отклик ГГДМ относится к прежнему "
                  "расписанию, и подписывать им новое нельзя: пересоберите кейс "
                  "и посчитайте заново.", file=sys.stderr)
            return 1

        rows = summary.monthly_rows(confirmed_dir / case_name, fluids=config.fluids, well_fluids=deck_fluids.by_well)
        if not rows:
            print(f"нет выгрузки в {confirmed_dir}", file=sys.stderr)
            return 1

        problems = summary.sanity_problems(
            rows,
            # Предел дебита — из таблицы ЭЦН нормативов ЭТОГО кейса, а не
            # умолчание 500 от Модели Z: на кейсе с верхним насосом до 320 м³/сут
            # подтверждение пропускало выгрузку с 480 молча (Н-18).
            max_liquid_rate=liquid_limit(
                economics.load_norms(config.economics.norms_file)[1]),
            min_reservoir_pressure=config.limits.min_reservoir_pressure_atm,
            max_reservoir_pressure=config.limits.max_reservoir_pressure_atm,
        )
        if problems:
            print(f"санити-чеки расчёта: нарушений {len(problems)}")
            for line in problems[:10]:
                print(f"  {line}")
            print("подтверждать ЧДД по нефизичной выгрузке нельзя", file=sys.stderr)
            return 1

        # ЧДД базового сценария берётся из полного расчёта, а не из прогноза:
        # сравнивать подтверждённое число с прогнозным нельзя, разойдётся сам
        # предмет сравнения.
        base_npv = economics.compute_npv(
            history, headers=economics.REQUIRED_COLUMNS,
            start_date=config.economics.start_date.isoformat(),
            norms_file=config.economics.norms_file,
        )
        confirmed = loop.confirm(result, rows, config.economics.start_date,
                                 base_rows=history,
                                 # условия кейса пересчитываются на выгрузке ГГДМ,
                                 # а не переносятся от прогноза суррогата (Р-171)
                                 limits=config.limits, fluids=config.fluids,
                                 norms_file=config.economics.norms_file)

        print("--- подтверждение полным расчётом ---")
        print(f"ЧДД базового сценария (ГГДМ):  {confirmed.base_chdd_m:12.1f} млн руб.")
        print(f"ЧДД сценария цикла (ГГДМ):     {confirmed.chdd_m:12.1f} млн руб.")
        # печатается gain_m самого результата, а не своя разность рядом: теперь
        # обе величины в нём из полного расчёта, и расходиться им негде
        print(f"прирост:                       {confirmed.gain_m:+12.1f} млн руб. "
              f"({100.0 * confirmed.gain_m / abs(confirmed.base_chdd_m):+.2f} %)")
        print()
        print(f"прогноз суррогата обещал:      {result.gain_m:+12.1f} млн руб.")
        print(f"ошибка прогноза:               "
              f"{result.gain_m - confirmed.gain_m:+12.1f} млн руб.")
        print()
        # обе величины берутся из эталонного расчётчика и потому заведомо за один
        # период — тот, что дисконтируется. Сумма по всей выгрузке рядом с ними
        # выглядит сопоставимой, но охватывает ещё и годы до начала управления,
        # одинаковые у всех сценариев: сравнение получается заниженным
        confirmed_npv = economics.compute_npv(
            rows, headers=economics.REQUIRED_COLUMNS,
            start_date=config.economics.start_date.isoformat(),
            norms_file=config.economics.norms_file,
        )
        print(f"нефть за период экономики (с {config.economics.start_date}): "
              f"{base_npv.oil_kt:.1f} → {confirmed_npv.oil_kt:.1f} тыс. т")
        print(f"жидкость:                  {base_npv.liquid_kt:.1f} → "
              f"{confirmed_npv.liquid_kt:.1f} тыс. т")
        print(f"закачка:                   {base_npv.injection_km3:.1f} → "
              f"{confirmed_npv.injection_km3:.1f} тыс. м3")
        print(f"подтверждено: {confirmed.confirmed}")

        # Объяснение пересобирается **по подтверждённому** результату. До 04.09
        # `explanation.json` писался только на `--out`, то есть по прогнозу; после
        # `run_all.py` рядом с подтверждённым кейсом лежал файл со словами
        # «ПРОГНОЗ, не подтверждён» — притом что Р-180 обещает «объяснение тем же
        # прогоном» (С6). Числа в объяснении при этом менялись: прогноз обещал
        # одно, ГГДМ ответил другое, и объяснять надо второе.
        explained = _write_explanation(Path(arguments.confirm), confirmed,
                                       selected.name, config)
        print(f"объяснение:   {explained}")

        # Условия кейса на подтверждённой выгрузке — их считает `loop.confirm`
        # по факту ГГДМ, а не по прогнозу (Р-171). Молчать о них здесь нельзя:
        # это последний шаг единой команды, и именно её вывод читает проверяющий.
        if confirmed.case_problems:
            print()
            print(f"УСЛОВИЯ КЕЙСА НАРУШЕНЫ ({len(confirmed.case_problems)}):",
                  file=sys.stderr)
            for line in confirmed.case_problems[:10]:
                print(f"  ! {line}", file=sys.stderr)

        # Ненулевой код при непригодном результате — не строгость, а честность.
        # README велит проверяющему запускать `scripts/run_all.py`, последний шаг
        # которого — этот самый `--confirm`; собственные замеры проекта дают на
        # этом пути −63…−70 % ЧДД против бездействия. Прогон, завершившийся
        # успехом на расписании втрое хуже истории, сообщает проверяющему
        # неправду (Р-191).
        if confirmed.case_problems or confirmed.gain_m <= 0:
            print()
            print("РЕЗУЛЬТАТ НЕПРИГОДЕН: "
                  + ("прирост не положителен" if confirmed.gain_m <= 0
                     else "нарушены условия кейса")
                  + ". Сдавать это расписание нельзя; заявляемым остаётся то, "
                    "что подтверждено (см. docs/submission.md).", file=sys.stderr)
            return 2
        return 0

    print(result.summary())
    print()

    if result.problems:
        print("расписание нарушает ограничения — кейс не собираю", file=sys.stderr)
        return 1

    case = build_case(
        arguments.out,
        config.case.model_dir,
        steps=list(result.steps),
        control_start=config.case.control_start,
        wells=schedule.wells,
        case_name=case_name,
    )
    marker = Path(arguments.out) / "cycle.json"
    marker.write_text(
        json.dumps({
            "surrogate_identity": surrogate_identity(arguments.surrogate_model, selected.name),
            "decision_identity": decision_identity(
                source=source,
                config_path=arguments.config,
                tune_rates=not arguments.no_tune_rates,
                multipliers=multipliers,
                search_seed=search_seed,
                base_schedule=arguments.base_schedule,
            ),
            "plan_md5": plan_digest(result.steps),
            "multipliers": list(multipliers) if multipliers else None,
            "tune_rates": not arguments.no_tune_rates,
            # Система отсчёта прогноза. Без неё сравнение «обещал против факта»
            # невозможно: маркеры, записанные при базе дисконтирования 2007,
            # несопоставимы с фактом, посчитанным при 1991, а по числу этого не
            # видно (Р-187). Источник нужен для той же цели: прирост считается
            # от прогона, с которого цикл стартовал, а не от произвольного.
            "start_date": config.economics.start_date.isoformat(),
            "source_run": Path(source).name,
            "norms_file": str(config.economics.norms_file),
            "forecast_chdd_m": result.chdd_m,
            "forecast_gain_m": result.gain_m,
            "safe_forecast_chdd_m": result.safe_chdd_m,
            "optimism_margin": margin.value,
            "margin_reason": margin.reason,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Плотности, на которых цикл ВЫБИРАЛ решения, до сих пор жили только в
    # прокрутке терминала: rows.csv в каталоге кейса не появляется — его пишет
    # export_rows.py уже по результату полного расчёта. Подпись едет вместе с
    # кейсом на ту машину, где его будут считать (Р-47, хвост).
    fluids_note = deck_fluids.write_case_note(
        arguments.out,
        fluids=config.fluids,
        model_dir=config.case.model_dir,
        forecast_fluids=config.fluids,
    )
    # Слой интерпретируемости — часть единой команды, а не только интерфейса.
    # До 03.09 объяснение рождалось лишь на странице цикла (`aios/web/app.py`),
    # то есть автоматический прогон (п. 3.2.3 ТЗ) слой не задействовал вовсе и
    # артефакта объяснения не оставлял. Объяснение пишется **тем же прогоном**,
    # что и решение, — это требование `CLAUDE.md`, и файл рядом с кейсом
    # доказывает его сам: другие числа туда попасть не могут (Р-180).
    explanation_file = _write_explanation(Path(arguments.out), result, selected.name, config)
    print(f"кейс собран: {case.data_file}")
    print(f"расписание:  {case.schedule_file}")
    print(f"чем считался прогноз: {marker}")
    print(f"на каких плотностях:  {fluids_note}")
    print(f"объяснение решения:   {explanation_file}")
    print()
    print("дальше — полный расчёт, без него прирост остаётся прогнозом:")
    # путь печатается относительным, когда кейс внутри репозитория, и абсолютным,
    # когда снаружи: команда должна копироваться как есть в обоих случаях
    data_path = case.data_file.resolve()
    shown = data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path
    print('  docker run --rm -v "$PWD:$PWD" -w "$PWD" aios \\')
    print(f"      scripts/run_flow.sh {shown} "
          f"{config.run.processes} --parsing-strictness=low")
    print("  " + confirmation_command(
        case_dir=arguments.out,
        source=source,
        config_path=arguments.config,
        no_tune_rates=arguments.no_tune_rates,
        surrogate_model=arguments.surrogate_model,
        base_schedule=arguments.base_schedule,
        multipliers=multipliers,
        search_seed=search_seed,
    ))
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
