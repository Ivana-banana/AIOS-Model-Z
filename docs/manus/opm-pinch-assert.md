# Анализ фактических материалов: OPM Flow 2026.04, `PINCHREG`, `MINPV` и расхождение с tNavigator

## Вывод в одном абзаце

Приложенные логи существенно уточняют картину. Ваш baseline с `PINCHREG` и `--parsing-strictness=low` **не означает отсутствие всех pinch-NNC**: в Flow 2026.04 код `MinpvProcessor` запускается и без global `PINCH`, и по direct-gap ветви способен добавить NNC; это согласуется с PRT-строкой `1 pinch-out connection generated`. Однако **региональные критерии `PINCHREG` действительно не применяются**. Замена на `PINCH 2 /` не является доказанной трансляцией reference case: она включает наиболее разрешающие defaults `GAP`, `PINCHGAP=1e20`, `TOPBOT`, `TOP` и расширила topology до **27 696** NNC, после чего сработал известный defect #505 в `opm-grid` 2026.04. Целевое исправление #935 уже существует, чисто накладывается на исходник `opm-grid release/2026.04/final`, и должно быть первой технической мерой. После него сравнение нужно вести как минимум по трём явно подписанным topology-сценариям, а не как единственное «OPM против tNavigator».

> **Статус 5%.** Данные файлов доказывают функциональное расхождение и crash-path, но пока не доказывают, какая именно часть разницы `5011` против `4771` тыс. т вызвана `PINCHREG`: сейчас неизвестны оба полных record `PINCHREG`, правило reference simulator при отсутствии массива `PINCHNUM`, число/список NNC и их transmissibility в tNavigator. Поэтому 5% — обоснованный **высокий риск совместимости**, но не выделенный количественный effect одной настройки.

## Факты из приложенных материалов и их проверка

| Наблюдение из файлов | Вердикт | Проверка / следствие |
|---|---|---|
| В исходном deck присутствуют `PINCHREG` (два record, threshold 2 m), `GRIDOPTS` item 3 = 3 и `MINPV 1`; `PINCHNUM` отсутствует. | **Подтверждённый input-факт.** | Flow 2026.04 классифицирует `PINCHREG` и `PINCHNUM` как critical unsupported; `--parsing-strictness=low` только понижает error до warning, но не реализует regional criteria.[1] [2] |
| Baseline без global `PINCH` пишет `1 pinch-out connection generated`. | **Согласуется с source; это не противоречие.** | В 2026.04 `processEclipseFormat` всегда вызывает `MinpvProcessor`. При no-PINCH он передаёт `z_tolerance=0` и `nogap=true`, но `max_gap` остаётся `getPinchMaxEmptyGap()` (parser default `1e20`). В direct-gap branch NNC добавляется при vertical gap и `computeGap < max_gap`.[3] [4] |
| Следовательно, `PINCHREG` в baseline «не создаёт региональную связность вовсе». | **Требует уточнения.** | Верно только для **regional PINCHREG semantics**. Неверно как утверждение об отсутствии любого pinch-NNC: наблюдаемый 1 NNC создаётся fallback-ветвью global grid preprocessing, без применения PINCHREG record.[3] [4] |
| После замены на `PINCH 2 /` создано 27 696 NNC и затем assertion в `EntityRep.hpp:120`; одинаково при MPI 1 и 4. | **Высокодостоверный воспроизводимый crash-case.** | `PINCH 2 /` раскрывается как `PINCHTHK=2`, `GAP`, `PINCHGAP=1e20`, `TOPBOT`, `TOP`; GAP разрешает перескок MINPV-deactivated cells шире threshold. Известный issue #505 фиксирует тот же `other_cell=-1` → `EntityRep::setValue()` path.[5] [6] [7] |
| Авария — не эффект MPI partitioning. | **Практически подтверждено для вашего кейса.** | Воспроизводимость на 1 rank исключает необходимость MPI-декомпозиции для trigger. Финально это утверждает только о данном deck/версии, а не обо всех CpGrid/MPI bugs. |
| `PINCH 2 /` является эквивалентом исходного PINCHREG. | **Не подтверждено; использовать это нельзя.** | Для эквивалентности нужны оба полных record и region assignment. Manual определяет `PINCHREG` через регионы `PINCHNUM`, а `GRIDOPTS` задаёт максимум records. Открытых данных о том, как tNavigator применяет два records при отсутствующем PINCHNUM, в приложениях нет.[8] |
| Flow 2026.04 можно безопасно продолжать с `--parsing-strictness=low`. | **Нет для production comparison.** | low переводит critical unsupported в warning. После удаления/замены `PINCHREG` нужно снять этот флаг либо явно использовать `--parsing-strictness=normal`, чтобы не скрывать другие critical unsupported keywords.[2] [9] |

## Что именно ломается в 2026.04

Путь в release source выглядит так: `processEclipseFormat()` создаёт `MinpvProcessor`, собирает `minpv_result.nnc` и помещает пары в `PinchNNC`. При построении topology `buildFaceToCell()` строит `global_to_local`, где отсутствующим локальным ячейкам соответствует `-1`, затем выполняет `other_cell = global_to_local[it->second]` и `cells[cellcount].setValue(other_cell, false)`. `EntityRep::setValue()` требует индекс `>=0`. Следовательно, отрицательный аргумент — не orientation flag, а **ненайденная локальная CpGrid cell** на стороне pinch-NNC.[5] [6]

Issue #505 позднее локализовал root cause: `MinpvProcessor` мог создать NNC, если одна из внешних клеток уже collapsed/нулевой толщины; затем preprocessing исключает её из topology. PR #935 вводит `isCollapsed()` и добавляет `!isCollapsed(...)` в оба места `result.add_nnc`, то есть задаёт policy «collapsed cell — barrier, NNC не создавать».[6] [7]

| Артефакт | Идентификатор | Статус относительно вашего образа 2026.04 |
|---|---|---|
| Release OPM Grid | `release/2026.04/final` → `6221c7b5cf206b1cec19e21966313c78d29d5d83` | Уязвимая release-линия. |
| Issue | `OPM/opm-grid#505` | Closed by #935. |
| Основная правка | `ef3fad537be6b94c35011f4014eb67fcf46be56c` | После 2026.04; отсутствует в вашем release. |
| Тесты | `7b665216985439a72fe908e574c6c2ffe5017e1a` | После 2026.04; absent from release. |
| Merge в master | `f8ebec3d2dce73f5de8deb32129ef47e9e4eb3d9`, 2026-08-04 | Fix присутствует в master. |

### Проверка возможности backport

Я создал отдельный worktree от `release/2026.04/final` и применил оба commit с `git cherry-pick --no-commit`. **Оба применились без конфликта и без whitespace errors.** Рабочая правка меняет только `opm/grid/MinpvProcessor.cpp` (+14/−1); test commit добавляет 34 строки в `tests/test_minpvprocessor.cpp`. Это сильное свидетельство, что узкий backport технически несложен. Однако это ещё не готовый Docker image: необходимо пересобрать совместимый OPM stack, выполнить тесты и зарегистрировать exact binary SHA.

## Что означают ваши текущие NNC counts

| Сценарий | Наблюдение | Корректная интерпретация |
|---|---:|---|
| Legacy: `PINCHREG` + `--parsing-strictness=low`, no global `PINCH` | 1 NNC | Это **не** реализация двух региональных records. Это отдельная no-PINCH/direct-gap ветвь Flow. |
| Пробный global `PINCH 2 /` на Flow 2026.04 | 27 696 NNC, abort | Это sensitivity с `GAP` и unlimited gap, а не verified proxy tNavigator. Число показывает, что PINCH/MINPV topology действительно может быть крупным источником production delta. |
| Patched Flow + исходный global `PINCH` | ещё не измерено | Первый meaningful candidate для stable OPM topology. Collapsed cells будут barriers. |
| Patched Flow + `PINCHGAP=0` | ещё не измерено | Диагностический extreme: выключает все PINCH NNC в этой логике. |

## Приоритетный план действий

| Приоритет | Гипотеза / цель | Основание | Точная строка или команда | Ожидаемый результат | Вероятность успеха |
|---:|---|---|---|---|---:|
| 1 | Устранить именно известный abort без изменения полного deck в обход причины. | #505/#935; fix чисто накладывается на Grid 2026.04.[6] [7] | Backport: `git cherry-pick ef3fad537be6b94c35011f4014eb67fcf46be56c 7b665216985439a72fe908e574c6c2ffe5017e1a` на `release/2026.04/final`, затем пересобрать coherent image; альтернативно — image, собранный после `f8ebec3d`. | `PINCH 2 /` перестаёт падать именно из-за collapsed outer cell; такая cell становится barrier. | Очень высокая |
| 2 | Получить корректный baseline без регионального keyword masking. | `PINCHREG` unsupported, low скрывает critical keywords.[1] [2] | После физического удаления `PINCHREG`/`PINCHNUM`: **не передавать** `--parsing-strictness=low`; либо `flow RUN.DATA --parsing-strictness=normal`. | Другие critical unsupported keywords снова видны. Сохранить как labelled OPM-no-regional baseline. | Высокая |
| 3 | Отделить вклад **всех** pinch-NNC от иных различий OPM/tNavigator. | В обоих add-NNC conditions используется strict `computeGap < max_gap`, а `computeGap >= 0`.[3] [4] | Полный record: `PINCH\n  2.0  NOGAP  0.0  TOPBOT  TOP /` | На patched Flow не будут генерироваться pinch-NNC. Это diagnostic lower-connectivity bound, не тождество tNavigator. | Очень высокая для topology isolation |
| 4 | Получить патченную global-PINCH sensitivity, но не выдавать её за reference-equivalent. | `PINCH 2 /` = GAP / 1e20 / TOPBOT / TOP.[4] | Только после #935: `PINCH\n  2.0  GAP  1.0E20  TOPBOT  TOP /` | Стабильный аналог уже сделанного stress-test; даст upper-connectivity OPM sensitivity. 1.0E20 необходимо оставить в deck units. | Высокая для запуска; низкая для equivalence tNavigator |
| 5 | Ограничить только MINPV-deactivated cells, если они толще 2 m. | GAP vs NOGAP documented through MINPV cells.[8] | `PINCH\n  2.0  NOGAP  1.0E20  TOPBOT  TOP /` | Меньше NNC, чем для GAP, только если relevant low-PV cells шире 2 m. Zero-thickness/collapsed crash без #935 этим не лечится. | Средняя |
| 6 | Проверить `PINCHCAL=ALL` только как физически мотивированную sensitivity. | ALL suppresses NNC when interval includes zero `PERMZ`/`MULTZ`; requires permeability input.[4] [8] | `PINCH\n  2.0  NOGAP  <ваш_физический_PINCHGAP>  ALL  TOP /` | Имеет смысл, только если record tNavigator действительно ALL и `PERMZ`/`MULTZ` физически заданы. Не использовать искусственный zero multiplier как fix. | Низкая без reference data |

### Почему я не предлагаю «точный global PINCH вместо PINCHREG»

Сейчас известны лишь порог 2 m, число records и `GRIDOPTS=3`. Не известны значения `PINCHOPT`, `PINCHGAP`, `PINCHCAL`, `PINCHMUL` обоих records. Кроме того, не найдено открытого первоисточника tNavigator, который определяет, как именно он обрабатывает два `PINCHREG` records без явного массива `PINCHNUM`. Строка `PINCH 2 /` поэтому допустима исключительно как **синтаксическая OPM sensitivity**, а не как migration mapping.

Минимальный набор данных, который нужно извлечь из reference side до утверждения equivalence:

| Требуемый артефакт | Зачем нужен |
|---|---|
| Оба полных record после `PINCHREG` с `/` terminator. | Восстановить option 2–5 и определить, был ли `PINCH 2 /` вообще близок к record 1. |
| Полный `GRIDOPTS` record, `PINCHNUM` из всех INCLUDE и default/summary reference parser. | Проверить, не находится ли PINCHNUM в include и как reference назначает regions. |
| tNavigator version/build и PRT/log вокруг grid preprocessing. | Установить фактический algorithm/version, а не предполагать Eclipse compatibility. |
| NNC report: count, cell pairs, transmissibilities до первого timestep. | Сравнить topology, а не только 31-летний FOPT. |
| Одинаковые `FOPT`, `FWPT`, `FGPT`, `FWCT` series и well cumulatives. | Локализовать момент divergence и связать его с affected wells/layers. |

## Оценка setup из приложенного файла

Setup содержит полезные operational findings: полностью завершённый run, foreground WSL/Docker запуск, `--shm-size=2g`, filesystem path consistency и исключение оборванных `.UNSMRY` из датасета. Они важны для reproducibility, особенно потому что незавершённый MPI run может оставить внешне читаемый, но усечённый summary файл.

Для этой конкретной задачи нужно внести три изменения в experiment harness. Во-первых, `--parsing-strictness=low` должен быть разрешён только в legacy baseline c original PINCHREG и фиксироваться в имени run; в translated/production scenarios он запрещён. Во-вторых, каталог результатов должен включать бинарную версию и topology label, например `runs/opm202604-p935/pinch-gap/`, `runs/opm202604-p935/pinch-off/`, `runs/opm202604-p935/no-regional/`. В-третьих, parser результатов обязан отвергать summary без нормального exit marker/expected last report date, а не только полагаться на наличие `.UNSMRY`.

## Рекомендуемая матрица из четырёх прогонов

| Run label | Бинарь | Deck transformation | Parsing strictness | Что изолирует |
|---|---|---|---|---|
| `L-legacy` | Flow 2026.04 current image | Исходный PINCHREG | `low` | Текущий сравниваемый baseline; **не** regional-equivalent. |
| `P-no-global` | Flow with #935 | PINCHREG/PINCHNUM удалены; no `PINCH` | `normal` | Сохраняет no-PINCH fallback (ваш observed 1 NNC), исключает masking. |
| `P-pinch-off` | Flow with #935 | `PINCH 2.0 NOGAP 0.0 TOPBOT TOP /` | `normal` | Нулевая pinch-NNC topology; lower-connectivity diagnostic. |
| `P-gap-unlimited` | Flow with #935 | `PINCH 2.0 GAP 1.0E20 TOPBOT TOP /` | `normal` | Ваш `PINCH 2 /` after crash fix; upper-connectivity sensitivity. |

Если tNavigator records окажутся, например, `NOGAP` с finite `PINCHGAP`, добавьте пятый run с **точно теми же five fields**, но продолжайте называть его `OPM-global-proxy`, пока NNC pairs/transmissibilities reference side не будут сопоставлены.

## References

[1]: https://github.com/OPM/opm-simulators/blob/21daa7b56d1081d97a084613ca289867e2c0e70b/opm/simulators/utils/UnsupportedFlowKeywords.cpp#L475-L481 "Flow 2026.04: PINCHNUM and PINCHREG are critical unsupported"
[2]: https://github.com/OPM/opm-simulators/blob/21daa7b56d1081d97a084613ca289867e2c0e70b/opm/simulators/flow/KeywordValidation.cpp#L110-L120 "Flow 2026.04: behaviour of --parsing-strictness=low"
[3]: https://github.com/OPM/opm-grid/blob/6221c7b5cf206b1cec19e21966313c78d29d5d83/opm/grid/cpgrid/processEclipseFormat.cpp#L179-L230 "OPM Grid 2026.04: MINPV/PINCH preprocessing is invoked and passes no-PINCH settings"
[4]: https://github.com/OPM/opm-grid/blob/6221c7b5cf206b1cec19e21966313c78d29d5d83/opm/grid/MinpvProcessor.cpp#L44-L57 "OPM Grid 2026.04: gap calculation" 
[5]: https://github.com/OPM/opm-grid/blob/6221c7b5cf206b1cec19e21966313c78d29d5d83/opm/grid/cpgrid/processEclipseFormat.cpp#L995-L1140 "OPM Grid 2026.04: global-to-local mapping and buildFaceToCell pinch NNC path"
[6]: https://github.com/OPM/opm-grid/issues/505 "OPM Grid issue #505"
[7]: https://github.com/OPM/opm-grid/pull/935 "OPM Grid PR #935"
[8]: https://github.com/OPM/opm-reference-manual/blob/release/2026-04/rev-0/parts/chapters/subsections/6.3/PINCH.fodt "OPM Flow Reference Manual 2026.04 — PINCH"
[9]: https://github.com/OPM/opm-simulators/blob/21daa7b56d1081d97a084613ca289867e2c0e70b/opm/simulators/utils/readDeck.cpp#L530-L550 "Flow 2026.04: accepted parsing-strictness values"
[10]: https://github.com/OPM/opm-reference-manual/blob/release/2026-04/rev-0/parts/chapters/subsections/6.3/PINCHREG.fodt "OPM Flow Reference Manual 2026.04 — PINCHREG relationship to PINCHNUM and GRIDOPTS"
