# OPM Flow: PINCH и PINCHREG — проверяемая сводка различий с Eclipse/tNavigator

**Автор:** Manus AI  
**Дата исследования:** 21 августа 2026 г.  
**Срез исходного кода:** `opm-grid` `685ce0c4626c92104852293dbe315acc9b6687a4`, `opm-simulators` `e4cdf79c2114cee531f47b56131dfbd68077e682`, `opm-common` `df76c5255222f56b08f38389138815376c4ecb74`.  
**Документационная база:** OPM Flow Reference Manual 2025.10 Rev-0 (11 декабря 2025 г.).

## Вывод в двух абзацах

**PINCH в OPM Flow поддерживается как глобальный механизм** создания вертикальных NNC через pinch-out; документированы параметры `PINCHTHK`, `PINCHOPT`, `PINCHGAP`, `PINCHCAL` и `PINCHMUL`. В частности, `PINCHCAL=ALL` предписывает гармоническое усреднение Z-transmissibility всех ячеек между двумя активными ячейками, а `PINCHMUL=ALL` учитывает минимум `MULTZ` по верхней активной и неактивным ячейкам колонки. Это непосредственно меняет связность и transmissibility, то есть физику потока. [1]

**Региональная схема PINCHREG/PINCHNUM в обычном режиме OPM Flow не является «молча игнорируемой»: она критически неподдерживаема.** Текущая таблица валидатора помечает `PINCHNUM`, `PINCHOUT`, `PINCHREG` и `PINCHXY` как critical; при нормальной parsing strictness Flow выдаёт ошибку и завершает расчёт. Только при явном `--parsing-strictness=low` critical-ошибка понижается до предупреждения; это не означает реализации регионального алгоритма и создаёт физически небезопасный deck. [1] [2] [3]

> **Главная практическая граница.** Открытые первоисточники позволяют строго сопоставить OPM с опубликованной Eclipse-терминологией и зафиксировать собственные исторические расхождения OPM. Они **не позволяют** утверждать «OPM отличается от tNavigator так-то» по каждому режиму PINCH/PINCHREG: официальные manuals и release notes tNavigator доступны через клиентский портал, а публичной спецификации этих ключевых слов компания не публикует. [4]

## 1. Методика и пределы доказательств

В исследование вошли официальное руководство OPM, опубликованные release notes, официальная рассылка OPM, публичный GitHub issue, история коммитов и текущий открытый исходный код. Для Eclipse найденная сторонняя копия *Technical Description 2014.1* не использована как доказательный первоисточник: это не официальный канал SLB. Основная научная статья об OPM Flow описывает состояние 2019.10 и не содержит самостоятельного разбора `PINCH`/`PINCHREG`; она приведена только для контекста. [5]

Ниже термин **«меняет физику»** означает, что ключ влияет на набор фаз/компонентов, начальное состояние, сеточную связность, NNC, transmissibility или рабочие скважинные граничные условия. Параметры печати, файлов, диагностики, распараллеливания и численного контроля выделены отдельно: они не задают физическую модель, хотя могут влиять на сходимость, дискретизацию по времени или воспроизводимость численного прогона.

| Обозначение в таблицах | Строгий смысл |
|---|---|
| **Физика: да** | Меняется постановка/связность/проводимость/фазовый или компонентный состав/рабочее управление скважиной. |
| **Физика: нет** | Описание OPM прямо говорит, что ключ не влияет на results, либо ключ относится лишь к отчётности, файлам, сообщениям или инструментальным настройкам. |
| **Не классифицировать как игнорируемый** | Ключ присутствует в некритической таблице валидатора, но отдельный обработчик или состояние deck его читает. Без исполнения конкретной версии Flow нельзя заключать, что данные отброшены. |
| **Критически неподдерживаемый** | В обычной strictness Flow не продолжает расчёт; это не «тихое игнорирование». |

## 2. Что именно делает PINCH в OPM Flow

Руководство OPM определяет `PINCH` как глобальный control генерации вертикальных NNC из-за pinch-out, в отличие от регионального `PINCHREG`+`PINCHNUM`. Исходный код grid-процессинга вызывает обработку `MINPV`/`PINCH`, материализует сформированные pinch-NNC и имеет отдельный путь вычисления transmissibility для `PINCHCAL=ALL`. [1] [6]

| Поле PINCH | Документированная семантика OPM | Физический эффект |
|---|---|---|
| `PINCHTHK` | Порог вертикальной толщины: NNC создаётся через неактивные ячейки тоньше порога. | **Да** — меняет наличие вертикального пути потока. [1, §6.3.175, pp. 670–671] |
| `PINCHOPT` (`GAP`/`NOGAP`) | Управляет NNC для ячеек, деактивированных `MINPV`; `GAP` допускает переход через такие ячейки, `NOGAP` требует соблюдения `PINCHTHK`. | **Да** — меняет барьер/связность через `MINPV`. [1] |
| `PINCHGAP` | Максимальная «пустая» толщина между соседними слоями, при которой может существовать ненулевая transmissibility. | **Да** — меняет существование NNC. [1] |
| `PINCHCAL` (`TOPBOT`/`ALL`) | `TOPBOT`: transmissibility из half-cell Z-transmissibility двух активных ячеек. `ALL`: гармоническое среднее Z-transmissibility всех ячеек между ними. | **Да** — меняет величину проводимости NNC. [1] |
| `PINCHMUL` (`TOP`/`ALL`) | При `MULTZ`: `TOP` использует верхнюю активную ячейку; `ALL` использует минимум `MULTZ` верхней и неактивных ячеек. Если `PINCHCAL=ALL`, `PINCHMUL` принудительно становится `TOP`. | **Да** — меняет проводимость NNC. [1] |

### Подтверждённая история различий и исправлений OPM

| Период / источник | Подтверждённый факт | Значение для сравнения |
|---|---|---|
| До 2020.10 | В 2020.10 OPM добавил поддержку обработки `MULTZ` в pinched-out column при `PINCHMUL=ALL`. | Старые Flow до этой доработки нельзя считать эквивалентными по этому режиму. [7] |
| 2021-04, issue #3145 | В issue зафиксировано, что Flow создавал pinch connections с default parameters при отсутствии `PINCH`; автор ожидал неизменного давления изолированных нижних слоёв. Issue остаётся открыт и не содержит ссыланного closing PR. | Это подтверждённый исторический риск; сам issue нельзя использовать как описание текущего исправленного поведения. [8] |
| 2021-06, commit `59ba192` | До фикса ячейки удалённых слоёв с collapsed cells между ними соединялись без `PINCH`; сообщение коммита прямо говорит, что это отличалось от reference simulator. Затем связь стала создаваться только при наличии `PINCH`. | Это наиболее прямое опубликованное доказательство прежнего расхождения OPM с reference simulator в **no-PINCH** случае. [9] |
| 2021.10 release notes / commit `b8bdcb1` | Релиз сообщает «Fix grid generation without PINCH»; commit уточняет корректную топологию boundary faces без PINCH. | Для практики обязательна проверка версии: результаты pre-2021.10 и после фикса могут отличаться. [10] [11] |
| 2023-04 | Commit `3d144ce` объявляет `PINCHOPT=NOGAP` fully supported; commit `553e568` добавляет option 3 `MAX_EMPTY_GAP`. | Для ранних версий нельзя предполагать идентичность по `NOGAP` и `PINCHGAP`. [12] [13] |
| 2023-04 → 2024-10 | В 2023 временно добавлялся явный throw для неподдержанного `PINCHCAL=ALL`, поскольку корректное гармоническое усреднение ещё не применялось; последующие фиксы охватили zero-permeability, а 2024.10 официально сообщает коррекцию transmissibility при option 4 `ALL`. | `PINCHCAL=ALL` — наиболее версионно-чувствительная комбинация; обязательны run-to-run regression checks против целевого симулятора. [14] [15] [16] |
| 2025-09 / 2026-07 | В историю вошли применение regional multiplier к PINCH NNC и запрет PINCH-NNC при collapsed ячейке; последний фикс прямо отмечает, что такая ячейка становится barrier. | Для свежих веток изменяются детали transmissibility и обработка вырожденной геометрии. [17] [18] |

Текущий source-path выполняет предварительную обработку даже когда ни `MINPV`, ни `PINCH` не заданы, но при неактивном `PINCH` передаёт нулевой порог и режим `nogap`; исторические commits выше показывают, что итоговое topology-правило при отсутствующем PINCH исправлялось неоднократно. Поэтому исходный факт «code path executes» **не равен** утверждению, что современный Flow создаёт NNC без PINCH. Для критичных моделей нужен regression deck именно на применяемом бинарнике. [6] [9] [11]

## 3. PINCHREG и PINCHNUM: заявленная семантика против статуса Flow

Справочник 2025.10 описывает `PINCHREG` как региональный аналог `PINCH`: `PINCHNUM` назначает номер региона каждой ячейке, а `PINCHREG` задаёт `NRPINC` записей с порогом, `GAP/NOGAP`, gap и вариантами расчёта transmissibility. В том же справочнике для `PINCHNUM`, `PINCHOUT`, `PINCHREG` и `PINCHXY` есть формулировка «not supported … would change results … simulation will be stopped». В разделе `PINCHREG` также осталась противоречащая ей фраза об автоматической генерации connections. [1, §§6.3.176–6.3.179, pp. 672–676]

**Для фактического статуса приоритетен валидатор и исполняемый код:** все четыре ключа записаны как critical unsupported. Валидатор печатает, что Flow terminates due to unsupported critical keywords, и разъясняет, что это keywords which would change simulator results if supported. Следовательно, противоречие в текстовом разделе PINCHREG — это дефект/несогласованность документации, а не доказательство поддержки. [2] [3]

| Ключевое слово Eclipse-формата | Нормальная strictness Flow | Физика, если ключ задан в reference simulator | Практическая оценка |
|---|---|---|---|
| `PINCHNUM` | **Critical unsupported: остановка.** | **Да:** задаёт региональное назначение для pinch criteria. | Не удалять и не игнорировать при миграции: требуется преобразование в эквивалентную глобальную/явную NNC-схему либо согласование модели. [1] [2] [3] |
| `PINCHREG` | **Critical unsupported: остановка.** | **Да:** region-specific NNC threshold/gap/transmissibility. | Главная документированная функциональная разница Flow относительно deck, использующего региональный pinch. [1] [2] [3] |
| `PINCHOUT` | **Critical unsupported: остановка.** | **Да:** включает вертикальные NNC с fixed threshold 0.001. | Нужно заменить на явно эквивалентный `PINCH`, если это соответствует требуемой семантике. [1] [2] |
| `PINCHXY` | **Critical unsupported: остановка.** | **Да:** управляет areal pinch-out options. | Не понижать strictness без отдельной физической верификации. [1] [2] |

## 4. tNavigator и Eclipse: что подтверждено и что нельзя утверждать

Официальная страница tNavigator прямо указывает, что user manuals и software release updates размещены в client support portal. Публичного текста разделов `PINCH`, `PINCHREG`, тестов совместимости либо changelog по этим ключам на сайте не опубликовано. [4]

| Сравнительное утверждение | Статус | Первичный источник / ограничение |
|---|---|---|
| Eclipse-совместимый deck с `PINCHREG`/`PINCHNUM` не имеет штатной региональной реализации в OPM Flow. | **Подтверждено для OPM Flow**. | OPM manual и critical validator. [1] [2] [3] |
| Исторически OPM соединял collapsed cells без `PINCH`, тогда как commit сравнивает это поведение с reference simulator, который не создавал face. | **Подтверждено для исторической версии OPM.** | Commit `59ba192`. [9] |
| tNavigator реализует PINCH/PINCHREG точно как Eclipse или именно так-то отличается от Eclipse. | **Не подтверждено открытым первоисточником.** | Официальные детали находятся за client portal. [4] |
| Текущая версия OPM Flow полностью совпадает с конкретной версии Eclipse/tNavigator для всех геометрических edge cases. | **Не подтверждено.** | Публичная история содержит многолетние фиксы no-PINCH, NOGAP, MAX_EMPTY_GAP, `PINCHCAL=ALL`, collapsed cells. [7] [9] [12] [13] [15] [18] |

## 5. Ключевые слова, которые Flow «молча игнорирует»: корректная трактовка

Фраза «молча игнорирует» нуждается в уточнении. В `UnsupportedFlowKeywords.cpp` есть **64 некритические** записи (`critical=false`). В стандартной strictness они должны быть **сообщены как warnings**, а не быть молчаливыми; только критические ключи по умолчанию превращаются в error. [2] [3]

Кроме того, наличие имени в этой таблице **само по себе не доказывает**, что Flow отбрасывает все его данные: для ряда Eclipse-300/experimental ключей отдельные `opm-common` handlers или state readers их разбирают. Поэтому ниже приведён именно полный аудит 64 некритических записей, а не небезопасное утверждение, что все они «тихо игнорируются».

### 5.1 Документированно не меняют физическую модель при игнорировании

Следующие строки относятся к output, логированию, диагностике, рестартам, численным/ресурсным настройкам или к redundant option. Руководство OPM для большинства из них прямо использует фразу **«has no effect on the results so it will be ignored»**; для группы отчётности это также следует из назначения ключа. [1] [2]

| Ключ | Назначение по OPM manual | Физика | Статус и источник |
|---|---|---|---|
| `AITS` | Intelligent time stepping on. | Нет; временная стратегия. | Некритический; manual: no effect. [1, §5.3.3] [2] |
| `AITSOFF` | Intelligent time stepping off. | Нет; временная стратегия. | Некритический; manual: no effect. [1, §5.3.4] [2] |
| `BIGMODEL` | Retired big-model option. | Нет. | Некритический; retired option. [1, §5.3.9] [2] |
| `COORDSYS` | Multiple-grid/radial-grid coordinate options. | Нет для поддерживаемого Flow scope; multiple grids не поддержаны. | Некритический; manual прямо говорит no effect. [1, §6.3.23] [2] |
| `DEBUG` | Выбор данных debug-file. | Нет. | Некритический; manual: no effect. [1, §6.3.28] [2] |
| `ECHO` | Echo input в print file. | Нет. | Некритический output control. [1] [2] |
| `EPSDBGS` | End-point-scaling debug для нескольких ячеек. | Нет. | Некритический; manual: no effect. [1, §8.3.58] [2] |
| `EPSDEBUG` | End-point-scaling debug для отдельных ячеек. | Нет. | Некритический; manual: no effect. [1, §8.3.59] [2] |
| `EXCEL` | Формат RSM для spreadsheet. | Нет. | Некритический; manual: no effect. [1, §11.3.9] [2] |
| `EXTRAPMS` | Предупреждения extrapolation. | Нет. | Некритический message control. [1] [2] |
| `FIPOWG` | Дополнительный отчёт FIP. | Нет. | Некритический; валидатор говорит report not available. [2] |
| `FULLIMP` | Fully implicit option. | Нет: Flow использует иную numerical scheme; ключ redundant. | Некритический; manual прямо говорит, что в Flow нет эффекта. [1, §5.3.50] [2] |
| `GDORIENT` | Grid orientation для post-processing. | Нет. | Некритический; manual: no effect. [1, §6.3.78] [2] |
| `MEMORY` | Memory-management control. | Нет. | Некритический runtime control. [2] |
| `MESSAGE` | Message control. | Нет. | Некритический. [2] |
| `MESSOPTS` | Severity message control. | Нет. | Некритический; manual: no effect. [1] [2] |
| `MONITOR` | Monitoring output. | Нет. | Некритический output control. [2] |
| `MSGFILE` | Message-file output. | Нет. | Некритический output control. [2] |
| `NEXT` | Maximum next timestep (alias `NEXTSTEP`). | Нет в модели; numerical timestep control. | Некритический; manual: no effect. [1, §12.3.163] [2] |
| `NOECHO` | Disable input echo. | Нет. | Некритический output control. [1] [2] |
| `NOWARNEP` | Disable endpoint-scaling warnings. | Нет. | Некритический warning control. [1, §8.3.195] [2] |
| `NOINSPEC` | Disable INIT index-file output. | Нет. | Некритический file-output control. [1] [2] |
| `NORSSPEC` | Disable RESTART index-file output. | Нет. | Некритический file-output control. [1] [2] |
| `NOWARN` | Disable warnings. | Нет. | Некритический warning control. [1] [2] |
| `NSTACK` | Stack length iterative linear solver. | Нет в физической постановке; numerical resource control. | Некритический. [1] [2] |
| `OUTSOL` | Retired restart-file output control. | Нет. | Некритический; manual: no effect. [1, §10.3.65] [2] |
| `PARAOPTS` | Parallel-run options. | Нет. | Некритический; manual: no effect. [1, §6.3.165] [2] |
| `PARALLEL` | Parallel-run configuration. | Нет. | Некритический runtime control. [2] |
| `RPTCPL` | Coupled-simulation report. | Нет. | Некритический reporting. [1, §5.3.124] [2] |
| `RPTGRIDL` | LGR GRID-section reporting. | Нет. | Некритический; manual: no effect. [1, §6.3.190] [2] |
| `RPTHM` | History-match reporting. | Нет. | Некритический reporting. [2] |
| `RPTHMG` | Group history-match-gradient reporting. | Нет. | Некритический; manual: no effect. [1, §12.3.203] [2] |
| `RPTHMD` | History-match-gradient output options. | Нет. | Некритический reporting. [1, §5.3.125] [2] |
| `RPTHMW` | Well history-match-gradient reporting. | Нет. | Некритический; manual: no effect. [1, §12.3.204] [2] |
| `RPTGRID` | GRID-section reporting. | Нет. | Некритический; manual: no effect. [1, §6.3.189] [2] |
| `RPTINIT` | INIT-file reporting. | Нет. | Некритический; manual: no effect. [1, §6.3.191] [2] |
| `RPTISOL` | Isolated-reservoir reporting. | Нет. | Некритический; manual: no effect. [1, §6.3.192] [2] |
| `RPTPROPS` | PROPS-section reporting. | Нет. | Некритический; manual: no effect. [1, §8.3.267] [2] |
| `RPTREGS` | REGIONS-section reporting. | Нет. | Некритический; manual: no effect. [1, §9.3.62] [2] |
| `RPTSOL` | SOLUTION-section reporting. | Нет. | Некритический output control. [1, §10.3.80] [2] |
| `RPTRUNSP` | Run-summary print control. | Нет. | Некритический reporting. [2] |
| `RPTSMRY` | Summary-list report. | Нет. | Некритический; manual: no effect. [1, §11.3.30] [2] |
| `RSSSPEC` | Restart specification output/index control. | Нет. | Некритический output control. [2] |
| `SAVE` | SAVE file for fast restarts. | Нет. | Некритический; manual: no effect. [1, §12.3.209] [2] |
| `SOLVDIRS` | Linear-solver principal directions. | Нет в постановке; solver tuning. | Некритический; manual: no effect. [1, §6.3.203] [2] |
| `TUNINGH` | History-match convergence tuning. | Нет в постановке; numerical tuning. | Некритический; manual: no effect. [1, §12.3.233] [2] |
| `TUNINGL` | LGR convergence tuning. | Нет в постановке; numerical tuning. | Некритический; manual: no effect. [1, §12.3.234] [2] |
| `TUNINGS` | Individual-LGR convergence tuning. | Нет в постановке; numerical tuning. | Некритический; manual: no effect. [1, §12.3.235] [2] |
| `UNIFOUTS` | Unified summary-file format. | Нет. | Некритический output control. [1, §5.3.156] [2] |
| `UNIFSAVE` | Unified SAVE-file format. | Нет. | Некритический output control. [1, §5.3.157] [2] |
| `WARN` | Enable warnings. | Нет. | Некритический warning control. [1] [2] |
| `WPOTCALC` | Well-potential calculation for stopped/shut wells. | Нет по manual; не применяется Flow. | Некритический; manual прямо говорит no effect. [1, §5.3.168] [2] |
| `ZIPP2OFF` | Disable commercial automatic timestep algorithm. | Нет в постановке; timestep control. | Некритический. [1, §12.3.362] [2] |
| `ZIPPY2` | Enable commercial automatic timestep algorithm. | Нет в постановке; timestep control. | Некритический. [1, §12.3.363] [2] |

### 5.2 Некритическая запись валидатора есть, но называть такие ключи «игнорируемыми» нельзя

Ниже приведены оставшиеся 14 из 64 записей `critical=false`. Для них обнаружен отдельный parser/state handler, explicit flow-side warning по части параметров или существенная связь с compositional/stream model. Они требуют проверки на **конкретной версии бинарника и конкретном deck**; исключать их до моделирования нельзя. [2] [19]

| Ключ | Почему нельзя отнести к безопасно игнорируемым | Физика |
|---|---|---|
| `COMPS` | `Runspec` читает число компонентов; release 2024.10 заявляет initial experimental compositional support. | **Да** — компонентная постановка. [16] [19] |
| `GASWAT` | `Runspec` трактует его как активацию и GAS, и WATER; manual 2025.10 прямо называет его OPM Flow keyword. | **Да** — набор фаз. [1, §5.3.53] [19] |
| `NCOMPS` | Manual определяет число compositional components; таблицы проверяют согласованность с `COMPS`. | **Да** — размер компонентной системы. [1, §8.3.192] [19] |
| `SOLID` | Eclipse-300 keyword; в текущем аудите найден parser specification, но не найдено достаточного открытого evidence полной flow-side реализации. | **Потенциально да**; не считать безопасно отброшенным. [2] [19] |
| `WELLSTRE` | Есть schedule handler, читающий stream name и composition. | **Да** для compositional injection. [19] |
| `WINJGAS` | Есть schedule handler; исходник явно отмечает, что items `MAKEUPGAS` и `STAGE` ещё не поддержаны. | **Да** для injection stream; поддержка частичная. [19] |
| `ZMFVD` | `TableManager` читает и валидирует composition-versus-depth tables и требует `COMPS`. | **Да** — compositional initialisation. [19] |
| `WSEGITER` | Schedule читает multi-segment-well iteration settings; Flow warning отдельно сообщает, что `MXWSIT` не supported. | **Нет в физике, но численно значимо**; поддержка не полная. [19] [20] |
| `FIELDSEP` | Некритическая запись есть, но в открытом manual/source-audit не найдено достаточного прямого описания текущей Flow-side обработки. | **Не подтверждено**; не относить автоматически к no-effect. [2] |
| `HWELLS` | Некритическая запись есть, но достаточного публичного evidence текущей Flow-side семантики в данном срезе не найдено. | **Не подтверждено**; нужна deck-проверка. [2] |
| `RPTCPL`–`RSSSPEC` (см. точные строки в §5.1) | Это output/reporting ключи; отдельные name-варианты и алиасы могут отсутствовать в текстовом manual, но остаются некритическими в валидаторе. | **Нет**, если используются только как reporting controls. [2] |
| `RPTSOL` | Manual описывает output/restart reporting; не устанавливает физические свойства. | **Нет.** [1, §10.3.80] [2] |
| `RPTRUNSP` | Некритический reporting control в валидаторе; отдельного описания в manual 2025.10 не локализовано. | **Нет, по назначению категории;** но версию формата output следует проверять. [2] |
| `NCOMPS`/`ZMFVD` в black-oil deck без `COMPS` | Наличие имени в таблице валидатора не заменяет применимость: parser проверяет согласованность compositional tables. | **Да при compositional run;** иначе ключ не должен использоваться как «молчаливый». [19] |

## 6. Практический чек-лист миграции Eclipse/tNavigator deck в Flow

| Проверка | Почему обязательна | Рекомендуемое действие |
|---|---|---|
| Найти `PINCHREG`, `PINCHNUM`, `PINCHOUT`, `PINCHXY`. | В обычном режиме Flow прекратит расчёт; механизм физически значим. | Не использовать `--parsing-strictness=low` как решение. Подготовить эквивалентную модель с global `PINCH`, explicit NNC либо согласованным upscaling и подтвердить на тестовом кейсе. [1] [2] [3] |
| Зафиксировать версию Flow. | Поведение no-PINCH, NOGAP, MAX_EMPTY_GAP и `PINCHCAL=ALL` менялось. | Записать `flow --version`, commit/package release и запустить один и тот же deck на целевом и reference binary. [7] [9] [12] [13] [15] |
| Выделить `PINCHCAL=ALL`, `PINCHMUL=ALL`, `MULTZ`/`MULTZ-`, `TRANZ`, `MULTREGT`. | Это режимы с историей фиксов transmissibility и multiplier application. | Сравнить количество PINCH-NNC и сами transmissibility до запуска прогноза; сохранять регрессионные `BPR`, rates и NNC report. [1] [15] [17] |
| Проверить collapsed/zero-volume cells и `MINPV/MINPVV`. | Именно эти edge cases породили no-PINCH и barrier fixes. | Сделать маленький 1×1×K regression deck и тесты с/без PINCH, GAP/NOGAP, zero permeability и collapsed cells. [8] [9] [18] |
| Не удалять «некритические» Eclipse-300 keys механически. | Часть ключей при `critical=false` всё же читается compositional/stream parser’ом. | Сначала классифицировать deck по `COMPS`, GASWAT, WELLSTRE, WINJGAS, ZMFVD; затем проверить лог Flow и results. [16] [19] |

## 7. Список источников

[1]: https://github.com/OPM/opm-reference-manual/releases/download/release%2F2025-10%2Frev-0/OPM_Flow_Reference_Manual_2025-10_Rev-0_compressed.pdf "OPM Flow Reference Manual 2025.10 Rev-0 — §§5.3, 6.3, 8.3, 10.3, 11.3, 12.3"
[2]: https://github.com/OPM/opm-simulators/blob/e4cdf79c2114cee531f47b56131dfbd68077e682/opm/simulators/utils/UnsupportedFlowKeywords.cpp "OPM Flow — noncritical and critical unsupported-keyword table"
[3]: https://github.com/OPM/opm-simulators/blob/e4cdf79c2114cee531f47b56131dfbd68077e682/opm/simulators/flow/KeywordValidation.cpp#L78-L137 "OPM Flow — validation behaviour for critical versus noncritical unsupported keywords"
[4]: https://rfdyn.com/resources-hub/ "tNavigator Resources Hub — manuals and software updates in client support portal"
[5]: https://arxiv.org/abs/1910.06059 "Rasmussen et al. (2019), The Open Porous Media Flow Reservoir Simulator"
[6]: https://github.com/OPM/opm-grid/blob/685ce0c4626c92104852293dbe315acc9b6687a4/opm/grid/cpgrid/processEclipseFormat.cpp#L179-L352 "OPM Grid — MINPV/PINCH preprocessing and PINCHCAL=ALL NNC construction"
[7]: https://opm-project.org/pipermail/opm/2020-November/000715.html "OPM 2020.10 Release Available — PINCHMUL=ALL change"
[8]: https://github.com/OPM/opm-simulators/issues/3145 "Issue #3145 — Unexpected behavior for PINCH"
[9]: https://github.com/OPM/opm-grid/commit/59ba192d9cef7cf77a260f92ea9f0544859e725a "OPM Grid commit — Do not connect cells from distant layers vertically without PINCH"
[10]: https://opm-project.org/?p=1412 "OPM Release 2021.10 — Fix grid generation without PINCH"
[11]: https://github.com/OPM/opm-grid/commit/b8bdcb118b931ae16819751b3b7262b3acd5f4c2 "OPM Grid commit — fix grid generation when there is no PINCH"
[12]: https://github.com/OPM/opm-simulators/commit/3d144ce4d7761038f673a310ba897d3c2fbda710 "OPM Simulators commit — PINCH option 2 NOGAP fully supported"
[13]: https://github.com/OPM/opm-grid/commit/553e568c9e93113eca384600dcf3ee8ca9c12728 "OPM Grid commit — support PINCH option 3 MAX_EMPTY_GAP"
[14]: https://github.com/OPM/opm-grid/commit/6bcb9a15f58f8c46bb579cf9a6acb17493954d5e "OPM Grid commit — throw for unsupported PINCH option 4 ALL"
[15]: https://opm-project.org/?p=1842 "OPM Flow 2024.10 release highlights — corrected transmissibilities for PINCH option 4 ALL"
[16]: https://github.com/OPM/opm-simulators/commit/e816ae4df3382b80a6af4f7887ed68472dd051d2 "OPM Simulators commit — correctly treat transmissibilities for PINCH option 4 ALL"
[17]: https://github.com/OPM/opm-simulators/commit/823f157aee7e275cd46ee5e7ab035335cd0ced95 "OPM Simulators commit — apply regional multiplier to PINCH NNC"
[18]: https://github.com/OPM/opm-grid/commit/ef3fad537be6b94c35011f4014eb67fcf46be56c "OPM Grid commit — skip PINCH NNC when a cell is collapsed"
[19]: https://github.com/OPM/opm-common/blob/df76c5255222f56b08f38389138815376c4ecb74/opm/input/eclipse/EclipseState/Runspec.cpp "OPM Common — RUNSPEC detection, including COMPS/GASWAT"
[20]: https://github.com/OPM/opm-simulators/blob/e4cdf79c2114cee531f47b56131dfbd68077e682/opm/simulators/flow/SimulatorFullyImplicit.cpp#L96-L124 "OPM Flow — unsupported tuning items including MXWSIT"

## Заключение

Публично проверяемая позиция такова: **использовать OPM Flow как drop-in замену Eclipse/tNavigator для deck с региональным `PINCHREG`/`PINCHNUM` нельзя**. Для глобального `PINCH` в Flow существуют поддержка и большой набор regression/bugfix history, но именно это означает необходимость версионного тестирования, особенно для no-PINCH, `MINPV`, `NOGAP`, `PINCHGAP`, `PINCHCAL=ALL`, `PINCHMUL=ALL`, `MULTZ-` и collapsed geometry. Для tNavigator без лицензированного manual/release note недопустимо дописывать детали реализации по памяти: открытый официальный сайт подтверждает наличие документации только в закрытом клиентском портале. [4]
