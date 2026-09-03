# Каталог открыто доступных Eclipse-деков для теста переносимости в OPM Flow

**Проверено: 21 августа 2026 г.**  
**Цель:** отобрать законно доступные Eclipse input deck'и для проверки контура «прочитать дек → сгенерировать `SCHEDULE` → запустить OPM Flow → выгрузить `.UNSMRY/.SMSPEC` → получить помесячные показатели скважин → посчитать экономику».

## База сравнения и методика

Предоставленная исходная модель имеет сетку **91×102×59**, 240 664 активных ячейки, **103 скважины**, единицы **METRIC**, трёхфазную black-oil физику с растворённым газом, 15 равновесных регионов и историю с 1991 г. В полном контуре есть 371 месячный отчётный шаг; сценарное управление меняется с 01.01.2007. В текущей поставке дек содержит `PINCHREG`, который Flow не поддерживает, поэтому базовый прогон требует `--parsing-strictness=low`; это означает, что сам исходный кейс нельзя считать строгим тестом полной совместимости.[1]

Параметры сетки, единицы, число обнаруженных скважин, ключевые слова и тип `SCHEDULE` ниже извлечены из непосредственно скачанных текстовых деков и их include-файлов. Число скважин — число уникальных имён из WELSPECS/управляющих записей, а не число строк controls. Статус Flow помечен как **«локально»** только если кейс был действительно запущен в Flow 2026.04. Условия замера: Ubuntu 24.04, 6 логических CPU, 3.8 GiB RAM. Малые тесты запускались как `mpirun -np 6 flow`; OLYMPUS — как `mpirun -np 1 flow --threads-per-process=6` ради памяти. Поэтому время на **8 ядрах**, если не приведён опубликованный benchmark, указано строго как **«не найдено»**; локальная цифра приведена лишь как ориентир масштаба.

> **Правило для acceptance-тестов:** не использовать `--parsing-strictness=low`. Иначе тест может формально завершиться, проигнорировав физически значимый ключ (как LGR/CARFIN или PINCHREG), а pipeline получит ложный «успех».

OPM заявляет, что его открытые наборы имеют Eclipse input-deck format и предназначены для validation/benchmark simulators.[2] Для `opm-data` по умолчанию применяются ODbL 1.0 к базе дека и DBCL 1.0 к содержимому, если не указано другое.[3] `opm-tests` предназначен для integration/regression testing Flow, но его README предупреждает, что не каждый файл каталога входит в тестовый набор, а отдельная корневая лицензия в проверенной ревизии **не найдена**.[4]

## Большая таблица-каталог

| Модель | Ссылка на первоисточник / загрузку | Лицензия и условия использования | Размер | Сетка | Скважин | Единицы | История / прогноз и `SCHEDULE` | Особенности дека относительно исходной модели | OPM Flow и время прогона | Пригодность | Одна строка обоснования |
|---|---|---|---|---|---:|---|---|---|---|---|---|
| **Egg Model, Eclipse realization** | [карточка 4TU][7]; [прямой `data.zip`][7a] | **4TU General Terms of Use**, не CC и не ODbL.[7] Перед коммерческим применением нужно проверить terms; DOI/авторов следует атрибутировать. | **47.0 MiB** | 60×60×7 = **25 200** | 12: 8 injector, 4 producer | METRIC | **Прогноз**; `TSTEP`, `WCONPROD`, `WCONINJE`, `COMPDAT`; WCONHIST нет. 120 report steps, 2011-06-15—2021-04-24. | Малый 3D Cartesian waterflood; 101 геореализация в наборе; TSTEP вместо DATES, небольшой фонд. | Дек успешно рассчитан с normal parsing. 8 ядер: **не найдено**; локально 6 MPI: **25 s**. Созданы SMSPEC/UNSMRY с WOPR/WWPR. | **Высокая** | Быстрый METRIC smoke-test, меняющий тип расписания и масштаб без сложной лицензии на ПО. |
| **OLYMPUS-1 (TNO / ISAPP)** | [репозиторий TNO][5]; [прямой `OLYMPUS_1.tar.gz`][5a] | **CC BY 4.0**; нужна атрибуция TNO и citation Fonseca et al. (2020).[5] | **6.2 MiB** | 118×181×16 = **341 728** | 18: 7 INJ, 11 PROD | METRIC | **Прогноз**; DATES 2016-01-01—2036-01-01, WCONPROD/WCONINJE, COMPDAT; WCONHIST нет. | Corner-point; faults + `MULTFLT`; `PVTNUM/SATNUM/EQLNUM/ROCKNUM`; `GRUPTREE`; Petrel-style include paths; 20 годовых report steps. | README прямо предназначает архив для OPM Flow.[5] Локально успешно: **105 s wall / 102.06 s simulation** на 1 MPI × 6 threads. 8 ядер: **не найдено**. | **Высокая** | Лицензионно чистый крупный corner-point forecast, который хорошо проверяет include-paths, fault multipliers и память. |
| **Brugge 60K, одна реализация** | [репозиторий TNO][6]; [прямой `tar.gz`][6a] | Не open-source: TNO Data Use Agreement, **только некоммерческое** применение; для коммерческого нужен контакт с TNO; acknowledgement обязателен.[6] | **13 MiB** | 139×48×9 = **60 048** | 30 | METRIC | **Прогноз / optimization benchmark**; DATES 2000-01-01—2010-01-01, WCONPROD/WCONINJE, COMPDAT; 37 report steps; WCONHIST нет. | Corner-point; `GRUPTREE`, `PVTNUM/SATNUM/EQLNUM`, VFP, PINCH; двухфлюидный вариант; сильно короче базового history case. | Локально успешно, normal parsing: **9 s** на 6 MPI. 8 ядер: **не найдено**. Flow сообщает об игнорировании MONITOR/NOECHO/ECHO/RPTSOL/COORDSYS. | **Средняя** | Технически отличный контроль corner-point и VFP, но юридически не годится для свободного коммерческого suite. |
| **Norne full benchmark** | [`opm-data/norne`][3a]; [общий архив `opm-data`][2a] | **ODbL 1.0** для дека, **DBCL 1.0** для содержимого по правилу `opm-data`.[3] При публичном распространении производной базы важны атрибуция и share-alike. | Каталог дека **31 MiB**; размер общего ZIP в первоисточнике **не найден** | 46×112×22 = **113 344** геометрических; **44 420** ячеек по MRST | 36 | METRIC | **История + управление:** `WCONHIST`, WCONPROD, WCONINJE, DATES. Локально: 247 report steps, 1997-11-06—2006-11-30. | Реальное месторождение; faulted corner-point; dissolved gas и vaporized oil, VFP, `MULTFLT/MULTZ/MULTREGT`, pressure-dependent rock, endpoint scaling, `GRUPTREE/GCONINJE`. | Официальный набор OPM.[2] Локально успешно в normal parsing: **155 s wall / 152.22 s simulation** на 6 MPI. 8 ядер: **не найдено**. | **Высокая** | Самый близкий к исходнику открытый кейс: METRIC, история, много скважин, сложная сетка и реальные operating changes. |
| **reduced_norne** | Официальная страница OPM перечисляет набор.[2] В проверенном `opm-data/master` каталог/прямой дек **не найден**. | Заявленный режим ожидаемо ODbL, но применимый файл и license header **не найдены**. | **Не найдено** | По OPM — та же Norne grid/petrophysics; конкретика из дека **не найдена** | **Не найдено** | **Не найдено** | По описанию few wells и simpler fluid; точные WCON* **не найдены**. | Удалены, в частности, некоторые transmissibility multipliers.[2] | Прогон и время: **не найдено**. | **Низкая** | Не включать в suite, пока не зафиксирована ревизия с полным исходником. |
| **SPE1, CASE1** | [`opm-data/spe1`][3b] | ODbL/DBCL.[3] | Каталог **164 KiB** | 10×10×3 = **300** | 2 | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE, COMPDAT; WCONHIST нет. 120 report steps, 2015-01-01—2025-01-01. | Малый black-oil gas injection; live oil (`PVTO`); FIELD + TSTEP вместо METRIC + DATES. | Официальный OPM case.[2] Локально успешно: **17 s** на 6 MPI; 8 ядер: **не найдено**. | **Высокая** | Минимальный тест FIELD-конверсии и прогнозного SCHEDULE, но с очень малым фондом. |
| **SPE2, SPIDER/RADIAL variants** | [`opm-tests/spe02`][4a]; [SPIDER-01][4b] | Для `opm-tests` отдельная лицензия **не найдена**.[4] | Каталог **344 KiB** | 10×1×15 = **150** | 3: OP01, CUT, PUT | FIELD | **Прогноз**; TSTEP, WCONPROD, WELTARG, COMPDAT; WCONHIST нет. | Радиальная/«spider» геометрия вместо обычной 3D Cartesian. | Конкретный SPIDER-01 в Flow 2026.04 не сошёлся на первом шаге после 10 timestep chops, около **7 s**. 8 ядер: не найдено. | **Низкая** | Полезен как expected-failure solver test, но не как готовый переносимый run-and-go case. |
| **SPE3, CASE1** | [`opm-data/spe3`][3c] | ODbL/DBCL.[3] | Каталог **36 KiB** | 9×9×4 = **324** | 2 | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE; WCONHIST нет. 175 report steps, 2015-01-01—2030-01-01. | Live gas / vaporized oil (`PVTG`), что меняет PVT-физику относительно базового dissolved-gas кейса. | Официальный OPM case.[2] Локально успешно: **18 s** на 6 MPI; 8 ядер: **не найдено**. | **Высокая** | Быстрый FIELD тест другого PVT-режима и простого TSTEP schedule. |
| **SPE4** | Официальный [SPE CSP][14] и OPM sources проверены; готовый открыто лицензированный Eclipse-дек для каталога **не найден**. | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | Не подменять номер случайной third-party копией без provenance и лицензии. | Flow/time: **не найдено**. | **Низкая** | Пробел в доступном legal coverage, а не рекомендуемая модель. |
| **SPE5, CASE1** | [`opm-data/spe5`][3d] | ODbL/DBCL.[3] | Каталог **24 KiB** | 7×7×3 = **147** | 3 | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE, COMPDAT; WCONHIST нет. 264 report steps, 1987-01-01—2009-01-01. | `SOLVENT`: расширенная физика, не чистый black-oil baseline. | Официальный OPM case.[2] Локально успешно: **39 s** на 6 MPI; 8 ядер: **не найдено**. | **Средняя** | Хорошо проверяет маршрутизацию экзотической физики, но не должен быть первым test case для oil economics. |
| **SPE6** | Готовый открыто лицензированный Eclipse-дек в проверенных источниках **не найден**. | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | См. ограничение coverage SPE CSP выше.[14] | Flow/time: **не найдено**. | **Низкая** | Нет проверяемого legal input для включения в suite. |
| **SPE7** | Готовый открыто лицензированный Eclipse-дек в проверенных источниках **не найден**. | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | См. ограничение coverage SPE CSP выше.[14] | Flow/time: **не найдено**. | **Низкая** | Нет проверяемого legal input для включения в suite. |
| **SPE8** | Готовый открыто лицензированный Eclipse-дек в проверенных источниках **не найден**. | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | См. ограничение coverage SPE CSP выше.[14] | Flow/time: **не найдено**. | **Низкая** | Нет проверяемого legal input для включения в suite. |
| **SPE9** | [`opm-data/spe9`][3e] | ODbL/DBCL.[3] | Каталог **1.1 MiB** | 24×25×15 = **9 000** | 26 по OPM; 27 имён обнаружено в тексте с учётом служебной записи | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE, COMPDAT; WCONHIST нет. 90 report steps / 900 days, 2015-01-01—2017-06-19. | Waterflood, сложные property curves и много переключений rate/BHP controls.[2] | Официальный OPM case.[2] Локально успешно: **13 s** на 6 MPI; 8 ядер: **не найдено**. | **Высокая** | Лучший первый legal smoke-test: FIELD + TSTEP + многоскважинные меняющиеся controls за секунды. |
| **SPE10 model 1** | [`opm-data/spe10model1`][3f] | ODbL/DBCL.[3] | Каталог **204 KiB** | 100×1×20 = **2 000** | 2 | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE; WCONHIST нет. Горизонт из дека отдельно не извлечён. | Upscaling benchmark с 1 ячейкой по Y; не похож на полноценный field schedule. | Официальный OPM case.[2] Локальный прогон в этой работе: **не найден**; 8 ядер: **не найдено**. | **Средняя** | Хорош для проверки grid/property path, слабее для теста realistic well management. |
| **SPE10 model 2** | [`opm-data/spe10model2`][3g] | ODbL/DBCL.[3] | Каталог **74 MiB** | 60×220×85 = **1 122 000** | 5 | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE; WCONHIST нет. | Очень крупный Cartesian upscaling benchmark с сильной гетерогенностью; OPM называет его tough problem.[2] | Локальная проверка и время 8 ядер: **не найдено**. | **Низкая** для 30 минут; **высокая** как отдельный stress-test | Нужен для проверки памяти/линейного решателя, но не для частого CI и не для calendar/history logic. |
| **PUNQ-S3, опубликованный Eclipse 100 вариант** | [Mendeley Data][9]; [прямой ZIP][9a] | **CC BY 4.0**: допустимо коммерческое использование при атрибуции.[9] | **66.7 KiB** в карточке, локально 68 KiB | 19×28×5 = **2 660** | 12 | METRIC | **Прогноз**; начальная DATES 1967-01-01 + TSTEP, WCONPROD/WCONINJE, COMPDAT; WCONHIST нет. | Малый METRIC FDO-дек, другой порядок таблиц/включений; истории нет. | Flow 2026.04 остановился на EQLOPTS, строка `2* 0 0 2 120 /`: parser error. 8 ядер: **не найдено**. | **Средняя** | Лицензия удобна и структура полезна, но до suite требуется минимальная нормализация синтаксиса и сверка с эталоном. |
| **`opm-tests/gconprod/T1L.DATA`** | [`opm-tests/gconprod`][4c] | Для `opm-tests` отдельная лицензия **не найдена**.[4] | Каталог **14 MiB** | 20×30×10 = **6 000** | 9 | METRIC | **Прогноз**; DATES 2020-01-01—2020-05-01 + TSTEP, WCONPROD/WCONINJE; WCONHIST нет. | `GRUPTREE`, `GCONPROD`, `GCONINJE`; короткий group-control schedule. | Regression-oriented case; локально успешно: **9 s** на 6 MPI. 8 ядер: **не найдено**. | **Высокая технически; средняя юридически** | Лучший быстрый METRIC тест групповых controls, если право применения `opm-tests` подтверждено. |
| **`opm-tests/spe9group/SPE9_CP_GROUP.DATA`** | [`opm-tests/spe9group`][4d] | Для `opm-tests` отдельная лицензия **не найдена**.[4] | Каталог **36 MiB** | 24×25×15 = **9 000** | 7 обнаруженных | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE; 30 report steps / 300 days; WCONHIST нет. | `GRUPTREE`, `GCONPROD`, `GCONINJE` при FIELD-единицах; наиболее контрастная готовая структура schedule. | Regression-oriented case; локально успешно: **4 s** на 6 MPI. В restart output есть предупреждение по WELSPECS. 8 ядер: **не найдено**. | **Высокая технически; средняя юридически** | Самый непохожий быстрый готовый SCHEDULE test, но нужен отдельный юридический review `opm-tests`. |
| **`opm-tests/msw/MSW-3D-01.DATA`** | [`opm-tests/msw`][4e] | Для `opm-tests` отдельная лицензия **не найдена**.[4] | Каталог **3.1 MiB** | 10×10×10 = **1 000** | 2 | METRIC | **Прогноз**; DATES и TSTEP, WCONPROD/WCONINJE, COMPDAT; WCONHIST нет. | `WELSEGS`/`COMPSEGS`: мультисегментные скважины. | Текущий вариант в Flow 2026.04 не стартовал: `COMPSEGS: the connection is not found`. 8 ядер: **не найдено**. | **Средняя** | Нужен как negative compatibility test MSW; этот variant не считать готовым pipeline benchmark. |
| **`opm-tests/lgr/SPE1CASE1_CARFIN_GR.DATA`** | [`opm-tests/lgr`][4f] | Для `opm-tests` отдельная лицензия **не найдена**.[4] | Каталог **2.2 MiB** | База 10×10×3 = **300** плюс LGR | 2 | FIELD | **Прогноз**; TSTEP, WCONPROD/WCONINJE; WCONHIST нет. | `CARFIN/ENDFIN`, `WELSPECL/COMPDATL`: LGR и LGR wells. | Flow 2026.04 normal parsing отвергает CARFIN, ENDFIN, WELSPECL, COMPDATL. `low` игнорирует физику. | **Низкая** для simulator test; **высокая** для fail-fast test | Идеален для проверки, что pipeline не выдаёт ложный успех при неподдерживаемом LGR. |
| **`opm-tests/actionx/ACTIONX_WCONHIST.DATA`** | [`opm-tests/actionx`][4g] | Для `opm-tests` отдельная лицензия **не найдена**.[4] | Каталог **47 MiB** | 13×22×11 = **3 146** | 2 | METRIC | **История**; DATES, WCONHIST/WCONINJH, ACTIONX/ENDACTIO; WCONPROD нет. | Conditional control через ACTIONX; include используют `$MODEL2/...`. | Представляет ACTIONX regression coverage.[4] Для прямого запуска вне test harness надо подставить `$MODEL2` include-paths; отдельное время: **не найдено**. | **Средняя** в harness; **низкая** «скачал-и-запустил» | Лучший тест сохранности ACTIONX и history controls при переписывании SCHEDULE. |
| **UNISIM-II-D deterministic BO** | [официальная страница UNISIM][10] | **ODbL 1.0** для деков и **DBCL 1.0** для contents, заявлено издателем.[10] | ZIP **3.4 MiB** | В скачанном файле `GRID CORNER 46 69 30` | **Не найдено** | Eclipse units: **не найдено** | Отдельно опубликован history-data archive на 516 days; Eclipse WCON* **не найдены**.[10] | Открытый benchmark Unicamp (Бразилия), но не Eclipse input. | Файл начинается `RESULTS SIMULATOR IMEX 201710`: это **CMG IMEX input**, не дек Flow/Eclipse. Прямой Flow run невозможен. | **Низкая** | Включён для закрытия UNISIM: лицензия открытая, но для данного pipeline нужен конвертер. |
| **CO2MULTISCALE ECLIPSE 100 decks, BGS** | [карточка BGS][11]; [resource locator][11a] | Общий портал — **OGL v3.0**, но поле Licence у конкретной записи — `Not set`; лицензия самого ресурса **не найдена**.[11] | **Не найдено** | **Не найдено** | **Не найдено** | **Не найдено** | История/WCON*: **не найдено**. | Серия CO₂ storage decks для исследования hysteresis; скорее boundary case, чем тест oilfield economics. | Flow/time: **не найдено**. | **Низкая** | Сначала нужно получить архив и проверить его лицензию/состав, затем отделить от oil-production suite. |

## Штатность Flow, расхождения и пригодность к выгрузке помесячных well data

| Группа | Модели | Вывод |
|---|---|---|
| **Подтверждённо прошли Flow 2026.04 с normal parsing** | SPE1, SPE3, SPE5, SPE9, Egg, Norne, Brugge, OLYMPUS-1, GCONPROD T1L, SPE9group | Это основной пул для немедленного теста чтения, вставки SUMMARY и выгрузки UNSMRY. Для всех успешно завершённых кейсов Flow создал `.SMSPEC` и `.UNSMRY`; в input/SMSPEC присутствуют per-well WOPR/WWPR, а для большинства также WGPR/WBHP. Brugge ограничен non-commercial terms, `opm-tests` требует отдельной проверки прав. |
| **Regression-oriented, но конкретный run требует настройки или не прошёл** | SPE2 SPIDER-01, MSW-3D-01, ACTIONX WCONHIST, LGR CARFIN | Это полезные **expected-failure** tests. Правильный результат pipeline — ясная диагностируемая ошибка, а не незаметное переписывание/игнорирование конструкции. |
| **Открытый Eclipse-текст, но parser conflict Flow** | PUNQ-S3 | CC BY и малый размер не заменяют проверку синтаксиса: сначала patch EQLOPTS и сравнение с исходным simulator/reference. |
| **Не прямой Eclipse/Flow case** | UNISIM-II-D IMEX; CO2MULTISCALE до получения файлов | Не включать в стартовый run suite без конвертации или верификации исходника. |

У Norne в successful normal run Flow вывел предупреждения об усечении fault names длиннее 8 символов в `FAULTS/MULTFLT`, а также о сервисных `NSTACK`, `NOECHO` и некоторых опциях `MESSAGES`. Brugge и OLYMPUS выдают warnings по MONITOR/NOECHO/ECHO/RPTSOL; `COORDSYS` игнорируется, потому что Flow сообщает об отсутствии multiple grid systems. Эти кейсы **завершились**, но не являются строгим доказательством численного совпадения с коммерческим Eclipse: для такого вывода нужен отдельный comparison summary vectors.

## Покрытие отличающихся конструкций дека

| Конструкция, которую нужно проверить | Рекомендуемый дек | Проверяемая гипотеза |
|---|---|---|
| FIELD вместо METRIC, TSTEP вместо DATES, меняющиеся controls | **SPE9** | Генератор и экономика не предполагают METRIC, календарные DATES или фиксированное число скважин. |
| `GRUPTREE`, `GCONPROD`, `GCONINJE` | **GCONPROD T1L**; для FIELD — **SPE9group** | Новый schedule не удаляет существующую group hierarchy и не конфликтует с group constraints. |
| Multi-segment wells: WELSEGS/COMPSEGS | **MSW-3D-01** | Pipeline должен сохранить topology или корректно остановиться до запуска. |
| LGR: CARFIN/ENDFIN, WELSPECL/COMPDATL | **SPE1CASE1_CARFIN_GR** | Unsupported LGR не должен быть молча проигнорирован в acceptance test. |
| Conditional historical control: ACTIONX + WCONHIST | **ACTIONX_WCONHIST** | Переписчик schedule не разрушает ACTIONX и не заменяет историю прогнозным control block. |
| Corner-point, faults, multipliers, реальная история | **Norne** | Pipeline выдерживает наиболее близкий к production structure дек. |
| Крупный heterogeneous Cartesian stress | **SPE10 model 2** | Слабые места по памяти/linear solver отделяются от ошибок SCHEDULE. |
| Petrel-style includes, fault multipliers, regions | **OLYMPUS-1** | Проверяется portability путей, include-дерева, regions и стратегии MPI/threads. |

## Открытые российские модели и учебно-соревновательные наборы

В проверенных русско- и англоязычных первоисточниках **не найден** дек российского происхождения, который одновременно содержит полный Eclipse-deck с include-файлами, имеет ясно опубликованную открытую лицензию и применим как прямой вход OPM Flow. Это не утверждение, что таких деков не существует; это означает, что их нельзя безопасно включить в воспроизводимый legal test suite без дополнительного первоисточника и лицензии.

Открытые учебно-соревновательные альтернативы представлены OLYMPUS (ISAPP optimization challenge, CC BY), Egg (стандартный synthetic test case), PUNQ-S3 (CC BY) и UNISIM-II-D (ODbL/DBCL, но IMEX). Brugge — важный SPE ATW benchmark, но его открытость ограничена некоммерческим соглашением.[5][6][7][9][10]

## Рекомендованная тройка

| Роль | Модель | Почему именно она |
|---|---|---|
| **Самый быстрый test переносимости за 30 минут** | **SPE9 из `opm-data`** | ODbL/DBCL; FIELD, TSTEP, 9 000 ячеек, многоскважинные controls. Локально — 13 s на 6 MPI. Это лучший guardrail против неявной зависимости от METRIC и DATES. |
| **Самый непохожий готовый тест** | **`opm-tests/spe9group/SPE9_CP_GROUP.DATA`** | FIELD + TSTEP + GRUPTREE/GCONPROD/GCONINJE; локально 4 s. Это наиболее сильный контраст по структуре SCHEDULE. Перед внешним/коммерческим использованием надо подтвердить лицензию `opm-tests`; если она не подтверждена, заменить на SPE9 либо GCONPROD T1L. |
| **Самый близкий к вашей модели** | **Norne full** | METRIC, реальная история, WCONHIST и forecast controls, corner-point/faults/multipliers/VFP/regions. Локально 155 s; наиболее полезный end-to-end regression для well vectors и экономики. |

Практический порядок: запускать **SPE9** на каждый PR; **Norne** — nightly; SPE9group/GCONPROD — в schedule-structure suite; ACTIONX/MSW/LGR/PUNQ — как отдельные expected-success/expected-failure contracts. Перед каждым запуском сохранять исходный SCHEDULE, выводить unsupported-keyword log и добавлять pipeline-required per-well vectors в SUMMARY.

## Воспроизводимые команды

```bash
# Юридически чистый основной источник OPM
# ODbL/DBCL: см. README в репозитории
git clone --depth 1 https://github.com/OPM/opm-data.git

# Самый быстрый FIELD + TSTEP test
mpirun -np 8 flow --output-dir=runs/spe9 opm-data/spe9/SPE9.DATA

# Наиболее близкий исторический test
mpirun -np 8 flow --output-dir=runs/norne opm-data/norne/NORNE_ATW2013.DATA

# Проверить, что Flow создал summary-файлы
ls runs/spe9/SPE9.{SMSPEC,UNSMRY}
ls runs/norne/NORNE_ATW2013.{SMSPEC,UNSMRY}
```

Запускать каждый case в отдельном `--output-dir`, не писать output в каталог исходного дека. Для каждого скачанного архива и ревизии Git полезно зафиксировать SHA256/commit и версию Flow рядом с эталонными summary-результатами.

## References

[1] Внутренний источник пользователя: приложенный `setup-simulator.md`.  
[2] [OPM — Open datasets](https://opm-project.org/?page_id=559).  
[2a] [OPM `opm-data` archive](https://github.com/OPM/opm-data/archive/master.zip).  
[3] [OPM `opm-data` README и лицензионные файлы](https://github.com/OPM/opm-data).  
[3a] [OPM Norne directory](https://github.com/OPM/opm-data/tree/master/norne).  
[3b] [OPM SPE1 directory](https://github.com/OPM/opm-data/tree/master/spe1).  
[3c] [OPM SPE3 directory](https://github.com/OPM/opm-data/tree/master/spe3).  
[3d] [OPM SPE5 directory](https://github.com/OPM/opm-data/tree/master/spe5).  
[3e] [OPM SPE9 directory](https://github.com/OPM/opm-data/tree/master/spe9).  
[3f] [OPM SPE10 model 1 directory](https://github.com/OPM/opm-data/tree/master/spe10model1).  
[3g] [OPM SPE10 model 2 directory](https://github.com/OPM/opm-data/tree/master/spe10model2).  
[4] [OPM `opm-tests` README](https://github.com/OPM/opm-tests).  
[4a] [OPM SPE02 tests](https://github.com/OPM/opm-tests/tree/master/spe02).  
[4b] [SPE02 SPIDER-01 deck](https://github.com/OPM/opm-tests/blob/master/spe02/SPE02-SPIDER-01.DATA).  
[4c] [OPM GCONPROD tests](https://github.com/OPM/opm-tests/tree/master/gconprod).  
[4d] [OPM SPE9 group-control tests](https://github.com/OPM/opm-tests/tree/master/spe9group).  
[4e] [OPM MSW tests](https://github.com/OPM/opm-tests/tree/master/msw).  
[4f] [OPM LGR tests](https://github.com/OPM/opm-tests/tree/master/lgr).  
[4g] [OPM ACTIONX tests](https://github.com/OPM/opm-tests/tree/master/actionx).  
[5] [TNO OLYMPUS repository and CC BY 4.0 statement](https://github.com/TNO/OLYMPUS).  
[5a] [OLYMPUS-1 direct archive](https://github.com/TNO/OLYMPUS/raw/main/OLYMPUS_1.tar.gz).  
[6] [TNO Brugge repository and Data Use Agreement](https://github.com/TNO/Brugge).  
[6a] [Brugge direct archive](https://github.com/TNO/Brugge/raw/master/BRUGGE60K_FY-SF-KM-1-1_JAN2017.tar.gz).  
[7] [4TU ResearchData — The Egg Model data files](https://data.4tu.nl/articles/_/12707642/1).  
[7a] [Egg direct `data.zip`](https://data.4tu.nl/file/0e53ddd5-33b7-44f2-9ca4-921880825e47/5867bc9e-4bf6-4e07-827c-4a6669a539ce).  
[8] [SINTEF MRST — Public data sets](https://www.sintef.no/projectweb/mrst/modules/mrst-core/data-sets/).  
[9] [Mendeley Data — PUNQ-S3 Eclipse 100 data](https://data.mendeley.com/datasets/nrrmpnkwzy/1).  
[9a] [PUNQ-S3 direct ZIP](https://data.mendeley.com/public-api/zip/nrrmpnkwzy/download/1).  
[10] [UNISIM-II-D official benchmark repository](https://www.unisim.cepetro.unicamp.br/benchmarks/br/unisim-ii/unisim-ii-d).  
[11] [BGS / UK National Data Library — CO2MULTISCALE Eclipse 100 decks](https://www.data.gov.uk/dataset/8b1eb1ae-73aa-40ea-8198-a494a07d9fb7/eclipse-100-input-decks-for-reservoir-models-developed-as-part-of-co2multiscale).  
[11a] [CO2MULTISCALE resource locator](https://ukccs-research.org/accessions/index.html#item75790).  
[12] [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1.0/).  
[13] [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1.0/).  
[14] [Society of Petroleum Engineers — Comparative Solution Project](https://www.spe.org/en/csp/).
