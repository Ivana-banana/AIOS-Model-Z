# Симулятор: установка и первый прогон

Решение команды: считаем **OPM Flow** на Linux. Организаторы допускают это явно и
закладывают расхождение с тНавигатором «в пару процентов» в допуск.

## Почему не на этой машине

Windows 10 Enterprise LTSC 2019 — сборка 17763 (версия 1809). WSL2 требует 1903+,
Docker Desktop современных версий — 21H2+. То есть штатный путь «OPM Flow под WSL2»
на этой машине не заводится без обновления ОС. Варианты, которые работают:

1. **Вторая ОС (Ubuntu 24.04) рядом с Windows** — выбранный путь. Нативная
   производительность, apt-пакет OPM ставится в три команды.
2. **Облачная VM** — если понадобится гонять DoE десятками прогонов параллельно.
   Дек и результаты гоняются через rsync, репозиторий один и тот же.
3. VirtualBox/Hyper-V с Ubuntu — заведётся, но на 547 тыс. ячеек × 371 шаг это
   заметно медленнее, и под пакетные прогоны не годится.

Разбор результатов (`aios/summary.py`, `resdata`) работает и на Windows: после
прогона достаточно забрать `.UNSMRY` и `.SMSPEC`.

## Установка OPM Flow (Ubuntu 22.04 / 24.04)

```bash
sudo apt-add-repository ppa:opm/ppa
sudo apt-get update
sudo apt-get install -y libopm-simulators-bin
flow --version
```

Для параллельного счёта дополнительно `sudo apt-get install -y mpi-default-bin`.

## Подготовка кейса

Материалы организаторов распаковываются в `vendor/Model_Z/` (см.
`docs/materials.md`). Дальше:

```bash
python -c "
from datetime import date
from aios.run import build_case
case = build_case('runs/base', 'vendor/Model_Z')
print(case.data_file)
"
scripts/run_flow.sh runs/base/RUN.DATA
```

Базовый кейс = история как есть, только с нашей SUMMARY-секцией: `Model_Z_summary.inc`
в поставке пустой, и без своих векторов симулятор не выгрузит ничего.

Свой сценарий:

```python
from datetime import date
from aios.run import build_case

case = build_case(
    "runs/scenario-01",
    "vendor/Model_Z",
    steps=my_steps,               # список ScheduleStep
    control_start=date(2007, 1, 1),
)
```

История до `control_start` сохраняется дословно — вместе с `WELLTRACK` и
`COMPDATMD`, которые задают траектории и вскрытия.

## Разбор результата

```python
from aios import economics, summary

rows = summary.monthly_rows("runs/base/RUN")
problems = summary.sanity_problems(rows)
assert not problems, problems[:10]

result = economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                               start_date="2007-01-01")
print(result.chdd_m)
```

## Главный риск первого прогона

Дек сгенерирован тНавигатором и содержит его собственные ключевые слова:
`WELLTRACK` (траектории, 103 блока) и `COMPDATMD` (вскрытия по измеренной глубине).
Сетка, свойства, регионы и начальные условия — стандартные и вопросов не вызывают,
но вскрытия — нет.

Если OPM Flow эти ключевые слова не поддерживает, он **не упадёт**: он их
пропустит, скважины останутся без соединений с пластом, и расчёт честно выдаст
нулевую добычу. Поэтому первый прогон проверяется не по факту «посчиталось», а по
трём вещам:

1. в логе нет строк про игнорирование `WELLTRACK` / `COMPDATMD`
   (`scripts/run_flow.sh` печатает их отдельно);
2. `FOPT` на конец истории отличен от нуля и по порядку величины совпадает с
   ориентиром организаторов — около 4,8 млн т нефти за 1991–2025
   (`docs/experiments.md`);
3. профили по нескольким скважинам похожи на историю, а не на прямую линию.

**Если вскрытия не читаются** — обходной путь есть: `WELLTRACK` даёт траекторию в
координатах X, Y, Z и MD, а `Model_Z_grid.inc` содержит `COORD`/`ZCORN`, то есть
геометрию ячеек. По ним восстанавливается `COMPDAT` (i, j, k и интервалы). Работа на
день, но начинать её надо, только убедившись, что она нужна: сначала прогон, потом
вывод.
