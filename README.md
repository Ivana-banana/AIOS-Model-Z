# AIOS · трек 2 · Модель Z

Мультиагентная система управления фондом скважин с нейросетевыми суррогатными
моделями. Хакатон AIOS (ПАО «Татнефть»). Целевая функция — максимум ЧДД (NPV) при
соблюдении технологических ограничений.

Результат работы системы: `wells_schedule.inc` (управляющие последовательности) и
заявленное значение ЧДД, которое организаторы независимо перепроверяют одним
расчётом на полной гидродинамической модели.

## С чего начать

| Документ | О чём |
|---|---|
| `docs/task.md` | постановка задачи, критерии оценки, регламент сдачи, календарь |
| `docs/method/diff.md` | две методики ЧДД: что изменилось и чем считаем |
| `PLAN.md` | план работ, вехи, риски |
| `CLAUDE.md` | правила работы над проектом |
| `GIT.md` | правила ведения Git |
| `docs/decisions.md` | инженерные решения с обоснованием |
| `docs/experiments.md` | журнал прогонов |

## Установка

Окружение лежит прямо в корне проекта (`Scripts/`, `Lib/`) и в репозиторий не
попадает. На чистой машине:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Материалы организаторов (дек Модели Z, эталонный расчётчик) распаковываются в
`vendor/` — см. `docs/materials.md`:

```powershell
Expand-Archive CHDD_PYTHON.zip       -DestinationPath vendor
Expand-Archive Model_Z_final_OPM.zip -DestinationPath vendor
Rename-Item vendor\Model_Z vendor\Model_Z_OPM
```

`Model_Z_final_OPM.zip` — действующая OPM-совместимая поставка дека от 14.08.2026.

## Проверка

```powershell
python -m ruff check .
python -m pytest -q
```

`pytest` гоняет и наши тесты, и тесты эталонного расчётчика организаторов — они
часть нашего прогона, а не чужие.

## Структура

```
aios/
  economics.py   ЧДД: тонкая обвязка над эталонным расчётчиком организаторов
  deck.py        разбор SCHEDULE-секции дека (DATES / WCONPROD / WCONINJE)
  schedule.py    генерация и валидация wells_schedule.inc
  summary.py     SUMMARY-секция и выгрузка результата в 14 колонок расчётчика
  run.py         сборка расчётного кейса поверх дека без его правки
  margin.py      экономический предел скважины — бейзлайн для отбора решений
scripts/
  build_case.py  собрать кейс:  python scripts/build_case.py runs/base
  run_flow.sh    запустить OPM Flow (Linux)
docs/
  task.md            постановка задачи и критерии
  architecture.md    архитектура МАС
  setup-simulator.md установка симулятора и первый прогон
  method/            методики ЧДД, выжимки лекций, разбор отличий
  decisions.md       инженерные решения
  experiments.md     журнал прогонов
tests/           наши тесты, включая пограничные случаи методики
vendor/
  CHDD_PYTHON/   эталонный расчётчик ЧДД организаторов — вендорен побайтово
  Model_Z_OPM/   действующий дек Модели Z (в репозиторий не коммитится)
```

## Сквозной прогон

```powershell
python scripts/build_case.py runs/base          # собрать кейс из дека
```
```bash
scripts/run_flow.sh runs/base/RUN.DATA          # посчитать (Linux, OPM Flow)
```
```python
from aios import economics, margin, summary

rows = summary.monthly_rows("runs/base/RUN")
assert not summary.sanity_problems(rows)
print(economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                            start_date="2007-01-01").chdd_m)
print(margin.shutdown_candidates(rows, margin.Norms.from_file())[:5])
```

## Главное правило

ЧДД считается **только** эталонным расчётчиком организаторов
(`vendor/CHDD_PYTHON/`). Своей формулы ЧДД в проекте нет и быть не должно:
расхождение заявленного значения с их пересчётом означает дисквалификацию.
Подробности — в `CLAUDE.md`.
