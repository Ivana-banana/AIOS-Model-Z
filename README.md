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
| `docs/onboarding-team.md` | **если ты в проекте первый раз — начни отсюда** |
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
pip install -r requirements-dev.txt
```

`requirements.txt` — рантайм решения (то, что попадает в Docker-образ),
`requirements-dev.txt` — плюс тесты, линтер и инструменты разбора материалов.

Материалы организаторов (дек Модели Z, эталонный расчётчик) распаковываются в
`vendor/` — см. `docs/materials.md`:

```powershell
Expand-Archive CHDD_PYTHON.zip       -DestinationPath vendor
Expand-Archive Model_Z_final_OPM.zip -DestinationPath vendor
Rename-Item vendor\Model_Z vendor\Model_Z_OPM
```

`Model_Z_final_OPM.zip` — действующая OPM-совместимая поставка дека от 14.08.2026.

## Расчёт в Docker

Симулятор и рантайм одной командой — так же поднимается облачная машина под
пакетные прогоны:

```bash
docker build -t aios -f docker/Dockerfile .
docker run --rm -v "$PWD:/work" -w /work aios python scripts/build_case.py
docker run --rm -v "$PWD:/work" -w /work aios scripts/run_flow.sh runs/base/RUN.DATA 8
```

Дек организаторов в образ не кладётся — монтируется томом. Про железо и оценку
времени счёта — `docs/setup-simulator.md`.

## Веб-платформа

```powershell
python -m uvicorn aios.web.app:app --reload
```

Открыть http://127.0.0.1:8000 — прогоны, экономика, экономический предел скважин,
запуск цикла управления с объяснением каждого решения. Страницы не тянут ни
одного внешнего ресурса: графики рисуются inline-SVG, скриптов нет вовсе — иначе
у проверяющего без интернета откроется пустая страница.

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
  policy.py      политика: от диагноза к управляющим действиям с объяснением
  surrogate.py   интерфейс суррогата и тривиальный бейзлайн (кривая обводнения)
  loop.py        цикл МАС: состояние → диагноз → решения → прогноз → ЧДД
  config.py      чтение config.toml
  web/           веб-платформа: FastAPI, inline-SVG графики, без внешних ресурсов
config.toml      всё, что влияет на число ЧДД
docker/          образ с OPM Flow и рантаймом
scripts/
  build_case.py  собрать кейс:  python scripts/build_case.py
  run_flow.sh    запустить OPM Flow (Linux)
  export_rows.py проверить прогон и собрать rows.csv для команды
  transcribe.py  расшифровка записей встреч организаторов
docs/
  onboarding-team.md подключение нового участника
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
from aios import config, economics, margin, summary

settings = config.load()
rows = summary.monthly_rows("runs/base/RUN", fluids=settings.fluids)
assert not summary.sanity_problems(rows)
print(economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                            start_date=settings.economics.start_date.isoformat()).chdd_m)
print(margin.shutdown_candidates(rows, margin.Norms.from_file())[:5])
```

## Главное правило

ЧДД считается **только** эталонным расчётчиком организаторов
(`vendor/CHDD_PYTHON/`). Своей формулы ЧДД в проекте нет и быть не должно:
расхождение заявленного значения с их пересчётом означает дисквалификацию.
Подробности — в `CLAUDE.md`.
