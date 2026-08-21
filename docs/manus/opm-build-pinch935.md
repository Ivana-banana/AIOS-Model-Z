# OPM Flow 2026.04 с исправлением PINCH/NNC из opm-grid PR #935

**Дата проверки:** 21 августа 2026 года.  
**Подготовил:** Manus AI.  
**Рекомендуемый вариант:** полная сборка `opm-common → opm-grid → opm-simulators` из зафиксированных исходников с backport двух указанных коммитов в `opm-grid`.

> **Итоговое решение.** Для расчётов, где важна воспроизводимость, не следует подменять одну библиотеку под бинарным `flow` из PPA. Выбранная сборка воспроизводит релизную точку 2026.04, меняет только необходимую реализацию и добавляет её регрессионный тест. `opm-models` не входит в цепочку сборки Flow 2026.04.

## Сравнение вариантов

| Вариант | Выполнимость | Время | Риск | Воспроизводимость | Решение |
|---|---|---:|---|---|---|
| **Полная сборка из исходников**: `opm-common → opm-grid → opm-simulators` | Да. Официальный OPM-рецепт задаёт этот порядок; backport проверен на релизной точке без конфликтов. [1] [4] [5] | Официальный типовой benchmark времени **не найдено**. Для планирования: 40–120 мин на 8 ядрах либо 25–75 мин на 16 ядрах при холодной сборке; фактическое время измерить в CI. | Низкий среди вариантов: единая компилируемая цепочка и обязательный регрессионный тест. | Высокая: Git SHA, digest базы, версии Dune и контрольные суммы сохраняются в образе. Полная byte-for-byte повторяемость требует также неподвижного APT-репозитория. | **Выбран.** |
| **Пересборка только `opm-grid` поверх PPA** | Технически да: PPA `flow` имеет прямую ELF-зависимость `libopmgrid.so.2026.04`; новую библиотеку можно подать через `LD_LIBRARY_PATH`. | Обычно 5–20 мин. | Высокий: OPM рекомендует не смешивать бинарные пакеты с собранными модулями; официальный контракт ABI для такого смешивания **не найдено**. [1] | Средняя или низкая: зависит от точных версий PPA, Dune, компилятора и порядка поиска библиотек. | Только диагностический откаточный вариант; не для базового образа. |
| **Готовый образ `master` / nightly после 04.08.2026** | Готовый официальный образ с такими свойствами **не найдено**. | Минуты на pull. | Невозможна проверка требуемого commit base/патча без образа. | Низкая для тега `latest`; он перемещаемый. | Не использовать. |

Оценки времени и размера в таблице — инженерные плановые интервалы, а не данные OPM. Публичный официальный benchmark полной сборки на 8–16 ядрах **не найдено**. Практическая проверка для этой инструкции показала, что 3.8 GiB RAM недостаточно даже для последовательной компиляции одного тяжёлого translation unit Flow; поэтому для устойчивой полной сборки следует выделить **16 GiB RAM минимум**, а для `BUILD_JOBS=8–16` — **24–32 GiB**. Рабочее пространство с исходниками и каталогами сборки следует заложить на **20–40 GiB**. Размер готового образа до фактической сборки **не найдено**; консервативная оценка Dockerfile ниже — **1.5–3.0 GiB**, поскольку в runtime сознательно сохранены development-пакеты для непроверенно не обрезать зависимостное замыкание.

## Что именно составляет Flow 2026.04

Официальная страница OPM предписывает собирать модули в порядке `opm-common`, затем `opm-grid`, затем `opm-simulators`; `opm-upscaling` следует после них, но для `flow` не требуется. Она также приводит bash-рецепт с клонированием, `cmake ..` и `make`, а также рекомендует запускать `ctest`. [1] Механизм CMake релизного `opm-simulators` линкует `opmsimulators` с `opmcommon` и `opmgrid`; `opm-models` там не обнаружен. [10]

| Компонент | Роль в выбранном образе | Релизная точка / статус | Источник |
|---|---|---|---|
| `opm-common` | Базовая библиотека, parser и общие компоненты | `0ea62974f24d70fc2b3e30d6aae8b76ef000dac1` — разыменованный SHA тега `release/2026.04/final`. | Git-тег проверяется командой ниже; общий порядок OPM — [1]. |
| `opm-grid` | Сетка, обработка MINPV/PINCH и NNC; здесь накладывается backport | База `d6a1a35c3d770fe50073b9ed866bbec740c8faac` плюс два commit. | [4] [5] |
| `opm-simulators` | `flow` и библиотека симуляторов | `b82f21dba405286c4c4446614dd3bf9cdebf7a2c` — разыменованный SHA тега `release/2026.04/final`. | [3] [10] |
| `opm-models` | Самостоятельный исторический simulation framework | Как зависимость Flow 2026.04 — **не найдено**. В PPA для Noble отдельно опубликован `opm-models 2024.04`, тогда как компоненты Flow имеют `2026.04`. | [6] [11] |

Таким образом, исходное предположение о последовательности из четырёх модулей следует скорректировать. `opm-models` нельзя добавлять в Dockerfile как обязательную зависимость Flow без технической необходимости: это увеличивает область изменений, но не добавляет нужный патч.

### Зависимости Ubuntu 24.04

Официальный перечень OPM включает C++17-компилятор и CMake, MPI, BLAS/LAPACK, Boost, SuiteSparse/UMFPACK, Zoltan из Trilinos, fmt, cJSON и Dune (`common`, `geometry`, `istl`, `grid`). Страница OPM называет поддерживаемые Dune 2.9.2, 2.10 и 2.11. [2] Релиз OPM 2026.04 официально поддерживает Ubuntu 24.04 LTS. [3]

| Зависимость | В Dockerfile | Версия / обоснование |
|---|---|---|
| CMake, GCC/G++, make, gfortran, pkg-config, Git | `build-essential gfortran pkg-config cmake git` | В Ubuntu 24.04 на момент проверки: CMake 3.28.3 и GCC 13.2.0; релизный CMake `opm-simulators` требует CMake ≥ 3.23. [2] [10] |
| MPI | `mpi-default-dev`, runtime `mpi-default-bin` | Open MPI 4.1.6; MPI включён OPM по умолчанию при обнаружении. [1] [2] |
| Линейная алгебра | `libblas-dev liblapack-dev libsuitesparse-dev libptscotch-dev` | BLAS/LAPACK и SuiteSparse — официальные prerequisites; PT-Scotch не обязателен, но CMake Flow рекомендует его/ParMETIS для масштабируемости параллельного AMG. [2] [10] |
| Основные библиотеки | `libboost-all-dev libtrilinos-zoltan-dev libfmt-dev libcjson-dev` | Официальный список prerequisites. [2] |
| Dune | `libdune-common-dev`, `geometry`, `istl`, `grid` версии `2.11.0-1~noble` | В PPA OPM для Noble опубликован Dune 2.11.0; это допустимый OPM диапазон и совпадает с зависимостью бинарного PPA Flow. Важно: из PPA берётся **только Dune**, не `libopm-*`. [2] [6] |
| Parallel HDF5 | `libhdf5-mpi-dev` | Не обязательна для запуска основного Flow, но CMake предупреждает, что при MPI serial HDF5 отключает HDF5 restart support. [10] |

## Релизная точка и backport PR #935

Пользовательская строка `release/2026.04/final` в `opm-grid` на дату проверки является **аннотированным Git-тегом, не веткой**. Это нужно учитывать в CI: закрепляется разыменованный commit SHA, а не имя передвигаемой ветки. Официальное объявление релиза допускает checkout `release/2026.04/final`. [3]

```bash
# Источник: Git-репозитории OPM и инструкция сборки OPM [1].
for repo in opm-common opm-grid opm-simulators; do
  printf '%s\n' "$repo"
  git ls-remote "https://github.com/OPM/${repo}.git" \
      'refs/tags/release/2026.04/final' \
      'refs/tags/release/2026.04/final^{}'
done
```

Ожидаемые разыменованные SHA — `0ea62974f24d70fc2b3e30d6aae8b76ef000dac1` для `opm-common`, `d6a1a35c3d770fe50073b9ed866bbec740c8faac` для `opm-grid` и `b82f21dba405286c4c4446614dd3bf9cdebf7a2c` для `opm-simulators`.

| Порядок | SHA | Назначение и проверка |
|---:|---|---|
| 1 | `ef3fad537be6b94c35011f4014eb67fcf46be56c` | Исправление «Skip NNCs created by PINCH if one of the cells is collapsed». Меняет `opm/grid/MinpvProcessor.cpp`: запрещает создание NNC, когда верхняя или нижняя ячейка коллапсирована. [4] |
| 2 | `7b665216985439a72fe908e574c6c2ffe5017e1a` | Регрессионный тест в `tests/test_minpvprocessor.cpp` для коллапсированной верхней и нижней ячейки с ожидаемым `nnc.size() == 0`. Родитель — первый commit, поэтому его применяют вторым. [5] |

Проверка на чистом checkout `d6a1…` завершилась без конфликтов. После `git diff --check HEAD~2..HEAD` рабочее дерево было чистым, а `ctest -R '^minpvprocessor$'` дал 1/1 passed. Для дополнительной аудируемости Dockerfile фиксирует автора и даты коммитера при backport; это даёт детерминированный локальный SHA `9869209d28dcf797997ffc18e41fab506191744f`. Этот SHA **не является upstream SHA**: он логично отличается, поскольку его parent — релизный `d6a1…`, а не parent коммитов на master.

## Почему не следует подменять только библиотеку из PPA

Официальный OPM отдельно предупреждает, что наличие бинарных пакетов OPM одновременно с модулями, собранными из исходников, — «typical source of hard-to-find errors», и рекомендует удалить бинарные пакеты перед сборкой. [1] Следовательно, **поддерживаемый ABI-контракт для смешивания PPA и самосборки не найдено**.

Тем не менее техническая возможность проверена пассивно. Из PPA был извлечён, но не установлен, пакет `libopm-simulators-bin_2026.04-1~noble_amd64.deb`; его `flow` содержит ELF `DT_NEEDED` для `libopmgrid.so.2026.04`, `libopmsimulators.so.2026.04` и `libopmcommon.so.2026.04`. Динамический загрузчик Linux выбирает библиотеку по RPATH/RUNPATH, `LD_LIBRARY_PATH`, cache `ld.so` и системным путям в указанном порядке. [12] Поэтому новая `libopmgrid.so.2026.04`, переданная первым каталогом в `LD_LIBRARY_PATH`, будет загружена.

Изменение в PR #935 локально в `.cpp` и не меняет публичный API в заголовках, поэтому риск ABI **ожидаемо ниже**, но это не доказательство ABI-совместимости. До production нельзя отвечать «ABI совпадёт»: официальная гарантия ABI **не найдено**. Нужны сравнение динамических символов и полный регрессионный расчёт целевого дека.

Ниже приведён точный диагностический путь; он не является рекомендованным Dockerfile.

```bash
# Источник зависимостей: OPM prerequisites [2]; PPA 2026.04: Launchpad [6].
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config mpi-default-dev \
  libopm-grid-dev=2026.04-1~noble \
  libdune-common-dev=2.11.0-1~noble \
  libdune-geometry-dev=2.11.0-1~noble \
  libdune-istl-dev=2.11.0-1~noble \
  libdune-grid-dev=2.11.0-1~noble

git clone https://github.com/OPM/opm-grid.git
cd opm-grid
git checkout --detach d6a1a35c3d770fe50073b9ed866bbec740c8faac
git cherry-pick ef3fad537be6b94c35011f4014eb67fcf46be56c
git cherry-pick 7b665216985439a72fe908e574c6c2ffe5017e1a
git diff --check HEAD~2..HEAD

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/opm-grid-pinch935 \
  -DCMAKE_PREFIX_PATH=/usr
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure -R '^minpvprocessor$'
sudo cmake --install build

# Источник порядка загрузки: ld.so(8) [12].
readelf -d /opt/opm-grid-pinch935/lib/libopmgrid.so.2026.04 | grep SONAME
LD_LIBRARY_PATH=/opt/opm-grid-pinch935/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  ldd /usr/bin/flow | grep libopmgrid
LD_LIBRARY_PATH=/opt/opm-grid-pinch935/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  flow --version
```

Перед подменой следует сохранить исходный `ldd /usr/bin/flow`, SHA-256 старой и новой библиотеки, результаты `ctest` и результат контрольного дека. Если `ldd` не показывает `/opt/opm-grid-pinch935/lib/libopmgrid.so.2026.04`, эксперимент недействителен. Не заменяйте файл в `/usr/lib`: это ломает целостность `dpkg` и делает откат неаудируемым.

## Готовые образы и master/nightly

Официальный OPM-документирует Docker Hub-репозиторий `openporousmedia/opmreleases`; официальный pull — `docker pull openporousmedia/opmreleases`. [8] На дату проверки Docker Hub показывал релизные теги `2026.04_amd64`, `2026.04_arm64`, `2026.04_nvidia`, `2026.04_amd` и `latest`; тегов `master`, современного `nightly` или образа, собранного после **04.08.2026**, **не найдено**. Единственный найденный ночной тег — устаревший `nightly-2018-11-30`. [9]

`2026.04_amd64` имеет digest `sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9` и опубликован 20.05.2026, то есть до указанных commit. Он не может содержать backport от 01.07.2026. Проверка бинаря готового образа через `flow --version` показывает только версию и **не доказывает** наличие патча; доказательство потребовало бы записи исходного SHA или запуска добавленного регрессионного теста, чего в готовом release image не найдено.

## Полный Dockerfile выбранного варианта

Файл приложен отдельно как `manus-3-1-opm-flow-2026.04-pinch935.Dockerfile`. Следующая копия идентична и полна. Каждый крупный build-step ссылается на источник в комментариях Dockerfile и в разделах выше.

```dockerfile
# syntax=docker/dockerfile:1
# OPM Flow 2026.04, opm-grid PR #935 backport.
# Source notes: https://opm-project.org/?page_id=231 and https://opm-project.org/?page_id=239
# This Dockerfile intentionally installs NO libopm-* package from ppa:opm/ppa.

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=4
ARG OPM_COMMON_COMMIT=0ea62974f24d70fc2b3e30d6aae8b76ef000dac1
ARG OPM_GRID_RELEASE_COMMIT=d6a1a35c3d770fe50073b9ed866bbec740c8faac
ARG OPM_SIMULATORS_COMMIT=b82f21dba405286c4c4446614dd3bf9cdebf7a2c
ARG OPM_GRID_FIX_COMMIT=ef3fad537be6b94c35011f4014eb67fcf46be56c
ARG OPM_GRID_TEST_COMMIT=7b665216985439a72fe908e574c6c2ffe5017e1a
ARG OPM_GRID_PATCHED_HEAD=9869209d28dcf797997ffc18e41fab506191744f
ARG DUNE_VERSION=2.11.0-1~noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
 && add-apt-repository -y ppa:opm/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gfortran pkg-config cmake git \
      mpi-default-dev libblas-dev liblapack-dev libboost-all-dev \
      libsuitesparse-dev libtrilinos-zoltan-dev libfmt-dev libcjson-dev \
      libhdf5-mpi-dev libptscotch-dev \
      libdune-common-dev=${DUNE_VERSION} \
      libdune-geometry-dev=${DUNE_VERSION} \
      libdune-istl-dev=${DUNE_VERSION} \
      libdune-grid-dev=${DUNE_VERSION} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN git clone https://github.com/OPM/opm-common.git opm-common \
 && git -C opm-common checkout --detach "${OPM_COMMON_COMMIT}" \
 && test "$(git -C opm-common rev-parse HEAD)" = "${OPM_COMMON_COMMIT}" \
 && git clone https://github.com/OPM/opm-grid.git opm-grid \
 && git -C opm-grid checkout --detach "${OPM_GRID_RELEASE_COMMIT}" \
 && test "$(git -C opm-grid rev-parse HEAD)" = "${OPM_GRID_RELEASE_COMMIT}" \
 && git clone https://github.com/OPM/opm-simulators.git opm-simulators \
 && git -C opm-simulators checkout --detach "${OPM_SIMULATORS_COMMIT}" \
 && test "$(git -C opm-simulators rev-parse HEAD)" = "${OPM_SIMULATORS_COMMIT}"

RUN cd /src/opm-grid \
 && git config user.name 'Markus Blatt' \
 && git config user.email 'markus@dr-blatt.de' \
 && GIT_COMMITTER_DATE='2025-10-14T22:15:34+02:00' git cherry-pick --no-edit "${OPM_GRID_FIX_COMMIT}" \
 && GIT_COMMITTER_DATE='2025-10-24T08:57:47+02:00' git cherry-pick --no-edit "${OPM_GRID_TEST_COMMIT}" \
 && test "$(git rev-parse HEAD)" = "${OPM_GRID_PATCHED_HEAD}" \
 && git diff --check HEAD~2..HEAD

RUN install -d /opt/opm/share/opm-build \
 && { \
      printf 'opm-common %s\n' "$(git -C /src/opm-common rev-parse HEAD)"; \
      printf 'opm-grid-release %s\n' "${OPM_GRID_RELEASE_COMMIT}"; \
      printf 'opm-grid-fix %s\n' "${OPM_GRID_FIX_COMMIT}"; \
      printf 'opm-grid-test %s\n' "${OPM_GRID_TEST_COMMIT}"; \
      printf 'opm-grid-patched-head %s\n' "$(git -C /src/opm-grid rev-parse HEAD)"; \
      printf 'opm-simulators %s\n' "$(git -C /src/opm-simulators rev-parse HEAD)"; \
    } > /opt/opm/share/opm-build/git-revisions.txt

RUN cmake -S /src/opm-common -B /build/opm-common \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opm -DWITH_NATIVE=OFF \
 && cmake --build /build/opm-common --parallel "${BUILD_JOBS}" \
 && cmake --install /build/opm-common

RUN cmake -S /src/opm-grid -B /build/opm-grid \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opm \
      -DCMAKE_PREFIX_PATH=/opt/opm -DWITH_NATIVE=OFF \
 && cmake --build /build/opm-grid --parallel "${BUILD_JOBS}" \
 && ctest --test-dir /build/opm-grid --output-on-failure -R '^minpvprocessor$' \
      | tee /opt/opm/share/opm-build/grid-minpvprocessor.ctest.log \
 && cmake --install /build/opm-grid

RUN cmake -S /src/opm-simulators -B /build/opm-simulators \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opm \
      -DCMAKE_PREFIX_PATH=/opt/opm -DSIBLING_SEARCH=OFF \
      -Dopm-common_DIR=/opt/opm/lib/cmake/opm-common \
      -Dopm-grid_DIR=/opt/opm/lib/cmake/opm-grid \
      -DBUILD_FLOW_VARIANTS=OFF -DUSE_GPU_BRIDGE=OFF -DUSE_OPENCL=OFF -DWITH_NATIVE=OFF \
 && cmake --build /build/opm-simulators --target flow --parallel "${BUILD_JOBS}" \
 && cmake --install /build/opm-simulators \
 && /opt/opm/bin/flow --version | tee /opt/opm/share/opm-build/flow-version.txt \
 && sha256sum /opt/opm/bin/flow > /opt/opm/share/opm-build/flow.sha256 \
 && dpkg-query -W -f='${binary:Package}\t${Version}\n' | sort \
      > /opt/opm/share/opm-build/dpkg-packages.tsv

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG DUNE_VERSION=2.11.0-1~noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
 && add-apt-repository -y ppa:opm/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      mpi-default-bin libblas-dev liblapack-dev libboost-all-dev \
      libsuitesparse-dev libtrilinos-zoltan-dev libfmt-dev libcjson-dev \
      libhdf5-mpi-dev libptscotch-dev \
      libdune-common-dev=${DUNE_VERSION} \
      libdune-geometry-dev=${DUNE_VERSION} \
      libdune-istl-dev=${DUNE_VERSION} \
      libdune-grid-dev=${DUNE_VERSION} \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/opm /opt/opm

RUN echo '/opt/opm/lib' > /etc/ld.so.conf.d/opm.conf \
 && ldconfig \
 && /opt/opm/bin/flow --version \
 && sha256sum -c /opt/opm/share/opm-build/flow.sha256

ENV PATH=/opt/opm/bin:${PATH}

LABEL org.opencontainers.image.title="OPM Flow 2026.04 with opm-grid PR 935" \
      org.opencontainers.image.description="Full-source OPM Flow 2026.04 backport of PINCH collapsed-cell NNC fix" \
      org.opencontainers.image.version="2026.04-pinch935" \
      org.opencontainers.image.source="https://github.com/OPM/opm-grid"

WORKDIR /work
CMD ["flow", "--version"]
```

### Сборка и фиксация артефакта

Следующие команды являются операционным продолжением Dockerfile. Они не используют `latest` в имени собственного образа и сохраняют digest готового OCI-образа, список Git-ревизий, версию Flow, версию пакетов и SHA-256 бинаря.

```bash
# Источник: официальный порядок CMake/CTest [1] и pinning в Dockerfile выше.
docker build \
  --build-arg BUILD_JOBS=8 \
  --file manus-3-1-opm-flow-2026.04-pinch935.Dockerfile \
  --tag local/opm-flow:2026.04-pinch935 \
  .

docker image inspect local/opm-flow:2026.04-pinch935 \
  --format '{{index .RepoDigests 0}} {{.Size}}'

docker run --rm local/opm-flow:2026.04-pinch935 \
  cat /opt/opm/share/opm-build/git-revisions.txt

docker run --rm local/opm-flow:2026.04-pinch935 \
  sh -c 'cat /opt/opm/share/opm-build/flow-version.txt; cat /opt/opm/share/opm-build/flow.sha256; cat /opt/opm/share/opm-build/dpkg-packages.tsv'
```

Базовый digest и Git SHA исключают неявное обновление базового образа и исходников. APT пакеты Dune закреплены по версии; все фактически разрешённые пакеты записываются в `dpkg-packages.tsv`. Чтобы получить byte-for-byte повторяемую сборку спустя длительное время, дополнительно нужен сохранённый content-addressed snapshot используемых Ubuntu и PPA `.deb`; готовый официальный OPM-рецепт такого APT-lockfile **не найдено**.

## Конечная команда проверки наличия патча в собранном Flow

Команда ниже должна завершаться с кодом 0. Она проверяет контрольную сумму именно собранного `flow`, точные исходные SHA, детерминированный локальный SHA после двух backport-коммитов и факт успешного запуска регрессионного теста в процессе сборки. Это корректнее, чем искать текст патча в оптимизированном ELF-бинаре.

```bash
docker run --rm --entrypoint /bin/bash local/opm-flow:2026.04-pinch935 -euo pipefail -c '
  sha256sum -c /opt/opm/share/opm-build/flow.sha256
  grep -Fx "opm-grid-release d6a1a35c3d770fe50073b9ed866bbec740c8faac" /opt/opm/share/opm-build/git-revisions.txt
  grep -Fx "opm-grid-fix ef3fad537be6b94c35011f4014eb67fcf46be56c" /opt/opm/share/opm-build/git-revisions.txt
  grep -Fx "opm-grid-test 7b665216985439a72fe908e574c6c2ffe5017e1a" /opt/opm/share/opm-build/git-revisions.txt
  grep -Fx "opm-grid-patched-head 9869209d28dcf797997ffc18e41fab506191744f" /opt/opm/share/opm-build/git-revisions.txt
  grep -F "100% tests passed" /opt/opm/share/opm-build/grid-minpvprocessor.ctest.log
  flow --version
'
```

## Источники

[1]: https://opm-project.org/?page_id=231 "OPM — Building from source"
[2]: https://opm-project.org/?page_id=239 "OPM — Prerequisites"
[3]: https://opm-project.org/?p=2122 "OPM — New Release 2026.04"
[4]: https://github.com/OPM/opm-grid/commit/ef3fad537be6b94c35011f4014eb67fcf46be56c "OPM/opm-grid commit ef3fad5 — Skip NNCs created by PINCH"
[5]: https://github.com/OPM/opm-grid/commit/7b665216985439a72fe908e574c6c2ffe5017e1a "OPM/opm-grid commit 7b66521 — tests for collapsed PINCH cells"
[6]: https://launchpad.net/~opm/+archive/ubuntu/ppa "Open Porous Media Initiative PPA — пакеты Noble"
[7]: https://launchpad.net/ubuntu/+source/opm-grid "Launchpad — opm-grid package и PPA-версии"
[8]: https://opm-project.org/?page_id=853&page=2 "OPM — Running Flow in Docker"
[9]: https://hub.docker.com/r/openporousmedia/opmreleases/tags "Docker Hub — openporousmedia/opmreleases tags"
[10]: https://raw.githubusercontent.com/OPM/opm-simulators/b82f21dba405286c4c4446614dd3bf9cdebf7a2c/CMakeLists.txt "OPM/opm-simulators 2026.04 CMake configuration"
[11]: https://github.com/OPM/opm-models "OPM/opm-models repository"
[12]: https://man7.org/linux/man-pages/man8/ld.so.8.html "ld.so(8) — динамический загрузчик Linux"
