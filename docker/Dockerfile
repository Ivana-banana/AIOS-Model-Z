# Среда расчёта: OPM Flow + наш код. Одна команда вместо установки симулятора
# руками — это и требование регламента к воспроизводимости, и способ поднять
# облачную машину под пакетные прогоны.
#
#   docker build --platform linux/amd64 -t aios -f docker/Dockerfile .
#
# `--platform linux/amd64` не украшение: ниже пакеты ставятся с
# `--require-hashes`, а в `requirements.lock` лежат хэши колёс только под
# linux-x86_64. На arm64 (Apple Silicon) сборка упадёт на несовпадении хэшей —
# и это правильное поведение lock-файла, а не его дефект.
#   docker run --rm -v "$PWD:/work" -w /work aios python scripts/build_case.py
#   docker run --rm -v "$PWD:/work" -w /work aios scripts/run_flow.sh runs/base/RUN.DATA 8
#
# Дек организаторов в образ НЕ кладём: он их, весит 78 МБ и монтируется томом.

# База закреплена по digest, а не только по тегу: `ubuntu:24.04` — движущаяся
# цель, её пересобирают, и «тот же Dockerfile» через месяц даёт другой образ.
# Правило одно на оба файла: `Dockerfile.pinch935` пинуется так же и этим же
# digest'ом.
#
# Проверено 22.08.2026 запросом в Docker Hub (без Docker, curl к registry API):
# манифест по этому digest существует, это multi-arch индекс, созданный
# 10.08.2026, и тег `24.04` на 22.08 указывает ровно на него.
# Обновление digest — осознанное действие: взять новый ответ
# `docker-content-digest` для тега, вписать сюда, пересобрать и перепроверить
# прогон, а не «подтянуть свежее» молча.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

# Тихая установка: иначе tzdata остановится на диалоге и сборка повиснет.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# ctranslate2/numpy иначе разбирают все ядра машины между собой при пакетных прогонах
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        gnupg \
    && add-apt-repository -y ppa:opm/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        libopm-simulators-bin \
        mpi-default-bin \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 запрещает pip в системный python (PEP 668), поэтому своё окружение.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Зависимости ставятся до копирования кода: слой переиспользуется между сборками.
#
# Ставится lock, а не requirements.txt: прямые пакеты запинены все, но resolver
# тянет ещё 31 транзитивный — включая NumPy, pandas и SciPy, которые приходят
# через resdata и scikit-learn. Дрейф не гипотетический: внешняя проверка 21.08
# разрешила SciPy 1.18.0, тот же файл 22.08 даёт 1.18.1.
#
# --require-hashes проверяет каждое колесо и заодно запрещает незапиненные
# строки: если lock перестанет покрывать граф, сборка упадёт, а не поставит
# что-нибудь свежее молча. Пересобрать lock — scripts/make_lock.py.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

COPY aios /opt/aios/aios
COPY scripts /opt/aios/scripts
COPY config.toml /opt/aios/config.toml
COPY vendor/CHDD_PYTHON /opt/aios/vendor/CHDD_PYTHON

# Лицензии едут вместе с образом, а не остаются в репозитории: получатель образа
# получает Flow и resdata под GPLv3, а §4 требует передать копию лицензии вместе
# с программой. `PROVENANCE_AND_RIGHTS.md` сюда намеренно не кладётся — это наш
# внутренний реестр прав, а не документ получателя.
COPY LICENSES /opt/aios/LICENSES
COPY LICENSE THIRD_PARTY_NOTICES.md MODIFICATIONS.md /opt/aios/
RUN chmod +x /opt/aios/scripts/*.sh

ENV PYTHONPATH=/opt/aios
WORKDIR /work

# Веб-платформа: docker run -p 8000:8000 aios uvicorn aios.web.app:app --host 0.0.0.0
EXPOSE 8000

# Сборка обязана падать здесь, а не на первом прогоне в облаке.
RUN flow --version && python -c "import resdata, openpyxl; print('resdata ok')"

CMD ["bash"]
