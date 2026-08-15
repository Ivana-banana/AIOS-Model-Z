# Среда расчёта: OPM Flow + наш код. Одна команда вместо установки симулятора
# руками — это и требование регламента к воспроизводимости, и способ поднять
# облачную машину под пакетные прогоны.
#
#   docker build -t aios -f docker/Dockerfile .
#   docker run --rm -v "$PWD:/work" -w /work aios python scripts/build_case.py
#   docker run --rm -v "$PWD:/work" -w /work aios scripts/run_flow.sh runs/base/RUN.DATA 8
#
# Дек организаторов в образ НЕ кладём: он их, весит 78 МБ и монтируется томом.

FROM ubuntu:24.04

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
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY aios /opt/aios/aios
COPY scripts /opt/aios/scripts
COPY config.toml /opt/aios/config.toml
COPY vendor/CHDD_PYTHON /opt/aios/vendor/CHDD_PYTHON
RUN chmod +x /opt/aios/scripts/*.sh

ENV PYTHONPATH=/opt/aios
WORKDIR /work

# Сборка обязана падать здесь, а не на первом прогоне в облаке.
RUN flow --version && python -c "import resdata, openpyxl; print('resdata ok')"

CMD ["bash"]
