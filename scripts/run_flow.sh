#!/usr/bin/env bash
# Запуск OPM Flow на собранном кейсе.
#
#   scripts/run_flow.sh runs/base/RUN.DATA 8
#
# Второй аргумент — число процессов MPI (по умолчанию все ядра машины).
#
# Скрипт делает три вещи, которых не делает сам симулятор:
#   1. меряет время счёта — оно важно не меньше результата: от него зависит,
#      сколько прогонов мы можем позволить себе на датасет суррогата;
#   2. печатает отдельным блоком всё, что симулятор не понял и пропустил. Flow
#      не падает на незнакомом ключевом слове — он его игнорирует и считает
#      дальше, поэтому «прогон без ошибок» ещё ничего не значит;
#   3. проверяет, что результат (.UNSMRY) вообще появился.
set -euo pipefail

DATA_FILE="${1:?укажите .DATA файл: scripts/run_flow.sh runs/base/RUN.DATA [процессов]}"
PROCESSES="${2:-$(nproc 2>/dev/null || echo 1)}"

if [[ ! -f "$DATA_FILE" ]]; then
    echo "нет такого файла: $DATA_FILE" >&2
    exit 1
fi
if ! command -v flow >/dev/null 2>&1; then
    echo "не найден flow (OPM Flow). Установка — docs/setup-simulator.md," >&2
    echo "или считайте в образе: docker build -t aios -f docker/Dockerfile ." >&2
    exit 1
fi

CASE_DIR="$(cd "$(dirname "$DATA_FILE")" && pwd)"
CASE_NAME="$(basename "${DATA_FILE%.*}")"
LOG_FILE="$CASE_DIR/$CASE_NAME.run.log"

echo "кейс:      $DATA_FILE"
echo "процессов: $PROCESSES"
echo "лог:       $LOG_FILE"
echo "flow:      $(flow --version 2>&1 | head -1)"
echo

START=$(date +%s)
set +e
if [[ "$PROCESSES" -gt 1 ]] && command -v mpirun >/dev/null 2>&1; then
    # --allow-run-as-root: в контейнере процесс идёт от root, без флага mpirun встаёт
    mpirun --allow-run-as-root -np "$PROCESSES" flow "$DATA_FILE" \
        --output-dir="$CASE_DIR" 2>&1 | tee "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
else
    flow "$DATA_FILE" --output-dir="$CASE_DIR" 2>&1 | tee "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
fi
set -e
ELAPSED=$(( $(date +%s) - START ))

echo
echo "время счёта: $((ELAPSED / 3600)) ч $(((ELAPSED % 3600) / 60)) мин $((ELAPSED % 60)) с (${ELAPSED} с)"
echo "код возврата: $STATUS"

echo
echo "--- пропущенные и непонятые ключевые слова ---"
# Flow сообщает о них по-разному в разных версиях, поэтому ловим все формулировки.
if grep -n -i -E "unsupported keyword|not supported|unrecognized|ignoring keyword|keyword .* is ignored|Unknown keyword" \
        "$LOG_FILE" "$CASE_DIR/$CASE_NAME.PRT" 2>/dev/null | sort -u -t: -k3 | head -40; then
    echo "(проверьте каждое: молча пропущенное ключевое слово меняет физику расчёта)"
else
    echo "нет"
fi

echo
echo "--- ошибки и предупреждения ---"
grep -c -i "error" "$LOG_FILE" 2>/dev/null | sed 's/^/строк с error: /' || true
grep -c -i "warning" "$LOG_FILE" 2>/dev/null | sed 's/^/строк с warning: /' || true

if [[ ! -f "$CASE_DIR/$CASE_NAME.UNSMRY" ]]; then
    echo
    echo "результата нет: $CASE_DIR/$CASE_NAME.UNSMRY не создан" >&2
    exit "${STATUS:-1}"
fi

echo
echo "результат: $CASE_DIR/$CASE_NAME.UNSMRY"
echo "дальше:    python scripts/export_rows.py $CASE_DIR"
exit "$STATUS"
