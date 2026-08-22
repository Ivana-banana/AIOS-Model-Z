#!/usr/bin/env bash
# Посчитать все ячейки плана коалиций подряд.
#
#   scripts/run_shapley.sh ~/runs/shapley 8
#
# Второй аргумент — число процессов MPI. Кейсы обязаны быть уже собраны:
#   python scripts/shapley_plan.py build ~/runs/shapley --merge-producers
#
# Уже посчитанные ячейки пропускаются по наличию `rows.csv`: прогон на 2,5 часа
# переживает перезапуск, и повторять сделанное незачем. Ошибка одной ячейки не
# останавливает остальные — план всё равно неполон, но лучше знать, сколько
# именно ячеек не сошлось, чем узнавать это по одной.
set -u

PLAN_DIR="${1:?укажите каталог плана: scripts/run_shapley.sh ~/runs/shapley [процессов]}"
PROCESSES="${2:-$(nproc 2>/dev/null || echo 1)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [[ ! -f "$PLAN_DIR/plan.json" ]]; then
    echo "нет плана: $PLAN_DIR/plan.json — сначала shapley_plan.py build" >&2
    exit 1
fi

failed=0
for dir in "$PLAN_DIR"/c*/; do
    case_name="$(basename "$dir")"
    if [[ -f "$dir/rows.csv" ]]; then
        echo "== $case_name уже посчитан, пропуск"
        continue
    fi
    echo "== $case_name начат $(date +%H:%M:%S)"
    if bash "$ROOT/scripts/run_flow.sh" "$dir/RUN.DATA" "$PROCESSES" \
            --parsing-strictness=low > "$dir/flow.out" 2>&1 \
       && (cd "$ROOT" && PYTHONPATH=. "$PYTHON" scripts/export_rows.py "$dir") \
            >> "$dir/flow.out" 2>&1; then
        echo "== $case_name готов $(date +%H:%M:%S)"
    else
        failed=$((failed + 1))
        echo "== $case_name НЕ СОШЁЛСЯ, хвост лога:"
        tail -5 "$dir/flow.out"
    fi
done

echo "== пройдено, не сошлось ячеек: $failed"
[[ $failed -eq 0 ]] || exit 1
