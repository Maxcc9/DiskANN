#!/usr/bin/env bash
# 依 ./inputFiles/build_configs.csv 批次建置 DiskANN index。
# 必要輸入：./inputFiles/build_configs.csv (build_id,build_R,build_L)
# 輸出位置: ./outputFiles/build/
# 可用 DRY_RUN=1 先驗證指令。

set -euo pipefail

usage() {
    cat <<'USAGE'
用法:
  bash build_batch.sh [build_csv] [dataset_name] [max_parallel_jobs]

參數:
  build_csv           預設 ./inputFiles/build_configs.csv（含表頭）
  dataset_name        預設 siftsmall，可覆寫資料路徑
  max_parallel_jobs   預設 2（建置較耗時，可酌情提高）

環境變數可覆寫:
  DATA_FILE, OUTPUT_DIR, DIST_FN, DATA_TYPE
  BUILD_B, BUILD_M, PQ_DISK_BYTES, BUILD_PQ_BYTES, NUM_THREADS
  APPEND_PARAMS=1 時使用 -A 自動附加參數到檔名前綴
  EXTRA_ARGS 可補充 build_disk_index 的其他參數
  DRY_RUN=1 時僅印出指令不執行 build_disk_index
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
BUILD_BIN="${APPS_DIR}/build_disk_index"

BUILD_CSV="${1:-${SCRIPT_DIR}/inputFiles/build_configs.csv}"
DATASET="${2:-siftsmall}"
MAX_PARALLEL="${3:-2}"
DRY_RUN="${DRY_RUN:-0}"
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"

BUILD_B="${BUILD_B:-0.2}"
BUILD_M="${BUILD_M:-1}"
PQ_DISK_BYTES="${PQ_DISK_BYTES:-0}"
BUILD_PQ_BYTES="${BUILD_PQ_BYTES:-0}"
NUM_THREADS="${NUM_THREADS:-8}"
APPEND_PARAMS="${APPEND_PARAMS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$BUILD_CSV" ]]; then
    echo "ERROR: 找不到 build CSV: $BUILD_CSV" >&2
    exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "ERROR: max_parallel_jobs 需為正整數" >&2
    exit 1
fi
if [[ "$APPEND_PARAMS" != "0" && "$APPEND_PARAMS" != "1" ]]; then
    echo "ERROR: APPEND_PARAMS 需為 0 或 1" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputFiles/build}"
DATA_FILE="${DATA_FILE:-${DISKANN_ROOT}/data/${DATASET}/${DATASET}_base.bin}"
mkdir -p "$OUTPUT_DIR"

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -x "$BUILD_BIN" ]]; then
        echo "ERROR: build_disk_index 不存在或不可執行: $BUILD_BIN" >&2
        echo "請先編譯: cmake --build build --target all -- -j" >&2
        exit 1
    fi
    if [[ ! -f "$DATA_FILE" ]]; then
        echo "ERROR: 找不到必要檔案: $DATA_FILE" >&2
        exit 1
    fi
else
    if [[ ! -x "$BUILD_BIN" ]]; then
        echo "WARN: DRY_RUN 模式忽略不存在的 build_disk_index: $BUILD_BIN" >&2
    fi
    if [[ ! -f "$DATA_FILE" ]]; then
        echo "WARN: DRY_RUN 模式忽略缺少檔案: $DATA_FILE" >&2
    fi
fi

strip_ws() { echo "$1" | tr -d '[:space:]'; }

run_one() {
    local build_id="$1" R="$2" L="$3"
    local prefix_base="${OUTPUT_DIR}/${DATASET}"
    local prefix_path
    local append_flag=()
    local display_prefix
    local log_file="${OUTPUT_DIR}/build_${build_id}.log"

    if [[ "$APPEND_PARAMS" == "1" ]]; then
        prefix_path="$prefix_base"
        append_flag=(-A)
        display_prefix="${prefix_base}_R${R}_L${L}_B${BUILD_B}_M${BUILD_M}"
    else
        prefix_path="${OUTPUT_DIR}/${DATASET}_R${R}_L${L}"
        display_prefix="${prefix_path}"
    fi

    echo "▶ Build #${build_id}: R=${R} L=${L} -> ${display_prefix}_disk.index"

    cmd=(
        "${BUILD_BIN}"
        --data_type "${DATA_TYPE}"
        --dist_fn "${DIST_FN}"
        --data_path "${DATA_FILE}"
        --index_path_prefix "${prefix_path}"
        -R "${R}"
        -L "${L}"
        -B "${BUILD_B}"
        -M "${BUILD_M}"
        --PQ_disk_bytes "${PQ_DISK_BYTES}"
        --build_PQ_bytes "${BUILD_PQ_BYTES}"
        --num_threads "${NUM_THREADS}"
    )
    cmd+=("${append_flag[@]}")
    if [[ -n "$EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        cmd+=($EXTRA_ARGS)
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN: %q ' "${cmd[@]}" > "$log_file"
        printf '\n' >> "$log_file"
        echo "✓ Build #${build_id} 完成 (dry-run)"
        return 0
    fi

    if ! "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null; then
        echo "✗ Build #${build_id} 失敗，請檢查 ${log_file}"
        return 1
    fi
    echo "✓ Build #${build_id} 完成"
}

running=0
fail=0
declare -A pid_to_sample
pids=()

exec 3< "$BUILD_CSV"
read -r _header <&3
while IFS=',' read -r build_id R L _rest <&3; do
    build_id=$(strip_ws "${build_id:-}")
    [[ -z "$build_id" ]] && continue
    R=$(strip_ws "${R:-}")
    L=$(strip_ws "${L:-}")

    if (( MAX_PARALLEL == 1 )); then
        if ! run_one "$build_id" "$R" "$L" 3<&-; then
            fail=1
        fi
    else
        run_one "$build_id" "$R" "$L" </dev/null 3<&- &
        pid=$!
        pid_to_sample[$pid]="build_${build_id}"
        running=$((running+1))
        pids+=("$pid")

        if (( running >= MAX_PARALLEL )); then
            oldest_pid="${pids[0]}"
            pids=("${pids[@]:1}")
            if ! wait "$oldest_pid"; then
                echo "WARN: ${pid_to_sample[$oldest_pid]:-unknown} 失敗，持續處理其餘樣本" >&2
                fail=1
            fi
            unset "pid_to_sample[$oldest_pid]"
            running=$((running-1))
        fi
    fi
done
exec 3<&-

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: ${pid_to_sample[$pid]:-unknown} 失敗，請檢查對應 log" >&2
        fail=1
    fi
    unset "pid_to_sample[$pid]"
    running=$((running-1))
done

echo ""
echo "=========================================="
echo "建置完成，索引位於: $OUTPUT_DIR"
if [[ "$APPEND_PARAMS" == "1" ]]; then
    echo "索引前綴格式: ${OUTPUT_DIR}/${DATASET}_R{R}_L{L}_B{B}_M{M}"
else
    echo "索引前綴格式: ${OUTPUT_DIR}/${DATASET}_R{R}_L{L}"
fi
echo "=========================================="

exit $fail
