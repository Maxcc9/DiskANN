#!/usr/bin/env bash
# 依 LHS CSV 批次執行 search_disk_index，可選擇乾跑 (DRY_RUN=1) 先驗證流程。

set -euo pipefail

usage() {
    cat <<'USAGE'
用法:
  bash run_batch.sh <lhs_csv> [dataset_name] [max_parallel_jobs]

參數:
  lhs_csv            gen_lhs.py 產生的樣本檔（含表頭）
  dataset_name       預設 siftsmall，可覆寫資料路徑
  max_parallel_jobs  預設 4

環境變數可覆寫:
  INDEX_PREFIX, QUERY_FILE, GT_FILE, RESULT_DIR, TOPK
  DRY_RUN=1 時僅印出指令不執行 search_disk_index
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }
[[ $# -lt 1 ]] && { usage; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
SEARCH_BIN="${APPS_DIR}/search_disk_index"

LHS_CSV="$1"
DATASET="${2:-siftsmall}"
MAX_PARALLEL="${3:-4}"
TOPK_OVERRIDE="${TOPK:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$LHS_CSV" ]]; then
    echo "ERROR: 找不到 LHS CSV: $LHS_CSV" >&2
    exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "ERROR: max_parallel_jobs 需為正整數" >&2
    exit 1
fi

INDEX_PREFIX="${INDEX_PREFIX:-${DISKANN_ROOT}/data/${DATASET}/index/disk/${DATASET}_R64_L100_B0.2_M1}"
QUERY_FILE="${QUERY_FILE:-${DISKANN_ROOT}/data/${DATASET}/${DATASET}_query.bin}"
GT_FILE="${GT_FILE:-${DISKANN_ROOT}/data/${DATASET}/${DATASET}_groundtruth.bin}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/batch_results}"

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -x "$SEARCH_BIN" ]]; then
        echo "ERROR: search_disk_index 不存在或不可執行: $SEARCH_BIN" >&2
        echo "請先編譯: cmake --build build --target all -- -j" >&2
        exit 1
    fi

    # 檢查動態庫缺失，避免跑第一筆就中斷
    ldd_missing=$(ldd "$SEARCH_BIN" | awk '/not found/ {print $1}')
    if [[ -n "$ldd_missing" ]]; then
        echo "ERROR: search_disk_index 缺少動態庫: $ldd_missing" >&2
        echo "請在執行環境安裝/設定上述庫 (例如 libaio)" >&2
        exit 1
    fi

    for f in "${INDEX_PREFIX}_disk.index" "$QUERY_FILE" "$GT_FILE"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: 找不到必要檔案: $f" >&2
            exit 1
        fi
    done
else
    if [[ ! -x "$SEARCH_BIN" ]]; then
        echo "WARN: DRY_RUN 模式忽略不存在的 search_disk_index: $SEARCH_BIN" >&2
    fi
    for f in "${INDEX_PREFIX}_disk.index" "$QUERY_FILE" "$GT_FILE"; do
        if [[ ! -f "$f" ]]; then
            echo "WARN: DRY_RUN 模式忽略缺少檔案: $f" >&2
        fi
    done
fi

mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "DiskANN 批次搜尋"
echo "=========================================="
echo "Dataset       : $DATASET"
echo "LHS Samples   : $LHS_CSV"
echo "Output Dir    : $RESULT_DIR"
echo "Max Parallel  : $MAX_PARALLEL"
echo "Top-K (K)     : ${TOPK_OVERRIDE:-'(CSV 或預設 10)'}"
echo "Dry Run       : $DRY_RUN"
echo ""

run_one_sample() {
    local sample_id="$1" L="$2" W="$3" cache="$4" io_limit="$5" threads="$6" k_value="$7"
    local prefix="${RESULT_DIR}/result_sample_${sample_id}"
    local log_file="${RESULT_DIR}/sample_${sample_id}.log"
    local stats_csv="${prefix}_search.csv"

    echo "▶ Sample #${sample_id}: L=${L} W=${W} cache=${cache} io_limit=${io_limit} T=${threads} K=${k_value}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY-RUN: ${SEARCH_BIN} --data_type float --dist_fn l2 --index_path_prefix ${INDEX_PREFIX} --query_file ${QUERY_FILE} --gt_file ${GT_FILE} --result_path ${prefix} --stats_csv_path ${stats_csv} --num_nodes_to_cache ${cache} --search_io_limit ${io_limit} --num_threads ${threads} -K ${k_value} -L ${L} -W ${W} -A" \
            > "${log_file}"
        echo "✓ Sample #${sample_id} 完成 (dry-run)"
        return 0
    fi

    if ! "${SEARCH_BIN}" \
        --data_type float \
        --dist_fn l2 \
        --index_path_prefix "${INDEX_PREFIX}" \
        --query_file "${QUERY_FILE}" \
        --gt_file "${GT_FILE}" \
        --result_path "${prefix}" \
        --stats_csv_path "${stats_csv}" \
        --num_nodes_to_cache "${cache}" \
        --search_io_limit "${io_limit}" \
        --num_threads "${threads}" \
        -K "${k_value}" \
        -L "${L}" \
        -W "${W}" \
        -A \
        > "${log_file}" 2>&1 < /dev/null; then
        echo "✗ Sample #${sample_id} 失敗，請檢查 ${log_file}"
        return 1
    fi
    echo "✓ Sample #${sample_id} 完成"
}

strip_ws() { echo "$1" | tr -d ' \t\r\n'; }

running=0
fail=0
declare -A pid_to_sample
pids=()

exec 3< "$LHS_CSV"
read -r _header <&3
while IFS=',' read -r sample_id L W cache io_limit threads csv_k _rest <&3; do
    sample_id=$(strip_ws "${sample_id:-}")
    [[ -z "$sample_id" ]] && continue

    L=$(strip_ws "${L:-}")
    W=$(strip_ws "${W:-}")
    cache=$(strip_ws "${cache:-}")
    io_limit=$(strip_ws "${io_limit:-}")
    threads=$(strip_ws "${threads:-}")
    csv_k=$(strip_ws "${csv_k:-}")

    k_value="${TOPK_OVERRIDE:-${csv_k:-10}}"

    if (( MAX_PARALLEL == 1 )); then
        if ! run_one_sample "$sample_id" "$L" "$W" "$cache" "$io_limit" "$threads" "$k_value"; then
            fail=1
        fi
    else
        run_one_sample "$sample_id" "$L" "$W" "$cache" "$io_limit" "$threads" "$k_value" </dev/null &
        pid=$!
        pid_to_sample[$pid]="sample_${sample_id}"
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
echo "批次搜尋完成，結果位於: $RESULT_DIR"
echo "可用指令整併結果:"
echo "  python collect.py --input_dir $RESULT_DIR --lhs_file $LHS_CSV --output results_all.csv"
echo "=========================================="

exit $fail
