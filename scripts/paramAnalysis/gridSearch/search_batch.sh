#!/usr/bin/env bash
# 依 ./inputFiles/search_configs.csv 批次對 ./outputFiles/build 內的索引進行搜尋。
# 必要輸入：./inputFiles/search_configs.csv (search_id,search_W,search_L,search_cache,search_thread)
# 輸出位置: ./outputFiles/search/
# 可用 DRY_RUN=1 先驗證指令。

set -euo pipefail

usage() {
    cat <<'USAGE'
用法:
  bash search_batch.sh [SEARCH_CSV] [DATASET] [MAX_PARALLEL]

參數:
  SEARCH_CSV     預設 ./inputFiles/search_configs.csv（含表頭）
  DATASET        預設自動從 index 檔名解析；可手動覆寫
  MAX_PARALLEL   預設 4

環境變數可覆寫:
  BUILD_DIR, OUTPUT_DIR, DATA_TYPE, DIST_FN
  QUERY_FILE, GT_FILE, TOPK, SEARCH_IO_LIMIT, THREAD_OVERRIDE
  APPEND_SEARCH_PARAMS=1 時使用 -A 自動附加搜尋參數到 result_path
  EXTRA_ARGS 可補充 search_disk_index 的其他參數
  ENABLE_IOSTAT=1 時為每筆樣本記錄 iostat
  IOSTAT_INTERVAL=1, IOSTAT_DEVICE, IOSTAT_DATA_PATH
  ENABLE_EXPANDED_NODES=1 時為每筆樣本輸出 expanded_nodes CSV
  EXPANDED_NODES_LIMIT=0 (0 = unlimited)
  DRY_RUN=1 時僅印出指令不執行 search_disk_index
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
SEARCH_BIN="${APPS_DIR}/search_disk_index"

SEARCH_CSV="${1:-${SCRIPT_DIR}/inputFiles/search_configs.csv}"
DATASET="${2:-}"
MAX_PARALLEL="${3:-4}"
DRY_RUN="${DRY_RUN:-0}"
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
TOPK="${TOPK:-10}"
SEARCH_IO_LIMIT="${SEARCH_IO_LIMIT:-}"
THREAD_OVERRIDE="${THREAD_OVERRIDE:-}"
APPEND_SEARCH_PARAMS="${APPEND_SEARCH_PARAMS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ENABLE_IOSTAT="${ENABLE_IOSTAT:-0}"
IOSTAT_INTERVAL="${IOSTAT_INTERVAL:-1}"
IOSTAT_DEVICE="${IOSTAT_DEVICE:-}"
IOSTAT_DATA_PATH="${IOSTAT_DATA_PATH:-}"
ENABLE_EXPANDED_NODES="${ENABLE_EXPANDED_NODES:-0}"
EXPANDED_NODES_LIMIT="${EXPANDED_NODES_LIMIT:-0}"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/outputFiles/build}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputFiles/search}"

if [[ ! -f "$SEARCH_CSV" ]]; then
    echo "ERROR: 找不到 SEARCH_CSV: $SEARCH_CSV" >&2
    exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "ERROR: MAX_PARALLEL 需為正整數" >&2
    exit 1
fi
if [[ "$ENABLE_IOSTAT" == "1" && "$MAX_PARALLEL" -ne 1 ]]; then
    echo "WARN: ENABLE_IOSTAT=1 建議單一序列執行，已將 MAX_PARALLEL 強制為 1" >&2
    MAX_PARALLEL=1
fi
if [[ "$APPEND_SEARCH_PARAMS" != "0" && "$APPEND_SEARCH_PARAMS" != "1" ]]; then
    echo "ERROR: APPEND_SEARCH_PARAMS 需為 0 或 1" >&2
    exit 1
fi
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: 找不到 BUILD_DIR 目錄: $BUILD_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -x "$SEARCH_BIN" ]]; then
        echo "ERROR: search_disk_index 不存在或不可執行: $SEARCH_BIN" >&2
        echo "請先編譯: cmake --build build --target all -- -j" >&2
        exit 1
    fi
else
    if [[ ! -x "$SEARCH_BIN" ]]; then
        echo "WARN: DRY_RUN 模式忽略不存在的 search_disk_index: $SEARCH_BIN" >&2
    fi
fi

strip_ws() { echo "$1" | tr -d '[:space:]'; }

resolve_iostat_device() {
    local path_hint="$1"
    if [[ -n "$IOSTAT_DEVICE" ]]; then
        echo "$IOSTAT_DEVICE"
        return 0
    fi
    if [[ -n "$IOSTAT_DATA_PATH" && -e "$IOSTAT_DATA_PATH" ]]; then
        df -P "$IOSTAT_DATA_PATH" | awk 'NR==2 {print $1}'
        return 0
    fi
    if [[ -n "$path_hint" && -e "$path_hint" ]]; then
        df -P "$path_hint" | awk 'NR==2 {print $1}'
        return 0
    fi
    echo ""
}

search_ids=()
search_ws=()
search_ls=()
search_ks=()
search_caches=()
search_threads=()

exec 3< "$SEARCH_CSV"
read -r _header <&3
while IFS=',' read -r search_id search_W search_L search_K search_cache search_thread _rest <&3; do
    search_id=$(strip_ws "${search_id:-}")
    [[ -z "$search_id" ]] && continue
    search_ids+=("$search_id")
    search_ws+=("$(strip_ws "${search_W:-}")")
    search_ls+=("$(strip_ws "${search_L:-}")")
    search_ks+=("$(strip_ws "${search_K:-}")")
    search_caches+=("$(strip_ws "${search_cache:-}")")
    search_threads+=("$(strip_ws "${search_thread:-}")")
done
exec 3<&-

if [[ "${#search_ids[@]}" -eq 0 ]]; then
    echo "ERROR: search_configs.csv 沒有任何可用的設定" >&2
    exit 1
fi

# 驗證數組長度一致
if [[ "${#search_ws[@]}" -ne "${#search_ids[@]}" || "${#search_ls[@]}" -ne "${#search_ids[@]}" || "${#search_ks[@]}" -ne "${#search_ids[@]}" || "${#search_caches[@]}" -ne "${#search_ids[@]}" || "${#search_threads[@]}" -ne "${#search_ids[@]}" ]]; then
    echo "ERROR: CSV 列數解析不匹配" >&2
    echo "  search_ids: ${#search_ids[@]}, search_ws: ${#search_ws[@]}, search_ls: ${#search_ls[@]}, search_ks: ${#search_ks[@]}, search_caches: ${#search_caches[@]}, search_threads: ${#search_threads[@]}" >&2
    exit 1
fi

shopt -s nullglob
index_files=("${BUILD_DIR}"/*_disk.index)
shopt -u nullglob
if [[ "${#index_files[@]}" -eq 0 ]]; then
    echo "ERROR: 在 $BUILD_DIR 找不到任何 *_disk.index" >&2
    exit 1
fi

run_one() {
    local index_prefix="$1" index_tag="$2" dataset_name="$3"
    local search_id="$4" W="$5" L="$6" K="$7" cache="$8" threads="$9"
    local result_dir="${OUTPUT_DIR}/${index_tag}"
    local result_prefix="${result_dir}/result_${index_tag}_S${search_id}"
    local log_file="${result_dir}/search_${search_id}.log"
    local stats_csv="${result_prefix}_W${W}_L${L}_cache${cache}_T${threads}_summary_stats.csv"
    local iostat_log="${stats_csv%_summary_stats.csv}_iostat.log"
    local expanded_nodes_csv="${stats_csv%_summary_stats.csv}_expanded_nodes.csv"
    local append_flag=()
    local query_file="${QUERY_FILE:-${DISKANN_ROOT}/data/${dataset_name}/${dataset_name}_query.bin}"
    local gt_file="${GT_FILE:-${DISKANN_ROOT}/data/${dataset_name}/${dataset_name}_groundtruth.bin}"
    local thread_value="${THREAD_OVERRIDE:-${threads}}"
    local K_value="${K_OVERRIDE:-${K}}"

    mkdir -p "$result_dir"

    if [[ "$DRY_RUN" != "1" ]]; then
        if [[ ! -f "${index_prefix}_disk.index" ]]; then
            echo "ERROR: 找不到 index 檔案: ${index_prefix}_disk.index" >&2
            return 1
        fi
        if [[ -z "${QUERY_FILE:-}" && ! -f "$query_file" ]]; then
            echo "ERROR: 找不到 query 檔案: $query_file" >&2
            return 1
        fi
        if [[ -z "${GT_FILE:-}" && ! -f "$gt_file" ]]; then
            echo "ERROR: 找不到 groundtruth 檔案: $gt_file" >&2
            return 1
        fi
    fi

    if [[ "$APPEND_SEARCH_PARAMS" == "1" ]]; then
        append_flag=(-A)
    fi

    cmd=(
        "${SEARCH_BIN}"
        --data_type "${DATA_TYPE}"
        --dist_fn "${DIST_FN}"
        --index_path_prefix "${index_prefix}"
        --query_file "${query_file}"
        --gt_file "${gt_file}"
        --result_path "${result_prefix}"
        --stats_csv_path "${stats_csv}"
        --num_nodes_to_cache "${cache}"
        --num_threads "${thread_value}"
        -K "${K_value}"
        -L "${L}"
        -W "${W}"
    )
    if [[ "$ENABLE_EXPANDED_NODES" == "1" ]]; then
        cmd+=(--record_expanded_nodes --expanded_nodes_path "${expanded_nodes_csv}" --expanded_nodes_limit "${EXPANDED_NODES_LIMIT}")
    fi
    if [[ -n "$SEARCH_IO_LIMIT" ]]; then
        cmd+=(--search_io_limit "${SEARCH_IO_LIMIT}")
    fi
    cmd+=("${append_flag[@]}")
    if [[ -n "$EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        cmd+=($EXTRA_ARGS)
    fi

    echo "▶ ${index_tag} / ${search_id}: L=${L} W=${W} cache=${cache} T=${thread_value} K=${K_value}"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN: %q ' "${cmd[@]}" > "$log_file"
        printf '\n' >> "$log_file"
        echo "✓ ${index_tag} / ${search_id} 完成 (dry-run)"
        return 0
    fi

    if [[ "$ENABLE_IOSTAT" == "1" ]]; then
        if ! command -v iostat >/dev/null 2>&1; then
            echo "WARN: ENABLE_IOSTAT=1 但找不到 iostat，略過記錄" >&2
        else
            local device
            device="$(resolve_iostat_device "${index_prefix}_disk.index")"
            if [[ -n "$device" ]]; then
                iostat -x "$IOSTAT_INTERVAL" "$device" > "$iostat_log" &
            else
                iostat -x "$IOSTAT_INTERVAL" > "$iostat_log" &
            fi
            iostat_pid=$!
        fi
    fi

    if ! "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null; then
        if [[ -n "${iostat_pid:-}" ]]; then
            kill "$iostat_pid" >/dev/null 2>&1 || true
        fi
        echo "✗ ${index_tag} / ${search_id} 失敗，請檢查 ${log_file}"
        return 1
    fi
    if [[ -n "${iostat_pid:-}" ]]; then
        kill "$iostat_pid" >/dev/null 2>&1 || true
    fi
    echo "✓ ${index_tag} / ${search_id} 完成"
}

running=0
fail=0
declare -A pid_to_job
pids=()

for index_file in "${index_files[@]}"; do
    index_name="$(basename "$index_file")"
    index_prefix="${index_file%_disk.index}"
    index_tag="$(basename "$index_prefix")"

    dataset_name=""
    build_R=""
    build_L=""
    build_B=""
    build_M=""

    if [[ "$index_name" =~ ^([^_]+)_R([0-9]+)_L([0-9]+)_B([0-9.]+)_M([0-9]+)_disk\.index$ ]]; then
        dataset_name="${BASH_REMATCH[1]}"
        build_R="${BASH_REMATCH[2]}"
        build_L="${BASH_REMATCH[3]}"
        build_B="${BASH_REMATCH[4]}"
        build_M="${BASH_REMATCH[5]}"
    elif [[ "$index_name" =~ ^([^_]+)_R([0-9]+)_L([0-9]+)_disk\.index$ ]]; then
        dataset_name="${BASH_REMATCH[1]}"
        build_R="${BASH_REMATCH[2]}"
        build_L="${BASH_REMATCH[3]}"
    fi

    if [[ -n "$DATASET" ]]; then
        dataset_name="$DATASET"
    fi
    if [[ -z "$dataset_name" ]]; then
        echo "WARN: 無法從檔名解析 dataset，略過: $index_name" >&2
        continue
    fi

    for i in "${!search_ids[@]}"; do
        search_id="${search_ids[$i]}"
        W="${search_ws[$i]}"
        L="${search_ls[$i]}"
        K="${search_ks[$i]}"
        cache="${search_caches[$i]}"
        threads="${search_threads[$i]}"

        if (( MAX_PARALLEL == 1 )); then
            if ! run_one "$index_prefix" "$index_tag" "$dataset_name" "$search_id" "$W" "$L" "$K" "$cache" "$threads"; then
                fail=1
            fi
        else
            run_one "$index_prefix" "$index_tag" "$dataset_name" "$search_id" "$W" "$L" "$K" "$cache" "$threads" </dev/null &
            pid=$!
            pid_to_job[$pid]="${index_tag}_${search_id}"
            running=$((running+1))
            pids+=("$pid")

            if (( running >= MAX_PARALLEL )); then
                oldest_pid="${pids[0]}"
                pids=("${pids[@]:1}")
                if ! wait "$oldest_pid"; then
                    echo "WARN: ${pid_to_job[$oldest_pid]:-unknown} 失敗，持續處理其餘樣本" >&2
                    fail=1
                fi
                unset "pid_to_job[$oldest_pid]"
                running=$((running-1))
            fi
        fi
    done
done

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: ${pid_to_job[$pid]:-unknown} 失敗，請檢查對應 log" >&2
        fail=1
    fi
    unset "pid_to_job[$pid]"
    running=$((running-1))
done

echo ""
echo "=========================================="
echo "批次搜尋完成，結果位於: $OUTPUT_DIR"
echo "=========================================="

exit $fail
