#!/usr/bin/env bash
# 依 ./inputFiles/search_configs.csv 批次對 ./outputFiles/build 內的索引進行搜尋。
# 必要輸入：./inputFiles/search_configs.csv (search_id,search_W,search_L,search_cache,search_thread)
# 輸出位置: ./outputFiles/search/
# 可用 DRY_RUN=1 先驗證指令。

set -euo pipefail

usage() {
    cat <<'USAGE'
用法:
  bash search_batch.sh [--search-csv PATH] [--dataset NAME] [--max-parallel N]

參數:
  --search-csv PATH
  --dataset NAME
  --max-parallel N

環境變數可覆寫:
  BUILD_DIR, OUTPUT_DIR, DATA_TYPE, DIST_FN
  QUERY_FILE, GT_FILE, SEARCH_IO_LIMIT, THREAD_OVERRIDE
  EXPERIMENT_TAG 追加到 OUTPUT_DIR，且在 BUILD_DIR 使用預設值時同步追加
  EXTRA_ARGS 可補充 search_disk_index 的其他參數
  ENABLE_IOSTAT=1 時為每筆樣本記錄 iostat
  IOSTAT_INTERVAL=1, IOSTAT_DEVICE, IOSTAT_DATA_PATH
  ENABLE_EXPANDED_NODES=1 時為每筆樣本輸出 expanded_nodes CSV
  EXPANDED_NODES_LIMIT=0 (0 = unlimited)
  SLEEP_SECONDS=0 每筆搜尋結束後 sleep 秒數（可用於降載）
  COOLDOWN_TEMP_C=60 設定後：每筆搜尋開始前確認 NVMe 溫度低於此值，並強制改為單工
  COOLDOWN_CHECK_INTERVAL=15 檢查間隔秒數
  TEMP_DEVICE=/dev/nvme0 供溫度檢查使用
  NVME_USE_SUDO=0 設為 1 時以 sudo 讀取 nvme smart-log（若無法直接讀取）
  DRY_RUN=1 時僅印出指令不執行 search_disk_index
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
SEARCH_BIN="${APPS_DIR}/search_disk_index"

SEARCH_CSV="${SCRIPT_DIR}/inputFiles/search_configs.csv"
DATASET=""
MAX_PARALLEL="4"

# Optional named args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --search-csv)
            SEARCH_CSV="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: 未知參數 $1" >&2
            exit 1
            ;;
        *)
            echo "ERROR: 不支援位置參數，請使用 --search-csv/--dataset/--max-parallel" >&2
            exit 1
            ;;
    esac
done
DRY_RUN="${DRY_RUN:-0}"
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
SEARCH_IO_LIMIT="${SEARCH_IO_LIMIT:-}"
THREAD_OVERRIDE="${THREAD_OVERRIDE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ENABLE_IOSTAT="${ENABLE_IOSTAT:-0}"
IOSTAT_INTERVAL="${IOSTAT_INTERVAL:-1}"
IOSTAT_DEVICE="${IOSTAT_DEVICE:-}"
IOSTAT_DATA_PATH="${IOSTAT_DATA_PATH:-}"
ENABLE_EXPANDED_NODES="${ENABLE_EXPANDED_NODES:-0}"
EXPANDED_NODES_LIMIT="${EXPANDED_NODES_LIMIT:-0}"
K_OVERRIDE="${K_OVERRIDE:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-}"
COOLDOWN_CHECK_INTERVAL="${COOLDOWN_CHECK_INTERVAL:-15}"
TEMP_DEVICE="${TEMP_DEVICE:-/dev/nvme0}"
NVME_USE_SUDO="${NVME_USE_SUDO:-0}"
COOLDOWN_ENABLED=0

if [[ -z "${BUILD_DIR+x}" ]]; then
    BUILD_DIR_DEFAULT=1
    BUILD_DIR="${SCRIPT_DIR}/outputFiles/build"
else
    BUILD_DIR_DEFAULT=0
    BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/outputFiles/build}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputFiles/search}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
if [[ -n "$EXPERIMENT_TAG" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/${EXPERIMENT_TAG}"
    if [[ "$BUILD_DIR_DEFAULT" -eq 1 ]]; then
        BUILD_DIR="${BUILD_DIR}/${EXPERIMENT_TAG}"
    fi
fi

if [[ ! -f "$SEARCH_CSV" ]]; then
    echo "ERROR: 找不到 SEARCH_CSV: $SEARCH_CSV" >&2
    exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "ERROR: MAX_PARALLEL 需為正整數" >&2
    exit 1
fi
if [[ -n "$COOLDOWN_TEMP_C" ]]; then
    if [[ "$COOLDOWN_TEMP_C" =~ ^[0-9]+$ ]]; then
        COOLDOWN_ENABLED=1
    else
        echo "WARN: COOLDOWN_TEMP_C=$COOLDOWN_TEMP_C 不是整數，忽略降溫控制" >&2
        COOLDOWN_TEMP_C=""
    fi
fi
if [[ "$ENABLE_IOSTAT" == "1" && "$MAX_PARALLEL" -ne 1 ]]; then
    echo "WARN: ENABLE_IOSTAT=1 建議單一序列執行，已將 MAX_PARALLEL 強制為 1" >&2
    MAX_PARALLEL=1
fi
if [[ "$COOLDOWN_ENABLED" == "1" && "$MAX_PARALLEL" -ne 1 ]]; then
    echo "WARN: 啟用 COOLDOWN_TEMP_C 時改為單工，已將 MAX_PARALLEL 強制為 1" >&2
    MAX_PARALLEL=1
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

get_nvme_temperature_c() {
    local dev="$1"
    if [[ -z "$dev" || ! -e "$dev" ]]; then
        echo ""
        return 0
    fi

    local nvme_cmd=(nvme)
    if [[ "$NVME_USE_SUDO" == "1" ]]; then
        nvme_cmd=(sudo -n nvme)
    fi

    if command -v "${nvme_cmd[-1]}" >/dev/null 2>&1; then
        local temp
        local output
        if output=$("${nvme_cmd[@]}" smart-log "$dev" 2>/dev/null); then
            temp=$(echo "$output" | awk '
                /^temperature/ {comp=$3}
                /^Temperature Sensor/ {
                    if ($5 ~ /^[0-9]+$/) {
                        if ($5 > max) max=$5
                    }
                }
                END {
                    if (max == "" && comp != "") max=comp
                    if (max != "") print max
                }')
            if [[ -n "$temp" ]]; then
                echo "$temp"
                return 0
            else
                echo "WARN: nvme smart-log 無法解析溫度（輸出為空），改用 hwmon 後備" >&2
            fi
        else
            echo "WARN: nvme smart-log 執行失敗，狀態碼 $?，改用 hwmon 後備" >&2
        fi
    fi

    # Fallback to sysfs hwmon (millidegree)
    local block_name
    block_name="$(basename "$dev")"
    local hwmon_files=("/sys/block/${block_name}/device/hwmon/"*/temp*_input)
    for f in "${hwmon_files[@]}"; do
        if [[ -f "$f" ]]; then
            local milli
            milli=$(cat "$f" 2>/dev/null)
            if [[ "$milli" =~ ^[0-9]+$ ]]; then
                echo $((milli / 1000))
                return 0
            fi
        fi
    done

    echo ""
}

block_until_cool() {
    local threshold="$1"
    local interval="$2"
    local dev="$3"

    if [[ "$COOLDOWN_ENABLED" -ne 1 ]]; then
        return 0
    fi

    while true; do
        local temp
        temp="$(get_nvme_temperature_c "$dev")"

        if [[ -z "$temp" ]]; then
            echo "WARN: 無法讀取 $dev 溫度，停止降溫等待" >&2
            return 0
        fi
        if ! [[ "$temp" =~ ^[0-9]+$ ]]; then
            echo "WARN: 讀到的溫度值非數字（$temp），停止降溫等待" >&2
            return 0
        fi

        if [[ "$temp" -lt "$threshold" ]]; then
            echo "INFO: $dev 溫度 ${temp}°C < ${threshold}°C，可開始下一筆搜尋" >&2
            return 0
        fi

        echo "INFO: $dev 溫度 ${temp}°C >= ${threshold}°C，等待 ${interval}s 後重試" >&2
        sleep "$interval"
    done
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
    local search_tag="${search_id}"
    if [[ "$search_tag" == S* ]]; then
        search_tag="${search_tag#S}"
    fi
    local log_file="${result_dir}/search_${search_id}.log"
    local query_file="${QUERY_FILE:-${DISKANN_ROOT}/data/${dataset_name}/${dataset_name}_query.bin}"
    local gt_file="${GT_FILE:-${DISKANN_ROOT}/data/${dataset_name}/${dataset_name}_groundtruth.bin}"
    local thread_value="${THREAD_OVERRIDE:-${threads}}"
    local K_value="${K_OVERRIDE:-${K}}"
    local result_prefix="${result_dir}/S${search_tag}_${index_tag}_W${W}_L${L}_K${K_value}_cache${cache}_T${threads}"
    local stats_csv="${result_prefix}_summary_stats.csv"
    local iostat_log="${stats_csv%_summary_stats.csv}_iostat.log"
    local expanded_nodes_csv="${stats_csv%_summary_stats.csv}_expanded_nodes.csv"

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

    block_until_cool "$COOLDOWN_TEMP_C" "$COOLDOWN_CHECK_INTERVAL" "$TEMP_DEVICE"

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
    if [[ "$SLEEP_SECONDS" =~ ^[0-9]+$ ]] && [[ "$SLEEP_SECONDS" -gt 0 ]]; then
        sleep "$SLEEP_SECONDS"
    fi
}

fail=0
declare -A pid_to_job
pids=()

for index_file in "${index_files[@]}"; do
    index_name="$(basename "$index_file")"
    index_prefix="${index_file%_disk.index}"
    index_tag="$(basename "$index_prefix")"

    dataset_name=""

    if [[ "$index_name" =~ ^([^_]+)_R([0-9]+)_L([0-9]+)_B([0-9.]+)_M([0-9]+)_disk\.index$ ]]; then
        dataset_name="${BASH_REMATCH[1]}"
    elif [[ "$index_name" =~ ^([^_]+)_R([0-9]+)_L([0-9]+)_disk\.index$ ]]; then
        dataset_name="${BASH_REMATCH[1]}"
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
            pids+=("$pid")

            if (( ${#pids[@]} >= MAX_PARALLEL )); then
                oldest_pid="${pids[0]}"
                pids=("${pids[@]:1}")
                if ! wait "$oldest_pid"; then
                    echo "WARN: ${pid_to_job[$oldest_pid]:-unknown} 失敗，持續處理其餘樣本" >&2
                    fail=1
                fi
                unset "pid_to_job[$oldest_pid]"
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
done

echo ""
echo "=========================================="
echo "批次搜尋完成，結果位於: $OUTPUT_DIR"
echo "=========================================="

exit $fail
