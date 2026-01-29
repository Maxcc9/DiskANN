#!/usr/bin/env bash
# 依 ./inputFiles/search_configs.csv 批次對 ./outputFiles/build 內的索引進行搜尋
# 核心邏輯：每次搜尋都是獨立單位，執行前依序進行：
#   1) 清除系統快取（drop_caches）
#   2) 檢測 NVMe 溫度，若超過閾值則等待
#   3) 執行實際搜尋
#   4) 收集監測指標（iostat/pidstat/mpstat）
#
# 必要輸入：./inputFiles/search_configs.csv (search_id,search_W,search_L,search_cache,search_thread)
# 輸出位置: ./outputFiles/search/

set -euo pipefail

# 記錄所有背景進程 PID，確保清理
declare -a BG_PIDS=()

# 清理函數：終止所有背景進程（包括子進程）
cleanup_bg_processes() {
    for pid in "${BG_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # 先嘗試溫和終止 (SIGTERM)
            kill -TERM "$pid" 2>/dev/null || true
            sleep 0.1
            # 如果還活著，強制終止 (SIGKILL)
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    BG_PIDS=()
    
    # 額外清理：終止任何可能的孤立 iostat/pidstat/mpstat 進程
    pkill -P $$ -f "iostat|pidstat|mpstat" 2>/dev/null || true
}

usage() {
    cat <<'USAGE'
用法:
  bash search_batch.sh [--search-csv PATH] [--dataset NAME] [--max-parallel N] [--repeat-count N] [--clean]

必要參數:
  --search-csv PATH     搜尋配置 CSV 路徑（預設：inputFiles/search_configs.csv 或 inputFiles/{EXPERIMENT_TAG}/search_configs.csv）
  --dataset NAME        資料集名稱（預設：自動推斷）
  --max-parallel N      最大並行搜尋數（預設：4，若啟用溫度控制強制改為 1）
  --repeat-count N      每組參數重複執行次數（預設：1）
  --clean               清除舊搜尋結果（預設：保留）

環境變數:
  實驗配置:
    EXPERIMENT_TAG        實驗名稱（自動推斷 BUILD_DIR/OUTPUT_DIR/DATASET/TEMP_DEVICE）
    BUILD_DIR             索引輸出資料夾
    OUTPUT_DIR            搜尋結果輸出資料夾
    DATASET               資料集名稱
    
  搜尋參數:
    DATA_TYPE             數據類型（預設：float）
    DIST_FN               距離函數（預設：l2）
    QUERY_FILE            查詢文件路徑
    GT_FILE               基準真值文件路徑
    SEARCH_IO_LIMIT       搜尋 I/O 限制
    THREAD_OVERRIDE       覆寫執行緒數
    K_OVERRIDE            覆寫 K 值
    EXTRA_ARGS            額外命令行參數
    SLEEP_SECONDS         每筆搜尋後 sleep 秒數（預設：0）
    
  快取與溫度控制（每次搜尋前執行）:
    CACHE_CLEAR_ENABLED   是否清除系統快取（預設：1）
    COOLDOWN_TEMP_C       NVMe 溫度閾值，單位°C（預設：60，0 表示禁用）
    COOLDOWN_CHECK_INTERVAL 溫度檢測間隔秒數（預設：15）
    TEMP_DEVICE           NVMe 設備路徑（預設：自動推斷）
    
  監測開關:
    ENABLE_IOSTAT         啟用 iostat 記錄（預設：1）
    ENABLE_PIDSTAT        啟用 pidstat 記錄（預設：1）
    ENABLE_WA_LOG         啟用 mpstat 記錄（預設：1）
    ENABLE_THREAD_TIMELINE    啟用線程時間軸（預設：1）
    ENABLE_READ_TRACE     啟用讀取追蹤（預設：1）
    ENABLE_EXPANDED_NODES 啟用展開節點記錄（預設：1）
    ENABLE_SUMMARY_STATS  啟用摘要統計（預設：1）
    ENABLE_PER_QUERY_STATS    啟用單查詢統計（預設：0）
    IOSTAT_INTERVAL       iostat 記錄間隔秒數（預設：1）
    PIDSTAT_INTERVAL      pidstat 記錄間隔秒數（預設：1）
    WA_INTERVAL           mpstat 記錄間隔秒數（預設：1）
    EXPANDED_NODES_LIMIT  展開節點記錄限制（預設：0）
    
  診斷模式:
    DRY_RUN               設置 1 以只列印命令不執行（預設：0）

使用範例:
  # 單次搜尋（自動檢測溫度，若 NVMe > 60°C 則等待）
  EXPERIMENT_TAG=sift01 bash search_batch.sh --clean

  # 多次重複搜尋（確保統計穩定性，每次搜尋前清除快取）
  EXPERIMENT_TAG=sift01 bash search_batch.sh --repeat-count 3 --clean

  # 自訂溫度閾值
  EXPERIMENT_TAG=sift01 COOLDOWN_TEMP_C=50 bash search_batch.sh --repeat-count 5 --clean
USAGE
}

# ==================== 命令行參數解析 ====================

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

# 基礎路徑設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
SEARCH_BIN="${APPS_DIR}/search_disk_index"

# ==================== 參數初始化 ====================

# 實驗標籤與配置路徑
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
if [[ -n "$EXPERIMENT_TAG" ]]; then
    SEARCH_CSV="${SCRIPT_DIR}/inputFiles/${EXPERIMENT_TAG}/search_configs.csv"
else
    SEARCH_CSV="${SCRIPT_DIR}/inputFiles/search_configs.csv"
fi

# 批次控制
DATASET="${DATASET:-}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
CLEAN="${CLEAN:-0}"

# 輸出與資料集
if [[ -z "${BUILD_DIR+x}" ]]; then
    BUILD_DIR="${SCRIPT_DIR}/outputFiles/build"
    [[ -n "$EXPERIMENT_TAG" ]] && BUILD_DIR="${BUILD_DIR}/${EXPERIMENT_TAG}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputFiles/search}"
[[ -n "$EXPERIMENT_TAG" ]] && OUTPUT_DIR="${OUTPUT_DIR}/${EXPERIMENT_TAG}"
[[ -z "$DATASET" ]] && DATASET="${EXPERIMENT_TAG:-}"

# 搜尋參數與監測
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
SEARCH_IO_LIMIT="${SEARCH_IO_LIMIT:-}"
THREAD_OVERRIDE="${THREAD_OVERRIDE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
K_OVERRIDE="${K_OVERRIDE:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"

# 監測開關
ENABLE_IOSTAT="${ENABLE_IOSTAT:-1}"
IOSTAT_INTERVAL="${IOSTAT_INTERVAL:-1}"
IOSTAT_DEVICE="${IOSTAT_DEVICE:-}"
IOSTAT_DATA_PATH="${IOSTAT_DATA_PATH:-}"
ENABLE_PIDSTAT="${ENABLE_PIDSTAT:-1}"
PIDSTAT_INTERVAL="${PIDSTAT_INTERVAL:-1}"
ENABLE_WA_LOG="${ENABLE_WA_LOG:-1}"
WA_INTERVAL="${WA_INTERVAL:-1}"
ENABLE_THREAD_TIMELINE="${ENABLE_THREAD_TIMELINE:-1}"
ENABLE_READ_TRACE="${ENABLE_READ_TRACE:-1}"
ENABLE_EXPANDED_NODES="${ENABLE_EXPANDED_NODES:-1}"
EXPANDED_NODES_LIMIT="${EXPANDED_NODES_LIMIT:-0}"
ENABLE_SUMMARY_STATS="${ENABLE_SUMMARY_STATS:-1}"
ENABLE_PER_QUERY_STATS="${ENABLE_PER_QUERY_STATS:-0}"

# 溫度與冷卻控制
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-60}"
COOLDOWN_CHECK_INTERVAL="${COOLDOWN_CHECK_INTERVAL:-15}"
TEMP_DEVICE="${TEMP_DEVICE:-}"

# 快取控制
CACHE_CLEAR_ENABLED="${CACHE_CLEAR_ENABLED:-1}"

# 診斷模式
DRY_RUN="${DRY_RUN:-0}"

# ==================== 命令行參數解析 ====================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --search-csv) SEARCH_CSV="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --repeat-count) REPEAT_COUNT="$2"; shift 2 ;;
        --clean) CLEAN="1"; shift ;;
        --) shift; break ;;
        -*) echo "ERROR: 未知參數 $1" >&2; exit 1 ;;
        *) echo "ERROR: 不支援位置參數，請使用命名參數" >&2; exit 1 ;;
    esac
done

# ==================== 參數驗證 ====================

[[ ! -f "$SEARCH_CSV" ]] && { echo "ERROR: 找不到 SEARCH_CSV: $SEARCH_CSV" >&2; exit 1; }
[[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]] && { echo "ERROR: MAX_PARALLEL 需為正整數" >&2; exit 1; }
[[ ! "$REPEAT_COUNT" =~ ^[0-9]+$ ]] || [[ "$REPEAT_COUNT" -lt 1 ]] && { echo "ERROR: REPEAT_COUNT 需為正整數" >&2; exit 1; }
[[ ! -d "$BUILD_DIR" ]] && { echo "ERROR: 找不到 BUILD_DIR 目錄: $BUILD_DIR" >&2; exit 1; }

# ==================== 工具函數 ====================

strip_ws() {
    echo "$1" | tr -d '[:space:]'
}

resolve_temp_device() {
    # 優先順序：環境變數（非預設） → OUTPUT_DIR 所在設備 → /dev/nvme0
    [[ -n "$TEMP_DEVICE" ]] && { echo "$TEMP_DEVICE"; return 0; }
    
    # 嘗試從 OUTPUT_DIR 推斷 NVMe 設備
    local check_dir="${OUTPUT_DIR:-${SCRIPT_DIR}}"
    if [[ -d "$check_dir" ]]; then
        local detected=$(df -P "$check_dir" 2>/dev/null | awk 'NR==2 {print $1}')
        [[ "$detected" =~ nvme ]] && { echo "$detected"; return 0; }
    fi
    
    echo "/dev/nvme0"  # 預設回退
}

get_nvme_temperature_c() {
    local dev="$1"
    [[ -z "$dev" ]] && return 1
    
    if command -v nvme >/dev/null 2>&1; then
        local temp=""
        
        # 方法 1：嘗試不用 sudo
        if output=$(nvme smart-log "$dev" 2>/dev/null); then
            temp=$(echo "$output" | awk '
                BEGIN {max_temp = 0}
                /[Tt]emperature/ && /:/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == ":") {
                            temp_val = $(i+1)
                            if (temp_val ~ /^[0-9]+$/ && temp_val > max_temp) {
                                max_temp = temp_val
                            }
                            break
                        }
                    }
                }
                END {if (max_temp > 0) print max_temp}
            ')
            [[ -n "$temp" ]] && echo "$temp" && return 0
        fi
        
        # 方法 2：自動嘗試用 sudo -n（已配置 sudoers 免密碼）
        if output=$(sudo -n nvme smart-log "$dev" 2>/dev/null); then
            temp=$(echo "$output" | awk '
                BEGIN {max_temp = 0}
                /[Tt]emperature/ && /:/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == ":") {
                            temp_val = $(i+1)
                            if (temp_val ~ /^[0-9]+$/ && temp_val > max_temp) {
                                max_temp = temp_val
                            }
                            break
                        }
                    }
                }
                END {if (max_temp > 0) print max_temp}
            ')
            [[ -n "$temp" ]] && echo "$temp" && return 0
        fi
    fi
    
    # 回退到 sysfs hwmon
    local block_name="$(basename "$dev" 2>/dev/null)"
    local hwmon_temp=""
    if [[ -d "/sys/block/${block_name}/device/hwmon" ]]; then
        for f in /sys/block/${block_name}/device/hwmon/*/temp*_input; do
            [[ -f "$f" ]] && hwmon_temp=$(cat "$f" 2>/dev/null | awk '{print int($1/1000); exit}') && break
        done
    fi
    [[ -n "$hwmon_temp" ]] && echo "$hwmon_temp" && return 0
    
    return 1
}

wait_for_temp_ok() {
    local threshold="$1"
    local interval="$2"
    local device="$3"
    
    # 若未設定溫度閾值或設備無效，直接通過
    [[ -z "$threshold" ]] || [[ -z "$device" ]] && return 0
    [[ ! "$threshold" =~ ^[0-9]+$ ]] && return 0
    [[ "$threshold" -le 0 ]] && return 0  # 0 表示禁用
    
    while true; do
        local temp=$(get_nvme_temperature_c "$device" 2>/dev/null || echo "")
        
        if [[ -z "$temp" ]]; then
            echo "WARN: 無法讀取 $device 溫度，跳過溫度檢測" >&2
            return 0
        fi
        
        if [[ "$temp" -lt "$threshold" ]]; then
            echo "INFO: NVMe $device 溫度 ${temp}°C < ${threshold}°C，可開始搜尋" >&2
            return 0
        fi
        
        echo "INFO: NVMe $device 溫度 ${temp}°C >= ${threshold}°C，等待 ${interval}s..." >&2
        sleep "$interval"
    done
}

clear_cache() {
    echo "INFO: 清除系統快取" >&2
    sync 2>/dev/null || true
    
    # 嘗試使用 tee（需配置 sudoers）
    if echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
        return 0
    fi
    
    # 回退到 sh -c（舊方法）
    if sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        return 0
    fi
    
    # 檢測是否在 Docker 容器內
    if [[ -f "/.dockerenv" ]]; then
        echo "WARN: 在 Docker 容器內無法清除快取（需要在主機上執行）" >&2
        echo "      建議使用 docker_run_search.sh 腳本，會自動清除快取後再執行容器" >&2
        return 0  # 不中止，允許繼續（warm cache 測試）
    fi
    
    echo "WARN: 清除快取失敗（需要 sudo 權限）。請執行：" >&2
    echo "  bash setup_sudoers.sh" >&2
    return 0  # 不中止，允許繼續
}

resolve_iostat_device() {
    [[ -n "$IOSTAT_DEVICE" ]] && { echo "$IOSTAT_DEVICE"; return 0; }
    [[ -n "$IOSTAT_DATA_PATH" ]] && [[ -e "$IOSTAT_DATA_PATH" ]] && { df -P "$IOSTAT_DATA_PATH" 2>/dev/null | awk 'NR==2 {print $1}'; return 0; }
    [[ -n "$1" ]] && [[ -e "$1" ]] && { df -P "$1" 2>/dev/null | awk 'NR==2 {print $1}'; return 0; }
    echo ""
}

infer_dataset_name() {
    local index_name="$1"
    local inferred=""

    if [[ "$index_name" =~ ^([^_]+)_R([0-9]+)_L([0-9]+)_B([0-9.]+)_M([0-9]+)_disk\.index$ ]]; then
        inferred="${BASH_REMATCH[1]}"
    elif [[ "$index_name" =~ ^([^_]+)_R([0-9]+)_L([0-9]+)_disk\.index$ ]]; then
        inferred="${BASH_REMATCH[1]}"
    elif [[ "$index_name" =~ ^([^_]+)_ ]]; then
        inferred="${BASH_REMATCH[1]}"
    fi

    echo "$inferred"
}

# ==================== 主程序初始化 ====================

echo "INFO: EXPERIMENT_TAG=$EXPERIMENT_TAG"
echo "INFO: REPEAT_COUNT=$REPEAT_COUNT"
echo "INFO: CACHE_CLEAR_ENABLED=$CACHE_CLEAR_ENABLED"
echo "INFO: COOLDOWN_TEMP_C=$COOLDOWN_TEMP_C"

# 監測衝突檢測與溫度控制初始化（集中強制單工邏輯）
force_serial_reason=""
if [[ "$COOLDOWN_TEMP_C" =~ ^[0-9]+$ ]] && [[ "$COOLDOWN_TEMP_C" -gt 0 ]]; then
    TEMP_DEVICE=$(resolve_temp_device)
    echo "INFO: 溫度控制啟用，設備：$TEMP_DEVICE，閾值：${COOLDOWN_TEMP_C}°C"
    force_serial_reason="溫度控制"
fi

if [[ "$ENABLE_IOSTAT" == "1" ]] && [[ -z "$force_serial_reason" ]]; then
    force_serial_reason="ENABLE_IOSTAT"
elif [[ "$ENABLE_WA_LOG" == "1" ]] && [[ -z "$force_serial_reason" ]]; then
    force_serial_reason="ENABLE_WA_LOG"
elif [[ "$REPEAT_COUNT" -gt 1 ]] && [[ -z "$force_serial_reason" ]]; then
    force_serial_reason="多次重複搜尋"
elif [[ "$CACHE_CLEAR_ENABLED" == "1" ]] && [[ -z "$force_serial_reason" ]]; then
    force_serial_reason="快取清除"
fi

if [[ -n "$force_serial_reason" && "$MAX_PARALLEL" -ne 1 ]]; then
    echo "WARN: $force_serial_reason 強制單工模式（MAX_PARALLEL=$MAX_PARALLEL → 1）"
    MAX_PARALLEL=1
fi

# 清除舊數據
[[ "$CLEAN" == "1" && -d "$OUTPUT_DIR" ]] && { echo "清除舊的搜尋結果: $OUTPUT_DIR"; rm -rf "$OUTPUT_DIR"; }
mkdir -p "$OUTPUT_DIR"

# 檢查可執行文件
[[ "$DRY_RUN" != "1" && ! -x "$SEARCH_BIN" ]] && { echo "ERROR: search_disk_index 不存在或不可執行: $SEARCH_BIN" >&2; exit 1; }

# ==================== 讀取搜尋配置 ====================

search_ids=() search_ws=() search_ls=() search_ks=() search_caches=() search_threads=()

exec 3< "$SEARCH_CSV"
read -r _header <&3 || true
while IFS=',' read -r search_id search_W search_L search_K search_cache search_thread _rest <&3 || [[ -n "${search_id:-}" ]]; do
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

[[ "${#search_ids[@]}" -eq 0 ]] && { echo "ERROR: search_configs.csv 無有效行" >&2; exit 1; }

# ==================== 搜尋執行函數 ====================

run_search() {
    local index_prefix="$1" index_tag="$2" dataset_name="$3"
    local search_id="$4" W="$5" L="$6" K="$7" cache="$8" threads="$9" repeat_idx="${10:-1}"
    
    # 準備結果路徑
    local result_dir="${OUTPUT_DIR}/${index_tag}"
    local search_tag="${search_id#S}"  # 移除前導 'S'
    local K_value="${K_OVERRIDE:-$K}"
    local thread_value="${THREAD_OVERRIDE:-$threads}"
    local result_prefix_basename="S${search_tag}_${index_tag}_W${W}_L${L}_K${K_value}_cache${cache}_T${thread_value}_${repeat_idx}"
    local result_subdir="${result_dir}/${result_prefix_basename}"
    local result_prefix="${result_subdir}/${result_prefix_basename}"
    
    mkdir -p "$result_subdir"
    local log_file="${result_subdir}/search.log"
    
    local qf="${QUERY_FILE:-${DISKANN_ROOT}/data/${dataset_name}/${dataset_name}_query.bin}"
    local gf="${GT_FILE:-${DISKANN_ROOT}/data/${dataset_name}/${dataset_name}_groundtruth.bin}"

    # ========== 驗證輸入文件 ==========
    if [[ "$DRY_RUN" != "1" ]]; then
        [[ ! -f "${index_prefix}_disk.index" ]] && { echo "ERROR: 找不到 index 檔案: ${index_prefix}_disk.index" >&2; return 1; }
        [[ ! -f "$qf" ]] && { echo "ERROR: 找不到 query 檔案: $qf" >&2; return 1; }
        [[ ! -f "$gf" ]] && { echo "ERROR: 找不到 groundtruth 檔案: $gf" >&2; return 1; }
    fi
    
    # ========== 階段 3：構建搜尋命令 ==========
    local cmd=(
        "${SEARCH_BIN}"
        --data_type "${DATA_TYPE}"
        --dist_fn "${DIST_FN}"
        --index_path_prefix "${index_prefix}"
        --query_file "${qf}"
        --gt_file "${gf}"
        --result_path "${result_prefix}"
        --num_nodes_to_cache "${cache}"
        --num_threads "${thread_value}"
        -K "${K_value}"
        -L "${L}"
        -W "${W}"
    )
    
    [[ "$ENABLE_SUMMARY_STATS" == "1" ]] && cmd+=(--stats_csv_path "${result_prefix}_summary_stats.csv")
    [[ "$ENABLE_PER_QUERY_STATS" == "1" ]] && cmd+=(--per_query_stats_path "${result_prefix}_query_stats.csv")
    [[ "$ENABLE_THREAD_TIMELINE" == "1" ]] && cmd+=(--thread_timeline_path "${result_prefix}_thread_timeline.csv")
    [[ "$ENABLE_READ_TRACE" == "1" ]] && cmd+=(--read_trace_path "${result_prefix}_read_trace.csv")
    [[ "$ENABLE_EXPANDED_NODES" == "1" ]] && cmd+=(--record_expanded_nodes --expanded_nodes_path "${result_prefix}_expanded_nodes.csv" --expanded_nodes_limit "${EXPANDED_NODES_LIMIT}")
    [[ -n "$SEARCH_IO_LIMIT" ]] && cmd+=(--search_io_limit "${SEARCH_IO_LIMIT}")
    [[ -n "$EXTRA_ARGS" ]] && cmd+=($EXTRA_ARGS)
    
    echo "▶ ${index_tag} / ${search_id}: W=${W} L=${L} K=${K_value} cache=${cache} T=${thread_value} [${repeat_idx}/${REPEAT_COUNT}]"
    
    # DRY RUN 模式
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY: %q ' "${cmd[@]}" > "$log_file"
        printf '\n' >> "$log_file"
        echo "✓ 完成 (dry-run)"
        return 0
    fi

    # ========== 階段 1：清除快取 ==========
    [[ "$CACHE_CLEAR_ENABLED" == "1" ]] && clear_cache
    
    # ========== 階段 2：溫度檢測 ==========
    wait_for_temp_ok "$COOLDOWN_TEMP_C" "$COOLDOWN_CHECK_INTERVAL" "$TEMP_DEVICE"
    
    # ========== 階段 4：執行搜尋 ==========
    local iostat_pid="" pidstat_pid="" wa_pid="" search_pid=""
    
    # 啟動 iostat（若啟用）
    if [[ "$ENABLE_IOSTAT" == "1" ]]; then
        local device
        device=$(resolve_iostat_device "${index_prefix}_disk.index")
        if command -v iostat >/dev/null 2>&1; then
            if [[ -n "$device" ]]; then
                iostat -x "$IOSTAT_INTERVAL" "$device" > "${result_prefix}_iostat.log" 2>&1 &
            else
                iostat -x "$IOSTAT_INTERVAL" > "${result_prefix}_iostat.log" 2>&1 &
            fi
            iostat_pid=$!
            BG_PIDS+=("$iostat_pid")
        fi
    fi
    
    # 執行搜尋與監測
    if [[ "$ENABLE_PIDSTAT" == "1" || "$ENABLE_WA_LOG" == "1" ]]; then
        # 背景執行搜尋並監測
        "${cmd[@]}" > "$log_file" 2>&1 &
        search_pid=$!
        BG_PIDS+=("$search_pid")
        
        [[ "$ENABLE_PIDSTAT" == "1" ]] && command -v pidstat >/dev/null 2>&1 && { 
            pidstat -t -u -d -r -w -p "$search_pid" "$PIDSTAT_INTERVAL" > "${result_prefix}_pidstat.log" 2>&1 &
            pidstat_pid=$!
            BG_PIDS+=("$pidstat_pid")
        }
        
        [[ "$ENABLE_WA_LOG" == "1" ]] && command -v mpstat >/dev/null 2>&1 && { 
            mpstat -P ALL "$WA_INTERVAL" > "${result_prefix}_wa.log" 2>&1 &
            wa_pid=$!
            BG_PIDS+=("$wa_pid")
        }
        
        # 等待搜尋完成
        wait "$search_pid"
        local ret=$?
        
        # 終止監測程序（先移除陣列中的 PID）
        local i
        for ((i=0; i<${#BG_PIDS[@]}; i++)); do
            [[ "${BG_PIDS[$i]}" == "$search_pid" || "${BG_PIDS[$i]}" == "$pidstat_pid" || "${BG_PIDS[$i]}" == "$wa_pid" ]] && unset 'BG_PIDS[$i]'
        done
        BG_PIDS=("${BG_PIDS[@]}")  # 重建陣列去除空洞
        
        # 用 SIGTERM 先溫和終止
        [[ -n "$pidstat_pid" ]] && kill -TERM "$pidstat_pid" 2>/dev/null || true
        [[ -n "$wa_pid" ]] && kill -TERM "$wa_pid" 2>/dev/null || true
        [[ -n "$iostat_pid" ]] && kill -TERM "$iostat_pid" 2>/dev/null || true
        
        # 等待一下讓進程優雅結束
        sleep 0.2
        
        # 確保強制終止任何殘留進程
        [[ -n "$pidstat_pid" ]] && kill -KILL "$pidstat_pid" 2>/dev/null || true
        [[ -n "$wa_pid" ]] && kill -KILL "$wa_pid" 2>/dev/null || true
        [[ -n "$iostat_pid" ]] && kill -KILL "$iostat_pid" 2>/dev/null || true
        
        [[ $ret -ne 0 ]] && { echo "✗ 失敗，見 $log_file"; return 1; }
    else
        # 直接執行搜尋（前台）
        if ! "${cmd[@]}" > "$log_file" 2>&1; then
            [[ -n "$iostat_pid" ]] && kill -KILL "$iostat_pid" 2>/dev/null || true
            echo "✗ 失敗，見 $log_file"
            return 1
        fi
        [[ -n "$iostat_pid" ]] && kill -TERM "$iostat_pid" 2>/dev/null || true
        sleep 0.1
        [[ -n "$iostat_pid" ]] && kill -KILL "$iostat_pid" 2>/dev/null || true
    fi
    
    echo "✓ 完成"
    [[ "$SLEEP_SECONDS" =~ ^[0-9]+$ ]] && [[ "$SLEEP_SECONDS" -gt 0 ]] && sleep "$SLEEP_SECONDS"
}

# ==================== 主循環 ====================

shopt -s nullglob
index_files=("${BUILD_DIR}"/*_disk.index)
shopt -u nullglob

[[ "${#index_files[@]}" -eq 0 ]] && { echo "ERROR: 找不到 *_disk.index 在 $BUILD_DIR" >&2; exit 1; }

fail=0
declare -A pid_to_job
pids=()

for index_file in "${index_files[@]}"; do
    index_prefix="${index_file%_disk.index}"
    index_tag="$(basename "$index_prefix")"
    
    # 推斷 dataset_name
    index_name="$(basename "$index_file")"
    dataset_name="$(infer_dataset_name "$index_name")"
    [[ -z "$dataset_name" && -n "$DATASET" ]] && dataset_name="$DATASET"
    if [[ -z "$dataset_name" ]]; then
        echo "WARN: 無法推斷 dataset，略過 $index_name" >&2
        continue
    fi
    
    # 遍歷所有搜尋配置和重複次數
    for i in "${!search_ids[@]}"; do
        for repeat_idx in $(seq 1 $REPEAT_COUNT); do
            if (( MAX_PARALLEL == 1 )); then
                # 單工模式：直接執行
                if ! run_search "$index_prefix" "$index_tag" "$dataset_name" "${search_ids[$i]}" "${search_ws[$i]}" "${search_ls[$i]}" "${search_ks[$i]}" "${search_caches[$i]}" "${search_threads[$i]}" "$repeat_idx"; then
                    fail=1
                fi
            else
                # 並行模式
                run_search "$index_prefix" "$index_tag" "$dataset_name" "${search_ids[$i]}" "${search_ws[$i]}" "${search_ls[$i]}" "${search_ks[$i]}" "${search_caches[$i]}" "${search_threads[$i]}" "$repeat_idx" </dev/null &
                pid=$!
                pid_to_job[$pid]="${index_tag}_${search_ids[$i]}_r${repeat_idx}"
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
done

# 等待所有剩餘的背景搜尋完成
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: ${pid_to_job[$pid]:-unknown} 失敗，請檢查對應 log" >&2
        fail=1
    fi
    unset "pid_to_job[$pid]"
done

# 最終輸出
echo ""
echo "=========================================="
echo "批次搜尋完成，結果位於: $OUTPUT_DIR"
echo "=========================================="

# 設定 EXIT 陷阱以確保清理
trap cleanup_bg_processes EXIT

exit $fail
