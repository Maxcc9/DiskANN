#!/usr/bin/env bash
# 外部量測 queue depth（avgqu-sz）用的包裝器：在執行指令期間持續記錄 iostat。

set -euo pipefail

usage() {
    cat <<'USAGE'
用法:
  bash measure_queue_depth.sh "<command>"
  bash measure_queue_depth.sh --clean

範例:
  bash measure_queue_depth.sh "bash search_batch.sh ./inputFiles/search_configs.csv"

環境變數可覆寫:
  IO_INTERVAL=1          iostat 取樣秒數
  OUTPUT_DIR=./outputFiles/analyze
  DEVICE=/dev/nvme0n1    指定欲監控的 block device（選填）
  DATA_PATH=...          若未指定 DEVICE，會用 data file 所在磁碟推估
  --clean                只刪除本腳本輸出的 iostat_*.log
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputFiles/analyze}"
IO_INTERVAL="${IO_INTERVAL:-1}"
DEVICE="${DEVICE:-}"
DATA_PATH="${DATA_PATH:-}"

CLEAN=0
args=()
for a in "$@"; do
    if [[ "$a" == "--clean" ]]; then
        CLEAN=1
    else
        args+=("$a")
    fi
done
set -- "${args[@]}"

if [[ "$CLEAN" == "1" ]]; then
    shopt -s nullglob
    logs=("${OUTPUT_DIR}"/iostat_*.log)
    shopt -u nullglob
    removed=0
    for f in "${logs[@]}"; do
        rm -f "$f"
        removed=$((removed + 1))
        echo "刪除: ${f}"
    done
    echo "完成：已刪除 ${removed} 個 iostat log"
    exit 0
fi

[[ $# -lt 1 ]] && { usage; exit 1; }

if ! command -v iostat >/dev/null 2>&1; then
    echo "ERROR: 找不到 iostat，請先安裝 sysstat" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
log_path="${OUTPUT_DIR}/iostat_${ts}.log"

if [[ -z "$DEVICE" && -n "$DATA_PATH" ]]; then
    # 由資料檔推估所在裝置（例如 /dev/nvme0n1p2）
    DEVICE="$(df -P "$DATA_PATH" | awk 'NR==2 {print $1}')"
fi

if [[ -n "$DEVICE" ]]; then
    echo "監控裝置: $DEVICE"
    iostat -x "$IO_INTERVAL" "$DEVICE" > "$log_path" &
else
    echo "未指定裝置，將記錄所有磁碟"
    iostat -x "$IO_INTERVAL" > "$log_path" &
fi

iostat_pid=$!

cleanup() {
    if kill -0 "$iostat_pid" >/dev/null 2>&1; then
        kill "$iostat_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "開始記錄 iostat -> $log_path"
echo "執行: $1"
bash -lc "$1"

echo "完成。iostat log: $log_path"
