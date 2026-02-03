#!/usr/bin/env bash
# 統計 expanded_nodes 的節點出現次數，取 Top-K 後再用 dump_disk_neighbors 輸出鄰居清單。
# 支援單檔與批次（資料夾）模式；每次執行只處理單一 TOPK。

set -euo pipefail  # -e: 任一指令失敗即退出；-u: 讀未定義變數視為錯誤；-o pipefail: 管線任一失敗即失敗。

usage() {
    cat <<'USAGE'
Usage:
  bash dump_topk_neighbors.sh <expanded_nodes_csv>     # Single file
  bash dump_topk_neighbors.sh <search_dir>             # Batch mode (all *_expanded_nodes.csv)
  EXPERIMENT_TAG=tag bash dump_topk_neighbors.sh       # Batch mode (search_dir defaults to outputFiles/search/<tag>)
  bash dump_topk_neighbors.sh --clean <expanded_nodes_csv>
  bash dump_topk_neighbors.sh --clean <search_dir>

Args:
  expanded_nodes_csv   Path to *_expanded_nodes.csv
  search_dir           Directory containing *_expanded_nodes.csv files

Env overrides:
  TOPK=10              Top-K nodes by frequency
  OUTPUT_DIR           Default: same dir as expanded_nodes_csv
  BUILD_DIR            Default: ./outputFiles/build
  EXPERIMENT_TAG       追加到預設 BUILD_DIR/SEARCH_DIR
  DATA_TYPE            Default: float
  DIST_FN              Default: l2
  DRY_RUN=1            Print commands only
  --clean              只刪除本腳本輸出的 node_counts / topk_nodes / neighbors
USAGE
}

# 用法旗標：若帶 -h/--help 就印 usage 然後結束。
[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

# 計算本腳本位置，方便推導專案根目錄與 build/apps 路徑。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
DUMP_BIN="${APPS_DIR}/dump_disk_neighbors"

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

# 解析輸入：沒有參數時，若有 EXPERIMENT_TAG 就用預設 outputFiles/search/<tag>；否則顯示用法。
if [[ $# -lt 1 ]]; then
    if [[ -n "${EXPERIMENT_TAG:-}" ]]; then
        INPUT_PATH="${SCRIPT_DIR}/outputFiles/search/${EXPERIMENT_TAG}"
    else
        usage
        exit 1
    fi
else
    INPUT_PATH="$1"
fi
TOPK="${TOPK:-10}"  # 取用環境變數 TOPK，未提供則預設 10。

# Batch mode: if input is a directory, process all *_expanded_nodes.csv files
if [[ -d "$INPUT_PATH" ]]; then
    SEARCH_DIR="$INPUT_PATH"
    echo "▶ Batch mode: processing all *_expanded_nodes.csv in ${SEARCH_DIR}"
    
    expanded_files=()  # 累積所有 *_expanded_nodes.csv
    while IFS= read -r -d '' f; do
        expanded_files+=("$f")
    done < <(find "$SEARCH_DIR" -type f -name "*_expanded_nodes.csv" -print0)
    
    if [[ "${#expanded_files[@]}" -eq 0 ]]; then
        echo "ERROR: No *_expanded_nodes.csv found in $SEARCH_DIR" >&2
        exit 1
    fi
    
    echo "Found ${#expanded_files[@]} file(s)"
    for expanded_csv in "${expanded_files[@]}"; do
        echo ""
        echo "▶ Processing: ${expanded_csv}"
        # Recursive call for each file
        env TOPK="${TOPK}" OUTPUT_DIR="${OUTPUT_DIR:-}" BUILD_DIR="${BUILD_DIR:-}" \
            DATA_TYPE="${DATA_TYPE:-float}" DIST_FN="${DIST_FN:-l2}" \
            EXPERIMENT_TAG="${EXPERIMENT_TAG:-}" DRY_RUN="${DRY_RUN:-0}" \
            bash "$0" ${CLEAN:+--clean} "${expanded_csv}"
    done
    echo ""
    echo "✓ Batch complete: processed ${#expanded_files[@]} file(s)"
    exit 0
fi

# Single file mode
EXPANDED_CSV="$INPUT_PATH"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$EXPANDED_CSV")}"  # 預設輸出到 CSV 所在資料夾
if [[ -z "${BUILD_DIR:-}" ]]; then
    BUILD_DIR_DEFAULT=1
    BUILD_DIR="${SCRIPT_DIR}/outputFiles/build"
else
    BUILD_DIR_DEFAULT=0
    BUILD_DIR="${BUILD_DIR}"
fi
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
if [[ -n "$EXPERIMENT_TAG" && "$BUILD_DIR_DEFAULT" -eq 1 ]]; then
    BUILD_DIR="${BUILD_DIR}/${EXPERIMENT_TAG}"
fi
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
DRY_RUN="${DRY_RUN:-0}"

# 基本輸入檢查
if [[ ! -f "$EXPANDED_CSV" ]]; then
    echo "ERROR: expanded_nodes_csv not found: $EXPANDED_CSV" >&2
    exit 1
fi

# 產出檔案命名
base_name="$(basename "$EXPANDED_CSV")"
base_prefix="${base_name%_expanded_nodes.csv}"
counts_csv="${OUTPUT_DIR}/${base_prefix}_node_counts.csv"
topk_nodes="${OUTPUT_DIR}/${base_prefix}_topk${TOPK}_nodes.txt"
neighbors_csv="${OUTPUT_DIR}/${base_prefix}_topk${TOPK}_neighbors.csv"

if [[ "$CLEAN" == "1" ]]; then
    removed=0
    for f in "$counts_csv" "$topk_nodes" "$neighbors_csv"; do
        if [[ -f "$f" ]]; then
            rm -f "$f"
            removed=$((removed + 1))
            echo "刪除: ${f}"
        fi
    done
    echo "完成：已刪除 ${removed} 個檔案"
    exit 0
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: build_dir not found: $BUILD_DIR" >&2
    exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
    # 非 DRY_RUN 必須確保執行檔存在且可執行
    if [[ ! -x "$DUMP_BIN" ]]; then
        echo "ERROR: dump_disk_neighbors not found or not executable: $DUMP_BIN" >&2
        echo "Build first: cmake --build build --target dump_disk_neighbors -- -j" >&2
        exit 1
    fi
else
    if [[ ! -x "$DUMP_BIN" ]]; then
        echo "WARN: DRY_RUN ignores missing dump_disk_neighbors: $DUMP_BIN" >&2
    fi
fi

# 內嵌 Python：統計節點出現次數並輸出 Top-K 清單
python3 - "$EXPANDED_CSV" "$counts_csv" "$topk_nodes" "$TOPK" <<'PY'
import csv
import sys
from collections import Counter

expanded_csv, counts_csv, topk_nodes, topk = sys.argv[1:5]
topk = int(topk)

counter = Counter()
with open(expanded_csv, newline="") as f:
    reader = csv.DictReader(f)
    # 驗證欄位是否存在，避免硬寫索引造成的脆弱性
    if reader.fieldnames is None or "node_id" not in reader.fieldnames:
        print(f"ERROR: 'node_id' column not found in {expanded_csv}", file=sys.stderr)
        if reader.fieldnames:
            print(f"Available columns: {reader.fieldnames}", file=sys.stderr)
        sys.exit(1)
    # 逐行計數每個 node_id 的展開次數
    for row in reader:
        if not row or not row.get("node_id"):
            continue
        node_id = row["node_id"].strip()
        if node_id:
            counter[node_id] += 1

# 按展開次數降序排列，相同次數按 node_id 數值升序
items = sorted(counter.items(), key=lambda x: (-x[1], int(x[0])))

# 輸出節點計數表
with open(counts_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "count"])
    writer.writerows(items)

# 輸出 Top-K 節點清單
with open(topk_nodes, "w") as f:
    for node_id, _count in items[:topk]:
        f.write(f"{node_id}\n")
PY

# 判斷 index_tag：若父資料夾命名像 search 的 run 目錄，則用上層（index tag）；否則就用目前資料夾名
run_dir="$(basename "$(dirname "$EXPANDED_CSV")")"
parent_dir="$(basename "$(dirname "$(dirname "$EXPANDED_CSV")")")"
if [[ "$run_dir" =~ _W[0-9]+_L[0-9]+_K[0-9]+_cache[0-9]+_T[0-9]+ ]]; then
    index_tag="$parent_dir"
else
    index_tag="$run_dir"
fi
index_prefix="${BUILD_DIR}/${index_tag}"  # dump_disk_neighbors 的 index 路徑前綴

# 組裝 dump_disk_neighbors 指令
cmd=(
    "${DUMP_BIN}"
    --data_type "${DATA_TYPE}"
    --dist_fn "${DIST_FN}"
    --index_path_prefix "${index_prefix}"
    --input_nodes "${topk_nodes}"
    --output_path "${neighbors_csv}"
)

echo "Counts: ${counts_csv}"
echo "Top-K nodes: ${topk_nodes}"
echo "Neighbors: ${neighbors_csv}"

# DRY_RUN 只列印命令，不實際執行
if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN: %q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

# 實際執行 dump_disk_neighbors
"${cmd[@]}"
