#!/bin/bash
# clean_topk_neighbors.sh - 清理 topk 相關檔案（不依賴 TOPK 值）
# 
# 自動掃描找到所有相關的 topk* 檔案並刪除，無需指定 TOPK

set -euo pipefail

usage() {
    cat <<'USAGE'
清理 topk 相關檔案 (無需指定 TOPK 值)

用法：
  bash clean_topk_neighbors.sh <search_dir>       # 清理整個資料夾
  bash clean_topk_neighbors.sh --dry-run <dir>   # 只列出要刪除的檔案，不實際刪除

範例：
  # 清理整個實驗
  bash clean_topk_neighbors.sh outputFiles/search/sift01/

  # 只看會刪除什麼
  bash clean_topk_neighbors.sh --dry-run outputFiles/search/sift01/

  # 清理特定參數配置
  bash clean_topk_neighbors.sh outputFiles/search/sift01/sift_R32_L128_B2_M2/
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

DRY_RUN=0
if [[ ${1:-} == "--dry-run" ]]; then
    DRY_RUN=1
    SEARCH_DIR="${2:-.}"
else
    SEARCH_DIR="${1:-.}"
fi

if [[ ! -d "$SEARCH_DIR" ]]; then
    echo "ERROR: Directory not found: $SEARCH_DIR" >&2
    exit 1
fi

echo "掃描目錄: $SEARCH_DIR"
echo ""

# 找到所有要刪除的檔案類型
# 1. *_node_counts.csv
# 2. *_topk*_nodes.txt  
# 3. *_topk*_neighbors.csv

files_to_delete=(
    $(find "$SEARCH_DIR" -name "*_node_counts.csv" -type f 2>/dev/null)
    $(find "$SEARCH_DIR" -name "*_topk*_nodes.txt" -type f 2>/dev/null)
    $(find "$SEARCH_DIR" -name "*_topk*_neighbors.csv" -type f 2>/dev/null)
)

if [[ ${#files_to_delete[@]} -eq 0 ]]; then
    echo "✓ 沒有找到要刪除的檔案"
    exit 0
fi

# 統計大小
total_size=0
for f in "${files_to_delete[@]}"; do
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
done

echo "找到 ${#files_to_delete[@]} 個檔案，合計 $(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo "$total_size bytes")"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "【DRY-RUN】以下檔案將被刪除："
    for f in "${files_to_delete[@]}"; do
        size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        printf "  %-80s %8s\n" "$(basename $f)" "$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "$size")"
    done
    echo ""
    echo "執行下列命令以實際刪除："
    echo "  bash clean_topk_neighbors.sh $SEARCH_DIR"
    exit 0
fi

# 實際刪除
echo "刪除中..."
removed=0
for f in "${files_to_delete[@]}"; do
    rm -f "$f"
    removed=$((removed + 1))
    if (( removed % 100 == 0 )); then
        echo "  已刪除 $removed 個檔案..."
    fi
done

echo "✓ 完成：已刪除 $removed 個檔案，釋放 $(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo "$total_size bytes")"
