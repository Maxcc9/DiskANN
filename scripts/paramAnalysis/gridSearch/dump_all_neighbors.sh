#!/usr/bin/env bash
# 將 outputFiles/search/ 內所有 *_expanded_nodes.csv 轉成對應的鄰居列表 CSV。

set -euo pipefail

usage() {
    cat <<'USAGE'
用法:
  bash dump_all_neighbors.sh [search_dir]

參數:
  search_dir          預設 ./outputFiles/search

環境變數可覆寫:
  BUILD_DIR           預設 ./outputFiles/build
  EXPERIMENT_TAG      追加到預設 SEARCH_DIR/BUILD_DIR
  DATA_TYPE           預設 float
  DIST_FN             預設 l2
  MAX_NODES           預設 0 (0 = all)
  KEEP_DUPLICATES     預設 0 (1 = keep duplicates)
  DRY_RUN=1           僅列印命令不執行
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
DUMP_BIN="${APPS_DIR}/dump_disk_neighbors"

SEARCH_DIR="${1:-${SCRIPT_DIR}/outputFiles/search}"
if [[ -z "${BUILD_DIR+x}" ]]; then
    BUILD_DIR_DEFAULT=1
    BUILD_DIR="${SCRIPT_DIR}/outputFiles/build"
else
    BUILD_DIR_DEFAULT=0
    BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/outputFiles/build}"
fi
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
if [[ -z "${1:-}" && -n "$EXPERIMENT_TAG" ]]; then
    SEARCH_DIR="${SEARCH_DIR}/${EXPERIMENT_TAG}"
fi
if [[ -n "$EXPERIMENT_TAG" && "$BUILD_DIR_DEFAULT" -eq 1 ]]; then
    BUILD_DIR="${BUILD_DIR}/${EXPERIMENT_TAG}"
fi
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
MAX_NODES="${MAX_NODES:-0}"
KEEP_DUPLICATES="${KEEP_DUPLICATES:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$SEARCH_DIR" ]]; then
    echo "ERROR: 找不到 search_dir: $SEARCH_DIR" >&2
    exit 1
fi
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: 找不到 build_dir: $BUILD_DIR" >&2
    exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -x "$DUMP_BIN" ]]; then
        echo "ERROR: dump_disk_neighbors 不存在或不可執行: $DUMP_BIN" >&2
        echo "請先編譯: cmake --build build --target dump_disk_neighbors -- -j" >&2
        exit 1
    fi
else
    if [[ ! -x "$DUMP_BIN" ]]; then
        echo "WARN: DRY_RUN 模式忽略不存在的 dump_disk_neighbors: $DUMP_BIN" >&2
    fi
fi

expanded_files=()
while IFS= read -r -d '' f; do
    expanded_files+=("$f")
done < <(find "$SEARCH_DIR" -type f -name "*_expanded_nodes.csv" -print0)

if [[ "${#expanded_files[@]}" -eq 0 ]]; then
    echo "ERROR: 找不到 *_expanded_nodes.csv 於 $SEARCH_DIR" >&2
    exit 1
fi

for expanded_csv in "${expanded_files[@]}"; do
    index_tag="$(basename "$(dirname "$expanded_csv")")"
    index_prefix="${BUILD_DIR}/${index_tag}"
    output_csv="${expanded_csv%_expanded_nodes.csv}_neighbors.csv"

    cmd=(
        "${DUMP_BIN}"
        --data_type "${DATA_TYPE}"
        --dist_fn "${DIST_FN}"
        --index_path_prefix "${index_prefix}"
        --input_nodes "${expanded_csv}"
        --output_path "${output_csv}"
        --max_nodes "${MAX_NODES}"
    )
    if [[ "$KEEP_DUPLICATES" == "1" ]]; then
        cmd+=(--keep_duplicates)
    fi

    echo "▶ Dump neighbors: ${expanded_csv} -> ${output_csv}"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN: %q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    "${cmd[@]}"
done

echo "完成：已輸出所有 neighbors CSV"
