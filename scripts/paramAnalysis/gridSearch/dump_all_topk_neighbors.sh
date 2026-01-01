#!/usr/bin/env bash
# Run dump_topk_neighbors.sh for every *_expanded_nodes.csv under outputFiles/search.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash dump_all_topk_neighbors.sh [search_dir]

Args:
  search_dir          Default: ./outputFiles/search

Env overrides:
  TOPK=10            Top-K nodes by frequency
  OUTPUT_DIR          Default: same dir as each expanded_nodes_csv
  BUILD_DIR           Default: ./outputFiles/build
  EXPERIMENT_TAG      追加到預設 SEARCH_DIR/BUILD_DIR
  DATA_TYPE           Default: float
  DIST_FN             Default: l2
  DRY_RUN=1           Print commands only
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="${1:-${SCRIPT_DIR}/outputFiles/search}"
TOPK="${TOPK:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
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
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$SEARCH_DIR" ]]; then
    echo "ERROR: 找不到 search_dir: $SEARCH_DIR" >&2
    exit 1
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
    echo "▶ Top-K neighbors: ${expanded_csv}"
    if [[ -n "$OUTPUT_DIR" ]]; then
        env TOPK="${TOPK}" BUILD_DIR="${BUILD_DIR}" DATA_TYPE="${DATA_TYPE}" DIST_FN="${DIST_FN}" \
            EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
            DRY_RUN="${DRY_RUN}" OUTPUT_DIR="${OUTPUT_DIR}" \
            bash "${SCRIPT_DIR}/dump_topk_neighbors.sh" "${expanded_csv}"
    else
        env TOPK="${TOPK}" BUILD_DIR="${BUILD_DIR}" DATA_TYPE="${DATA_TYPE}" DIST_FN="${DIST_FN}" \
            EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
            DRY_RUN="${DRY_RUN}" \
            bash "${SCRIPT_DIR}/dump_topk_neighbors.sh" "${expanded_csv}"
    fi
done

echo "完成：已處理所有 expanded_nodes CSV"
