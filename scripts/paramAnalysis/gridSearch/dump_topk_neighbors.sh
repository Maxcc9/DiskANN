#!/usr/bin/env bash
# Count expanded node frequency then dump neighbors for Top-K nodes.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash dump_topk_neighbors.sh <expanded_nodes_csv>

Args:
  expanded_nodes_csv   Path to *_expanded_nodes.csv

Env overrides:
  TOPK=10             Top-K nodes by frequency
  OUTPUT_DIR           Default: same dir as expanded_nodes_csv
  BUILD_DIR            Default: ./outputFiles/build
  EXPERIMENT_TAG       追加到預設 BUILD_DIR
  DATA_TYPE            Default: float
  DIST_FN              Default: l2
  DRY_RUN=1            Print commands only
USAGE
}

[[ ${1:-} == "-h" || ${1:-} == "--help" ]] && { usage; exit 0; }
[[ $# -lt 1 ]] && { usage; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APPS_DIR="${DISKANN_ROOT}/build/apps"
DUMP_BIN="${APPS_DIR}/dump_disk_neighbors"

EXPANDED_CSV="$1"
TOPK="${TOPK:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$EXPANDED_CSV")}"
if [[ -z "${BUILD_DIR+x}" ]]; then
    BUILD_DIR_DEFAULT=1
    BUILD_DIR="${SCRIPT_DIR}/outputFiles/build"
else
    BUILD_DIR_DEFAULT=0
    BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/outputFiles/build}"
fi
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
if [[ -n "$EXPERIMENT_TAG" && "$BUILD_DIR_DEFAULT" -eq 1 ]]; then
    BUILD_DIR="${BUILD_DIR}/${EXPERIMENT_TAG}"
fi
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$EXPANDED_CSV" ]]; then
    echo "ERROR: expanded_nodes_csv not found: $EXPANDED_CSV" >&2
    exit 1
fi
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: build_dir not found: $BUILD_DIR" >&2
    exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
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

base_name="$(basename "$EXPANDED_CSV")"
base_prefix="${base_name%_expanded_nodes.csv}"
counts_csv="${OUTPUT_DIR}/${base_prefix}_node_counts.csv"
topk_nodes="${OUTPUT_DIR}/${base_prefix}_topk${TOPK}_nodes.txt"
neighbors_csv="${OUTPUT_DIR}/${base_prefix}_topk${TOPK}_neighbors.csv"

python3 - "$EXPANDED_CSV" "$counts_csv" "$topk_nodes" "$TOPK" <<'PY'
import csv
import sys
from collections import Counter

expanded_csv, counts_csv, topk_nodes, topk = sys.argv[1:5]
topk = int(topk)

counter = Counter()
with open(expanded_csv, newline="") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if not row:
            continue
        node_id = row[-1].strip()
        if not node_id:
            continue
        counter[node_id] += 1

items = sorted(counter.items(), key=lambda x: (-x[1], int(x[0])))

with open(counts_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "count"])
    writer.writerows(items)

with open(topk_nodes, "w") as f:
    for node_id, _count in items[:topk]:
        f.write(f"{node_id}\n")
PY

index_tag="$(basename "$(dirname "$EXPANDED_CSV")")"
index_prefix="${BUILD_DIR}/${index_tag}"

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

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN: %q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
