#!/usr/bin/env python3
"""
Deduplicate rows in a collected_all_*.csv by key columns, keeping the median-qps row.
"""

import argparse
import csv
import math
from pathlib import Path

DEFAULT_KEY_COLS = [
    "build_R",
    "build_L",
    "search_K",
    "search_L",
    "search_W",
    "search_T",
    "actual_cached_nodes",
]


def _parse_qps(val):
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except ValueError:
        return None


def pick_median_row(rows, qps_key="qps"):
    """
    rows: list of dicts
    returns: one dict (median by qps; if even, upper middle). If all qps invalid, keep first.
    """
    parsed = []
    for r in rows:
        qps = _parse_qps(r.get(qps_key))
        parsed.append((qps, r))

    valid = [item for item in parsed if item[0] is not None and not math.isnan(item[0])]
    if not valid:
        return rows[0]

    valid.sort(key=lambda x: x[0])
    mid = len(valid) // 2
    return valid[mid][1]


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate collected_all CSV by key columns, keeping median-qps row."
    )
    parser.add_argument("-i", "--input", required=True, help="Input collected_all_*.csv")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--key-cols",
        default=",".join(DEFAULT_KEY_COLS),
        help="Comma-separated key columns used for grouping",
    )
    parser.add_argument(
        "--qps-col",
        default="qps",
        help="Column name for QPS (default: qps)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_median" + input_path.suffix)

    key_cols = [c.strip() for c in args.key_cols.split(",") if c.strip()]
    if not key_cols:
        raise SystemExit("ERROR: key-cols is empty")

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("ERROR: input has no header")
        fieldnames = list(reader.fieldnames)
        for c in key_cols:
            if c not in fieldnames:
                raise SystemExit(f"ERROR: key column not found: {c}")
        if args.qps_col not in fieldnames:
            raise SystemExit(f"ERROR: qps column not found: {args.qps_col}")

        groups = {}
        for row in reader:
            key = tuple(row.get(c, "") for c in key_cols)
            groups.setdefault(key, []).append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in groups.values():
            chosen = pick_median_row(rows, qps_key=args.qps_col)
            writer.writerow(chosen)

    print(f"Wrote {output_path} (groups: {len(groups)})")


if __name__ == "__main__":
    main()
