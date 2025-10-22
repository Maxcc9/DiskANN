#!/usr/bin/env python3
"""
Utility script to sweep DiskANN search parameters over a lambda range and collect results.

Example:
    python scripts/run_search_sweep.py \
        --search-bin build/apps/search_disk_index \
        --data-type float \
        --dist-fn l2 \
        --index-path-prefix sift/index_R32_L50 \
        --query-file sift/sift_query.bin \
        --gt-file sift/sift_groundtruth.bin \
        --result-path-prefix sift/sweep_run \
        --summary-csv sift/sweep_summary.csv \
        --K-start 10 --K-end 20 --K-step 5 \
        --L-start 20 --L-end 60 --L-step 20 \
        --beamwidth-start 4 --beamwidth-end 8 --beamwidth-step 4 \
        --num-nodes-to-cache-start 5000 --num-nodes-to-cache-end 10000 --num-nodes-to-cache-step 5000 \
        --num-threads-start 16 --num-threads-end 32 --num-threads-step 16 \
        --lambda-start 0.0 \
        --lambda-end 2.0 \
        --lambda-step 0.5
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from decimal import Decimal, getcontext
from typing import Iterable, List, Tuple


def decimal_range(start: float, end: float, step: float) -> Iterable[float]:
    """Generate floating point range using Decimal for robustness."""
    getcontext().prec = 12
    dec_start = Decimal(str(start))
    dec_end = Decimal(str(end))
    dec_step = Decimal(str(step))
    if dec_step <= 0:
        raise ValueError("lambda-step must be positive.")

    current = dec_start
    while current <= dec_end + Decimal("1e-12"):
        yield float(current)
        current += dec_step


def parse_table(output: str) -> List[Tuple[int, int, float, float, float, float, float, float, float, float]]:
    """Extract table rows from search_disk_index output."""
    rows: List[Tuple[int, int, float, float, float, float, float, float, float, float]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        # expect at least the base metrics (8 values). optional recall columns follow.
        if len(tokens) < 8:
            continue
        try:
            L_val = int(tokens[0])
            beamwidth = int(tokens[1])
            qps = float(tokens[2])
            mean_latency = float(tokens[3])
            p999_latency = float(tokens[4])
            mean_ios = float(tokens[5])
            mean_io_us = float(tokens[6])
            cpu_s = float(tokens[7])
            recall = float(tokens[8]) if len(tokens) >= 9 else float("nan")
            recall_at_1 = float(tokens[9]) if len(tokens) >= 10 else float("nan")
        except ValueError:
            continue
        rows.append((L_val, beamwidth, qps, mean_latency, p999_latency, mean_ios, mean_io_us, cpu_s, recall, recall_at_1))
    return rows


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def build_command(args: argparse.Namespace, lambda_value: float, result_path_prefix: str) -> List[str]:
    cmd = [
        args.search_bin,
        "--data_type",
        args.data_type,
        "--dist_fn",
        args.dist_fn,
        "--index_path_prefix",
        args.index_path_prefix,
        "--query_file",
        args.query_file,
        "--gt_file",
        args.gt_file,
        "--result_path",
        result_path_prefix,
        "-K",
        str(args.K),
        "-L",
        *[str(l) for l in args.Ls],
        "-W",
        str(args.beamwidth),
        "--num_nodes_to_cache",
        str(args.num_nodes_to_cache),
        "--num_threads",
        str(args.num_threads),
        "--lambda",
        str(lambda_value),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep DiskANN search parameters over lambda range.")
    parser.add_argument("--search-bin", default="build/apps/search_disk_index", help="Path to search_disk_index binary.")
    parser.add_argument("--data-type", required=True, help="Data type (e.g., float, uint8, int8).")
    parser.add_argument("--dist-fn", required=True, help="Distance function (e.g., l2, mips).")
    parser.add_argument("--index-path-prefix", required=True, help="Index path prefix.")
    parser.add_argument("--query-file", required=True, help="Query file path.")
    parser.add_argument("--gt-file", required=True, help="Ground truth file path.")
    parser.add_argument("--result-path-prefix", required=True, help="Prefix for result files.")
    parser.add_argument("--summary-csv", required=True, help="Output CSV summarizing results.")
    parser.add_argument("--K", type=int, help="Top-K value for search.")
    parser.add_argument("--K-start", type=int)
    parser.add_argument("--K-end", type=int)
    parser.add_argument("--K-step", type=int)
    parser.add_argument("--Ls", nargs="+", type=int, help="Explicit list of L values (space separated).")
    parser.add_argument("--L-start", type=int)
    parser.add_argument("--L-end", type=int)
    parser.add_argument("--L-step", type=int)
    parser.add_argument("--beamwidth", type=int, help="Beamwidth parameter (W).")
    parser.add_argument("--beamwidth-start", type=int)
    parser.add_argument("--beamwidth-end", type=int)
    parser.add_argument("--beamwidth-step", type=int)
    parser.add_argument("--num-nodes-to-cache", type=int, help="Number of nodes to cache.")
    parser.add_argument("--num-nodes-to-cache-start", type=int)
    parser.add_argument("--num-nodes-to-cache-end", type=int)
    parser.add_argument("--num-nodes-to-cache-step", type=int)
    parser.add_argument("--num-threads", type=int, help="Number of threads.")
    parser.add_argument("--num-threads-start", type=int)
    parser.add_argument("--num-threads-end", type=int)
    parser.add_argument("--num-threads-step", type=int)
    parser.add_argument("--lambda-start", type=float, required=True, help="Lambda sweep start (inclusive).")
    parser.add_argument("--lambda-end", type=float, required=True, help="Lambda sweep end (inclusive).")
    parser.add_argument("--lambda-step", type=float, required=True, help="Lambda sweep step.")

    args = parser.parse_args()

    if not os.path.exists(args.search_bin):
        print(f"Error: search binary '{args.search_bin}' not found.", file=sys.stderr)
        sys.exit(1)

    ensure_parent_dir(args.summary_csv)

    def build_int_values(single: int, start: int, end: int, step: int, name: str) -> List[int]:
        if start is None and end is None and step is None:
            if single is None:
                print(f"Error: must provide either --{name} or --{name}-start/--{name}-end/--{name}-step.", file=sys.stderr)
                sys.exit(1)
            return [single]
        if None in (start, end, step):
            print(f"Error: {name}-start, {name}-end, and {name}-step must all be provided together.", file=sys.stderr)
            sys.exit(1)
        if step <= 0:
            print(f"Error: {name}-step must be positive.", file=sys.stderr)
            sys.exit(1)
        if start > end:
            print(f"Error: {name}-start must be <= {name}-end.", file=sys.stderr)
            sys.exit(1)
        values = list(range(start, end + 1, step))
        return values

    K_values = build_int_values(args.K, args.K_start, args.K_end, args.K_step, "K")

    L_lists: List[List[int]]
    if args.L_start is not None or args.L_end is not None or args.L_step is not None:
        if None in (args.L_start, args.L_end, args.L_step):
            print("Error: L-start, L-end, and L-step must all be provided together.", file=sys.stderr)
            sys.exit(1)
        if args.L_step <= 0:
            print("Error: L-step must be positive.", file=sys.stderr)
            sys.exit(1)
        if args.L_start > args.L_end:
            print("Error: L-start must be <= L-end.", file=sys.stderr)
            sys.exit(1)
        L_lists = [[value] for value in range(args.L_start, args.L_end + 1, args.L_step)]
    elif args.Ls:
        L_lists = [args.Ls]
    else:
        print("Error: must provide either --Ls or the L start/end/step range.", file=sys.stderr)
        sys.exit(1)

    beamwidth_values = build_int_values(
        args.beamwidth, args.beamwidth_start, args.beamwidth_end, args.beamwidth_step, "beamwidth"
    )
    cache_values = build_int_values(
        args.num_nodes_to_cache,
        args.num_nodes_to_cache_start,
        args.num_nodes_to_cache_end,
        args.num_nodes_to_cache_step,
        "num-nodes-to-cache",
    )
    thread_values = build_int_values(
        args.num_threads, args.num_threads_start, args.num_threads_end, args.num_threads_step, "num-threads"
    )

    all_rows: List[dict] = []
    lambda_values = list(decimal_range(args.lambda_start, args.lambda_end, args.lambda_step))

    for k_val in K_values:
        for L_values in L_lists:
            L_label = "-".join(str(x) for x in L_values)
            for beamwidth_val in beamwidth_values:
                for cache_val in cache_values:
                    for thread_val in thread_values:
                        for lambda_value in lambda_values:
                            suffix = (
                                f"K{k_val}_L{L_label}_W{beamwidth_val}_C{cache_val}_T{thread_val}_lambda{lambda_value:g}"
                            )
                            result_prefix = f"{args.result_path_prefix}_{suffix}"
                            ensure_parent_dir(result_prefix)
                            cmd_args_dict = vars(args).copy()
                            cmd_args_dict.update(
                                {
                                    "K": k_val,
                                    "Ls": L_values,
                                    "beamwidth": beamwidth_val,
                                    "num_nodes_to_cache": cache_val,
                                    "num_threads": thread_val,
                                }
                            )
                            cmd_args = argparse.Namespace(**cmd_args_dict)
                            cmd = build_command(
                                cmd_args,
                                lambda_value,
                                result_prefix,
                            )

                            print(f"\n=== Running {suffix} ===")
                            print("Command:", " ".join(cmd))

                            completed = subprocess.run(cmd, capture_output=True, text=True)
                            if completed.returncode != 0:
                                print("Command failed. Stdout and stderr follow.", file=sys.stderr)
                                print(completed.stdout, file=sys.stderr)
                                print(completed.stderr, file=sys.stderr)
                                sys.exit(completed.returncode)

                            table_rows = parse_table(completed.stdout)
                            if not table_rows:
                                print("Warning: no table rows parsed from output; check the command output.", file=sys.stderr)

                            for row in table_rows:
                                record = {
                                    "lambda": lambda_value,
                                    "L": row[0],
                                    "L_config": " ".join(str(x) for x in L_values),
                                    "beamwidth": row[1],
                                    "K": k_val,
                                    "num_nodes_to_cache": cache_val,
                                    "num_threads": thread_val,
                                    "QPS": row[2],
                                    "mean_latency_us": row[3],
                                    "p999_latency_us": row[4],
                                    "mean_ios": row[5],
                                    "mean_io_us": row[6],
                                    "cpu_seconds": row[7],
                                    "recall_at_K": row[8],
                                    "recall_at_1": row[9],
                                    "result_path_prefix": result_prefix,
                                }
                                all_rows.append(record)

    fieldnames = [
        "lambda",
        "L",
        "L_config",
        "beamwidth",
        "K",
        "num_nodes_to_cache",
        "num_threads",
        "QPS",
        "mean_latency_us",
        "p999_latency_us",
        "mean_ios",
        "mean_io_us",
        "cpu_seconds",
        "recall_at_K",
        "recall_at_1",
        "result_path_prefix",
    ]

    with open(args.summary_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSummary written to {args.summary_csv}")


if __name__ == "__main__":
    main()
