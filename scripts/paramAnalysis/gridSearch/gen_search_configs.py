#!/usr/bin/env python3

import csv
import os
import argparse

SEARCH_W_LIST = [1, 2, 4, 8, 16]
ALPHA_LIST = [2, 3, 4]          # search_L = alpha * search_W
CACHE_RATIO_LIST = [0, 0.01, 0.02, 0.05, 0.10]
THREAD_LIST = [1, 2, 4, 8, 16]            # 會自動補 max_cores
K_LIST = [10, 100]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--max_cores", type=int, required=True)
    args = parser.parse_args()

    dataset_size = args.dataset_size
    max_cores = args.max_cores

    # preserve order while avoiding duplicates (e.g., max_cores already in THREAD_LIST)
    thread_list = list(dict.fromkeys(THREAD_LIST + [max_cores]))

    rows = []
    seen = set()
    sid = 1

    for W in SEARCH_W_LIST:
        for K in K_LIST:
            # L is derived from alpha and W, but only L is emitted; de-dup L values
            l_list = sorted({max(alpha * W, K) for alpha in ALPHA_LIST})
            for L in l_list:

                for cache_ratio in CACHE_RATIO_LIST:
                    cache_nodes = int(cache_ratio * dataset_size)

                    for T in thread_list:
                        key = (W, L, K, cache_nodes, T)
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append({
                            "search_id": f"S{sid}",
                            "search_W": W,
                            "search_L": L,
                            "search_K": K,
                            "search_cache": cache_nodes,
                            "search_thread": T,
                        })
                        sid += 1

    # 讀取 EXPERIMENT_TAG，決定輸出資料夾（固定輸出到本腳本下的 inputFiles）
    base_dir = os.path.join(os.path.dirname(__file__), "inputFiles")
    experiment_tag = os.environ.get("EXPERIMENT_TAG", "")
    if experiment_tag:
        output_dir = os.path.join(base_dir, experiment_tag)
        output_file = os.path.join(output_dir, "search_configs.csv")
    else:
        output_dir = base_dir
        output_file = os.path.join(base_dir, "search_configs.csv")

    # Auto-create directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "search_id",
                "search_W",
                "search_L",
                "search_K",
                "search_cache",
                "search_thread"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} search configs → {output_file}")

if __name__ == "__main__":
    main()
