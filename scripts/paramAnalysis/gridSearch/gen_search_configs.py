#!/usr/bin/env python3

import csv
import argparse

SEARCH_W_LIST = [1, 2, 4, 8, 16]
ALPHA_LIST = [2, 3, 4]          # search_L = alpha * search_W
CACHE_RATIO_LIST = [0, 0.01, 0.02, 0.05, 0.10]
THREAD_LIST = [1, 4]            # 會自動補 max_cores
K_LIST = [10, 100]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--max_cores", type=int, required=True)
    args = parser.parse_args()

    dataset_size = args.dataset_size
    max_cores = args.max_cores

    thread_list = THREAD_LIST + [max_cores]

    rows = []
    sid = 1

    for W in SEARCH_W_LIST:
        for alpha in ALPHA_LIST:
            for K in K_LIST:
                L = max(alpha * W, K)

                for cache_ratio in CACHE_RATIO_LIST:
                    cache_nodes = int(cache_ratio * dataset_size)

                    for T in thread_list:
                        rows.append({
                            "search_id": f"S{sid}",
                            "search_W": W,
                            "search_L": L,
                            "search_K": K,
                            "search_cache": cache_nodes,
                            "search_thread": T,
                        })
                        sid += 1

    with open("./inputFiles/search_configs.csv", "w", newline="") as f:
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

    print(f"Generated {len(rows)} search configs → search_configs.csv")

if __name__ == "__main__":
    main()
