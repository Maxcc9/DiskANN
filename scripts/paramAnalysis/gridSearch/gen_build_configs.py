#!/usr/bin/env python3

import csv
import os

# === Build parameter candidates ===
BUILD_R_LIST = [16, 32, 64, 128]
BUILD_L_LIST = [32, 64, 128, 256]

# Optional: 手動挑選代表性組合（推薦）
SELECTED = {
    (16, 32),
    (16, 64),
    (32, 64),
    (32, 128),
    (64, 128),
    (64, 256),
    (128, 256),  # optional
}

def main():
    rows = []
    build_id = 1

    for R in BUILD_R_LIST:
        for L in BUILD_L_LIST:
            if L < R:
                continue
            if SELECTED and (R, L) not in SELECTED:
                continue

            rows.append({
                "build_id": f"B{build_id}",
                "build_R": R,
                "build_L": L
            })
            build_id += 1

    # 讀取 EXPERIMENT_TAG，決定輸出資料夾
    experiment_tag = os.environ.get("EXPERIMENT_TAG", "")
    if experiment_tag:
        output_dir = f"./inputFiles/{experiment_tag}"
        output_file = f"{output_dir}/build_configs.csv"
    else:
        output_dir = "./inputFiles"
        output_file = "./inputFiles/build_configs.csv"

    # Auto-create directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["build_id", "build_R", "build_L"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} build configs → {output_file}")

if __name__ == "__main__":
    main()
