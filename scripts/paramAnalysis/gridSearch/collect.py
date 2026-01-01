#!/usr/bin/env python3
"""
這個script取得所有在./outputFiles/search/內所有資料夾內的所有 _summary_stats.csv 檔案，
並將它們彙總成一個大的CSV檔案，方便後續分析。

用法：
  python collect.py [output_file]

參數：
  output_file  彙總結果輸出檔案，預設 ./outputFiles/analyze/collected_stats_{timestamp}.csv
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path
from datetime import datetime
import pandas as pd


def find_summary_files(search_dir):
    """遞迴查找所有 *_summary_stats.csv 檔案"""
    pattern = os.path.join(search_dir, "**", "*_summary_stats.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def extract_index_info(file_path):
    """從檔案路徑提取 index 名稱"""
    # 路徑格式: ./outputFiles/search/{index_name}/{result_file}
    parts = Path(file_path).parts
    if len(parts) >= 3:
        return parts[-2]  # 返回倒數第二個部分（資料夾名稱）
    return "unknown"


def parse_expanded_stats(expanded_csv):
    if not os.path.isfile(expanded_csv):
        return {}
    try:
        df = pd.read_csv(expanded_csv)
    except Exception:
        return {}
    if df.empty:
        return {"expanded_rows": 0, "expanded_unique_nodes": 0, "expanded_unique_queries": 0}
    for col in ("query_id", "order", "node_id"):
        if col not in df.columns:
            return {}
    expanded_rows = len(df)
    unique_nodes = df["node_id"].nunique()
    unique_queries = df["query_id"].nunique()
    per_query_counts = df.groupby("query_id", sort=False).size()
    per_query_max_order = df.groupby("query_id", sort=False)["order"].max() + 1
    return {
        "expanded_rows": int(expanded_rows),
        "expanded_unique_nodes": int(unique_nodes),
        "expanded_unique_queries": int(unique_queries),
        "expanded_revisit_ratio": float(1.0 - (unique_nodes / expanded_rows)) if expanded_rows else 0.0,
        "expanded_per_query_mean": float(per_query_counts.mean()),
        "expanded_per_query_p50": float(per_query_counts.quantile(0.50)),
        "expanded_per_query_p90": float(per_query_counts.quantile(0.90)),
        "expanded_per_query_max": int(per_query_counts.max()),
        "expanded_steps_mean": float(per_query_max_order.mean()),
        "expanded_steps_p50": float(per_query_max_order.quantile(0.50)),
        "expanded_steps_p90": float(per_query_max_order.quantile(0.90)),
        "expanded_steps_max": int(per_query_max_order.max()),
    }


def parse_node_counts(node_counts_csv):
    if not os.path.isfile(node_counts_csv):
        return {}
    try:
        df = pd.read_csv(node_counts_csv)
    except Exception:
        return {}
    if df.empty or "count" not in df.columns:
        return {}
    counts = df["count"]
    total = float(counts.sum())
    sorted_counts = counts.sort_values(ascending=False).reset_index(drop=True)
    def share_top(n):
        if total == 0:
            return 0.0
        return float(sorted_counts.head(min(n, len(sorted_counts))).sum() / total)
    return {
        "node_counts_total": float(total),
        "node_counts_unique": int(len(df)),
        "node_counts_top1": float(sorted_counts.iloc[0]) if len(sorted_counts) else 0.0,
        "node_counts_top1_share": share_top(1),
        "node_counts_top10_share": share_top(10),
        "node_counts_top100_share": share_top(100),
    }


def parse_iostat_log(iostat_log):
    if not os.path.isfile(iostat_log):
        return {}
    device_blocks = {}
    current_header = None
    try:
        with open(iostat_log, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    current_header = None
                    continue
                if line.startswith("Device"):
                    current_header = line.split()
                    continue
                if current_header:
                    parts = line.split()
                    if len(parts) != len(current_header):
                        continue
                    device = parts[0]
                    values = {}
                    for key, val in zip(current_header[1:], parts[1:]):
                        try:
                            values[key] = float(val)
                        except ValueError:
                            values[key] = None
                    device_blocks.setdefault(device, []).append(values)
    except Exception:
        return {}

    if not device_blocks:
        return {}

    devices = sorted(device_blocks.keys())
    device_choice = devices[0]
    if len(devices) > 1:
        def score(dev):
            rows = device_blocks[dev]
            rkbs = [r.get("rkB/s") for r in rows if r.get("rkB/s") is not None]
            rs = [r.get("r/s") for r in rows if r.get("r/s") is not None]
            if rkbs:
                return sum(rkbs) / len(rkbs)
            if rs:
                return sum(rs) / len(rs)
            return 0.0
        device_choice = max(devices, key=score)

    rows = device_blocks[device_choice]
    columns = set().union(*[r.keys() for r in rows])
    stats = {
        "iostat_device": device_choice,
        "iostat_device_multi": int(len(devices) > 1),
        "iostat_device_list": ",".join(devices),
    }
    for col in sorted(columns):
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        stats[f"iostat_{col}_mean"] = float(sum(vals) / len(vals))
        stats[f"iostat_{col}_max"] = float(max(vals))
    return stats


def parse_topk_files(base_prefix, node_counts_csv):
    topk_rows = []
    neighbors_files = glob.glob(f"{base_prefix}_topk*_neighbors.csv")
    nodes_files = glob.glob(f"{base_prefix}_topk*_nodes.txt")
    nodes_map = {}
    for path in nodes_files:
        match = re.search(r"_topk(\d+)_nodes\.txt$", path)
        if match:
            nodes_map[int(match.group(1))] = path

    node_counts_df = None
    if node_counts_csv and os.path.isfile(node_counts_csv):
        try:
            node_counts_df = pd.read_csv(node_counts_csv)
        except Exception:
            node_counts_df = None

    for path in sorted(neighbors_files):
        match = re.search(r"_topk(\d+)_neighbors\.csv$", path)
        if not match:
            continue
        topk = int(match.group(1))
        row = {
            "topk_k": topk,
        }
        row_paths = {
            "topk_neighbors_path": path,
            "topk_nodes_path": nodes_map.get(topk, ""),
        }
        try:
            df = pd.read_csv(path)
            row["topk_neighbor_edges"] = int(len(df))
            if not df.empty and "node_id" in df.columns:
                row["topk_unique_nodes"] = int(df["node_id"].nunique())
            if not df.empty and "neighbor_id" in df.columns:
                row["topk_unique_neighbors"] = int(df["neighbor_id"].nunique())
            if not df.empty and "degree" in df.columns:
                degree_per_node = df.groupby("node_id")["degree"].first()
                row["topk_degree_mean"] = float(degree_per_node.mean())
                row["topk_degree_min"] = float(degree_per_node.min())
                row["topk_degree_max"] = float(degree_per_node.max())
                for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99):
                    row[f"topk_degree_p{p}"] = float(degree_per_node.quantile(p / 100.0))
        except Exception:
            pass

        if row_paths["topk_nodes_path"] and node_counts_df is not None:
            try:
                with open(row_paths["topk_nodes_path"], "r", encoding="utf-8") as f:
                    nodes = [line.strip() for line in f if line.strip()]
                if nodes and "node_id" in node_counts_df.columns and "count" in node_counts_df.columns:
                    counts_map = dict(zip(node_counts_df["node_id"].astype(str), node_counts_df["count"]))
                    total = float(node_counts_df["count"].sum())
                    topk_total = sum(counts_map.get(n, 0.0) for n in nodes)
                    row["topk_cover_ratio"] = float(topk_total / total) if total else 0.0
            except Exception:
                pass

        row.update(row_paths)
        topk_rows.append(row)

    summary = {
        "topk_variants_count": len(topk_rows),
        "topk_k_list": ",".join(str(r["topk_k"]) for r in topk_rows),
        "topk_neighbors_files": ";".join(r["topk_neighbors_path"] for r in topk_rows),
        "topk_nodes_files": ";".join(r["topk_nodes_path"] for r in topk_rows if r["topk_nodes_path"]),
    }
    return topk_rows, summary


def collect_summary_stats(search_dir, output_file):
    """
    蒐集所有 summary_stats.csv 並彙總到一個檔案
    
    Args:
        search_dir: search 輸出目錄
        output_file: 彙總輸出檔案路徑
    
    Returns:
        成功處理的檔案數量
    """
    # 查找所有 summary_stats.csv 檔案
    summary_files = find_summary_files(search_dir)
    
    if not summary_files:
        print(f"警告: 在 {search_dir} 內找不到任何 *_summary_stats.csv 檔案", file=sys.stderr)
        return 0
    
    print(f"找到 {len(summary_files)} 個 summary_stats.csv 檔案")
    
    all_data = []
    topk_data = []
    row_id = 1
    
    for summary_file in summary_files:
        try:
            df = pd.read_csv(summary_file)
            base_prefix = summary_file[: -len("_summary_stats.csv")]
            expanded_csv = f"{base_prefix}_expanded_nodes.csv"
            node_counts_csv = f"{base_prefix}_node_counts.csv"
            iostat_log = f"{base_prefix}_iostat.log"
            topk_rows, topk_summary = parse_topk_files(base_prefix, node_counts_csv)

            expanded_stats = parse_expanded_stats(expanded_csv)
            node_count_stats = parse_node_counts(node_counts_csv)
            iostat_stats = parse_iostat_log(iostat_log)

            extra_cols = {
                "run_prefix": os.path.basename(base_prefix),
                "expanded_nodes_path": expanded_csv if os.path.isfile(expanded_csv) else "",
                "node_counts_path": node_counts_csv if os.path.isfile(node_counts_csv) else "",
                "iostat_log_path": iostat_log if os.path.isfile(iostat_log) else "",
            }
            extra_cols.update(expanded_stats)
            extra_cols.update(node_count_stats)
            extra_cols.update(iostat_stats)
            extra_cols.update(topk_summary)

            # 添加 id 列（在最前面）
            ids = list(range(row_id, row_id + len(df)))
            df.insert(0, "id", ids)
            row_id += len(df)

            for key, value in extra_cols.items():
                df[key] = value
            
            all_data.append(df)
            index_name = extract_index_info(summary_file)
            print(f"  ✓ 已讀取: {summary_file} (index: {index_name}, 行數: {len(df)})")

            for row in topk_rows:
                row["run_prefix"] = os.path.basename(base_prefix)
                row["summary_stats_path"] = summary_file
                topk_data.append(row)
            
        except Exception as e:
            print(f"  ✗ 讀取失敗: {summary_file} - {e}", file=sys.stderr)
            continue
    
    if not all_data:
        print("錯誤: 沒有成功讀取任何檔案", file=sys.stderr)
        return 0
    
    # 合併所有資料
    print(f"\n正在合併 {len(all_data)} 個資料框...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"合併後總列數: {len(combined_df)}")
    print(f"列名: {list(combined_df.columns)}")
    
    # 儲存到輸出檔案
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ 彙總完成，已儲存至: {output_file}")
    print(f"  總行數: {len(combined_df)}")
    print(f"  總列數: {len(combined_df.columns)}")
    
    return len(all_data), topk_data


def main():
    parser = argparse.ArgumentParser(
        description="蒐集 search 結果中的 summary_stats.csv 並彙總到單一檔案"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="彙總結果輸出檔案 (預設: ./outputFiles/analyze/collected_stats_{timestamp}.csv)"
    )
    parser.add_argument(
        "-d", "--search-dir",
        default="./outputFiles/search",
        help="search 輸出目錄 (預設: ./outputFiles/search)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="顯示詳細資訊"
    )
    parser.add_argument(
        "--topk-output",
        default=None,
        help="Top-K 彙總結果輸出檔案 (預設: ./outputFiles/analyze/collected_topk_{timestamp}.csv)"
    )
    
    args = parser.parse_args()
    
    # 轉為絕對路徑，必要時加入 EXPERIMENT_TAG
    search_dir = args.search_dir
    experiment_tag = os.environ.get("EXPERIMENT_TAG", "")
    if experiment_tag and args.search_dir == "./outputFiles/search":
        search_dir = os.path.join(search_dir, experiment_tag)
    search_dir = os.path.abspath(search_dir)
    
    # 生成預設輸出檔案名稱（帶時間戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_tag = Path(search_dir).name
    if args.output is None:
        output_file = os.path.abspath(
            f"./outputFiles/analyze/collected_stats_{search_tag}_{timestamp}.csv"
        )
    else:
        output_file = os.path.abspath(args.output)
    if args.topk_output is None:
        topk_output = os.path.abspath(
            f"./outputFiles/analyze/collected_topk_{search_tag}_{timestamp}.csv"
        )
    else:
        topk_output = os.path.abspath(args.topk_output)
    
    # 檢查輸入目錄
    if not os.path.isdir(search_dir):
        print(f"錯誤: search 目錄不存在或不是目錄: {search_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 建立輸出目錄
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"搜尋目錄: {search_dir}")
    print(f"輸出檔案: {output_file}")
    print(f"Top-K 輸出檔案: {topk_output}")
    print("-" * 60)
    
    # 執行蒐集
    count, topk_data = collect_summary_stats(search_dir, output_file)
    
    if count == 0:
        sys.exit(1)

    if topk_data:
        topk_df = pd.DataFrame(topk_data)
        path_cols = [c for c in topk_df.columns if c.endswith("_path")]
        other_cols = [c for c in topk_df.columns if c not in path_cols]
        topk_df = topk_df[other_cols + path_cols]
        os.makedirs(os.path.dirname(topk_output), exist_ok=True)
        topk_df.to_csv(topk_output, index=False)
        print(f"\n✓ Top-K 彙總完成，已儲存至: {topk_output}")
        print(f"  總行數: {len(topk_df)}")
        print(f"  總列數: {len(topk_df.columns)}")
    
    # 顯示統計資訊
    print("\n" + "=" * 60)
    print("統計資訊:")
    print("=" * 60)
    try:
        df = pd.read_csv(output_file)
        path_cols = [c for c in df.columns if c.endswith("_path") or c.endswith("_files")]
        other_cols = [c for c in df.columns if c not in path_cols]
        df = df[other_cols + path_cols]
        df.to_csv(output_file, index=False)
        print(f"總列數: {len(df)}")
        print(f"ID 範圍: {df['id'].min()} - {df['id'].max()}")
        print(f"列名: {list(df.columns)[:15]}...")
        print("\n前 10 行:")
        print(df.head(10).to_string())
    except Exception as e:
        print(f"無法讀取輸出檔案: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
