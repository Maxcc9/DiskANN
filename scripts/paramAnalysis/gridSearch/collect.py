#!/usr/bin/env python3
"""
這個script取得所有在./outputFiles/search/內所有資料夾內的所有 _summary_stats.csv 檔案，
並將它們彙總成一個大的CSV檔案，方便後續分析。

用法：
  python collect.py [output_file]

參數：
  output_file  彙總結果輸出檔案，預設 ./outputFiles/analyze/{search_tag}/collected_stats_{timestamp}.csv
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


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
    """解析 expanded_nodes.csv，計算被展開節點的各項統計"""
    if not os.path.isfile(expanded_csv):
        return {}
    try:
        df = pd.read_csv(expanded_csv)
    except Exception:
        return {}
    if df.empty:
        return {"expanded_nodes_total": 0, "expanded_nodes_unique": 0}
    for col in ("query_id", "order", "node_id"):
        if col not in df.columns:
            return {}
    
    nodes_total = len(df)
    nodes_unique = df["node_id"].nunique()
    
    # 計算節點被展開次數的統計（原 parse_node_counts 的功能）
    node_counts = df["node_id"].value_counts().sort_values(ascending=False)
    total_expansions = float(node_counts.sum())
    
    def share_top(n):
        if total_expansions == 0:
            return 0.0
        return float(node_counts.head(min(n, len(node_counts))).sum() / total_expansions)
    
    return {
        "expanded_nodes_total": int(nodes_total),
        "expanded_nodes_unique": int(nodes_unique),
        "expanded_nodes_revisit_ratio": float(1.0 - (nodes_unique / nodes_total)) if nodes_total else 0.0,
 
        "expanded_node_hottest_count": float(node_counts.iloc[0]) if len(node_counts) else 0.0,
        "expanded_node_top1_share": share_top(1),
        "expanded_node_top10_share": share_top(10),
        "expanded_node_top100_share": share_top(100),
        "expanded_node_top1000_share": share_top(1000),
        "expanded_node_top10000_share": share_top(10000),
    }


def parse_iostat_log(iostat_log):
    """解析 iostat log：選出最忙碌裝置，計算各欄位的平均與最大值"""
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
                    # 新的一批 iostat 輸出，先記錄欄位名稱
                    current_header = line.split()
                    continue
                if current_header:
                    parts = line.split()
                    if len(parts) != len(current_header):
                        continue
                    device = parts[0]
                    # 將本行的數值依欄位名稱轉成 float，無法解析的設為 None
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
        # 若有多個裝置，挑讀量最高者（rkB/s，退而求其次 r/s）
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
    # 對所選裝置的每個欄位，計算 mean / gmean / var / std / iqr / cv / percentiles
    for col in sorted(columns):
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        vals_arr = np.array(vals, dtype=float)
        stats[f"iostat_{col}_mean"] = float(np.mean(vals_arr))
        stats[f"iostat_{col}_gmean"] = float(np.exp(np.mean(np.log(vals_arr + 1e-10))))  # 幾何平均數
        stats[f"iostat_{col}_var"] = float(np.var(vals_arr))
        stats[f"iostat_{col}_std"] = float(np.std(vals_arr))
        stats[f"iostat_{col}_iqr"] = float(np.quantile(vals_arr, 0.75) - np.quantile(vals_arr, 0.25))
        mean_val = stats[f"iostat_{col}_mean"]
        stats[f"iostat_{col}_cv"] = float(stats[f"iostat_{col}_std"] / abs(mean_val)) if mean_val != 0 else 0.0
        percentiles = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
        for p in percentiles:
            q = float(np.quantile(vals_arr, p))
            key = "p999" if p == 0.999 else f"p{int(p * 100)}"
            stats[f"iostat_{col}_{key}"] = q
    return stats


def parse_topk_files(base_prefix, node_counts_csv):
    """
    解析 Top-K 相關輸出檔案，彙整單一 K 的圖統計與覆蓋率。

    參數：
      - base_prefix：某次 run 的前綴（不含尾端的 _summary_stats.csv）
      - node_counts_csv：節點被展開次數的統計檔（可選），用以計算覆蓋率

    返回：(topk_rows, summary)
    """
    topk_rows = []
    neighbors_files = sorted(glob.glob(f"{base_prefix}_topk*_neighbors.csv"))
    
    if not neighbors_files:
        return topk_rows, {}
    if len(neighbors_files) > 1:
        print(f"WARN: multiple topk neighbors files found; using first: {neighbors_files[0]}", file=sys.stderr)
    
    # 預先載入節點計數資料與 nodes 檔案映射
    node_counts_df = None
    counts_map = {}
    total_count = 0.0
    if node_counts_csv and os.path.isfile(node_counts_csv):
        try:
            node_counts_df = pd.read_csv(node_counts_csv)
            if not node_counts_df.empty and "node_id" in node_counts_df.columns and "count" in node_counts_df.columns:
                counts_map = dict(zip(node_counts_df["node_id"].astype(str), node_counts_df["count"]))
                total_count = float(node_counts_df["count"].sum())
        except Exception:
            pass
    
    neighbors_path = neighbors_files[0]
    m = re.search(r"_topk(\d+)_neighbors\.csv$", neighbors_path)
    if not m:
        return topk_rows, {}

    topk = int(m.group(1))
    row = {"topk_k": topk, "topk_neighbors_path": neighbors_path}

    try:
        df = pd.read_csv(neighbors_path)
        if df.empty:
            topk_rows.append(row)
            return topk_rows, {}

        required_cols = {"node_id", "neighbor_id", "degree"}
        if not required_cols.issubset(df.columns):
            topk_rows.append(row)
            return topk_rows, {}

        # 基本統計
        row["topk_expanded_neighbor_count"] = int(len(df))
        row["topk_expanded_unique_count"] = int(df["node_id"].nunique())
        row["topk_expanded_unique_neighbors_count"] = int(df["neighbor_id"].nunique())

        # 度數統計
        degree_per_node = df.groupby("node_id")["degree"].first()
        row["topk_expanded_degree_mean"] = float(degree_per_node.mean())

        for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
            row[f"topk_expanded_degree_p{p}"] = float(degree_per_node.quantile(p / 100.0))

    except Exception:
        topk_rows.append(row)
        return topk_rows, {}

    # 計算覆蓋率
    nodes_path = f"{base_prefix}_topk{topk}_nodes.txt"
    if os.path.isfile(nodes_path):
        row["topk_nodes_path"] = nodes_path
    if os.path.isfile(nodes_path) and counts_map and total_count > 0:
        try:
            with open(nodes_path, "r", encoding="utf-8") as f:
                topk_total = sum(counts_map.get(line.strip(), 0.0) for line in f if line.strip())
            row["topk_expanded_coverage_ratio"] = float(topk_total / total_count)
        except Exception:
            pass

    topk_rows.append(row)
    return topk_rows, {}


def collect_summary_stats(search_dir, output_file=None, verbose=False):
    """
    蒐集所有 summary_stats.csv 並彙總到一個檔案
    
    Args:
        search_dir: search 輸出目錄
    output_file: 彙總輸出檔案路徑 (為空則不寫檔)
        verbose: 是否顯示詳細資訊
    
    Returns:
        combined_df, topk_data
    """
    # 查找所有 summary_stats.csv 檔案
    summary_files = find_summary_files(search_dir)
    
    if not summary_files:
        print(f"警告: 在 {search_dir} 內找不到任何 *_summary_stats.csv 檔案", file=sys.stderr)
        return None, []
    
    if verbose:
        print(f"找到 {len(summary_files)} 個 summary_stats.csv 檔案")
    else:
        print(f"處理 {len(summary_files)} 個檔案...", end='', flush=True)
    
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
            iostat_stats = parse_iostat_log(iostat_log)

            extra_cols = {
                "run_prefix": os.path.basename(base_prefix),
                "summary_stats_path": summary_file,
                "expanded_nodes_path": expanded_csv if os.path.isfile(expanded_csv) else "",
                "node_counts_path": node_counts_csv if os.path.isfile(node_counts_csv) else "",
                "iostat_log_path": iostat_log if os.path.isfile(iostat_log) else "",
            }
            extra_cols.update(expanded_stats)
            extra_cols.update(iostat_stats)
            extra_cols.update(topk_summary)

            # 添加 id 列（在最前面）
            ids = list(range(row_id, row_id + len(df)))
            df.insert(0, "id", ids)
            row_id += len(df)

            if extra_cols:
                extra_df = pd.DataFrame({key: [value] * len(df) for key, value in extra_cols.items()})
                df = pd.concat([df, extra_df], axis=1)
            
            all_data.append(df)
            index_name = extract_index_info(summary_file)
            if verbose:
                print(f"  ✓ 已讀取: {summary_file} (index: {index_name}, 行數: {len(df)})")

            for row in topk_rows:
                row["run_prefix"] = os.path.basename(base_prefix)
                row["summary_stats_path"] = summary_file
                topk_data.append(row)
            
        except Exception as e:
            print(f"  ✗ 讀取失敗: {summary_file} - {e}", file=sys.stderr)
            continue
    
    if not verbose:
        print(" 完成")
    
    if not all_data:
        print("錯誤: 沒有成功讀取任何檔案", file=sys.stderr)
        return None, []
    
    # 合併所有資料
    if verbose:
        print(f"\n正在合併 {len(all_data)} 個資料框...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print(f"合併後總列數: {len(combined_df)}")
        print(f"列名: {list(combined_df.columns)}")
    
    # 儲存到輸出檔案
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"✓ 彙總完成: {output_file} ({len(combined_df)} 行, {len(combined_df.columns)} 列)")
    
    return combined_df, topk_data


def main():
    parser = argparse.ArgumentParser(
        description="蒐集 search 結果中的 summary_stats.csv 並彙總到單一檔案"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="彙總結果輸出檔案 (預設: ./outputFiles/analyze/{search_tag}/collected_stats_{timestamp}.csv)"
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
            f"./outputFiles/analyze/{search_tag}/collected_all_{search_tag}_{timestamp}.csv"
        )
    else:
        output_file = os.path.abspath(args.output)
    
    # 檢查輸入目錄
    if not os.path.isdir(search_dir):
        print(f"錯誤: search 目錄不存在或不是目錄: {search_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 建立輸出目錄
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"搜尋目錄: {search_dir}")
        print(f"輸出檔案: {output_file}")
        print("-" * 60)
    
    # 執行蒐集
    summary_df, topk_data = collect_summary_stats(search_dir, output_file=None, verbose=args.verbose)
    if summary_df is None or summary_df.empty:
        sys.exit(1)

    final_df = summary_df
    ordered_cols = list(summary_df.columns)
    if topk_data:
        topk_df = pd.DataFrame(topk_data)
        topk_extra_cols_base = [
            "topk_k",
            "topk_neighbors_path",
            "topk_nodes_path",
            "topk_expanded_neighbor_count",
            "topk_expanded_unique_count",
            "topk_expanded_unique_neighbors_count",
            "topk_expanded_degree_mean",
            "topk_expanded_coverage_ratio",
        ]
        topk_extra_cols = [c for c in topk_extra_cols_base if c in topk_df.columns]
        topk_extra_cols += [c for c in topk_df.columns if c.startswith("topk_expanded_degree_p")]
        ordered_cols.extend([c for c in topk_extra_cols if c not in ordered_cols])
        # 按 run_prefix 和 summary_stats_path left join（一個 summary 對應多個 topk_k）
        final_df = summary_df.merge(
            topk_df[["run_prefix", "summary_stats_path"] + topk_extra_cols],
            on=["run_prefix", "summary_stats_path"],
            how="left"
        )

    # 移除不需要的路徑欄位
    drop_cols = [
        "summary_stats_path",
        "expanded_nodes_path",
        "node_counts_path",
        "iostat_log_path",
        "topk_neighbors_files",
        "topk_nodes_files",
        "topk_neighbors_path",
        "topk_nodes_path",
    ]
    final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])
    ordered_cols = [c for c in ordered_cols if c in final_df.columns and c not in drop_cols]
    remaining_cols = [c for c in final_df.columns if c not in ordered_cols]
    final_df = final_df[ordered_cols + remaining_cols]

    # 僅在同一 iostat 欄位族群全為 0 時才移除
    iostat_cols = [c for c in final_df.columns if c.startswith("iostat_")]
    iostat_groups = {}
    for col in iostat_cols:
        if not pd.api.types.is_numeric_dtype(final_df[col]):
            continue
        parts = col.split("_", 2)
        if len(parts) < 3:
            continue
        base = parts[1]
        iostat_groups.setdefault(base, []).append(col)

    drop_iostat_cols = []
    for base, cols in iostat_groups.items():
        all_zero = True
        for col in cols:
            series = pd.to_numeric(final_df[col], errors="coerce").fillna(0.0)
            if series.abs().max() > 1e-9:
                all_zero = False
                break
        if all_zero:
            drop_iostat_cols.extend(cols)

    if drop_iostat_cols:
        final_df = final_df.drop(columns=drop_iostat_cols)

    final_df.to_csv(output_file, index=False)
    print(f"✓ 完整彙總：{output_file} ({len(final_df)} 行, {len(final_df.columns)} 列)")
    
    # 顯示統計資訊
    if args.verbose:
        print("\n" + "=" * 60)
        print("統計資訊:")
        print("=" * 60)
        try:
            print(f"總列數: {len(final_df)}")
            print(f"ID 範圍: {final_df['id'].min()} - {final_df['id'].max()}")
            print(f"列名: {list(final_df.columns)[:15]}...")
            print("\n前 10 行:")
            print(final_df.head(10).to_string())
        except Exception as e:
            print(f"無法讀取輸出檔案: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
