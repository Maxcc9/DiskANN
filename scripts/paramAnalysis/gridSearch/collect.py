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
import csv
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
    # 路徑格式: ./outputFiles/search/{index_name}/{run_dir}/{result_file}
    parts = Path(file_path).parts
    if len(parts) >= 4:
        return parts[-3]  # 返回 index 資料夾名稱
    if len(parts) >= 3:
        return parts[-2]
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


def _compute_numeric_stats(prefix, values):
    vals_arr = np.array(values, dtype=float)
    stats = {}
    stats[f"{prefix}_mean"] = float(np.mean(vals_arr))
    stats[f"{prefix}_gmean"] = float(np.exp(np.mean(np.log(vals_arr + 1e-10))))
    stats[f"{prefix}_var"] = float(np.var(vals_arr))
    stats[f"{prefix}_std"] = float(np.std(vals_arr))
    stats[f"{prefix}_iqr"] = float(np.quantile(vals_arr, 0.75) - np.quantile(vals_arr, 0.25))
    mean_val = stats[f"{prefix}_mean"]
    stats[f"{prefix}_cv"] = float(stats[f"{prefix}_std"] / abs(mean_val)) if mean_val != 0 else 0.0
    percentiles = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
    for p in percentiles:
        q = float(np.quantile(vals_arr, p))
        key = "p999" if p == 0.999 else f"p{int(p * 100)}"
        stats[f"{prefix}_{key}"] = q
    return stats


def parse_pidstat_log(pidstat_log):
    """解析 pidstat log：彙總 per-thread CPU/IO/等待等統計"""
    if not os.path.isfile(pidstat_log):
        return {}
    columns = {}
    tid_set = set()
    header = None
    try:
        with open(pidstat_log, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    header = None
                    continue
                if line.startswith("Average:"):
                    continue
                parts = line.split()
                if "UID" in parts and ("PID" in parts or "TGID" in parts) and "TID" in parts:
                    header = parts
                    continue
                if not header:
                    continue
                if len(parts) < len(header):
                    continue
                if len(parts) > len(header):
                    parts = parts[-len(header):]
                row = dict(zip(header, parts))
                tid = row.get("TID") or row.get("tid")
                if tid:
                    tid_set.add(tid)
                for key, val in row.items():
                    if key in ("UID", "PID", "TGID", "TID", "tid", "Command", "CPU", "%guest"):
                        continue
                    try:
                        columns.setdefault(key, []).append(float(val))
                    except ValueError:
                        continue
    except Exception:
        return {}

    if not columns:
        return {}
    stats = {
        "pidstat_thread_count": int(len(tid_set)) if tid_set else 0,
    }
    for col, vals in columns.items():
        if not vals:
            continue
        stats.update(_compute_numeric_stats(f"pidstat_{col}", vals))
    return stats


def parse_wa_log(wa_log):
    """解析 mpstat log：彙總 %iowait (CPU wa)"""
    if not os.path.isfile(wa_log):
        return {}
    header = None
    iowait_vals = []
    try:
        with open(wa_log, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    header = None
                    continue
                if line.startswith("Linux"):
                    continue
                parts = line.split()
                if "CPU" in parts and "%iowait" in parts:
                    header = parts
                    continue
                if not header:
                    continue
                if len(parts) < len(header):
                    continue
                if len(parts) > len(header):
                    parts = parts[-len(header):]
                row = dict(zip(header, parts))
                cpu = row.get("CPU")
                if cpu != "all":
                    continue
                val = row.get("%iowait")
                if val is None:
                    continue
                try:
                    iowait_vals.append(float(val))
                except ValueError:
                    continue
    except Exception:
        return {}
    if not iowait_vals:
        return {}
    return _compute_numeric_stats("wa_%iowait", iowait_vals)


def parse_thread_timeline(thread_timeline_csv):
    """解析 thread_timeline.csv：彙總每個 query 的執行時間與 thread 數量"""
    if not os.path.isfile(thread_timeline_csv):
        return {}
    try:
        df = pd.read_csv(thread_timeline_csv)
    except Exception:
        return {}
    if df.empty:
        return {}
    if "duration_us" not in df.columns:
        return {}
    stats = {}
    duration_vals = pd.to_numeric(df["duration_us"], errors="coerce").dropna().values
    if duration_vals.size > 0:
        stats.update(_compute_numeric_stats("thread_timeline_duration_us", duration_vals))
    if "os_tid" in df.columns:
        stats["thread_timeline_os_tid_unique"] = int(pd.Series(df["os_tid"]).nunique())
    if "thread_id" in df.columns:
        stats["thread_timeline_thread_id_unique"] = int(pd.Series(df["thread_id"]).nunique())
    return stats


def parse_read_trace(read_trace_csv, window_ms_list):
    """解析 read_trace.csv：統計時間窗內重複讀取"""
    if not os.path.isfile(read_trace_csv):
        return {}
    events_by_node = {}
    total_reads = 0
    cache_hits = 0
    disk_reads = 0
    try:
        with open(read_trace_csv, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}
            required = {"ts_ns", "node_id", "os_tid", "is_cache_hit"}
            if not required.issubset(set(reader.fieldnames)):
                return {}
            for row in reader:
                try:
                    ts = int(row["ts_ns"])
                    node_id = int(row["node_id"])
                    tid = int(row["os_tid"])
                    is_cache_hit = int(row["is_cache_hit"]) == 1
                except (KeyError, ValueError):
                    continue
                events_by_node.setdefault(node_id, []).append((ts, tid, is_cache_hit))
                total_reads += 1
                if is_cache_hit:
                    cache_hits += 1
                else:
                    disk_reads += 1
    except Exception:
        return {}

    if not events_by_node:
        return {}

    def calc_window_stats(events, window_ns):
        repeat_reads = 0
        repeat_multi_thread = 0
        max_window_size = 0
        max_unique_threads = 0
        per_node_max_unique = []
        per_node_max_unique_map = {}
        per_node_repeat_mt = {}
        per_node_total = {}
        per_node_unique_threads = {}

        for node_id, node_events in events.items():
            node_events.sort(key=lambda x: x[0])
            start = 0
            thread_counts = {}
            node_max_unique = 0
            node_repeat_mt = 0
            thread_set = set()
            for i, (ts, tid) in enumerate(node_events):
                while start < i and (ts - node_events[start][0] > window_ns):
                    old_tid = node_events[start][1]
                    count = thread_counts.get(old_tid, 0)
                    if count <= 1:
                        thread_counts.pop(old_tid, None)
                    else:
                        thread_counts[old_tid] = count - 1
                    start += 1
                if i > start:
                    repeat_reads += 1
                    if len(thread_counts) > 0 and not (len(thread_counts) == 1 and tid in thread_counts):
                        repeat_multi_thread += 1
                        node_repeat_mt += 1
                thread_counts[tid] = thread_counts.get(tid, 0) + 1
                thread_set.add(tid)
                window_size = i - start + 1
                if window_size > max_window_size:
                    max_window_size = window_size
                if len(thread_counts) > max_unique_threads:
                    max_unique_threads = len(thread_counts)
                if len(thread_counts) > node_max_unique:
                    node_max_unique = len(thread_counts)
            per_node_max_unique.append(node_max_unique)
            per_node_max_unique_map[node_id] = node_max_unique
            per_node_repeat_mt[node_id] = node_repeat_mt
            per_node_total[node_id] = len(node_events)
            per_node_unique_threads[node_id] = len(thread_set)
        return {
            "repeat_reads": repeat_reads,
            "repeat_multi_thread": repeat_multi_thread,
            "max_window_size": max_window_size,
            "max_unique_threads": max_unique_threads,
            "per_node_max_unique": per_node_max_unique,
            "per_node_max_unique_map": per_node_max_unique_map,
            "per_node_repeat_mt": per_node_repeat_mt,
            "per_node_total": per_node_total,
            "per_node_unique_threads": per_node_unique_threads,
        }

    stats = {
        "read_trace_total_reads": int(total_reads),
        "read_trace_unique_nodes": int(len(events_by_node)),
        "read_trace_cache_hits": int(cache_hits),
        "read_trace_disk_reads": int(disk_reads),
        "read_trace_cache_hit_ratio": float(cache_hits / total_reads) if total_reads else 0.0,
        "read_trace_disk_read_ratio": float(disk_reads / total_reads) if total_reads else 0.0,
    }

    window_ms_list = [int(v) for v in window_ms_list if str(v).strip()]
    if not window_ms_list:
        window_ms_list = [50]
    stats["read_trace_window_ms_list"] = ",".join(str(v) for v in window_ms_list)

    topk = int(os.environ.get("READ_TRACE_TOPK", "100"))
    hot_window_ms = window_ms_list[0]
    hot_window_ns = hot_window_ms * 1_000_000
    hot_stats = None

    for window_ms in window_ms_list:
        window_ns = int(window_ms) * 1_000_000
        window_stats = calc_window_stats(
            {k: [(ts, tid) for ts, tid, _is_cache_hit in v] for k, v in events_by_node.items()},
            window_ns,
        )
        stats[f"read_trace_window_ms_{window_ms}"] = int(window_ms)
        stats[f"read_trace_repeat_reads_ms{window_ms}"] = int(window_stats["repeat_reads"])
        stats[f"read_trace_repeat_ratio_ms{window_ms}"] = (
            float(window_stats["repeat_reads"] / total_reads) if total_reads else 0.0
        )
        stats[f"read_trace_repeat_multi_thread_reads_ms{window_ms}"] = int(window_stats["repeat_multi_thread"])
        stats[f"read_trace_repeat_multi_thread_ratio_ms{window_ms}"] = (
            float(window_stats["repeat_multi_thread"] / total_reads) if total_reads else 0.0
        )
        stats[f"read_trace_max_window_size_ms{window_ms}"] = int(window_stats["max_window_size"])
        stats[f"read_trace_max_unique_threads_ms{window_ms}"] = int(window_stats["max_unique_threads"])
        if window_stats["per_node_max_unique"]:
            stats[f"read_trace_max_unique_threads_mean_ms{window_ms}"] = float(
                np.mean(window_stats["per_node_max_unique"])
            )
            stats[f"read_trace_max_unique_threads_p95_ms{window_ms}"] = float(
                np.quantile(window_stats["per_node_max_unique"], 0.95)
            )
        if window_ms == hot_window_ms:
            hot_stats = window_stats

        disk_events = {}
        for node_id, node_events in events_by_node.items():
            filtered = [(ts, tid) for ts, tid, is_cache_hit in node_events if not is_cache_hit]
            if filtered:
                disk_events[node_id] = filtered
        disk_total_reads = sum(len(v) for v in disk_events.values())
        disk_stats = calc_window_stats(disk_events, window_ns)
        stats[f"read_trace_repeat_reads_disk_ms{window_ms}"] = int(disk_stats["repeat_reads"])
        stats[f"read_trace_repeat_ratio_disk_ms{window_ms}"] = (
            float(disk_stats["repeat_reads"] / disk_total_reads) if disk_total_reads else 0.0
        )
        stats[f"read_trace_repeat_multi_thread_reads_disk_ms{window_ms}"] = int(disk_stats["repeat_multi_thread"])
        stats[f"read_trace_repeat_multi_thread_ratio_disk_ms{window_ms}"] = (
            float(disk_stats["repeat_multi_thread"] / disk_total_reads) if disk_total_reads else 0.0
        )

    if hot_stats:
        base_prefix = read_trace_csv[: -len("_read_trace.csv")]
        hot_rows = []
        for node_id, total in hot_stats["per_node_total"].items():
            node_events = events_by_node.get(node_id, [])
            node_cache_hits = sum(1 for _ts, _tid, is_cache_hit in node_events if is_cache_hit)
            node_disk_reads = len(node_events) - node_cache_hits
            hot_rows.append(
                {
                    "node_id": int(node_id),
                    "total_reads": int(total),
                    "disk_reads": int(node_disk_reads),
                    "cache_hits": int(node_cache_hits),
                    "repeat_multi_thread_reads": int(hot_stats["per_node_repeat_mt"].get(node_id, 0)),
                    "unique_threads": int(hot_stats["per_node_unique_threads"].get(node_id, 0)),
                    "max_unique_threads_window": int(hot_stats["per_node_max_unique_map"].get(node_id, 0)),
                }
            )
        hot_rows.sort(
            key=lambda r: (r["repeat_multi_thread_reads"], r["total_reads"], r["unique_threads"]), reverse=True
        )
        if hot_rows:
            hot_path = f"{base_prefix}_read_trace_hot_nodes_{hot_window_ms}ms_top{topk}.csv"
            with open(hot_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(hot_rows[0].keys()))
                writer.writeheader()
                writer.writerows(hot_rows[:topk])
            topk_total_reads = sum(r["total_reads"] for r in hot_rows[:topk])
            topk_repeat_mt = sum(r["repeat_multi_thread_reads"] for r in hot_rows[:topk])
            stats["read_trace_hot_nodes_topk"] = int(topk)
            stats["read_trace_hot_nodes_read_share"] = float(topk_total_reads / total_reads) if total_reads else 0.0
            stats["read_trace_hot_nodes_repeat_mt_share"] = (
                float(topk_repeat_mt / total_reads) if total_reads else 0.0
            )

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
    
    window_env = os.environ.get("READ_TRACE_WINDOWS_MS", os.environ.get("READ_TRACE_WINDOW_MS", "50"))
    read_trace_window_ms_list = [v.strip() for v in window_env.split(",") if v.strip()]
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
            pidstat_log = f"{base_prefix}_pidstat.log"
            wa_log = f"{base_prefix}_wa.log"
            thread_timeline_csv = f"{base_prefix}_thread_timeline.csv"
            read_trace_csv = f"{base_prefix}_read_trace.csv"
            topk_rows, topk_summary = parse_topk_files(base_prefix, node_counts_csv)

            expanded_stats = parse_expanded_stats(expanded_csv)
            iostat_stats = parse_iostat_log(iostat_log)
            pidstat_stats = parse_pidstat_log(pidstat_log)
            wa_stats = parse_wa_log(wa_log)
            thread_timeline_stats = parse_thread_timeline(thread_timeline_csv)
            read_trace_stats = parse_read_trace(read_trace_csv, window_ms_list=read_trace_window_ms_list)

            extra_cols = {
                "run_prefix": os.path.basename(base_prefix),
                "summary_stats_path": summary_file,
                "expanded_nodes_path": expanded_csv if os.path.isfile(expanded_csv) else "",
                "node_counts_path": node_counts_csv if os.path.isfile(node_counts_csv) else "",
                "iostat_log_path": iostat_log if os.path.isfile(iostat_log) else "",
                "pidstat_log_path": pidstat_log if os.path.isfile(pidstat_log) else "",
                "wa_log_path": wa_log if os.path.isfile(wa_log) else "",
                "thread_timeline_path": thread_timeline_csv if os.path.isfile(thread_timeline_csv) else "",
                "read_trace_path": read_trace_csv if os.path.isfile(read_trace_csv) else "",
            }
            extra_cols.update(expanded_stats)
            extra_cols.update(iostat_stats)
            extra_cols.update(pidstat_stats)
            extra_cols.update(wa_stats)
            extra_cols.update(thread_timeline_stats)
            extra_cols.update(read_trace_stats)
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
        "pidstat_log_path",
        "wa_log_path",
        "thread_timeline_path",
        "read_trace_path",
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
