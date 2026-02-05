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
import time
import multiprocessing as mp
import functools
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

_CURRENT_SUMMARY_FILE = ""
_GMEAN_WARNED = set()
DEFAULT_MEDIAN_KEY_COLS = [
    "build_R",
    "build_L",
    "search_K",
    "search_L",
    "search_W",
    "search_T",
    "actual_cached_nodes",
]


def _safe_remove_file(path):
    """Best-effort remove; ignore missing/permission/transient errors."""
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def find_summary_files(search_dir):
    """遞迴查找所有 *_summary_stats.csv 檔案"""
    pattern = os.path.join(search_dir, "**", "*_summary_stats.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def _parse_float_or_none(val):
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        out = float(s)
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


def _select_median_summary_files(summary_files, key_cols, qps_col="qps", verbose=False):
    """
    只讀 summary_stats.csv 的首列，依 key_cols 分組後挑 qps 中位數對應的 summary_file。
    偶數筆時取 upper middle（與 median_dedupe.py 一致）。
    """
    groups = {}
    scanned = 0
    for summary_file in summary_files:
        try:
            with open(summary_file, newline="") as f:
                reader = csv.DictReader(f)
                first_row = next(reader, None)
                if first_row is None:
                    continue
                key = tuple(first_row.get(c, "") for c in key_cols)
                qps = _parse_float_or_none(first_row.get(qps_col))
                groups.setdefault(key, []).append((qps, summary_file))
                scanned += 1
        except Exception:
            continue

    selected = []
    for _key, items in groups.items():
        valid = [(q, s) for q, s in items if q is not None]
        if not valid:
            selected.append(items[0][1])
            continue
        valid.sort(key=lambda x: (x[0], x[1]))
        mid = len(valid) // 2
        selected.append(valid[mid][1])

    selected = sorted(set(selected))
    if verbose:
        print(f"Median-first: 掃描 {scanned} 個 summary，分組 {len(groups)}，保留 {len(selected)} 個 run")
    return selected


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
    bad_mask = (~np.isfinite(vals_arr)) | (vals_arr <= -1e-10)
    if np.any(bad_mask):
        key = (_CURRENT_SUMMARY_FILE, prefix)
        if key not in _GMEAN_WARNED:
            _GMEAN_WARNED.add(key)
            bad_count = int(np.sum(bad_mask))
            sample = vals_arr[bad_mask][:5]
            src = _CURRENT_SUMMARY_FILE if _CURRENT_SUMMARY_FILE else "<unknown summary>"
            print(
                f"[WARN] invalid gmean input: file={src}, prefix={prefix}, "
                f"bad_count={bad_count}, sample={sample.tolist()}",
                file=sys.stderr,
            )
    with np.errstate(invalid="ignore", divide="ignore"):
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
                        fval = float(val)
                        # pidstat 在程序結束尾端可能用 -1 表示無效 I/O 速率，避免污染統計
                        if key in ("kB_rd/s", "kB_wr/s", "kB_ccwr/s") and fval < 0:
                            continue
                        columns.setdefault(key, []).append(fval)
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


def _parse_window_ms_list(window_ms_list):
    parsed = []
    for v in window_ms_list:
        if str(v).strip():
            try:
                value = float(v)
                if value > 0:
                    parsed.append(value)
            except ValueError:
                continue
    if not parsed:
        parsed = [0.5]
    return parsed


def _window_label(window_ms):
    label = f"{window_ms:g}"
    return label.replace(".", "p")


def parse_read_trace(read_trace_csv, window_ms_list, emit_intermediate=True):
    """解析 read_trace.csv：統計時間窗內重複讀取"""
    if not os.path.isfile(read_trace_csv):
        return {}, None, {
            "window_stats_paths": {},
            "window_stats_frames": {},
            "node_stats_paths": [],
            "hot_nodes_path": "",
        }
    events = []
    node_cache_hits = {}
    node_disk_reads = {}
    total_reads = 0
    cache_hits = 0
    disk_reads = 0
    try:
        with open(read_trace_csv, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}, None, {
                    "window_stats_paths": {},
                    "window_stats_frames": {},
                    "node_stats_paths": [],
                    "hot_nodes_path": "",
                }
            required = {"ts_ns", "node_id", "os_tid", "is_cache_hit"}
            if not required.issubset(set(reader.fieldnames)):
                # 支援舊格式（包含 omp_tid, read_bytes）與新格式
                if not {"ts_ns", "node_id", "is_cache_hit"}.issubset(set(reader.fieldnames)):
                    return {}, None, {
                        "window_stats_paths": {},
                        "window_stats_frames": {},
                        "node_stats_paths": [],
                        "hot_nodes_path": "",
                    }
            for row in reader:
                try:
                    ts = int(row["ts_ns"])
                    node_id = int(row["node_id"])
                    tid = int(row["os_tid"]) if "os_tid" in row and row["os_tid"] else -1
                    is_cache_hit = int(row["is_cache_hit"]) == 1
                except (KeyError, ValueError):
                    continue
                events.append((ts, node_id, tid, is_cache_hit))
                total_reads += 1
                if is_cache_hit:
                    cache_hits += 1
                    node_cache_hits[node_id] = node_cache_hits.get(node_id, 0) + 1
                else:
                    disk_reads += 1
                    node_disk_reads[node_id] = node_disk_reads.get(node_id, 0) + 1
    except Exception:
        return {}, None, {
            "window_stats_paths": {},
            "window_stats_frames": {},
            "node_stats_paths": [],
            "hot_nodes_path": "",
        }

    if not events:
        return {}, None, {
            "window_stats_paths": {},
            "window_stats_frames": {},
            "node_stats_paths": [],
            "hot_nodes_path": "",
        }

    stats = {
        "read_trace_total_reads": int(total_reads),
        "read_trace_unique_nodes": int(len(set(list(node_cache_hits.keys()) + list(node_disk_reads.keys())))),
        "read_trace_cache_hits": int(cache_hits),
        "read_trace_disk_reads": int(disk_reads),
        "read_trace_cache_hit_ratio": float(cache_hits / total_reads) if total_reads else 0.0,
        "read_trace_disk_read_ratio": float(disk_reads / total_reads) if total_reads else 0.0,
    }

    window_ms_list = _parse_window_ms_list(window_ms_list)
    stats["read_trace_window_ms_list"] = ",".join(f"{v:g}" for v in window_ms_list)
    write_node_stats = emit_intermediate and (os.environ.get("READ_TRACE_NODE_STATS", "1") == "1")
    write_window_stats = emit_intermediate and (os.environ.get("READ_TRACE_WINDOW_STATS", "1") == "1")
    write_hot_nodes = emit_intermediate
    artifacts = {
        "window_stats_paths": {},
        "window_stats_frames": {},
        "node_stats_paths": [],
        "hot_nodes_path": "",
    }

    topk = int(os.environ.get("READ_TRACE_TOPK", "100"))
    hot_window_ms = window_ms_list[0]
    hot_stats = None

    events.sort(key=lambda x: x[0])
    t0 = events[0][0]

    # 低記憶體優先：每個 window 逐一處理，避免同時維護多個 window 的大型聚合結構
    for window_ms in window_ms_list:
        window_label = _window_label(window_ms)
        window_ns = int(window_ms * 1_000_000)
        windows = {}
        for ts, node_id, tid, is_cache_hit in events:
            win = int((ts - t0) // window_ns)
            w = windows.setdefault(
                win,
                {
                    "total": 0,
                    "disk_total": 0,
                    "node_counts": {},
                    "node_threads": {},
                    "node_thread_counts": {},
                    "disk_node_counts": {},
                    "disk_node_threads": {},
                },
            )
            w["total"] += 1
            w["node_counts"][node_id] = w["node_counts"].get(node_id, 0) + 1
            w["node_thread_counts"][(node_id, tid)] = w["node_thread_counts"].get((node_id, tid), 0) + 1
            w["node_threads"].setdefault(node_id, set()).add(tid)
            if not is_cache_hit:
                w["disk_total"] += 1
                w["disk_node_counts"][node_id] = w["disk_node_counts"].get(node_id, 0) + 1
                w["disk_node_threads"].setdefault(node_id, set()).add(tid)

        window_repeat_reads = []
        window_repeat_multi_thread_reads = []
        window_max_node_reads = []
        window_max_node_reads_ratio = []
        window_max_same_thread_reads = []
        window_max_same_thread_reads_ratio = []
        window_max_multi_thread_reads = []
        window_max_multi_thread_reads_ratio = []
        window_max_unique_threads = []
        window_node_reads = []
        window_node_read_ratios = []
        window_node_threads = []
        window_node_thread_ratios = []
        window_disk_repeat_reads = []
        window_disk_repeat_multi_thread_reads = []
        window_stats_rows = []

        per_node_total = {}
        per_node_repeat_mt = {}
        per_node_unique_threads = {}
        for win, w in windows.items():
            total = w["total"]
            if total == 0:
                continue
            node_counts = w["node_counts"]
            node_threads = w["node_threads"]
            node_thread_counts = w["node_thread_counts"]

            repeat_reads = sum(c - 1 for c in node_counts.values() if c > 1)
            repeat_multi_thread = sum(
                c for nid, c in node_counts.items()
                if c > 1 and len(node_threads.get(nid, set())) >= 2
            )
            max_node_reads = max(node_counts.values()) if node_counts else 0
            max_same_thread = max(node_thread_counts.values()) if node_thread_counts else 0
            max_multi_thread = max(
                (c for nid, c in node_counts.items() if len(node_threads.get(nid, set())) >= 2),
                default=0,
            )
            max_unique_threads = max((len(tset) for tset in node_threads.values()), default=0)
            window_total_threads = len({t for tset in node_threads.values() for t in tset})

            window_repeat_reads.append(repeat_reads)
            window_repeat_multi_thread_reads.append(repeat_multi_thread)
            window_max_node_reads.append(max_node_reads)
            window_max_node_reads_ratio.append(max_node_reads / total)
            window_max_same_thread_reads.append(max_same_thread)
            window_max_same_thread_reads_ratio.append(max_same_thread / total)
            window_max_multi_thread_reads.append(max_multi_thread)
            window_max_multi_thread_reads_ratio.append(max_multi_thread / total)
            window_max_unique_threads.append(max_unique_threads)
            window_node_reads.extend(list(node_counts.values()))
            window_node_read_ratios.extend([c / total for c in node_counts.values()])
            window_node_threads.extend([len(tset) for tset in node_threads.values()])
            if window_total_threads > 0:
                window_node_thread_ratios.extend([len(tset) / window_total_threads for tset in node_threads.values()])

            disk_counts = w["disk_node_counts"]
            disk_threads = w["disk_node_threads"]
            disk_repeat_reads = sum(c - 1 for c in disk_counts.values() if c > 1)
            disk_repeat_mt = sum(
                c for nid, c in disk_counts.items()
                if c > 1 and len(disk_threads.get(nid, set())) >= 2
            )
            window_disk_repeat_reads.append(disk_repeat_reads)
            window_disk_repeat_multi_thread_reads.append(disk_repeat_mt)

            for nid, c in node_counts.items():
                per_node_total[nid] = per_node_total.get(nid, 0) + c
                if len(node_threads.get(nid, set())) >= 2 and c > 1:
                    per_node_repeat_mt[nid] = per_node_repeat_mt.get(nid, 0) + c
                per_node_unique_threads.setdefault(nid, set()).update(node_threads.get(nid, set()))

            window_stats_rows.append(
                {
                    "window_id": int(win),
                    "window_start_ns": int(t0 + win * window_ns),
                    "window_end_ns": int(t0 + (win + 1) * window_ns),
                    "total_reads": int(total),
                    "repeat_reads": int(repeat_reads),
                    "repeat_multi_thread_reads": int(repeat_multi_thread),
                    "max_node_reads": int(max_node_reads),
                    "max_node_reads_ratio": float(max_node_reads / total),
                    "max_same_thread_reads": int(max_same_thread),
                    "max_same_thread_reads_ratio": float(max_same_thread / total),
                    "max_multi_thread_reads": int(max_multi_thread),
                    "max_multi_thread_reads_ratio": float(max_multi_thread / total),
                    "max_unique_threads": int(max_unique_threads),
                }
            )

        if window_stats_rows:
            window_df = pd.DataFrame(window_stats_rows).sort_values("window_id")
            artifacts["window_stats_frames"][window_label] = window_df
            if write_window_stats:
                base_prefix = read_trace_csv[: -len("_read_trace.csv")]
                window_stats_path = f"{base_prefix}_read_trace_window_{window_label}ms_stats.csv"
                window_df.to_csv(window_stats_path, index=False)
                artifacts["window_stats_paths"][window_label] = window_stats_path

        stats[f"read_trace_repeat_reads_ms{window_label}"] = int(sum(window_repeat_reads))
        stats[f"read_trace_repeat_ratio_ms{window_label}"] = (
            float(sum(window_repeat_reads) / total_reads) if total_reads else 0.0
        )
        stats[f"read_trace_repeat_multi_thread_reads_ms{window_label}"] = int(sum(window_repeat_multi_thread_reads))
        stats[f"read_trace_repeat_multi_thread_ratio_ms{window_label}"] = (
            float(sum(window_repeat_multi_thread_reads) / total_reads) if total_reads else 0.0
        )
        if window_max_unique_threads:
            stats.update(_compute_numeric_stats(
                f"read_trace_max_unique_threads_ms{window_label}",
                window_max_unique_threads,
            ))
        if window_max_node_reads:
            stats.update(_compute_numeric_stats(
                f"read_trace_node_window_reads_ms{window_label}",
                window_max_node_reads,
            ))
        if window_max_node_reads_ratio:
            stats.update(_compute_numeric_stats(
                f"read_trace_node_window_reads_ratio_ms{window_label}",
                window_max_node_reads_ratio,
            ))
        if window_max_same_thread_reads:
            stats.update(_compute_numeric_stats(
                f"read_trace_node_same_thread_reads_ms{window_label}",
                window_max_same_thread_reads,
            ))
        if window_max_same_thread_reads_ratio:
            stats.update(_compute_numeric_stats(
                f"read_trace_node_same_thread_reads_ratio_ms{window_label}",
                window_max_same_thread_reads_ratio,
            ))
        if window_max_multi_thread_reads:
            stats.update(_compute_numeric_stats(
                f"read_trace_node_multi_thread_reads_ms{window_label}",
                window_max_multi_thread_reads,
            ))
        if window_max_multi_thread_reads_ratio:
            stats.update(_compute_numeric_stats(
                f"read_trace_node_multi_thread_reads_ratio_ms{window_label}",
                window_max_multi_thread_reads_ratio,
            ))
        if window_node_reads:
            stats.update(_compute_numeric_stats(
                f"read_trace_window_node_reads_ms{window_label}",
                window_node_reads,
            ))
        if window_node_read_ratios:
            stats.update(_compute_numeric_stats(
                f"read_trace_window_node_read_ratio_ms{window_label}",
                window_node_read_ratios,
            ))
        if window_node_threads:
            stats.update(_compute_numeric_stats(
                f"read_trace_window_node_threads_ms{window_label}",
                window_node_threads,
            ))
        if window_node_thread_ratios:
            stats.update(_compute_numeric_stats(
                f"read_trace_window_node_thread_ratio_ms{window_label}",
                window_node_thread_ratios,
            ))

        stats[f"read_trace_repeat_reads_disk_ms{window_label}"] = int(sum(window_disk_repeat_reads))
        stats[f"read_trace_repeat_ratio_disk_ms{window_label}"] = (
            float(sum(window_disk_repeat_reads) / disk_reads) if disk_reads else 0.0
        )
        stats[f"read_trace_repeat_multi_thread_reads_disk_ms{window_label}"] = int(sum(window_disk_repeat_multi_thread_reads))
        stats[f"read_trace_repeat_multi_thread_ratio_disk_ms{window_label}"] = (
            float(sum(window_disk_repeat_multi_thread_reads) / disk_reads) if disk_reads else 0.0
        )

        if window_ms == hot_window_ms:
            hot_stats = {
                "per_node_total": per_node_total,
                "per_node_repeat_mt": per_node_repeat_mt,
                "per_node_unique_threads": {k: len(v) for k, v in per_node_unique_threads.items()},
            }

        if write_node_stats:
            base_prefix = read_trace_csv[: -len("_read_trace.csv")]
            node_stats_path = f"{base_prefix}_read_trace_window_{window_label}ms_node_stats.csv"
            rows = []
            for node_id, total in per_node_total.items():
                rows.append(
                    {
                        "node_id": int(node_id),
                        "total_reads": int(total),
                        "disk_reads": int(node_disk_reads.get(node_id, 0)),
                        "cache_hits": int(node_cache_hits.get(node_id, 0)),
                        "unique_threads": int(len(per_node_unique_threads.get(node_id, set()))),
                        "repeat_multi_thread_reads": int(per_node_repeat_mt.get(node_id, 0)),
                    }
                )
            if rows:
                with open(node_stats_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                artifacts["node_stats_paths"].append(node_stats_path)

    if hot_stats:
        base_prefix = read_trace_csv[: -len("_read_trace.csv")]
        hot_rows = []
        for node_id, total in hot_stats["per_node_total"].items():
            hot_rows.append(
                {
                    "node_id": int(node_id),
                    "total_reads": int(total),
                    "disk_reads": int(node_disk_reads.get(node_id, 0)),
                    "cache_hits": int(node_cache_hits.get(node_id, 0)),
                    "repeat_multi_thread_reads": int(hot_stats["per_node_repeat_mt"].get(node_id, 0)),
                    "unique_threads": int(hot_stats["per_node_unique_threads"].get(node_id, 0)),
                }
            )
        hot_rows.sort(
            key=lambda r: (r["repeat_multi_thread_reads"], r["total_reads"], r["unique_threads"]), reverse=True
        )
        if hot_rows and write_hot_nodes:
            hot_window_label = _window_label(hot_window_ms)
            hot_path = f"{base_prefix}_read_trace_hot_nodes_{hot_window_label}ms_top{topk}.csv"
            with open(hot_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(hot_rows[0].keys()))
                writer.writeheader()
                writer.writerows(hot_rows[:topk])
            artifacts["hot_nodes_path"] = hot_path
            topk_total_reads = sum(r["total_reads"] for r in hot_rows[:topk])
            topk_repeat_mt = sum(r["repeat_multi_thread_reads"] for r in hot_rows[:topk])
            stats["read_trace_hot_nodes_topk"] = int(topk)
            stats["read_trace_hot_nodes_read_share"] = float(topk_total_reads / total_reads) if total_reads else 0.0
            stats["read_trace_hot_nodes_repeat_mt_share"] = (
                float(topk_repeat_mt / total_reads) if total_reads else 0.0
            )

    return stats, t0, artifacts


def parse_thread_timeline_windows(thread_timeline_csv, window_ms_list, window_start_ns=None, emit_intermediate=True):
    """解析 thread_timeline.csv：彙總每個時間窗的 latency 分佈"""
    if not os.path.isfile(thread_timeline_csv):
        return {}, {}
    try:
        df = pd.read_csv(thread_timeline_csv)
    except Exception:
        return {}, {}
    if df.empty:
        return {}, {}

    for col in ("start_time_ns", "duration_us"):
        if col not in df.columns:
            return {}, {}

    df = df.copy()
    df["start_time_ns"] = pd.to_numeric(df["start_time_ns"], errors="coerce")
    df["duration_us"] = pd.to_numeric(df["duration_us"], errors="coerce")
    df = df.dropna(subset=["start_time_ns", "duration_us"])
    if df.empty:
        return {}, {}

    if window_start_ns is not None:
        min_start_ns = int(df["start_time_ns"].min())
        df["aligned_start_ns"] = df["start_time_ns"] - min_start_ns + int(window_start_ns)
    else:
        df["aligned_start_ns"] = df["start_time_ns"]

    window_ms_list = _parse_window_ms_list(window_ms_list)

    write_window_stats = emit_intermediate and (os.environ.get("THREAD_TIMELINE_WINDOW_STATS", "1") == "1")
    window_paths = {}
    window_frames = {}

    for window_ms in window_ms_list:
        window_label = _window_label(window_ms)
        window_ns = int(window_ms * 1_000_000)
        window_id = ((df["aligned_start_ns"] - int(window_start_ns or 0)) // window_ns).astype("int64")
        df_win = df.assign(window_id=window_id)
        grouped = df_win.groupby("window_id")["duration_us"]
        rows = []
        for win, durations in grouped:
            vals = durations.values
            if vals.size == 0:
                continue
            start_ns = int((window_start_ns or 0) + win * window_ns)
            end_ns = int(start_ns + window_ns)
            rows.append(
                {
                    "window_id": int(win),
                    "window_start_ns": start_ns,
                    "window_end_ns": end_ns,
                    "query_count": int(vals.size),
                    "latency_mean_us": float(np.mean(vals)),
                    "latency_p50_us": float(np.quantile(vals, 0.50)),
                    "latency_p95_us": float(np.quantile(vals, 0.95)),
                    "latency_p99_us": float(np.quantile(vals, 0.99)),
                    "latency_p100_us": float(np.quantile(vals, 1.0)),
                }
            )

        if rows:
            window_df = pd.DataFrame(rows).sort_values("window_id")
            window_frames[window_label] = window_df
            if write_window_stats:
                base_prefix = thread_timeline_csv[: -len("_thread_timeline.csv")]
                window_path = f"{base_prefix}_thread_timeline_window_{window_label}ms_latency.csv"
                window_df.to_csv(window_path, index=False)
                window_paths[window_label] = window_path

    return window_paths, window_frames


def compute_window_correlations(read_trace_window_df, latency_window_df, window_label):
    """計算 read_trace window stats 與 latency window stats 的相關性"""
    if read_trace_window_df is None or read_trace_window_df.empty:
        return {}
    if latency_window_df is None or latency_window_df.empty:
        return {}
    read_df = read_trace_window_df.copy()
    if read_df.empty or "window_id" not in read_df.columns:
        return {}

    read_df = read_df.copy()
    if "window_start_ns" not in read_df.columns or "window_end_ns" not in read_df.columns:
        return {}
    read_df["repeat_ratio"] = read_df["repeat_reads"] / read_df["total_reads"].replace(0, np.nan)
    read_df["repeat_multi_thread_ratio"] = read_df["repeat_multi_thread_reads"] / read_df["total_reads"].replace(0, np.nan)

    latency_df = latency_window_df.copy()
    if "window_id" not in latency_df.columns:
        return {}
    merged = pd.merge(read_df, latency_df, on="window_id", how="inner")
    if merged.empty:
        return {}

    def safe_corr(a, b):
        import warnings
        a_vals = pd.to_numeric(a, errors="coerce")
        b_vals = pd.to_numeric(b, errors="coerce")
        mask = a_vals.notna() & b_vals.notna()
        if mask.sum() < 3:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float(a_vals[mask].corr(b_vals[mask], method="pearson"))

    metrics = {
        "repeat_ratio": "repeat_ratio",
        "repeat_multi_thread_ratio": "repeat_multi_thread_ratio",
        "max_node_reads_ratio": "max_node_reads_ratio",
        "max_same_thread_reads_ratio": "max_same_thread_reads_ratio",
        "max_multi_thread_reads_ratio": "max_multi_thread_reads_ratio",
        "max_unique_threads": "max_unique_threads",
        "total_reads": "total_reads",
    }
    latency_metrics = [
        "latency_mean_us",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "latency_p100_us",
    ]

    stats = {}
    for metric_key, col in metrics.items():
        if col not in merged.columns:
            continue
        for latency_col in latency_metrics:
            if latency_col not in merged.columns:
                continue
            stat_key = f"read_trace_window_corr_{metric_key}_vs_{latency_col}_ms{window_label}"
            stats[stat_key] = safe_corr(merged[col], merged[latency_col])

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

        # degree 欄位現已可選（新格式不包含）
        required_cols = {"node_id", "neighbor_id"}
        if not required_cols.issubset(df.columns):
            topk_rows.append(row)
            return topk_rows, {}

        # 基本統計
        row["topk_expanded_neighbor_count"] = int(len(df))
        row["topk_expanded_unique_count"] = int(df["node_id"].nunique())
        row["topk_expanded_unique_neighbors_count"] = int(df["neighbor_id"].nunique())

        # 度數統計：計算每個節點出現的次數（即其度數）
        # 支援兩種格式：
        # 1. 新格式：無 degree 欄位，透過 groupby().size() 計算
        # 2. 舊格式：有 degree 欄位，透過 groupby().first() 讀取（結果相同）
        if "degree" in df.columns:
            degree_per_node = df.groupby("node_id")["degree"].first()
        else:
            degree_per_node = df.groupby("node_id").size()
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


def _process_one_summary_file(summary_file, read_trace_window_ms_list, cleanup=False, timing=False):
    """
    Module-level function to process a single summary_stats.csv file.
    This must be at module level (not nested) to be pickleable by multiprocessing.Pool.
    
    Args:
        summary_file: Path to _summary_stats.csv file
        read_trace_window_ms_list: List of window ms values for read trace analysis
    
    Returns:
        Dict with processing results
    """
    global _CURRENT_SUMMARY_FILE
    _CURRENT_SUMMARY_FILE = summary_file
    t0_total = time.perf_counter()
    timing_stats = {}
    try:
        t0 = time.perf_counter()
        df = pd.read_csv(summary_file)
        timing_stats["read_summary"] = time.perf_counter() - t0
        if df.empty:
            return {
                "summary_file": summary_file,
                "rows": [],
                "cols": list(df.columns),
                "extra_cols": [],
                "topk_rows": [],
                "timing": timing_stats,
            }
        base_prefix = summary_file[: -len("_summary_stats.csv")]
        index_tag = extract_index_info(summary_file)
        expanded_csv = f"{base_prefix}_expanded_nodes.csv"
        node_counts_csv = f"{base_prefix}_node_counts.csv"
        iostat_log = f"{base_prefix}_iostat.log"
        pidstat_log = f"{base_prefix}_pidstat.log"
        wa_log = f"{base_prefix}_wa.log"
        thread_timeline_csv = f"{base_prefix}_thread_timeline.csv"
        read_trace_csv = f"{base_prefix}_read_trace.csv"
        t0 = time.perf_counter()
        topk_rows, topk_summary = parse_topk_files(base_prefix, node_counts_csv)
        timing_stats["parse_topk"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        expanded_stats = parse_expanded_stats(expanded_csv)
        timing_stats["parse_expanded"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        iostat_stats = parse_iostat_log(iostat_log)
        timing_stats["parse_iostat"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pidstat_stats = parse_pidstat_log(pidstat_log)
        timing_stats["parse_pidstat"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        wa_stats = parse_wa_log(wa_log)
        timing_stats["parse_wa"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        thread_timeline_stats = parse_thread_timeline(thread_timeline_csv)
        timing_stats["parse_thread_timeline"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        read_trace_stats, read_trace_t0_ns, read_trace_artifacts = parse_read_trace(
            read_trace_csv,
            window_ms_list=read_trace_window_ms_list,
            emit_intermediate=(not cleanup),
        )
        timing_stats["parse_read_trace"] = time.perf_counter() - t0
        # node/hot 統計檔在 collect 階段後續不再使用，可立即刪除
        if cleanup:
            for p in read_trace_artifacts.get("node_stats_paths", []):
                _safe_remove_file(p)
            _safe_remove_file(read_trace_artifacts.get("hot_nodes_path", ""))

        extra_cols = {
            "run_prefix": os.path.basename(base_prefix),
            "index_tag": index_tag,
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
        t0 = time.perf_counter()
        if read_trace_t0_ns is not None and os.path.isfile(thread_timeline_csv):
            window_latency_paths, window_latency_frames = parse_thread_timeline_windows(
                thread_timeline_csv,
                window_ms_list=read_trace_window_ms_list,
                window_start_ns=read_trace_t0_ns,
                emit_intermediate=(not cleanup),
            )
            window_list_str = read_trace_stats.get(
                "read_trace_window_ms_list",
                ",".join(read_trace_window_ms_list),
            )
            extra_cols["thread_timeline_window_ms_list"] = window_list_str
            for window_label, latency_df in window_latency_frames.items():
                read_trace_window_df = read_trace_artifacts.get("window_stats_frames", {}).get(window_label)
                corr_stats = compute_window_correlations(read_trace_window_df, latency_df, window_label)
                extra_cols.update(corr_stats)
                if cleanup:
                    _safe_remove_file(read_trace_artifacts.get("window_stats_paths", {}).get(window_label, ""))
                    _safe_remove_file(window_latency_paths.get(window_label, ""))
        elif cleanup:
            # 沒做 correlation 的情況下，window stats 與 latency window 檔也可立即刪除
            for p in read_trace_artifacts.get("window_stats_paths", {}).values():
                _safe_remove_file(p)
            base_prefix = summary_file[: -len("_summary_stats.csv")]
            for window_ms in _parse_window_ms_list(read_trace_window_ms_list):
                window_label = _window_label(window_ms)
                latency_path = f"{base_prefix}_thread_timeline_window_{window_label}ms_latency.csv"
                _safe_remove_file(latency_path)

        if cleanup:
            # 保險：刪除尚未被逐窗清掉的 read_trace window stats
            for p in read_trace_artifacts.get("window_stats_paths", {}).values():
                _safe_remove_file(p)
        timing_stats["window_corr"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rows = df.to_dict(orient="records")
        extra_keys = list(extra_cols.keys())
        for row in rows:
            for key, value in extra_cols.items():
                row[key] = value

        for row in topk_rows:
            row["run_prefix"] = os.path.basename(base_prefix)
            row["index_tag"] = index_tag
            row["summary_stats_path"] = summary_file
        timing_stats["assemble_rows"] = time.perf_counter() - t0
        timing_stats["total"] = time.perf_counter() - t0_total

        return {
            "summary_file": summary_file,
            "rows": rows,
            "cols": list(df.columns),
            "extra_cols": extra_keys,
            "topk_rows": topk_rows,
            "timing": timing_stats if timing else {},
        }
    except Exception as e:
        return {
            "summary_file": summary_file,
            "error": str(e),
            "rows": [],
            "cols": [],
            "extra_cols": [],
            "topk_rows": [],
            "timing": timing_stats if timing else {},
        }
    finally:
        _CURRENT_SUMMARY_FILE = ""


def collect_summary_stats(
    search_dir,
    output_file=None,
    verbose=False,
    workers=None,
    cleanup=False,
    median_first=False,
    median_key_cols=None,
    median_qps_col="qps",
    timing=False,
):
    """
    蒐集所有 summary_stats.csv 並彙總到一個檔案
    
    Args:
        search_dir: search 輸出目錄
    output_file: 彙總輸出檔案路徑 (為空則不寫檔)
        verbose: 是否顯示詳細資訊
        workers: Number of worker processes for parallelization (None = auto-detect from COLLECT_WORKERS env var)
        cleanup: 若為 True，於每個 summary 檔處理完成後立即刪除 collect 產生的中間檔
        median_first: 若為 True，先按 key 分組挑選 qps 中位數 run 再進行完整分析
        median_key_cols: median_first 使用的 key 欄位
        median_qps_col: median_first 使用的 qps 欄位
    
    Returns:
        combined_df, topk_data
    """
    # 查找所有 summary_stats.csv 檔案
    summary_files = find_summary_files(search_dir)
    
    if not summary_files:
        print(f"警告: 在 {search_dir} 內找不到任何 *_summary_stats.csv 檔案", file=sys.stderr)
        return None, []

    if median_first:
        key_cols = median_key_cols if median_key_cols else list(DEFAULT_MEDIAN_KEY_COLS)
        summary_files = _select_median_summary_files(
            summary_files,
            key_cols=key_cols,
            qps_col=median_qps_col,
            verbose=verbose,
        )
        if not summary_files:
            print("錯誤: median-first 後沒有可處理的 summary 檔案", file=sys.stderr)
            return None, []
    
    window_env = os.environ.get("READ_TRACE_WINDOWS_MS", os.environ.get("READ_TRACE_WINDOW_MS", "0.5"))
    read_trace_window_ms_list = [v.strip() for v in window_env.split(",") if v.strip()]
    
    # Determine number of workers
    if workers is None:
        workers_env = os.environ.get("COLLECT_WORKERS", "")
        if workers_env.strip().isdigit():
            workers = int(workers_env.strip())
        else:
            # 自動模式：避免直接打滿核心造成磁碟 I/O 爭用
            workers = min(os.cpu_count() or 1, 8)
    workers = max(1, int(workers))
    workers = min(workers, len(summary_files))
    
    if verbose:
        print(f"找到 {len(summary_files)} 個 summary_stats.csv 檔案")
        if workers > 1:
            print(f"使用 {workers} 個 worker 進行平行處理...")
    
    all_rows = []
    topk_data = []
    row_id = 1
    failed_count = 0
    timing_totals = {}
    timing_count = 0

    def consume_result(res):
        nonlocal failed_count, row_id, all_rows, topk_data, timing_totals, timing_count
        if res.get("error"):
            failed_count += 1
            if verbose:
                print(f"  ✗ 讀取失敗: {res['summary_file']} - {res['error']}", file=sys.stderr)
            return

        rows = res.get("rows", [])
        if not rows:
            return

        for r in rows:
            r["id"] = row_id
            row_id += 1
            all_rows.append(r)

        if verbose:
            index_name = extract_index_info(res["summary_file"])
            print(f"  ✓ 已讀取: {res['summary_file']} (index: {index_name}, 行數: {len(rows)})")

        topk_data.extend(res.get("topk_rows", []))
        if timing and res.get("timing"):
            timing_count += 1
            for k, v in res["timing"].items():
                timing_totals[k] = timing_totals.get(k, 0.0) + float(v)

    # Process files with multiprocessing if workers > 1
    if workers > 1:
        # Use fork if available (faster), otherwise use spawn (more compatible but slower)
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            process_func = functools.partial(
                _process_one_summary_file,
                read_trace_window_ms_list=read_trace_window_ms_list,
                cleanup=cleanup,
                timing=timing,
            )
            for res in tqdm(
                pool.imap(process_func, summary_files),
                total=len(summary_files),
                desc="處理檔案",
                unit="file",
                disable=verbose,
            ):
                consume_result(res)
    else:
        for sf in tqdm(summary_files, desc="處理檔案", unit="file", disable=verbose):
            consume_result(_process_one_summary_file(sf, read_trace_window_ms_list, cleanup=cleanup, timing=timing))

    if not all_rows:
        print("錯誤: 沒有成功讀取任何檔案", file=sys.stderr)
        return None, []

    # 合併所有資料
    if verbose:
        print(f"\n正在合併資料（rows={len(all_rows)}）...")
    combined_df = pd.DataFrame(all_rows)
    if "id" in combined_df.columns:
        ordered = ["id"] + [c for c in combined_df.columns if c != "id"]
        combined_df = combined_df[ordered]
    
    if verbose:
        print(f"合併後總列數: {len(combined_df)}")
        print(f"列名: {list(combined_df.columns)}")
    
    # 儲存到輸出檔案
    if output_file:
        combined_df.to_csv(output_file, index=False)
        success_msg = f"✓ 彙總完成: {output_file} ({len(combined_df)} 行, {len(combined_df.columns)} 列)"
        if failed_count > 0:
            success_msg += f" (失敗: {failed_count})"
        print(success_msg)
    if timing and timing_count > 0:
        print("\nTiming Summary (avg sec/run):")
        for k in sorted(timing_totals.keys()):
            print(f"  {k}: {timing_totals[k] / timing_count:.4f}")
    
    return combined_df, topk_data


def _process_one_summary_patch(summary_file, read_trace_window_ms_list):
    """計算單一 summary 的指定 window 欄位 patch（僅 read_trace / corr 類）。"""
    try:
        base_prefix = summary_file[: -len("_summary_stats.csv")]
        run_prefix = os.path.basename(base_prefix)
        index_tag = extract_index_info(summary_file)
        read_trace_csv = f"{base_prefix}_read_trace.csv"
        thread_timeline_csv = f"{base_prefix}_thread_timeline.csv"
        labels = [_window_label(v) for v in _parse_window_ms_list(read_trace_window_ms_list)]

        read_trace_stats, read_trace_t0_ns, read_trace_artifacts = parse_read_trace(
            read_trace_csv,
            window_ms_list=read_trace_window_ms_list,
            emit_intermediate=False,
        )
        patch = {"run_prefix": run_prefix, "index_tag": index_tag}

        for k, v in read_trace_stats.items():
            if any(f"_ms{lbl}" in k for lbl in labels):
                patch[k] = v

        if read_trace_t0_ns is not None and os.path.isfile(thread_timeline_csv):
            _paths, window_latency_frames = parse_thread_timeline_windows(
                thread_timeline_csv,
                window_ms_list=read_trace_window_ms_list,
                window_start_ns=read_trace_t0_ns,
                emit_intermediate=False,
            )
            for lbl, latency_df in window_latency_frames.items():
                rt_df = read_trace_artifacts.get("window_stats_frames", {}).get(lbl)
                corr_stats = compute_window_correlations(rt_df, latency_df, lbl)
                patch.update(corr_stats)

        return {"run_prefix": run_prefix, "patch": patch}
    except Exception as e:
        return {"run_prefix": "", "patch": {}, "error": str(e), "summary_file": summary_file}


def patch_window_columns(
    search_dir,
    input_csv,
    output_csv,
    window_ms_list,
    workers=None,
    verbose=False,
):
    """
    對既有 collected CSV 做增量 patch：只補指定 READ_TRACE window 的欄位。
    目前以 run_prefix 作為 merge key。
    """
    summary_files = find_summary_files(search_dir)
    if not summary_files:
        raise SystemExit(f"錯誤: 在 {search_dir} 內找不到 *_summary_stats.csv")

    summary_by_run_prefix = {}
    for s in summary_files:
        rp = os.path.basename(s[: -len("_summary_stats.csv")])
        if rp not in summary_by_run_prefix:
            summary_by_run_prefix[rp] = s

    base_df = pd.read_csv(input_csv)
    if "run_prefix" not in base_df.columns:
        raise SystemExit(f"錯誤: {input_csv} 缺少 run_prefix 欄位，無法 patch")

    use_index_tag = "index_tag" in base_df.columns
    if use_index_tag:
        keys = set(zip(base_df["run_prefix"].astype(str), base_df["index_tag"].astype(str)))
        target_files = []
        for rp, it in keys:
            s = summary_by_run_prefix.get(rp)
            if s and extract_index_info(s) == it:
                target_files.append(s)
    else:
        target_run_prefixes = [rp for rp in base_df["run_prefix"].astype(str).unique() if rp in summary_by_run_prefix]
        target_files = [summary_by_run_prefix[rp] for rp in target_run_prefixes]
    if not target_files:
        raise SystemExit("錯誤: 目標 CSV 與 search_dir 沒有可對應的 run_prefix")

    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    workers = max(1, int(workers))
    workers = min(workers, len(target_files))

    if verbose:
        print(f"Patch-window: 目標 run {len(target_files)}，workers={workers}")

    if workers > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            process_func = functools.partial(_process_one_summary_patch, read_trace_window_ms_list=window_ms_list)
            results = list(
                tqdm(
                    pool.imap(process_func, target_files),
                    total=len(target_files),
                    desc="Patch window",
                    unit="file",
                    disable=verbose,
                )
            )
    else:
        results = [
            _process_one_summary_patch(s, window_ms_list)
            for s in tqdm(target_files, total=len(target_files), desc="Patch window", unit="file", disable=verbose)
        ]

    patch_rows = []
    failed = 0
    for r in results:
        if r.get("error"):
            failed += 1
            if verbose:
                print(f"  ✗ Patch 失敗: {r.get('summary_file', '')} - {r['error']}", file=sys.stderr)
            continue
        if r.get("patch"):
            patch_rows.append(r["patch"])

    if not patch_rows:
        raise SystemExit("錯誤: 沒有可寫入的 patch 欄位資料")

    patch_df = pd.DataFrame(patch_rows)
    if use_index_tag and "index_tag" in patch_df.columns:
        patch_df = patch_df.drop_duplicates(subset=["run_prefix", "index_tag"], keep="last")
        merged = base_df.merge(patch_df, on=["run_prefix", "index_tag"], how="left", suffixes=("", "_patch"))
    else:
        patch_df = patch_df.drop_duplicates(subset=["run_prefix"], keep="last")
        merged = base_df.merge(patch_df, on="run_prefix", how="left", suffixes=("", "_patch"))

    # 若原欄位已存在，覆蓋為 patch 值（非空）
    for c in list(merged.columns):
        if not c.endswith("_patch"):
            continue
        base_col = c[: -len("_patch")]
        if base_col in merged.columns:
            merged[base_col] = merged[c].combine_first(merged[base_col])
            merged.drop(columns=[c], inplace=True)
        else:
            merged.rename(columns={c: base_col}, inplace=True)

    merged.to_csv(output_csv, index=False)
    msg = f"✓ Patch 完成：{output_csv} ({len(merged)} 行, {len(merged.columns)} 列)"
    if failed > 0:
        msg += f" (失敗: {failed})"
    print(msg)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="用於平行化的 worker 數量 (預設: auto=min(CPU,8)，或由 COLLECT_WORKERS 環境變數覆寫)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="每個 summary 檔處理完成後立即清理 collect 產生的中間統計檔案"
    )
    parser.add_argument(
        "--median-first",
        action="store_true",
        help="先依參數組合挑 qps 中位數 run，再做完整彙總（可大幅減少重複分析）",
    )
    parser.add_argument(
        "--median-key-cols",
        default=",".join(DEFAULT_MEDIAN_KEY_COLS),
        help="搭配 --median-first，指定分組鍵（逗號分隔）",
    )
    parser.add_argument(
        "--median-qps-col",
        default="qps",
        help="搭配 --median-first，指定 qps 欄位名稱",
    )
    parser.add_argument(
        "--patch-window",
        action="store_true",
        help="對既有 collected CSV 補算指定 READ_TRACE_WINDOWS_MS 欄位並 merge",
    )
    parser.add_argument(
        "--patch-input",
        default=None,
        help="搭配 --patch-window：要被更新的既有 collected CSV 路徑",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="輸出 collect 各主要階段耗時摘要（秒）",
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

    window_env = os.environ.get("READ_TRACE_WINDOWS_MS", os.environ.get("READ_TRACE_WINDOW_MS", "0.5"))
    read_trace_window_ms_list = [v.strip() for v in window_env.split(",") if v.strip()]

    if args.patch_window:
        if not args.patch_input:
            print("錯誤: --patch-window 需要 --patch-input", file=sys.stderr)
            sys.exit(1)
        patch_input = os.path.abspath(args.patch_input)
        if not os.path.isfile(patch_input):
            print(f"錯誤: patch input 不存在: {patch_input}", file=sys.stderr)
            sys.exit(1)
        if args.output is None:
            output_file = patch_input
        patch_window_columns(
            search_dir=search_dir,
            input_csv=patch_input,
            output_csv=output_file,
            window_ms_list=read_trace_window_ms_list,
            workers=args.workers,
            verbose=args.verbose,
        )
        return
    
    # 執行蒐集
    median_key_cols = [c.strip() for c in args.median_key_cols.split(",") if c.strip()]
    if args.median_first and not median_key_cols:
        print("錯誤: --median-key-cols 為空", file=sys.stderr)
        sys.exit(1)

    summary_df, topk_data = collect_summary_stats(
        search_dir,
        output_file=None,
        verbose=args.verbose,
        workers=args.workers,
        cleanup=args.cleanup,
        median_first=args.median_first,
        median_key_cols=median_key_cols,
        median_qps_col=args.median_qps_col,
        timing=args.timing,
    )
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
        if args.verbose:
            print(f"  移除全零 iostat 欄位: {len(drop_iostat_cols)} 個")

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
