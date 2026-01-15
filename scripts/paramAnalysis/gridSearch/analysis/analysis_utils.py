from __future__ import annotations

import numpy as np


def _set_default(df, name, value):
    if name not in df.columns:
        df[name] = value


def _alias_column(df, old_name, new_name):
    if old_name not in df.columns and new_name in df.columns:
        df[old_name] = df[new_name]


def normalize_columns(df):
    """Backfill legacy column names for analysis notebooks."""
    # Expanded nodes / node count aliases
    _alias_column(df, "expanded_revisit_ratio", "expanded_nodes_revisit_ratio")
    _alias_column(df, "node_counts_total", "expanded_nodes_total")
    _alias_column(df, "node_counts_unique", "expanded_nodes_unique")
    _alias_column(df, "node_counts_top1_share", "expanded_node_top1_share")
    _alias_column(df, "node_counts_top10_share", "expanded_node_top10_share")
    _alias_column(df, "node_counts_top100_share", "expanded_node_top100_share")
    _alias_column(df, "node_counts_top1000_share", "expanded_node_top1000_share")
    _alias_column(df, "node_counts_top10000_share", "expanded_node_top10000_share")

    # Derived expanded per-query metrics (not available in new output)
    if "expanded_per_query_mean" not in df.columns:
        if "expanded_nodes_total" in df.columns and "num_queries" in df.columns:
            df["expanded_per_query_mean"] = df["expanded_nodes_total"] / df["num_queries"]
        else:
            df["expanded_per_query_mean"] = np.nan
    _set_default(df, "expanded_per_query_p50", np.nan)
    _set_default(df, "expanded_per_query_p90", np.nan)
    _set_default(df, "expanded_steps_mean", np.nan)
    _set_default(df, "expanded_steps_p50", np.nan)
    _set_default(df, "expanded_steps_p90", np.nan)

    # Out-degree aliases
    for suffix in ("mean", "p50", "p90", "p95", "p99", "p999", "p0", "p1", "p5", "p10", "p25", "p75", "p100"):
        _alias_column(df, f"out_degree_{suffix}", f"expanded_node_out_degree_{suffix}")

    # Queue depth aliases
    for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        _alias_column(df, f"queue_depth_p{p}", f"frontier_queue_depth_mean_p{p}")

    # Latency aliases
    for p in (50, 90, 95, 99, 999):
        _alias_column(df, f"latency_p{p}_us", f"latency_us_p{p}")
    _alias_column(df, "latency_mean_us", "latency_us_mean")

    # Top-K aliases
    _alias_column(df, "topk_neighbor_edges", "topk_expanded_neighbor_count")
    _alias_column(df, "topk_unique_nodes", "topk_expanded_unique_count")
    _alias_column(df, "topk_unique_neighbors", "topk_expanded_unique_neighbors_count")
    _alias_column(df, "topk_degree_mean", "topk_expanded_degree_mean")
    _set_default(df, "topk_neighbors_path", "")
    _set_default(df, "topk_nodes_path", "")
    _set_default(df, "summary_stats_path", "")
    _alias_column(df, "topk_cover_ratio", "topk_expanded_coverage_ratio")
    for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        _alias_column(df, f"topk_degree_p{p}", f"topk_expanded_degree_p{p}")

    return df
