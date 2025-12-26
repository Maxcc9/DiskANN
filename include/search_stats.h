// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>

namespace diskann
{
// Aggregated per-run stats for in-memory search flows.
struct MemoryStatRow
{
    uint32_t L = 0;
    double qps = 0;
    double mean_latency = 0;
    double latency_999 = 0;
    double mean_cmps = 0;
    double recall = 0;
    double hop_mean = 0;
    double hop_p50 = 0;
    double hop_p90 = 0;
    double hop_p95 = 0;
    double hop_p99 = 0;
    uint64_t hop_max = 0;
    double visited_mean = 0;
    double visited_p50 = 0;
    double visited_p90 = 0;
    double visited_p95 = 0;
    double visited_p99 = 0;
    uint64_t visited_max = 0;
};

// Aggregated per-run stats for disk-based search flows.
struct DiskStatRow
{
    std::string dataset_name;
    std::string data_type;
    // double build_alpha = 0;  // 磁碟版沒開放建置參數調整
    uint32_t build_R = 0;
    uint32_t build_L = 0;
    double build_B = 0;
    double build_M = 0;

    uint32_t search_K = 0;
    uint32_t search_L = 0;
    uint32_t search_W = 0;
    uint32_t search_T = 0;
    uint32_t search_io_limit = 0;

    uint32_t num_queries = 0;
    uint32_t dataset_size = 0;
    uint32_t vector_dim = 0;

    uint32_t actual_cached_nodes = 0;
    double qps = 0;

    double out_degree_mean = 0;
    double out_degree_p0 = 0;
    double out_degree_p1 = 0;
    double out_degree_p5 = 0;
    double out_degree_p10 = 0;
    double out_degree_p25 = 0;
    double out_degree_p50 = 0;
    double out_degree_p75 = 0;
    double out_degree_p90 = 0;
    double out_degree_p95 = 0;
    double out_degree_p99 = 0;
    double out_degree_max = 0;

    double mean_latency = 0;
    double latency_p50 = 0;
    double latency_p75 = 0;
    double latency_p90 = 0;
    double latency_p95 = 0;
    double latency_p99 = 0;
    double latency_999 = 0;
    double latency_max = 0;

    double ios_mean = 0;
    double ios_p50 = 0;
    double ios_p75 = 0;
    double ios_p90 = 0;
    double ios_p95 = 0;
    double ios_p99 = 0;
    double ios_max = 0;

    double io_us_mean = 0;
    double io_us_p50 = 0;
    double io_us_p75 = 0;
    double io_us_p90 = 0;
    double io_us_p95 = 0;
    double io_us_p99 = 0;
    double io_us_max = 0;

    double cpu_us_mean = 0;
    double cpu_us_p50 = 0;
    double cpu_us_p75 = 0;
    double cpu_us_p90 = 0;
    double cpu_us_p95 = 0;
    double cpu_us_p99 = 0;
    double cpu_us_max = 0;

    double sort_us_mean = 0;
    double sort_us_p50 = 0;
    double sort_us_p75 = 0;
    double sort_us_p90 = 0;
    double sort_us_p95 = 0;
    double sort_us_p99 = 0;
    double sort_us_max = 0;

    // double reorder_cpu_us_mean = 0;

    double read_size_mean = 0;
    double read_size_p50 = 0;
    double read_size_p75 = 0;
    double read_size_p90 = 0;
    double read_size_p95 = 0;
    double read_size_p99 = 0;
    double read_size_max = 0;

    double compares_mean = 0;
    double compares_p50 = 0;
    double compares_p75 = 0;
    double compares_p90 = 0;
    double compares_p95 = 0;
    double compares_p99 = 0;
    double compares_max = 0;

    double recall_mean = 0;
    double recall_p0 = 0;
    double recall_p1 = 0;
    double recall_p5 = 0;
    double recall_p10 = 0;
    double recall_p25 = 0;
    double recall_p50 = 0;
    double recall_p75 = 0;
    double recall_p90 = 0;
    double recall_max = 0;
    
    double cache_hit_rate_mean = 0;
    double cache_hit_rate_p0 = 0;
    double cache_hit_rate_p1 = 0;
    double cache_hit_rate_p5 = 0;
    double cache_hit_rate_p10 = 0;
    double cache_hit_rate_p25 = 0;
    double cache_hit_rate_p50 = 0;
    double cache_hit_rate_p75 = 0;
    double cache_hit_rate_p90 = 0;
    double cache_hit_rate_max = 0;

    double hop_mean = 0;
    double hop_p50 = 0;
    double hop_p75 = 0;
    double hop_p90 = 0;
    double hop_p95 = 0;
    double hop_p99 = 0;
    uint64_t hop_max = 0;

    double visited_mean = 0;
    double visited_p50 = 0;
    double visited_p75 = 0;
    double visited_p90 = 0;
    double visited_p95 = 0;
    double visited_p99 = 0;
    uint64_t visited_max = 0;
};
} // namespace diskann
