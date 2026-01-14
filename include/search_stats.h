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
};
} // namespace diskann
