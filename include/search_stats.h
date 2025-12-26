// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>

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
    bool has_recall = false;
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
    uint32_t L = 0;
    uint32_t beamwidth = 0;
    double qps = 0;
    double mean_latency = 0;
    double latency_p50 = 0;
    double latency_p90 = 0;
    double latency_p95 = 0;
    double latency_p99 = 0;
    double latency_999 = 0;
    double latency_max = 0;
    double mean_ios = 0;
    double mean_io_us = 0;
    double mean_cpu_us = 0;
    double mean_sort_us = 0;
    double mean_reorder_cpu_us = 0;
    bool has_recall = false;
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
} // namespace diskann
