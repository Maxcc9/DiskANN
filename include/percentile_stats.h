// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO (dram 快取命中不會增加 io_us)
    float cpu_us = 0;   // total time spent in CPU (PQ distance calculations + filtering)
    float sort_us = 0;  // time spent sorting candidates after search
    float reorder_cpu_us = 0;  // time spent in full-precision distance calculations and final sort

    uint64_t n_4k = 0;         // # of 4kB reads
    uint64_t n_8k = 0;         // # of 8kB reads
    uint64_t n_12k = 0;        // # of 12kB reads
    uint64_t n_16k = 0;        // # of 16kB reads
    uint64_t n_ios = 0;        // total # of IOs issued
    uint64_t read_size = 0;    // total # of bytes read
    uint64_t n_cmps_saved = 0; // # cmps saved (not implemented yet)
    uint64_t n_cmps = 0;       // # cmps (查詢向量與候選節點作距離比較的次數)
    uint64_t n_cache_hits = 0; // # cache_hits (快取命中次數)
    uint64_t n_hops = 0;       // # search hops
    uint64_t visited_nodes = 0; // # unique visited nodes in search
    uint64_t queue_depth_sum = 0;   // sum of IO batch sizes across iterations
    uint64_t queue_depth_count = 0; // number of IO batches issued
    uint64_t queue_depth_max = 0;   // max IO batch size in a query
    uint64_t visited_out_degree_sum = 0;   // sum of outdegree for expanded nodes
    uint64_t visited_out_degree_count = 0; // number of expanded nodes counted
    uint64_t visited_out_degree_max = 0;   // max outdegree among expanded nodes
    bool expanded_nodes_enabled = false; // enable recording expanded node ids
    uint32_t expanded_nodes_limit = 0; // max expanded nodes to record (0 = unlimited)
    uint32_t expanded_nodes_dropped = 0; // count of dropped expanded nodes due to limit
    std::vector<uint32_t> expanded_nodes; // expanded node ids for analysis
    unsigned thread_id = 0;     // thread executing the query
    unsigned recall_match_count = 0; // # of matches against ground truth @K
};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}

template <typename T>
inline T get_max_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    T max_val = member_fn(stats[0]);
    for (uint64_t i = 1; i < len; i++)
    {
        max_val = (std::max)(max_val, member_fn(stats[i]));
    }
    return max_val;
}
} // namespace diskann
