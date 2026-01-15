// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#include <type_traits>
#include <utility>
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
    float io_us = 0;    // time spent in read() syscalls (includes OS page cache hits; app-level cache hits skip read() entirely)
    float cpu_us = 0;   // total time spent in CPU (PQ distance calculations + filtering)
    float sort_us = 0;  // time spent sorting candidates after search
    float reorder_cpu_us = 0;  // time spent in full-precision distance calculations and final sort

    uint64_t n_4k = 0;         // # of 4kB reads
    uint64_t n_8k = 0;         // # of 8kB reads
    uint64_t n_12k = 0;        // # of 12kB reads
    uint64_t n_16k = 0;        // # of 16kB reads

    uint64_t n_ios = 0;        // total # of logical IO requests (independent read units: nodes in search, vectors in reorder)
    uint64_t read_size = 0;    // total # of bytes read (actual SSD transfer size)
    uint64_t n_ios_search = 0;    // # of logical IO requests during search phase (frontier nodes)
    uint64_t n_ios_reorder = 0;   // # of logical IO requests during reorder phase (full-precision vectors)
    uint64_t n_cmps_saved = 0; // # cmps saved (not implemented yet)
    uint64_t n_cmps = 0;       // # cmps (查詢向量與候選節點作距離比較的次數)
    uint64_t n_cache_hits = 0; // # cache_hits (快取命中次數)
    uint64_t n_hops = 0;       // # search hops

    uint64_t visited_nodes_count = 0; // # unique visited nodes in search

    // Per-query queue depth statistics (aggregated from all frontier IO batches)
    uint64_t frontier_io_iterations = 0;  // # of non-empty frontier IO batches
    double frontier_queue_depths_mean = 0.0;  // average frontier IO batch size
    uint64_t frontier_queue_depths_max = 0;  // maximum frontier IO batch size
    uint64_t frontier_queue_depths_min = UINT64_MAX;  // minimum frontier IO batch size (init to max value)
    uint64_t reorder_io_iterations = 0;  // # of non-empty reorder IO batches
    double reorder_queue_depths_mean = 0.0;  // average reorder IO batch size
    uint64_t reorder_queue_depths_max = 0;  // maximum reorder IO batch size
    uint64_t reorder_queue_depths_min = UINT64_MAX;  // minimum reorder IO batch size (init to max value)
    
    std::vector<uint64_t> visited_out_degrees; // out-degree of each expanded node (for accurate percentile analysis)
    
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

    const float clamped = std::max(0.0f, std::min(1.0f, percentile));
    const uint64_t last = (len > 0) ? (len - 1) : 0;
    // Use p * (n - 1) so p99 doesn't collapse to max when n == 100.
    uint64_t idx = (len > 0) ? static_cast<uint64_t>(clamped * last) : 0;
    auto retval = vals[idx];
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
    // Deprecated: use get_percentile_stats with percentile = 1.0f instead
    return get_percentile_stats<T>(stats, len, 1.0f, member_fn);
}

// Collect all values from a vector field across all queries and compute percentile
template <typename T, typename VectorFn>
inline double get_percentile_from_vector_field(QueryStats *stats, uint64_t num_queries, float percentile,
                                               VectorFn &&vector_fn)
{
    using VecRef = decltype(vector_fn(std::declval<const QueryStats &>()));
    static_assert(std::is_lvalue_reference<VecRef>::value,
                  "vector_fn must return const std::vector<T>& (no temporaries).");
    static_assert(std::is_same<typename std::remove_reference<VecRef>::type, const std::vector<T>>::value,
                  "vector_fn must return const std::vector<T>& (matching element type).");

    std::vector<T> all_vals;
    for (uint64_t i = 0; i < num_queries; i++)
    {
        const auto &vec = vector_fn(stats[i]);
        all_vals.insert(all_vals.end(), vec.begin(), vec.end());
    }

    if (all_vals.empty())
        return 0.0;

    std::sort(all_vals.begin(), all_vals.end(), [](const T &left, const T &right) { return left < right; });
    const float clamped = std::max(0.0f, std::min(1.0f, percentile));
    const uint64_t last = (all_vals.size() > 0) ? (all_vals.size() - 1) : 0;
    // Use p * (n - 1) so p99 doesn't collapse to max when n == 100.
    uint64_t idx = (all_vals.size() > 0) ? static_cast<uint64_t>(clamped * last) : 0;
    return static_cast<double>(all_vals[idx]);
}

// Compute mean from vector field across all queries
template <typename T, typename VectorFn>
inline double get_mean_from_vector_field(QueryStats *stats, uint64_t num_queries, VectorFn &&vector_fn)
{
    using VecRef = decltype(vector_fn(std::declval<const QueryStats &>()));
    static_assert(std::is_lvalue_reference<VecRef>::value,
                  "vector_fn must return const std::vector<T>& (no temporaries).");
    static_assert(std::is_same<typename std::remove_reference<VecRef>::type, const std::vector<T>>::value,
                  "vector_fn must return const std::vector<T>& (matching element type).");

    std::vector<T> all_vals;
    for (uint64_t i = 0; i < num_queries; i++)
    {
        const auto &vec = vector_fn(stats[i]);
        all_vals.insert(all_vals.end(), vec.begin(), vec.end());
    }

    if (all_vals.empty())
        return 0.0;

    double sum = 0;
    for (const auto &val : all_vals)
        sum += static_cast<double>(val);
    return sum / all_vals.size();
}

// Get max from vector field across all queries (uses percentile 1.0)
template <typename T, typename VectorFn>
inline T get_max_from_vector_field(QueryStats *stats, uint64_t num_queries, VectorFn &&vector_fn)
{
    return static_cast<T>(get_percentile_from_vector_field<T>(stats, num_queries, 1.0f, vector_fn));
}
} // namespace diskann
