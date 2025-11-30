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
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads (not implemented yet)
    unsigned n_12k = 0;        // # of 12kB reads (not implemented yet)
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved (not implemented yet)
    unsigned n_cmps = 0;       // # cmps (查詢向量與候選節點作距離比較的次數)
    unsigned n_cache_hits = 0; // # cache_hits (快取命中次數)
    unsigned n_hops = 0;       // # search hops
    unsigned visited_nodes = 0; // # unique visited nodes in search
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
