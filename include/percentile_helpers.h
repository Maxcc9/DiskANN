// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include "percentile_stats.h"

namespace diskann
{

// 百分位數配置
struct PercentileSet
{
    std::vector<float> percentiles = {0.0f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f, 0.999f, 1.0f};
    
    // 默認構造函數：使用默認百分位數
    PercentileSet() = default;
    
    // 自定義構造函數：支持自訂百分位數列表
    explicit PercentileSet(const std::vector<float> &p) : percentiles(p) {}
};

// 存儲單個指標的完整統計結果
template <typename T>
struct MetricStats
{
    T mean = 0;
    T geometric_mean = 0;  // 幾何平均數 (對延遲等對數正態分佈的指標更穩健)
    std::map<float, T> percentiles; // key: percentile value (0.0, 0.01, 0.5, ..., 1.0)
    
    // 輔助函數：按名稱獲取百分位數
    T get_percentile(float p) const
    {
        auto it = percentiles.find(p);
        return (it != percentiles.end()) ? it->second : T(0);
    }
    
    // min/max 就是 p0/p100
    T min() const { return get_percentile(0.0f); }
    T max() const { return get_percentile(1.0f); }
    
    T p0() const { return get_percentile(0.0f); }
    T p1() const { return get_percentile(0.01f); }
    T p5() const { return get_percentile(0.05f); }
    T p10() const { return get_percentile(0.1f); }
    T p25() const { return get_percentile(0.25f); }
    T p50() const { return get_percentile(0.5f); }
    T p75() const { return get_percentile(0.75f); }
    T p90() const { return get_percentile(0.9f); }
    T p95() const { return get_percentile(0.95f); }
    T p99() const { return get_percentile(0.99f); }
    T p999() const { return get_percentile(0.999f); }
    T p100() const { return get_percentile(1.0f); }
    
    // 生成 CSV header
    std::string to_csv_header(const std::string &prefix, const PercentileSet &pset) const
    {
        std::ostringstream oss;
        oss << prefix << "_mean," << prefix << "_gmean";
        for (float p : pset.percentiles)
        {
            oss << "," << prefix << "_p" << percentile_to_string(p);
        }
        return oss.str();
    }
    
    // 生成 CSV 數據行
    std::string to_csv_values(const PercentileSet &pset) const
    {
        std::ostringstream oss;
        // 對於整數類型，不使用小數點；對於浮點類型，使用 4 位精度
        if constexpr (std::is_integral<T>::value) {
            oss << mean << "," << geometric_mean;
            for (float p : pset.percentiles)
            {
                auto it = percentiles.find(p);
                oss << "," << (it != percentiles.end() ? it->second : T(0));
            }
        } else {
            oss << std::fixed << std::setprecision(4) << mean << "," << geometric_mean;
            for (float p : pset.percentiles)
            {
                auto it = percentiles.find(p);
                oss << "," << std::fixed << std::setprecision(4) << (it != percentiles.end() ? it->second : T(0));
            }
        }
        return oss.str();
    }
    
private:
    static std::string percentile_to_string(float p)
    {
        if (p == 0.999f) return "999";  // 特殊：保留三位
        int pct = static_cast<int>(p * 100 + 0.5f);
        return std::to_string(pct);
    }
};

// 計算單個指標的所有統計值（使用迴圈）
template <typename T, typename ExtractorFunc>
MetricStats<T> compute_metric_stats(QueryStats *stats, uint64_t query_num, const PercentileSet &pset,
                                    ExtractorFunc extractor)
{
    MetricStats<T> result;
    
    // 計算平均值
    result.mean = get_mean_stats<T>(stats, query_num, extractor);
    
    // 計算幾何平均數 (適用於對數正態分佈的指標，如延遲)
    {
        const double eps = 1e-6;
        double sum_log = 0.0;
        for (uint64_t qi = 0; qi < query_num; qi++)
        {
            double value = std::max<double>(static_cast<double>(extractor(stats[qi])), eps);
            sum_log += std::log(value);
        }
        result.geometric_mean = static_cast<T>(std::exp(sum_log / static_cast<double>(query_num)));
    }
    
    // 計算所有百分位數（迴圈自動包含 p0 和 p100）
    for (float p : pset.percentiles)
    {
        result.percentiles[p] = get_percentile_stats<T>(stats, query_num, p, extractor);
    }
    
    return result;
}

// 特殊版本：從 vector field 計算統計
template <typename T, typename ExtractorFunc>
MetricStats<double> compute_metric_stats_from_vector(QueryStats *stats, uint64_t query_num,
                                                     const PercentileSet &pset, ExtractorFunc extractor)
{
    MetricStats<double> result;
    
    result.mean = get_mean_from_vector_field<T>(stats, query_num, extractor);
    
    // 計算幾何平均數（從 vector field）
    {
        const double eps = 1e-6;
        double sum_log = 0.0;
        uint64_t total_count = 0;
        for (uint64_t qi = 0; qi < query_num; qi++)
        {
            const auto &vec = extractor(stats[qi]);
            for (const auto &val : vec)
            {
                double value = std::max<double>(static_cast<double>(val), eps);
                sum_log += std::log(value);
                total_count++;
            }
        }
        result.geometric_mean = (total_count > 0) 
            ? std::exp(sum_log / static_cast<double>(total_count)) 
            : 0.0;
    }
    
    for (float p : pset.percentiles)
    {
        result.percentiles[p] = get_percentile_from_vector_field<T>(stats, query_num, p, extractor);
    }
    
    return result;
}

// 管理多個指標的統計結果（用於動態 CSV 生成）
template <typename T>
struct MetricsCollection
{
    std::map<std::string, MetricStats<T>> metrics;  // key: metric name (e.g., "latency", "ios")
    std::vector<std::string> order; // preserve insertion order for CSV output
    PercentileSet pset;
    
    MetricsCollection(const PercentileSet &p) : pset(p) {}
    
    void add(const std::string &name, const MetricStats<T> &stats)
    {
        if (metrics.find(name) == metrics.end())
        {
            order.push_back(name);
        }
        metrics[name] = stats;
    }
    
    // 生成完整的 CSV header
    std::string to_csv_header() const
    {
        std::ostringstream oss;
        bool first = true;
        for (const auto &name : order)
        {
            auto it = metrics.find(name);
            if (it == metrics.end())
            {
                continue;
            }
            if (!first) oss << ",";
            oss << it->second.to_csv_header(name, pset);
            first = false;
        }
        return oss.str();
    }
    
    // 生成完整的 CSV 數據行
    std::string to_csv_values() const
    {
        std::ostringstream oss;
        bool first = true;
        for (const auto &name : order)
        {
            auto it = metrics.find(name);
            if (it == metrics.end())
            {
                continue;
            }
            if (!first) oss << ",";
            oss << it->second.to_csv_values(pset);
            first = false;
        }
        return oss.str();
    }
};

} // namespace diskann
