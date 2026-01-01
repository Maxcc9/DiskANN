// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include <regex>
#include <type_traits>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "search_stats.h"
#include "program_options_utils.hpp"

#include <fstream>
#include <unordered_set>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false,
                      const std::string &stats_csv_path = "", const bool append_search_params = false)
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    // if (num_nodes_to_cache > 0)
    //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
    //     num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    const uint32_t actual_cached_nodes = static_cast<uint32_t>(node_list.size());
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "Mean IO (us)" << std::setw(16) << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "=================================================================="
                     "================================================================="
                  << std::endl;

    std::vector<diskann::DiskStatRow> stats_summary;
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    auto split_dir_and_base = [](const std::string &path, std::string &dir, std::string &base) {
        dir.clear();
        base.clear();
        if (path.empty())
        {
            dir = ".";
            return;
        }
        if (path.back() == '/')
        {
            dir = path.substr(0, path.size() - 1);
            if (dir.empty())
                dir = ".";
            return;
        }
        auto pos = path.find_last_of("/\\");
        if (pos == std::string::npos)
        {
            dir = ".";
            base = path;
        }
        else
        {
            dir = path.substr(0, pos);
            if (dir.empty())
                dir = ".";
            base = path.substr(pos + 1);
        }
    };

    auto get_basename = [](const std::string &path) {
        auto pos = path.find_last_of("/\\");
        if (pos == std::string::npos)
            return path;
        return path.substr(pos + 1);
    };

    std::string result_dir, user_prefix;
    split_dir_and_base(result_output_prefix, result_dir, user_prefix);
    const std::string index_basename = get_basename(index_path_prefix);

    struct ParsedIndexParams
    {
        std::string dataset_name;
        uint32_t build_R = 0;
        uint32_t build_L = 0;
        double build_B = 0;
        double build_M = 0;
    };

    auto parse_index_params = [](const std::string &basename) {
        ParsedIndexParams params;
        params.dataset_name = basename;

        std::smatch match;
        const std::regex r_regex(R"(_R(\d+))");
        const std::regex l_regex(R"(_L(\d+))");
        const std::regex b_regex(R"(_B([0-9eE+.\-]+))");
        const std::regex m_regex(R"(_M(\d+))");

        if (std::regex_search(basename, match, r_regex))
        {
            params.build_R = static_cast<uint32_t>(std::stoul(match[1].str()));
            if (match.position(0) > 0)
            {
                params.dataset_name = basename.substr(0, match.position(0));
            }
        }
        if (std::regex_search(basename, match, l_regex))
        {
            params.build_L = static_cast<uint32_t>(std::stoul(match[1].str()));
        }
        if (std::regex_search(basename, match, b_regex))
        {
            params.build_B = std::stod(match[1].str());
        }
        if (std::regex_search(basename, match, m_regex))
        {
            params.build_M = std::stod(match[1].str());
        }
        return params;
    };

    const auto index_params = parse_index_params(index_basename);

    auto build_param_suffix = [&](uint32_t bw_value) {
        std::ostringstream oss;
        if (append_search_params)
        {
            oss << "_W" << bw_value << "_cache" << num_nodes_to_cache << "_T" << num_threads;
        }
        return oss.str();
    };

    auto base_output_prefix = [&](uint32_t bw_value) {
        std::ostringstream oss;
        oss << result_dir << "/";
        if (!user_prefix.empty())
            oss << user_prefix << "_";
        oss << index_basename << build_param_suffix(bw_value);
        return oss.str();
    };

    auto make_result_prefix_for_l = [&](uint32_t l_value, uint32_t bw_value) {
        std::ostringstream oss;
        oss << base_output_prefix(bw_value) << "_L" << l_value;
        return oss.str();
    };

    double best_recall = 0.0;
    std::string per_query_csv_path = stats_csv_path;
    
    if (per_query_csv_path.empty())
    {
        per_query_csv_path = base_output_prefix(optimized_beamwidth) + 
        "_query_stats.csv";
    }
    std::ofstream per_query_csv(per_query_csv_path, std::ios::out | std::ios::trunc);
    if (!per_query_csv.is_open())
    {
        diskann::cerr << "Failed to open per-query stats csv file: " << 
        per_query_csv_path << std::endl;
    }
    else
    {
        per_query_csv << "query_id,L,beamwidth,thread_id,total_us,io_us,cpu_us,sort_us,reorder_cpu_us,"
                      << "n_ios,n_4k,n_8k,n_12k,n_16k,read_size,n_cmps,n_cache_hits,n_hops,"
                      << "visited_nodes,recall_match_count,queue_depth_mean,queue_depth_max,"
                      << "visited_out_degree_mean,visited_out_degree_max\n";
    }

    auto compute_recall_matches = [&](uint32_t query_idx, uint32_t test_id) -> uint32_t {
        if (!calc_recall_flag)
            return 0;
        const uint32_t *gt_q = gt_ids + ((uint64_t)query_idx * gt_dim);
        const uint32_t *res_q = query_result_ids[test_id].data() + ((uint64_t)query_idx * recall_at);
        uint32_t limit = std::min<uint32_t>(gt_dim, recall_at);
        std::unordered_set<uint32_t> gt_set(gt_q, gt_q + limit);
        uint32_t matches = 0;
        for (uint32_t i = 0; i < recall_at; i++)
        {
            if (gt_set.find(res_q[i]) != gt_set.end())
            {
                matches++;
            }
        }
        return matches;
    };

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            stats[i].thread_id = (unsigned)omp_get_thread_num();
            if (!filtered_search)
            {
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        if (calc_recall_flag)
        {
            for (uint32_t qi = 0; qi < query_num; qi++)
            {
                stats[qi].recall_match_count = compute_recall_matches(qi, test_id);
            }
        }

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });
        double log_mean_latency = 0.0;
        {
            const double eps = 1e-6;
            double sum_log = 0.0;
            for (uint32_t qi = 0; qi < query_num; qi++)
            {
                double value = std::max<double>(stats[qi].total_us, eps);
                sum_log += std::log(value);
            }
            log_mean_latency = std::exp(sum_log / static_cast<double>(query_num));
        }

        auto latency_p0 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p1 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p5 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p10 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p25 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p50 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p75 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p90 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p95 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_p99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_max = diskann::get_max_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p0 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p1 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p5 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p10 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p25 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p50 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p75 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p90 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p95 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_p99 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });
        auto ios_max = diskann::get_max_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_ios); });

        auto mean_cpuus =
            diskann::get_mean_stats<float>(stats, query_num, [](const diskann::QueryStats &stats) {
                return stats.cpu_us;
            });
        auto cpu_us_p0 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p1 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p5 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p10 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p25 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p50 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p75 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p90 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p95 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_p99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto cpu_us_max = diskann::get_max_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_sort_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p0 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p1 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p5 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p10 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p25 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p50 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p75 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p90 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p95 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_p99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto sort_us_max = diskann::get_max_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.sort_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p0 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p1 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p5 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p10 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p25 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p50 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p75 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p90 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p95 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_p99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.io_us; });
        auto io_us_max = diskann::get_max_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto read_size_mean = diskann::get_mean_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p0 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p1 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p5 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p10 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p25 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p50 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p75 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p90 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p95 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_p99 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });
        auto read_size_max = diskann::get_max_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.read_size); });

        auto queue_depth_mean = diskann::get_mean_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p0 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p1 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p5 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p10 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p25 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p50 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p75 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p90 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p95 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_p99 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) {
                return stats.queue_depth_count > 0
                           ? static_cast<double>(stats.queue_depth_sum) /
                                 static_cast<double>(stats.queue_depth_count)
                           : 0.0;
            });
        auto queue_depth_max = diskann::get_max_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.queue_depth_max); });

        auto compares_mean = diskann::get_mean_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p0 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p1 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p5 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p10 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p25 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p50 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p75 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p90 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p95 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_p99 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });
        auto compares_max = diskann::get_max_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) { return static_cast<double>(stats.n_cmps); });

        auto out_degree_mean = diskann::get_mean_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p0 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p1 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p5 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p10 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p25 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p50 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p75 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p90 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p95 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_p99 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });
        auto out_degree_max = diskann::get_max_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) {
                return stats.visited_out_degree_count > 0
                           ? static_cast<double>(stats.visited_out_degree_sum) /
                                 static_cast<double>(stats.visited_out_degree_count)
                           : 0.0;
            });

        auto cache_hit_rate_mean = diskann::get_mean_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p0 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p1 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p5 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p10 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p25 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p50 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p75 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p90 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p95 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_p99 = diskann::get_percentile_stats<double>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });
        auto cache_hit_rate_max = diskann::get_max_stats<double>(
            stats, query_num, [](const diskann::QueryStats &stats) {
                const double denom = static_cast<double>(stats.n_ios + stats.n_cache_hits);
                return denom > 0.0 ? static_cast<double>(stats.n_cache_hits) / denom : 0.0;
            });

        auto hop_mean = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p0 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p1 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p5 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p10 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p25 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p50 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p75 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p90 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p95 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p99 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_max = diskann::get_max_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_hops; });

        auto visited_mean = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p0 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.0f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p1 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.01f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p5 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.05f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p10 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.1f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p25 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.25f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p50 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p75 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.75f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p90 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p95 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p99 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_max = diskann::get_max_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });

        std::vector<double> thread_busy_us(num_threads, 0.0);
        double wall_time_us = diff.count() * 1e6;
        if (wall_time_us > 0.0)
        {
            for (uint32_t qi = 0; qi < query_num; qi++)
            {
                unsigned tid = stats[qi].thread_id;
                if (tid < num_threads)
                {
                    thread_busy_us[tid] += static_cast<double>(stats[qi].total_us);
                }
            }
        }
        double thread_util_mean = 0.0;
        double thread_util_p0 = 0.0;
        double thread_util_p1 = 0.0;
        double thread_util_p5 = 0.0;
        double thread_util_p10 = 0.0;
        double thread_util_p25 = 0.0;
        double thread_util_p50 = 0.0;
        double thread_util_p75 = 0.0;
        double thread_util_p90 = 0.0;
        double thread_util_p95 = 0.0;
        double thread_util_p99 = 0.0;
        double thread_util_max = 0.0;
        if (wall_time_us > 0.0 && num_threads > 0)
        {
            std::vector<double> thread_utils(num_threads, 0.0);
            for (uint32_t tid = 0; tid < num_threads; tid++)
            {
                double util = thread_busy_us[tid] / wall_time_us;
                thread_utils[tid] = util;
                thread_util_mean += util;
                if (util > thread_util_max)
                {
                    thread_util_max = util;
                }
            }
            thread_util_mean /= static_cast<double>(num_threads);
            std::sort(thread_utils.begin(), thread_utils.end());
            auto pick_util = [&](double percentile) -> double {
                if (thread_utils.empty())
                {
                    return 0.0;
                }
                size_t idx = static_cast<size_t>(percentile * thread_utils.size());
                if (idx >= thread_utils.size())
                {
                    idx = thread_utils.size() - 1;
                }
                return thread_utils[idx];
            };
            thread_util_p0 = pick_util(0.0);
            thread_util_p1 = pick_util(0.01);
            thread_util_p5 = pick_util(0.05);
            thread_util_p10 = pick_util(0.1);
            thread_util_p25 = pick_util(0.25);
            thread_util_p50 = pick_util(0.5);
            thread_util_p75 = pick_util(0.75);
            thread_util_p90 = pick_util(0.9);
            thread_util_p95 = pick_util(0.95);
            thread_util_p99 = pick_util(0.99);
        }

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
        if (calc_recall_flag)
        {
            auto recall_ratio = [recall_at](const diskann::QueryStats &stats) {
                return static_cast<double>(stats.recall_match_count) / static_cast<double>(recall_at);
            };
            recall_mean = diskann::get_mean_stats<double>(stats, query_num, recall_ratio);
            recall_p0 = diskann::get_percentile_stats<double>(stats, query_num, 0.0f, recall_ratio);
            recall_p1 = diskann::get_percentile_stats<double>(stats, query_num, 0.01f, recall_ratio);
            recall_p5 = diskann::get_percentile_stats<double>(stats, query_num, 0.05f, recall_ratio);
            recall_p10 = diskann::get_percentile_stats<double>(stats, query_num, 0.1f, recall_ratio);
            recall_p25 = diskann::get_percentile_stats<double>(stats, query_num, 0.25f, recall_ratio);
            recall_p50 = diskann::get_percentile_stats<double>(stats, query_num, 0.5f, recall_ratio);
            recall_p75 = diskann::get_percentile_stats<double>(stats, query_num, 0.75f, recall_ratio);
            recall_p90 = diskann::get_percentile_stats<double>(stats, query_num, 0.9f, recall_ratio);
            recall_p95 = diskann::get_percentile_stats<double>(stats, query_num, 0.95f, recall_ratio);
            recall_p99 = diskann::get_percentile_stats<double>(stats, query_num, 0.99f, recall_ratio);
            recall_max = diskann::get_max_stats<double>(stats, query_num, recall_ratio);
            best_recall = std::max(recall_mean, best_recall);
        }

        diskann::DiskStatRow row;
        if (std::is_same<T, float>::value)
        {
            row.data_type = "float";
        }
        else if (std::is_same<T, int8_t>::value)
        {
            row.data_type = "int8";
        }
        else
        {
            row.data_type = "uint8";
        }
        row.search_K = recall_at;
        row.search_L = L;
        row.search_W = optimized_beamwidth;
        row.search_T = num_threads;
        row.search_io_limit = search_io_limit;
        row.dataset_name = index_params.dataset_name;
        row.build_R = index_params.build_R;
        row.build_L = index_params.build_L;
        row.build_B = index_params.build_B;
        row.build_M = index_params.build_M;
        row.num_queries = static_cast<uint32_t>(query_num);
        row.dataset_size = static_cast<uint32_t>(_pFlashIndex->get_num_points());
        row.vector_dim = static_cast<uint32_t>(_pFlashIndex->get_data_dim());
        row.actual_cached_nodes = actual_cached_nodes;
        row.qps = qps;
        row.out_degree_mean = out_degree_mean;
        row.out_degree_p0 = out_degree_p0;
        row.out_degree_p1 = out_degree_p1;
        row.out_degree_p5 = out_degree_p5;
        row.out_degree_p10 = out_degree_p10;
        row.out_degree_p25 = out_degree_p25;
        row.out_degree_p50 = out_degree_p50;
        row.out_degree_p75 = out_degree_p75;
        row.out_degree_p90 = out_degree_p90;
        row.out_degree_p95 = out_degree_p95;
        row.out_degree_p99 = out_degree_p99;
        row.out_degree_max = out_degree_max;
        row.mean_latency = mean_latency;
        row.log_mean_latency = log_mean_latency;
        row.latency_p0 = latency_p0;
        row.latency_p1 = latency_p1;
        row.latency_p5 = latency_p5;
        row.latency_p10 = latency_p10;
        row.latency_p25 = latency_p25;
        row.latency_p50 = latency_p50;
        row.latency_p75 = latency_p75;
        row.latency_p90 = latency_p90;
        row.latency_p95 = latency_p95;
        row.latency_p99 = latency_p99;
        row.latency_999 = latency_999;
        row.latency_max = latency_max;
        row.ios_mean = mean_ios;
        row.ios_p0 = ios_p0;
        row.ios_p1 = ios_p1;
        row.ios_p5 = ios_p5;
        row.ios_p10 = ios_p10;
        row.ios_p25 = ios_p25;
        row.ios_p50 = ios_p50;
        row.ios_p75 = ios_p75;
        row.ios_p90 = ios_p90;
        row.ios_p95 = ios_p95;
        row.ios_p99 = ios_p99;
        row.ios_max = ios_max;
        row.io_us_mean = mean_io_us;
        row.io_us_p0 = io_us_p0;
        row.io_us_p1 = io_us_p1;
        row.io_us_p5 = io_us_p5;
        row.io_us_p10 = io_us_p10;
        row.io_us_p25 = io_us_p25;
        row.io_us_p50 = io_us_p50;
        row.io_us_p75 = io_us_p75;
        row.io_us_p90 = io_us_p90;
        row.io_us_p95 = io_us_p95;
        row.io_us_p99 = io_us_p99;
        row.io_us_max = io_us_max;
        row.cpu_us_mean = mean_cpuus;
        row.cpu_us_p0 = cpu_us_p0;
        row.cpu_us_p1 = cpu_us_p1;
        row.cpu_us_p5 = cpu_us_p5;
        row.cpu_us_p10 = cpu_us_p10;
        row.cpu_us_p25 = cpu_us_p25;
        row.cpu_us_p50 = cpu_us_p50;
        row.cpu_us_p75 = cpu_us_p75;
        row.cpu_us_p90 = cpu_us_p90;
        row.cpu_us_p95 = cpu_us_p95;
        row.cpu_us_p99 = cpu_us_p99;
        row.cpu_us_max = cpu_us_max;
        row.sort_us_mean = mean_sort_us;
        row.sort_us_p0 = sort_us_p0;
        row.sort_us_p1 = sort_us_p1;
        row.sort_us_p5 = sort_us_p5;
        row.sort_us_p10 = sort_us_p10;
        row.sort_us_p25 = sort_us_p25;
        row.sort_us_p50 = sort_us_p50;
        row.sort_us_p75 = sort_us_p75;
        row.sort_us_p90 = sort_us_p90;
        row.sort_us_p95 = sort_us_p95;
        row.sort_us_p99 = sort_us_p99;
        row.sort_us_max = sort_us_max;
        row.read_size_mean = read_size_mean;
        row.read_size_p0 = read_size_p0;
        row.read_size_p1 = read_size_p1;
        row.read_size_p5 = read_size_p5;
        row.read_size_p10 = read_size_p10;
        row.read_size_p25 = read_size_p25;
        row.read_size_p50 = read_size_p50;
        row.read_size_p75 = read_size_p75;
        row.read_size_p90 = read_size_p90;
        row.read_size_p95 = read_size_p95;
        row.read_size_p99 = read_size_p99;
        row.read_size_max = read_size_max;
        row.queue_depth_mean = queue_depth_mean;
        row.queue_depth_p0 = queue_depth_p0;
        row.queue_depth_p1 = queue_depth_p1;
        row.queue_depth_p5 = queue_depth_p5;
        row.queue_depth_p10 = queue_depth_p10;
        row.queue_depth_p25 = queue_depth_p25;
        row.queue_depth_p50 = queue_depth_p50;
        row.queue_depth_p75 = queue_depth_p75;
        row.queue_depth_p90 = queue_depth_p90;
        row.queue_depth_p95 = queue_depth_p95;
        row.queue_depth_p99 = queue_depth_p99;
        row.queue_depth_max = queue_depth_max;
        row.compares_mean = compares_mean;
        row.compares_p0 = compares_p0;
        row.compares_p1 = compares_p1;
        row.compares_p5 = compares_p5;
        row.compares_p10 = compares_p10;
        row.compares_p25 = compares_p25;
        row.compares_p50 = compares_p50;
        row.compares_p75 = compares_p75;
        row.compares_p90 = compares_p90;
        row.compares_p95 = compares_p95;
        row.compares_p99 = compares_p99;
        row.compares_max = compares_max;
        row.recall_mean = recall_mean;
        row.recall_p0 = recall_p0;
        row.recall_p1 = recall_p1;
        row.recall_p5 = recall_p5;
        row.recall_p10 = recall_p10;
        row.recall_p25 = recall_p25;
        row.recall_p50 = recall_p50;
        row.recall_p75 = recall_p75;
        row.recall_p90 = recall_p90;
        row.recall_p95 = recall_p95;
        row.recall_p99 = recall_p99;
        row.recall_max = recall_max;
        row.cache_hit_rate_mean = cache_hit_rate_mean;
        row.cache_hit_rate_p0 = cache_hit_rate_p0;
        row.cache_hit_rate_p1 = cache_hit_rate_p1;
        row.cache_hit_rate_p5 = cache_hit_rate_p5;
        row.cache_hit_rate_p10 = cache_hit_rate_p10;
        row.cache_hit_rate_p25 = cache_hit_rate_p25;
        row.cache_hit_rate_p50 = cache_hit_rate_p50;
        row.cache_hit_rate_p75 = cache_hit_rate_p75;
        row.cache_hit_rate_p90 = cache_hit_rate_p90;
        row.cache_hit_rate_p95 = cache_hit_rate_p95;
        row.cache_hit_rate_p99 = cache_hit_rate_p99;
        row.cache_hit_rate_max = cache_hit_rate_max;
        row.hop_mean = hop_mean;
        row.hop_p0 = hop_p0;
        row.hop_p1 = hop_p1;
        row.hop_p5 = hop_p5;
        row.hop_p10 = hop_p10;
        row.hop_p25 = hop_p25;
        row.hop_p50 = hop_p50;
        row.hop_p75 = hop_p75;
        row.hop_p90 = hop_p90;
        row.hop_p95 = hop_p95;
        row.hop_p99 = hop_p99;
        row.hop_max = hop_max;
        row.visited_mean = visited_mean;
        row.visited_p0 = visited_p0;
        row.visited_p1 = visited_p1;
        row.visited_p5 = visited_p5;
        row.visited_p10 = visited_p10;
        row.visited_p25 = visited_p25;
        row.visited_p50 = visited_p50;
        row.visited_p75 = visited_p75;
        row.visited_p90 = visited_p90;
        row.visited_p95 = visited_p95;
        row.visited_p99 = visited_p99;
        row.visited_max = visited_max;
        row.thread_util_mean = thread_util_mean;
        row.thread_util_p0 = thread_util_p0;
        row.thread_util_p1 = thread_util_p1;
        row.thread_util_p5 = thread_util_p5;
        row.thread_util_p10 = thread_util_p10;
        row.thread_util_p25 = thread_util_p25;
        row.thread_util_p50 = thread_util_p50;
        row.thread_util_p75 = thread_util_p75;
        row.thread_util_p90 = thread_util_p90;
        row.thread_util_p95 = thread_util_p95;
        row.thread_util_p99 = thread_util_p99;
        row.thread_util_max = thread_util_max;
        stats_summary.push_back(row);

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall_mean << std::endl;
        }
        else
            diskann::cout << std::endl;

        // diskann::cout << "    HopCount mean/median/p90/p95/p99/max: " << hop_mean << "/" << hop_p50 << "/" << hop_p90
        //               << "/" << hop_p95 << "/" << hop_p99 << "/" << hop_max << std::endl;
        // diskann::cout << "    VisitedNodes mean/median/p90/p95/p99/max: " << visited_mean << "/" << visited_p50 << "/"
        //               << visited_p90 << "/" << visited_p95 << "/" << visited_p99 << "/" << visited_max << std::endl;

        if (per_query_csv.is_open())
        {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(3);
            for (uint32_t qi = 0; qi < query_num; qi++)
            {
                stats[qi].recall_match_count = compute_recall_matches(qi, test_id);
                const double queue_depth_mean =
                    stats[qi].queue_depth_count > 0
                        ? static_cast<double>(stats[qi].queue_depth_sum) /
                              static_cast<double>(stats[qi].queue_depth_count)
                        : 0.0;
                const double visited_out_degree_mean =
                    stats[qi].visited_out_degree_count > 0
                        ? static_cast<double>(stats[qi].visited_out_degree_sum) /
                              static_cast<double>(stats[qi].visited_out_degree_count)
                        : 0.0;
                oss << qi << "," << L << "," << optimized_beamwidth << "," << stats[qi].thread_id << ","
                    << stats[qi].total_us << "," << stats[qi].io_us << "," << stats[qi].cpu_us << ","
                    << stats[qi].sort_us << "," << stats[qi].reorder_cpu_us << ","
                    << stats[qi].n_ios << "," << stats[qi].n_4k << "," << stats[qi].n_8k << "," << stats[qi].n_12k
                    << "," << stats[qi].n_16k << "," << stats[qi].read_size << "," << stats[qi].n_cmps << ","
                    << stats[qi].n_cache_hits << "," << stats[qi].n_hops << "," << stats[qi].visited_nodes << ","
                    << stats[qi].recall_match_count << "," << queue_depth_mean << "," << stats[qi].queue_depth_max
                    << "," << visited_out_degree_mean << "," << stats[qi].visited_out_degree_max << "\n";
            }
            per_query_csv << oss.str();
        }
        delete[] stats;
    }

    std::string csv_path = base_output_prefix(optimized_beamwidth) + "_summary_stats.csv";
    std::ofstream csv_stream(csv_path, std::ios::out | std::ios::trunc);
    if (!csv_stream.is_open())
    {
        diskann::cerr << "Failed to open stats csv file: " << csv_path << std::endl;
    }
    else
    {
        csv_stream << "dataset_name,data_type,build_R,build_L,build_B,build_M,search_K,search_L,search_W,search_T,"
                      "search_io_limit,num_queries,dataset_size,vector_dim,actual_cached_nodes,qps,"
                      "out_degree_mean,out_degree_p0,out_degree_p1,out_degree_p5,out_degree_p10,out_degree_p25,"
                      "out_degree_p50,out_degree_p75,out_degree_p90,out_degree_p95,out_degree_p99,out_degree_max,"
                      "mean_latency_us,log_mean_latency_us,latency_p0_us,latency_p1_us,latency_p5_us,latency_p10_us,latency_p25_us,latency_p50_us,latency_p75_us,latency_p90_us,latency_p95_us,latency_p99_us,"
                      "latency_p999_us,latency_max_us,ios_mean,ios_p0,ios_p1,ios_p5,ios_p10,ios_p25,ios_p50,ios_p75,ios_p90,ios_p95,ios_p99,ios_max,"
                      "io_us_mean,io_us_p0,io_us_p1,io_us_p5,io_us_p10,io_us_p25,io_us_p50,io_us_p75,io_us_p90,io_us_p95,io_us_p99,io_us_max,"
                      "cpu_us_mean,cpu_us_p0,cpu_us_p1,cpu_us_p5,cpu_us_p10,cpu_us_p25,cpu_us_p50,cpu_us_p75,cpu_us_p90,cpu_us_p95,cpu_us_p99,cpu_us_max,"
                      "sort_us_mean,sort_us_p0,sort_us_p1,sort_us_p5,sort_us_p10,sort_us_p25,sort_us_p50,sort_us_p75,sort_us_p90,sort_us_p95,sort_us_p99,sort_us_max,"
                      "read_size_mean,read_size_p0,read_size_p1,read_size_p5,read_size_p10,read_size_p25,read_size_p50,read_size_p75,read_size_p90,read_size_p95,read_size_p99,"
                      "read_size_max,queue_depth_mean,queue_depth_p0,queue_depth_p1,queue_depth_p5,queue_depth_p10,queue_depth_p25,queue_depth_p50,queue_depth_p75,queue_depth_p90,queue_depth_p95,queue_depth_p99,queue_depth_max,compares_mean,compares_p0,compares_p1,compares_p5,compares_p10,compares_p25,compares_p50,compares_p75,compares_p90,compares_p95,compares_p99,"
                      "compares_max,recall_mean,recall_p0,recall_p1,recall_p5,recall_p10,recall_p25,"
                      "recall_p50,recall_p75,recall_p90,recall_p95,recall_p99,recall_max,cache_hit_rate_mean,cache_hit_rate_p0,"
                      "cache_hit_rate_p1,cache_hit_rate_p5,cache_hit_rate_p10,cache_hit_rate_p25,cache_hit_rate_p50,"
                      "cache_hit_rate_p75,cache_hit_rate_p90,cache_hit_rate_p95,cache_hit_rate_p99,cache_hit_rate_max,hop_mean,hop_p0,hop_p1,hop_p5,hop_p10,hop_p25,hop_p50,hop_p75,hop_p90,"
                      "hop_p95,hop_p99,hop_max,visited_mean,visited_p0,visited_p1,visited_p5,visited_p10,visited_p25,visited_p50,visited_p75,visited_p90,visited_p95,"
                      "visited_p99,visited_max,thread_util_mean,thread_util_p0,thread_util_p1,thread_util_p5,thread_util_p10,thread_util_p25,thread_util_p50,thread_util_p75,thread_util_p90,thread_util_p95,thread_util_p99,thread_util_max\n";
        csv_stream << std::fixed << std::setprecision(4);
        for (const auto &row : stats_summary)
        {
            csv_stream << row.dataset_name << "," << row.data_type << "," << row.build_R << "," << row.build_L << ","
                       << row.build_B << "," << row.build_M << "," << row.search_K << "," << row.search_L << ","
                       << row.search_W << "," << row.search_T << "," << row.search_io_limit << "," << row.num_queries
                       << ","
                       << row.dataset_size << "," << row.vector_dim << "," << row.actual_cached_nodes << ","
                       << row.qps << "," << row.out_degree_mean << "," << row.out_degree_p0 << "," << row.out_degree_p1
                       << "," << row.out_degree_p5 << "," << row.out_degree_p10 << "," << row.out_degree_p25 << ","
                       << row.out_degree_p50 << "," << row.out_degree_p75 << "," << row.out_degree_p90 << ","
                       << row.out_degree_p95 << "," << row.out_degree_p99 << "," << row.out_degree_max << ","
                       << row.mean_latency << "," << row.log_mean_latency << "," << row.latency_p0 << "," << row.latency_p1 << ","
                       << row.latency_p5 << "," << row.latency_p10 << "," << row.latency_p25 << "," << row.latency_p50 << "," << row.latency_p75 << "," << row.latency_p90
                       << "," << row.latency_p95 << "," << row.latency_p99 << "," << row.latency_999 << ","
                       << row.latency_max << "," << row.ios_mean << "," << row.ios_p0 << "," << row.ios_p1 << ","
                       << row.ios_p5 << "," << row.ios_p10 << "," << row.ios_p25 << "," << row.ios_p50 << ","
                       << row.ios_p75 << "," << row.ios_p90 << "," << row.ios_p95 << "," << row.ios_p99 << "," << row.ios_max << ","
                       << row.io_us_mean << "," << row.io_us_p0 << "," << row.io_us_p1 << "," << row.io_us_p5 << ","
                       << row.io_us_p10 << "," << row.io_us_p25 << "," << row.io_us_p50 << "," << row.io_us_p75 << "," << row.io_us_p90
                       << "," << row.io_us_p95 << "," << row.io_us_p99 << "," << row.io_us_max << ","
                       << row.cpu_us_mean << "," << row.cpu_us_p0 << "," << row.cpu_us_p1 << "," << row.cpu_us_p5 << ","
                       << row.cpu_us_p10 << "," << row.cpu_us_p25 << "," << row.cpu_us_p50 << "," << row.cpu_us_p75 << "," << row.cpu_us_p90
                       << "," << row.cpu_us_p95 << "," << row.cpu_us_p99 << "," << row.cpu_us_max << ","
                       << row.sort_us_mean << "," << row.sort_us_p0 << "," << row.sort_us_p1 << "," << row.sort_us_p5 << ","
                       << row.sort_us_p10 << "," << row.sort_us_p25 << "," << row.sort_us_p50 << "," << row.sort_us_p75 << ","
                       << row.sort_us_p90 << "," << row.sort_us_p95 << "," << row.sort_us_p99 << "," << row.sort_us_max << ","
                       << row.read_size_mean << "," << row.read_size_p0 << "," << row.read_size_p1 << "," << row.read_size_p5 << ","
                       << row.read_size_p10 << "," << row.read_size_p25 << "," << row.read_size_p50 << "," << row.read_size_p75 << ","
                       << row.read_size_p90 << "," << row.read_size_p95 << "," << row.read_size_p99 << "," << row.read_size_max << ","
                       << row.queue_depth_mean << "," << row.queue_depth_p0 << ","
                       << row.queue_depth_p1 << "," << row.queue_depth_p5 << "," << row.queue_depth_p10 << ","
                       << row.queue_depth_p25 << "," << row.queue_depth_p50 << "," << row.queue_depth_p75 << ","
                       << row.queue_depth_p90 << "," << row.queue_depth_p95 << "," << row.queue_depth_p99 << ","
                       << row.queue_depth_max << ","
                       << row.compares_mean << "," << row.compares_p0 << "," << row.compares_p1 << "," << row.compares_p5 << ","
                       << row.compares_p10 << "," << row.compares_p25 << "," << row.compares_p50 << "," << row.compares_p75 << ","
                       << row.compares_p90 << "," << row.compares_p95 << "," << row.compares_p99 << "," << row.compares_max << ","
                       << row.recall_mean << "," << row.recall_p0 << "," << row.recall_p1
                       << "," << row.recall_p5 << "," << row.recall_p10 << "," << row.recall_p25 << ","
                       << row.recall_p50 << "," << row.recall_p75 << "," << row.recall_p90 << "," << row.recall_p95 << ","
                       << row.recall_p99 << "," << row.recall_max
                       << "," << row.cache_hit_rate_mean << "," << row.cache_hit_rate_p0 << ","
                       << row.cache_hit_rate_p1 << "," << row.cache_hit_rate_p5 << "," << row.cache_hit_rate_p10
                       << "," << row.cache_hit_rate_p25 << "," << row.cache_hit_rate_p50 << ","
                       << row.cache_hit_rate_p75 << "," << row.cache_hit_rate_p90 << "," << row.cache_hit_rate_p95 << ","
                       << row.cache_hit_rate_p99 << "," << row.cache_hit_rate_max
                       << "," << row.hop_mean << "," << row.hop_p0 << "," << row.hop_p1 << "," << row.hop_p5 << "," << row.hop_p10
                       << "," << row.hop_p25 << "," << row.hop_p50 << "," << row.hop_p75 << "," << row.hop_p90
                       << "," << row.hop_p95 << "," << row.hop_p99 << "," << row.hop_max << "," << row.visited_mean
                       << "," << row.visited_p0 << "," << row.visited_p1 << "," << row.visited_p5 << "," << row.visited_p10 << ","
                       << row.visited_p25 << "," << row.visited_p50 << "," << row.visited_p75 << "," << row.visited_p90 << ","
                       << row.visited_p95 << "," << row.visited_p99 << "," << row.visited_max << ","
                       << row.thread_util_mean << "," << row.thread_util_p0 << "," << row.thread_util_p1 << ","
                       << row.thread_util_p5 << "," << row.thread_util_p10 << "," << row.thread_util_p25 << ","
                       << row.thread_util_p50 << "," << row.thread_util_p75 << "," << row.thread_util_p90 << ","
                       << row.thread_util_p95 << "," << row.thread_util_p99 << "," << row.thread_util_max << "\n";
        }
    }

    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string base_prefix = make_result_prefix_for_l(L, optimized_beamwidth);
        std::string cur_result_path = base_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = base_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
        label_type, query_filters_file, stats_csv_path;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    bool append_search_params = false;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()(
            "stats_csv_path", po::value<std::string>(&stats_csv_path)->default_value(std::string("")),
            "Path to write per-query stats (CSV) for spreadsheet analysis. Defaults to <result_path>_query_stats.csv");
        optional_configs.add_options()("append_search_params_to_result_path,A", po::bool_switch(&append_search_params),
                                       "Append _W<beamwidth>_cache<num_nodes_to_cache>_T<num_threads>_L<L> to "
                                       "result_path for outputs");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    stats_csv_path, append_search_params);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    stats_csv_path, append_search_params);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    stats_csv_path, append_search_params);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data, stats_csv_path,
                                                append_search_params);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data, stats_csv_path,
                                                 append_search_params);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data, stats_csv_path,
                                                  append_search_params);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
