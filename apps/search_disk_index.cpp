// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

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
                      << "n_ios,n_4k,n_8k,n_12k,n_16k,n_cache_hits,n_hops,"
                      << "visited_nodes,recall_match_count\n";
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

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        auto mean_sort_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.sort_us; });
        auto mean_reorder_cpu_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.reorder_cpu_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto hop_mean = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_hops; });
        auto hop_p50 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.n_hops; });
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
        auto visited_p50 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.5f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p90 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.9f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p95 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.95f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_p99 = diskann::get_percentile_stats<uint32_t>(
            stats, query_num, 0.99f, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });
        auto visited_max = diskann::get_max_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.visited_nodes; });

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        diskann::DiskStatRow row;
        row.L = L;
        row.beamwidth = optimized_beamwidth;
        row.qps = qps;
        row.mean_latency = mean_latency;
        row.latency_999 = latency_999;
        row.mean_ios = mean_ios;
        row.mean_io_us = mean_io_us;
        row.mean_cpu_us = mean_cpuus;
        row.mean_sort_us = mean_sort_us;
        row.mean_reorder_cpu_us = mean_reorder_cpu_us;
        row.has_recall = calc_recall_flag;
        row.recall = recall;
        row.hop_mean = hop_mean;
        row.hop_p50 = hop_p50;
        row.hop_p90 = hop_p90;
        row.hop_p95 = hop_p95;
        row.hop_p99 = hop_p99;
        row.hop_max = hop_max;
        row.visited_mean = visited_mean;
        row.visited_p50 = visited_p50;
        row.visited_p90 = visited_p90;
        row.visited_p95 = visited_p95;
        row.visited_p99 = visited_p99;
        row.visited_max = visited_max;
        stats_summary.push_back(row);

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;

        diskann::cout << "    HopCount mean/median/p90/p95/p99/max: " << hop_mean << "/" << hop_p50 << "/" << hop_p90
                      << "/" << hop_p95 << "/" << hop_p99 << "/" << hop_max << std::endl;
        diskann::cout << "    VisitedNodes mean/median/p90/p95/p99/max: " << visited_mean << "/" << visited_p50 << "/"
                      << visited_p90 << "/" << visited_p95 << "/" << visited_p99 << "/" << visited_max << std::endl;

        if (per_query_csv.is_open())
        {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(3);
            for (uint32_t qi = 0; qi < query_num; qi++)
            {
                stats[qi].recall_match_count = compute_recall_matches(qi, test_id);
                oss << qi << "," << L << "," << optimized_beamwidth << "," << stats[qi].thread_id << ","
                    << stats[qi].total_us << "," << stats[qi].io_us << "," << stats[qi].cpu_us << ","
                    << stats[qi].sort_us << "," << stats[qi].reorder_cpu_us << ","
                    << stats[qi].n_ios << "," << stats[qi].n_4k << "," << stats[qi].n_8k << "," << stats[qi].n_12k
                    << "," << stats[qi].n_16k << "," << stats[qi].n_cache_hits << "," << stats[qi].n_hops << ","
                    << stats[qi].visited_nodes << "," << stats[qi].recall_match_count << "\n";
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
        csv_stream << "L,beamwidth,qps,mean_latency_us,latency_p999_us,mean_ios,mean_io_us,mean_cpu_us,mean_sort_us,mean_reorder_cpu_us,recall,"
                   << "hop_mean,hop_p50,hop_p90,hop_p95,hop_p99,hop_max,"
                   << "visited_mean,visited_p50,visited_p90,visited_p95,visited_p99,visited_max\n";
        csv_stream << std::fixed << std::setprecision(3);
        for (const auto &row : stats_summary)
        {
            csv_stream << row.L << "," << row.beamwidth << "," << row.qps << "," << row.mean_latency << ","
                       << row.latency_999 << "," << row.mean_ios << "," << row.mean_io_us << "," << row.mean_cpu_us
                       << "," << row.mean_sort_us << "," << row.mean_reorder_cpu_us << ",";
            if (row.has_recall)
            {
                csv_stream << row.recall;
            }
            csv_stream << "," << row.hop_mean << "," << row.hop_p50 << "," << row.hop_p90 << "," << row.hop_p95 << ","
                       << row.hop_p99 << "," << row.hop_max << "," << row.visited_mean << "," << row.visited_p50 << ","
                       << row.visited_p90 << "," << row.visited_p95 << "," << row.visited_p99 << "," << row.visited_max
                       << "\n";
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
