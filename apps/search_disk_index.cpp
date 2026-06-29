// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include <limits>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

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
                      const float et_theta = std::numeric_limits<float>::max(), const float et_dk = 0.0f,
                      const uint32_t hop_budget = std::numeric_limits<uint32_t>::max(),
                      const float et_sat_gamma = 1.0f, const uint32_t et_sat_delta = 0,
                      const double neighbor_cache_gb = 0.0,
                      const float et_theta_exact = std::numeric_limits<float>::max(),
                      const uint32_t et_ref_rank = 0, const uint32_t et_min_hops = 0,
                      const bool oracle_hops = false, const uint32_t et_conv_delta = 0,
                      const uint32_t et_conv_width = 0, const bool dump_features = false,
                      const std::string &self_ids_file = "",
                      const float et_verify_alpha = std::numeric_limits<float>::max(),
                      const uint32_t et_verify_patience = 1,
                      const bool et_exact_led = false,
                      const uint32_t et_exact_patience = 1,
                      const float et_exact_beta = std::numeric_limits<float>::max())
{
    // Leave-one-out self-exclusion ids (one per query; base-vector-as-query training).
    std::vector<uint32_t> self_ids;
    if (!self_ids_file.empty() && file_exists(self_ids_file))
    {
        std::ifstream sf(self_ids_file, std::ios::binary);
        int32_t sn = 0, sd = 0; sf.read((char *)&sn, 4); sf.read((char *)&sd, 4);
        self_ids.resize((size_t)sn * sd);
        sf.read((char *)self_ids.data(), (size_t)sn * sd * sizeof(uint32_t));
    }
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
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    if (neighbor_cache_gb > 0.0)
    {
        size_t capacity_bytes = static_cast<size_t>(neighbor_cache_gb * 1024.0 * 1024.0 * 1024.0);
        _pFlashIndex->init_bounded_neighbor_cache(capacity_bytes);
        diskann::cout << "[BNC] Bounded neighbor cache enabled: " << neighbor_cache_gb << " GB" << std::endl;
    }

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
                                             warmup_result_dists.data() + (i * 1), 4, et_theta, hop_budget,
                                             et_sat_gamma, et_sat_delta);
        }
        diskann::cout << "..done" << std::endl;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "P50 Latency" << std::setw(16) << "P99 Latency"
                  << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
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

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

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
        std::vector<std::vector<float>> feat_logs(dump_features ? query_num : 0);

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {
                // Use the full overload to support et_theta_exact (guaranteed ET via exact kth).
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L,
                    query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at),
                    optimized_beamwidth, false, (LabelT)0,
                    std::numeric_limits<uint32_t>::max(),
                    et_theta, hop_budget, et_sat_gamma, et_sat_delta,
                    et_theta_exact, et_conv_delta,
                    use_reorder_data, stats + i,
                    ((oracle_hops || dump_features) && gt_ids != nullptr) ? gt_ids + (size_t)i * gt_dim : nullptr,
                    et_ref_rank, et_min_hops, et_conv_width,
                    dump_features ? &feat_logs[i] : nullptr,
                    self_ids.empty() ? std::numeric_limits<uint32_t>::max() : self_ids[i],
                    et_verify_alpha, et_verify_patience, et_exact_led, et_exact_patience, et_exact_beta);
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
                    query + (i * query_aligned_dim), recall_at, L,
                    query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at),
                    optimized_beamwidth, true, label_for_search,
                    std::numeric_limits<uint32_t>::max(),
                    et_theta, hop_budget, et_sat_gamma, et_sat_delta,
                    et_theta_exact, et_conv_delta,
                    use_reorder_data, stats + i, nullptr, et_ref_rank, et_min_hops, et_conv_width);
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

        auto latency_50 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.500, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.990, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_hops = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                           [](const diskann::QueryStats &stats) { return stats.n_beam_hops; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.io_us; });

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_50 << std::setw(16) << latency_99
                      << std::setw(16) << latency_999 << std::setw(16) << mean_ios << std::setw(12) << mean_hops
                      << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;

        // Save per-query hop counts for oracle hop post-hoc analysis
        {
            std::string hops_path = result_output_prefix + "_" + std::to_string(L) + "_hops_uint32.bin";
            std::vector<uint32_t> per_query_hops(query_num);
            for (size_t qi = 0; qi < query_num; qi++)
                per_query_hops[qi] = stats[qi].n_beam_hops;
            diskann::save_bin<uint32_t>(hops_path, per_query_hops.data(), query_num, 1);

            if (oracle_hops)
            {
                std::string opath = result_output_prefix + "_" + std::to_string(L) + "_oraclehops_uint32.bin";
                std::vector<uint32_t> per_query_oracle(query_num);
                for (size_t qi = 0; qi < query_num; qi++)
                    per_query_oracle[qi] = stats[qi].oracle_hops; // 0 = never reached full recall
                diskann::save_bin<uint32_t>(opath, per_query_oracle.data(), query_num, 1);
            }

            if (dump_features)
            {
                // Combined per-hop feature file: [int32 n_queries][int32 feat_dim]
                // then per query: [int32 n_hops][n_hops*feat_dim floats].
                std::string fpath = result_output_prefix + "_" + std::to_string(L) + "_features.bin";
                std::ofstream ff(fpath, std::ios::binary);
                int32_t nq = (int32_t)query_num, fdim = 17;
                ff.write((char *)&nq, 4);
                ff.write((char *)&fdim, 4);
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    int32_t nh = (int32_t)(feat_logs[qi].size() / fdim);
                    ff.write((char *)&nh, 4);
                    ff.write((char *)feat_logs[qi].data(), feat_logs[qi].size() * sizeof(float));
                }
                ff.close();
                diskann::cout << "Wrote features: " << fpath << std::endl;
            }
        }

        delete[] stats;
    }

    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
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
        label_type, query_filters_file;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;
    float et_theta = 1e9f;
    float et_dk = 0.0f;
    uint32_t hop_budget = std::numeric_limits<uint32_t>::max();
    float et_sat_gamma = 1.0f;
    uint32_t et_sat_delta = 0;
    double neighbor_cache_gb = 0.0;
    float et_theta_exact = 1e9f;
    uint32_t et_ref_rank = 0;
    uint32_t et_min_hops = 0;
    bool oracle_hops = false;
    uint32_t et_conv_delta = 0;
    uint32_t et_conv_width = 0;
    bool dump_features = false;
    std::string self_ids_file;
    float et_verify_alpha = std::numeric_limits<float>::max();
    uint32_t et_verify_patience = 1;
    bool et_exact_led = false;
    uint32_t et_exact_patience = 1;
    float et_exact_beta = std::numeric_limits<float>::max();

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
        optional_configs.add_options()("et_theta", po::value<float>(&et_theta)->default_value(1e9f),
                                       "Early termination theta. Default value: 1e9");
        optional_configs.add_options()("et_dk", po::value<float>(&et_dk)->default_value(0.0f),
                                       "Early termination additive margin. Default value: 0.0");
        optional_configs.add_options()(
            "hop_budget", po::value<uint32_t>(&hop_budget)->default_value(std::numeric_limits<uint32_t>::max()),
            "Early termination hop budget. Default value: uint32::max()");
        optional_configs.add_options()("et_sat_gamma", po::value<float>(&et_sat_gamma)->default_value(1.0f),
                                       "Top-k saturation gamma threshold [0,1]. Default 1.0 = disabled.");
        optional_configs.add_options()("et_sat_delta", po::value<uint32_t>(&et_sat_delta)->default_value(0),
                                       "Top-k saturation consecutive stable hops needed. Default 0 = disabled.");
        optional_configs.add_options()("neighbor_cache_gb",
                                       po::value<double>(&neighbor_cache_gb)->default_value(0.0),
                                       "Bounded neighbor cache size in GB (0 = disabled). Shared across queries.");
        optional_configs.add_options()("et_theta_exact", po::value<float>(&et_theta_exact)->default_value(1e9f),
                                       "Guaranteed ET: stop when best_unexp_pq > kth_exact * theta. Default 1e9 (disabled).");
        optional_configs.add_options()("et_ref_rank", po::value<uint32_t>(&et_ref_rank)->default_value(0),
                                       "θ-ET reference rank: compare best unexpanded vs the ref_rank-th candidate "
                                       "instead of the k-th. 0 = use k (classic). Larger = wider attention window "
                                       "toward L (more conservative ET).");
        optional_configs.add_options()("et_min_hops", po::value<uint32_t>(&et_min_hops)->default_value(0),
                                       "ET grace period: θ-ET only activates after a query has run this many "
                                       "hops. Set to the profiled P50/P75 hop count so only the long-running "
                                       "tail queries get terminated. 0 = no grace (ET from hop 0).");
        optional_configs.add_options()("oracle_hops", po::bool_switch(&oracle_hops),
                                       "Analysis only: record per-query oracle hop (earliest hop whose exact "
                                       "top-K matches ground truth) and dump to *_oraclehops_uint32.bin. "
                                       "Requires gt_file. O(N log N)/hop overhead.");
        optional_configs.add_options()("et_conv_delta", po::value<uint32_t>(&et_conv_delta)->default_value(0),
                                       "Patience ET: terminate when the exact top-K set is unchanged for this "
                                       "many consecutive hops. 0 = disabled.");
        optional_configs.add_options()("et_conv_width", po::value<uint32_t>(&et_conv_width)->default_value(0),
                                       "Patience convergence window: track stability of the top-M set "
                                       "(M=et_conv_width) instead of top-K. 0 = use K. Wider M = more reliable "
                                       "converged signal (catches true NNs still rising through ranks K..M).");
        optional_configs.add_options()("self_ids_file",
                                       po::value<std::string>(&self_ids_file)->default_value(std::string("")),
                                       "Leave-one-out: bin [n][1][n uint32] of each query's own base id to "
                                       "exclude from results (base-vector-as-query training). Empty = off.");
        optional_configs.add_options()("dump_features", po::bool_switch(&dump_features),
                                       "Analysis only: dump per-hop raw features (8/hop) for learned-ET training "
                                       "to *_features.bin. Requires gt_file. Heavy (sorts full_retset each hop).");
        optional_configs.add_options()("et_verify_alpha",
                                       po::value<float>(&et_verify_alpha)->default_value(std::numeric_limits<float>::max()),
                                       "Predict-then-verify ET: layer-2 exact alpha (FLT_MAX = disabled). θ-ET (et_theta) "
                                       "becomes a PQ predictor; stop only if min(this-hop exact) > kth_exact * alpha.");
        optional_configs.add_options()("et_verify_patience",
                                       po::value<uint32_t>(&et_verify_patience)->default_value(1),
                                       "Predict-then-verify ET: consecutive hops where both layers agree before "
                                       "stopping (1 = stop on first trigger).");
        optional_configs.add_options()("et_exact_led", po::bool_switch(&et_exact_led),
                                       "ET order = exact-led: decide at hop top using PREVIOUS hop's exact "
                                       "(free, no verify-beam) then PQ look-ahead. Needs et_verify_alpha & et_theta.");
        optional_configs.add_options()("et_exact_patience",
                                       po::value<uint32_t>(&et_exact_patience)->default_value(1),
                                       "Dual-rail ET patience: consecutive hops where BOTH rails fire before stopping. "
                                       "Use et_conv_width=2k. 1 = stop on first joint trigger.");
        optional_configs.add_options()("et_exact_beta",
                                       po::value<float>(&et_exact_beta)->default_value(std::numeric_limits<float>::max()),
                                       "Dual-rail ET: exact-convergence rail beta (FLT_MAX=disabled). Fires when "
                                       "k2-th EXACT < (best EXACT)*beta (top collapsed to plateau). PQ rail uses et_theta_exact (=alpha).");

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
                    et_theta, et_dk, hop_budget, et_sat_gamma, et_sat_delta, neighbor_cache_gb, et_theta_exact,
                    et_ref_rank, et_min_hops);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    et_theta, et_dk, hop_budget, et_sat_gamma, et_sat_delta, neighbor_cache_gb, et_theta_exact,
                    et_ref_rank, et_min_hops);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    et_theta, et_dk, hop_budget, et_sat_gamma, et_sat_delta, neighbor_cache_gb, et_theta_exact,
                    et_ref_rank, et_min_hops);
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
                                                fail_if_recall_below, query_filters, use_reorder_data, et_theta,
                                                et_dk, hop_budget, et_sat_gamma, et_sat_delta, neighbor_cache_gb,
                                                et_theta_exact, et_ref_rank, et_min_hops, oracle_hops, et_conv_delta, et_conv_width,
                    dump_features, self_ids_file, et_verify_alpha, et_verify_patience, et_exact_led, et_exact_patience, et_exact_beta);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data, et_theta,
                                                 et_dk, hop_budget, et_sat_gamma, et_sat_delta, neighbor_cache_gb,
                                                 et_theta_exact, et_ref_rank, et_min_hops, oracle_hops, et_conv_delta, et_conv_width,
                    dump_features, self_ids_file, et_verify_alpha, et_verify_patience, et_exact_led, et_exact_patience, et_exact_beta);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data, et_theta,
                                                  et_dk, hop_budget, et_sat_gamma, et_sat_delta, neighbor_cache_gb,
                                                 et_theta_exact, et_ref_rank, et_min_hops, oracle_hops, et_conv_delta, et_conv_width,
                    dump_features, self_ids_file, et_verify_alpha, et_verify_patience, et_exact_led, et_exact_patience, et_exact_beta);
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
