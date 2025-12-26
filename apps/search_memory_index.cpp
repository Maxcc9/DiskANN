// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <unordered_set>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"
#include "percentile_stats.h"
#include "search_stats.h"

namespace po = boost::program_options;

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below,
                        const std::string &stats_csv_path = "")
{
    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

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

    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .is_dynamic_index(dynamic)
                      .is_enable_tags(tags)
                      .is_concurrent_consolidate(false)
                      .is_pq_dist_build(false)
                      .is_use_opq(false)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;

    if (metric == diskann::FAST_L2)
        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<diskann::QueryStats> stats(query_num);
    std::vector<diskann::MemoryStatRow> stats_summary;
    std::string per_query_csv_path =
        stats_csv_path.empty() ? result_path_prefix + "_query_stats.csv" : stats_csv_path;
    std::ofstream per_query_csv(per_query_csv_path, std::ios::out | std::ios::trunc);
    if (!per_query_csv.is_open())
    {
        diskann::cerr << "Failed to open per-query stats csv file: " << per_query_csv_path << std::endl;
    }
    else
    {
        per_query_csv << "query_id,L,beamwidth,thread_id,total_us,io_us,cpu_us,n_ios,read_size,n_cmps,n_cache_hits,"
                      << "n_hops,visited_nodes,recall_match_count\n";
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

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::fill(stats.begin(), stats.end(), diskann::QueryStats());
        std::vector<T *> res = std::vector<T *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            stats[i].thread_id = (unsigned)omp_get_thread_num();
            auto qs = std::chrono::high_resolution_clock::now();
            diskann::QueryStats *qstat = stats.data() + i;
            if (filtered_search && !tags)
            {
                std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                auto retval = index->search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                         query_result_ids[test_id].data() + i * recall_at,
                                                         query_result_dists[test_id].data() + i * recall_at, qstat);
            }
            else if (metric == diskann::FAST_L2)
            {
                index->search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                    query_result_ids[test_id].data() + i * recall_at);
            }
            else if (tags)
            {
                if (!filtered_search)
                {
                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, false, "", qstat);
                }
                else
                {
                    std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, true, raw_filter,
                                            qstat);
                }

                for (int64_t r = 0; r < (int64_t)recall_at; r++)
                {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            }
            else
            {
                index->search(query + i * query_aligned_dim, recall_at, L,
                              query_result_ids[test_id].data() + i * recall_at, nullptr, qstat);
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            float latency_us = (float)(diff.count() * 1000000);
            qstat->total_us = latency_us;
            qstat->cpu_us = latency_us;
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        auto mean_latency = diskann::get_mean_stats<float>(
            stats.data(), query_num, [](const diskann::QueryStats &s) { return s.total_us; });
        auto latency_999 = diskann::get_percentile_stats<float>(
            stats.data(), query_num, 0.999f, [](const diskann::QueryStats &s) { return s.total_us; });
        auto mean_cmps = diskann::get_mean_stats<uint32_t>(
            stats.data(), query_num, [](const diskann::QueryStats &s) { return s.n_cmps; });
        auto hop_mean = diskann::get_mean_stats<uint32_t>(
            stats.data(), query_num, [](const diskann::QueryStats &s) { return s.n_hops; });
        auto hop_p50 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.5f, [](const diskann::QueryStats &s) { return s.n_hops; });
        auto hop_p90 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.9f, [](const diskann::QueryStats &s) { return s.n_hops; });
        auto hop_p95 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.95f, [](const diskann::QueryStats &s) { return s.n_hops; });
        auto hop_p99 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.99f, [](const diskann::QueryStats &s) { return s.n_hops; });
        auto hop_max = diskann::get_max_stats<uint32_t>(
            stats.data(), query_num, [](const diskann::QueryStats &s) { return s.n_hops; });

        auto visited_mean = diskann::get_mean_stats<uint32_t>(
            stats.data(), query_num, [](const diskann::QueryStats &s) { return s.visited_nodes; });
        auto visited_p50 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.5f, [](const diskann::QueryStats &s) { return s.visited_nodes; });
        auto visited_p90 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.9f, [](const diskann::QueryStats &s) { return s.visited_nodes; });
        auto visited_p95 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.95f, [](const diskann::QueryStats &s) { return s.visited_nodes; });
        auto visited_p99 = diskann::get_percentile_stats<uint32_t>(
            stats.data(), query_num, 0.99f, [](const diskann::QueryStats &s) { return s.visited_nodes; });
        auto visited_max = diskann::get_max_stats<uint32_t>(
            stats.data(), query_num, [](const diskann::QueryStats &s) { return s.visited_nodes; });

        diskann::MemoryStatRow row;
        row.L = L;
        row.qps = displayed_qps;
        row.mean_latency = mean_latency;
        row.latency_999 = latency_999;
        row.mean_cmps = mean_cmps;
        if (!recalls.empty())
            row.recall = recalls.back();
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

        if (tags && !filtered_search)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_999;
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << (float)mean_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15) << (float)latency_999;
        }
        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;

        std::cout << "    HopCount mean/median/p90/p95/p99/max: " << hop_mean << "/" << hop_p50 << "/" << hop_p90 << "/"
                  << hop_p95 << "/" << hop_p99 << "/" << hop_max << std::endl;
        std::cout << "    VisitedNodes mean/median/p90/p95/p99/max: " << visited_mean << "/" << visited_p50 << "/"
                  << visited_p90 << "/" << visited_p95 << "/" << visited_p99 << "/" << visited_max << std::endl;

        if (per_query_csv.is_open())
        {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(4);
            for (uint32_t qi = 0; qi < query_num; qi++)
            {
                stats[qi].recall_match_count = compute_recall_matches(qi, test_id);
                oss << qi << "," << L << "," << L << "," << stats[qi].thread_id << "," << stats[qi].total_us << ","
                    << stats[qi].io_us << "," << stats[qi].cpu_us << "," << stats[qi].n_ios << ","
                    << stats[qi].read_size << "," << stats[qi].n_cmps << "," << stats[qi].n_cache_hits << ","
                    << stats[qi].n_hops << "," << stats[qi].visited_nodes << "," << stats[qi].recall_match_count
                    << "\n";
            }
            per_query_csv << oss.str();
        }
    }

    std::string csv_path = result_path_prefix + "_summary_stats.csv";
    std::ofstream csv_stream(csv_path, std::ios::out | std::ios::trunc);
    if (!csv_stream.is_open())
    {
        diskann::cerr << "Failed to open stats csv file: " << csv_path << std::endl;
    }
    else
    {
        csv_stream << "L,qps,mean_latency_us,latency_p999_us,mean_cmps,recall,"
                   << "hop_mean,hop_p50,hop_p90,hop_p95,hop_p99,hop_max,"
                   << "visited_mean,visited_p50,visited_p90,visited_p95,visited_p99,visited_max\n";
        csv_stream << std::fixed << std::setprecision(4);
        for (const auto &row : stats_summary)
        {
            csv_stream << row.L << "," << row.qps << "," << row.mean_latency << "," << row.latency_999 << ","
                       << row.mean_cmps << ",";
            if (calc_recall_flag)
            {
                csv_stream << row.recall;
            }
            csv_stream << "," << row.hop_mean << "," << row.hop_p50 << "," << row.hop_p90 << "," << row.hop_p95 << ","
                       << row.hop_p99 << "," << row.hop_max << "," << row.visited_mean << "," << row.visited_p50 << ","
                       << row.visited_p90 << "," << row.visited_p95 << "," << row.visited_p99 << "," << row.visited_max
                       << "\n";
        }
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }

    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, filter_label, label_type,
        query_filters_file, stats_csv_path;
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
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
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()(
            "stats_csv_path", po::value<std::string>(&stats_csv_path)->default_value(std::string("")),
            "Path to write per-query stats (CSV) for spreadsheet analysis. Defaults to <result_path>_query_stats.csv");

        // Output controls
        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified "
                                      "recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in "
                                      "the output table");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs).add(output_controls);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
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
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    if (dynamic && not tags)
    {
        std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
        return -1;
    }

    if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0)
    {
        std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
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
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, fail_if_recall_below, stats_csv_path);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters, fail_if_recall_below, stats_csv_path);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float, uint16_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                            num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                            show_qps_per_thread, query_filters, fail_if_recall_below,
                                                            stats_csv_path);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                   num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                   show_qps_per_thread, query_filters, fail_if_recall_below,
                                                   stats_csv_path);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, query_filters, fail_if_recall_below,
                                                    stats_csv_path);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, query_filters, fail_if_recall_below,
                                                  stats_csv_path);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
