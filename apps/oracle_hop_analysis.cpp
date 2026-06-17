// oracle_hop_analysis: measure oracle hop count on SIFT100M
//
// For each query, runs search THREE times:
//   1. Full search (no ET)  → final result IDs, total_hops
//   2. Oracle replay        → stability_hop: first hop where all K final-result IDs are in retset
//   3. ET search (theta)    → et_hops, et_recall
//
// "Stability hop" (oracle) = first hop after which the final output is already determined.
// Any search work beyond that hop is provably wasted.
// θ-ET savings vs oracle savings shows ET efficiency.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "disk_utils.h"
#include "linux_aligned_file_reader.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"

namespace po = boost::program_options;

template <typename T>
int run_analysis(const std::string &index_prefix, const std::string &query_file, const std::string &gt_file,
                 const uint64_t K, const uint64_t L, const uint64_t W, const uint32_t T_threads,
                 const float et_theta, const float et_theta_exact,
                 const uint32_t et_conv_delta,
                 const float sat_gamma, const uint32_t sat_delta,
                 const size_t bnc_bytes,
                 const std::string &out_csv, const std::string &out_summary)
{
    // ── Load query ──────────────────────────────────────────────────────
    T *queries = nullptr;
    size_t nq, qdim, qdim_aligned;
    diskann::load_aligned_bin<T>(query_file, queries, nq, qdim, qdim_aligned);
    std::cout << "Queries: " << nq << " × " << qdim << " (aligned " << qdim_aligned << ")\n";

    // ── Load ground truth ────────────────────────────────────────────────
    uint32_t *gt_ids = nullptr;
    float    *gt_dists = nullptr;
    size_t    gt_num, gt_dim;
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    std::cout << "Ground truth: " << gt_num << " queries × " << gt_dim << " NNs\n";
    if (gt_dim < K)
    {
        std::cerr << "ERROR: gt_dim=" << gt_dim << " < K=" << K << "\n";
        return -1;
    }

    // ── Load index ───────────────────────────────────────────────────────
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<T>> index(
        new diskann::PQFlashIndex<T>(reader, diskann::Metric::L2));
    int ret = index->load(T_threads, index_prefix.c_str());
    if (ret != 0) { std::cerr << "ERROR: index load failed\n"; return -1; }

    if (bnc_bytes > 0)
    {
        std::cout << "Initialising BNC cache: " << bnc_bytes / 1024 / 1024 << " MB...\n";
        index->init_bounded_neighbor_cache(bnc_bytes);
        std::cout << "BNC ready.\n";
    }
    std::cout << "Index loaded. Running oracle hop analysis...\n";

    // ── Per-query buffers ────────────────────────────────────────────────
    std::vector<uint64_t> res_ids_full(K), res_ids_et(K);
    std::vector<float>    res_dists_full(K), res_dists_et(K);

    std::vector<uint32_t> total_hops_vec(nq), oracle_hops_vec(nq), et_hops_vec(nq);
    std::vector<uint32_t> et_recall_correct(nq);

    diskann::QueryStats stats_full, stats_oracle, stats_et;
    std::vector<uint32_t> oracle_ids(K);  // final result IDs cast to uint32_t

    for (size_t qi = 0; qi < nq; qi++)
    {
        const T        *q  = queries + qi * qdim_aligned;
        const uint32_t *gt = gt_ids + qi * gt_dim;

        // ── Pass 1: full search → get final result IDs and total hops ──
        stats_full = diskann::QueryStats{};
        index->cached_beam_search(q, K, L, res_ids_full.data(), res_dists_full.data(), W,
                                  false, (uint32_t)0,
                                  std::numeric_limits<uint32_t>::max(),
                                  std::numeric_limits<float>::max(),   // et_theta: disabled
                                  std::numeric_limits<uint32_t>::max(),
                                  1.0f, 0,
                                  std::numeric_limits<float>::max(),   // et_theta_exact: disabled
                                  0,                                   // et_conv_delta: disabled
                                  false,
                                  &stats_full, nullptr);

        total_hops_vec[qi] = stats_full.n_beam_hops;
        for (uint64_t k = 0; k < K; k++)
            oracle_ids[k] = (uint32_t)res_ids_full[k];

        // ── Pass 2: oracle replay — first hop where full_retset top-K = oracle_ids ──
        stats_oracle = diskann::QueryStats{};
        index->cached_beam_search(q, K, L, res_ids_full.data(), res_dists_full.data(), W,
                                  false, (uint32_t)0,
                                  std::numeric_limits<uint32_t>::max(),
                                  std::numeric_limits<float>::max(),   // et_theta: disabled
                                  std::numeric_limits<uint32_t>::max(),
                                  1.0f, 0,
                                  std::numeric_limits<float>::max(),   // et_theta_exact: disabled
                                  0,                                   // et_conv_delta: disabled
                                  false,
                                  &stats_oracle, oracle_ids.data());

        oracle_hops_vec[qi] = stats_oracle.oracle_hops;

        // ── Pass 3: ET search (theta-ET / exact-kth-ET / sat-ET / conv-ET) ────
        stats_et = diskann::QueryStats{};
        index->cached_beam_search(q, K, L, res_ids_et.data(), res_dists_et.data(), W,
                                  false, (uint32_t)0,
                                  std::numeric_limits<uint32_t>::max(),
                                  et_theta,
                                  std::numeric_limits<uint32_t>::max(),
                                  sat_gamma, sat_delta,
                                  et_theta_exact, et_conv_delta,
                                  false,
                                  &stats_et, nullptr);

        et_hops_vec[qi] = stats_et.n_beam_hops;

        // ET recall: compare ET output against GT
        uint32_t correct = 0;
        for (uint64_t k = 0; k < K; k++)
            for (uint64_t g = 0; g < K; g++)
                if (res_ids_et[k] == gt[g]) { correct++; break; }
        et_recall_correct[qi] = correct;

        if ((qi + 1) % 1000 == 0)
            std::cout << "  " << (qi + 1) << "/" << nq << " queries done\n";
    }

    // ── Compute aggregate statistics ─────────────────────────────────────
    auto mean_f = [](const std::vector<uint32_t> &v) {
        return (double)std::accumulate(v.begin(), v.end(), 0ULL) / v.size();
    };
    auto percentile = [](std::vector<uint32_t> v, double p) -> double {
        std::sort(v.begin(), v.end());
        return v[(size_t)(p * v.size())];
    };

    // oracle_hops == 0 should not happen (stability must be reached if search completes)
    size_t oracle_found = 0;
    for (auto h : oracle_hops_vec) if (h > 0) oracle_found++;

    double mean_total  = mean_f(total_hops_vec);
    double mean_oracle = mean_f(oracle_hops_vec);
    double mean_et     = mean_f(et_hops_vec);
    double mean_et_recall = (double)std::accumulate(et_recall_correct.begin(), et_recall_correct.end(), 0ULL) / (nq * K);

    std::cout << "\n";
    std::cout << "=======================================================\n";
    std::cout << "Oracle Hop Analysis — SIFT100M, L=" << L << " W=" << W << " K=" << K << "\n";
    std::cout << "=======================================================\n";
    std::cout << "  Queries: " << nq << "  oracle_found: " << oracle_found
              << " (" << 100.0 * oracle_found / nq << "%)\n\n";

    std::cout << "  Mean total hops:   " << mean_total
              << "  P50=" << percentile(total_hops_vec, 0.50)
              << "  P99=" << percentile(total_hops_vec, 0.99) << "\n";
    std::cout << "  Mean oracle hops:  " << mean_oracle
              << "  P50=" << percentile(oracle_hops_vec, 0.50)
              << "  P99=" << percentile(oracle_hops_vec, 0.99) << "\n";
    std::cout << "  Mean ET hops:      " << mean_et
              << "  P50=" << percentile(et_hops_vec, 0.50)
              << "  P99=" << percentile(et_hops_vec, 0.99) << "\n";
    std::cout << "  ET recall@K:       " << mean_et_recall * 100 << "%\n\n";

    double waste_oracle = (mean_total - mean_oracle) / mean_total * 100.0;
    double waste_et     = (mean_total - mean_et)     / mean_total * 100.0;
    std::cout << "  Waste fraction (oracle): " << waste_oracle << "%\n";
    std::cout << "  Waste fraction (ET):     " << waste_et     << "%\n\n";

    // CDF of oracle hops
    std::cout << "  CDF: % queries with oracle_hops ≤ H:\n";
    for (uint32_t h : {1u,2u,3u,5u,8u,10u,15u,20u,25u,30u,40u,50u})
    {
        size_t cnt = 0;
        for (auto v : oracle_hops_vec) if (v > 0 && v <= h) cnt++;
        std::cout << "    ≤" << h << ": " << 100.0 * cnt / oracle_found << "%\n";
    }
    std::cout << "  CDF: % queries with ET_hops ≤ H:\n";
    for (uint32_t h : {10u,20u,30u,40u,50u,80u,100u,120u,150u})
    {
        size_t cnt = 0;
        for (auto v : et_hops_vec) if (v <= h) cnt++;
        std::cout << "    ≤" << h << ": " << 100.0 * cnt / nq << "%\n";
    }
    std::cout << "=======================================================\n";

    // ── Save summary CSV ─────────────────────────────────────────────────
    {
        std::ofstream sf(out_summary);
        sf << "metric,mean,p25,p50,p75,p99\n";
        auto pct = [&](const std::vector<uint32_t> &v, double p) { return percentile(v, p); };
        sf << "total_hops,"  << mean_total  << "," << pct(total_hops_vec,0.25)  << ","
           << pct(total_hops_vec,0.50)  << "," << pct(total_hops_vec,0.75)  << "," << pct(total_hops_vec,0.99)  << "\n";
        sf << "oracle_hops," << mean_oracle << "," << pct(oracle_hops_vec,0.25) << ","
           << pct(oracle_hops_vec,0.50) << "," << pct(oracle_hops_vec,0.75) << "," << pct(oracle_hops_vec,0.99) << "\n";
        sf << "et_hops,"     << mean_et     << "," << pct(et_hops_vec,0.25)     << ","
           << pct(et_hops_vec,0.50)     << "," << pct(et_hops_vec,0.75)     << "," << pct(et_hops_vec,0.99)     << "\n";
        sf << "waste_oracle_pct,"  << waste_oracle  << ",,,,\n";
        sf << "waste_et_pct,"      << waste_et      << ",,,,\n";
        sf << "et_recall_pct,"     << mean_et_recall*100 << ",,,,\n";
        std::cout << "Summary saved: " << out_summary << "\n";
    }

    // ── Save per-query CSV ───────────────────────────────────────────────
    {
        std::ofstream qf(out_csv);
        qf << "query_id,total_hops,oracle_hops,et_hops,waste_oracle_pct,waste_et_pct,et_recall_k\n";
        for (size_t qi = 0; qi < nq; qi++)
        {
            double wo = oracle_hops_vec[qi] > 0 ?
                        100.0 * (total_hops_vec[qi] - oracle_hops_vec[qi]) / std::max(total_hops_vec[qi], 1u) : -1;
            double we = 100.0 * (total_hops_vec[qi] - et_hops_vec[qi]) / std::max(total_hops_vec[qi], 1u);
            qf << qi << "," << total_hops_vec[qi] << "," << oracle_hops_vec[qi] << ","
               << et_hops_vec[qi] << "," << wo << "," << we << "," << et_recall_correct[qi] << "\n";
        }
        std::cout << "Per-query CSV saved: " << out_csv << "\n";
    }

    diskann::aligned_free(queries);
    delete[] gt_ids;
    delete[] gt_dists;
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, index_prefix, query_file, gt_file, out_csv, out_summary;
    uint64_t    K, L, W, T_threads, bnc_mb, sat_delta, et_conv_delta_u64;
    float       et_theta, et_theta_exact, sat_gamma;

    po::options_description desc("oracle_hop_analysis options");
    desc.add_options()
        ("help,h",         "print help")
        ("data_type",      po::value<std::string>(&data_type)->required(),    "float/uint8/int8")
        ("index_path_prefix", po::value<std::string>(&index_prefix)->required(), "index prefix")
        ("query_file",     po::value<std::string>(&query_file)->required(),   "query .bin")
        ("gt_file",        po::value<std::string>(&gt_file)->required(),      "ground truth .bin")
        ("K",              po::value<uint64_t>(&K)->default_value(10),        "recall@K")
        ("L",              po::value<uint64_t>(&L)->default_value(150),       "search list size")
        ("W",              po::value<uint64_t>(&W)->default_value(8),         "beam width")
        ("num_threads",    po::value<uint64_t>(&T_threads)->default_value(1), "threads (use 1 for determinism)")
        ("et_theta",       po::value<float>(&et_theta)->default_value(std::numeric_limits<float>::max()), "theta-ET: stop when best_unexp_pq > kth_pq * theta (default: disabled)")
        ("et_theta_exact", po::value<float>(&et_theta_exact)->default_value(std::numeric_limits<float>::max()), "exact-kth ET: stop when best_unexp_pq > kth_exact * theta (default: disabled)")
        ("sat_gamma",      po::value<float>(&sat_gamma)->default_value(1.0f), "saturation overlap fraction [0,1] (default: 1.0=all match)")
        ("sat_delta",      po::value<uint64_t>(&sat_delta)->default_value(0), "saturation patience hops (0=disabled)")
        ("et_conv_delta",  po::value<uint64_t>(&et_conv_delta_u64)->default_value(0), "exact-conv ET: stop when full_retset top-K unchanged for N hops (0=disabled)")
        ("bnc_mb",         po::value<uint64_t>(&bnc_mb)->default_value(0),    "BNC cache size in MB (0=disabled)")
        ("out_csv",        po::value<std::string>(&out_csv)->required(),      "per-query output CSV")
        ("out_summary",    po::value<std::string>(&out_summary)->required(),  "summary CSV");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc; return 0; }
    po::notify(vm);

    size_t bnc_bytes = (size_t)bnc_mb * 1024 * 1024;

    if (data_type == "uint8")
        return run_analysis<uint8_t>(index_prefix, query_file, gt_file, K, L, W, T_threads, et_theta, et_theta_exact, (uint32_t)et_conv_delta_u64, sat_gamma, (uint32_t)sat_delta, bnc_bytes, out_csv, out_summary);
    else if (data_type == "float")
        return run_analysis<float>(index_prefix, query_file, gt_file, K, L, W, T_threads, et_theta, et_theta_exact, (uint32_t)et_conv_delta_u64, sat_gamma, (uint32_t)sat_delta, bnc_bytes, out_csv, out_summary);
    else if (data_type == "int8")
        return run_analysis<int8_t>(index_prefix, query_file, gt_file, K, L, W, T_threads, et_theta, et_theta_exact, (uint32_t)et_conv_delta_u64, sat_gamma, (uint32_t)sat_delta, bnc_bytes, out_csv, out_summary);
    else
    {
        std::cerr << "Unknown data_type: " << data_type << "\n";
        return -1;
    }
}
