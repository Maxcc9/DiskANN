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
#include <sstream>
#include <vector>

#include "disk_utils.h"
#include "linux_aligned_file_reader.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"

namespace po = boost::program_options;

// One row of the ET-config grid (--grid_file). Pass-1 (full search) and pass-2 (oracle
// replay) never depend on any of these fields, so a whole grid can share one index load
// and one pass-1/pass-2 pass per query, looping only pass-3 (the cheap in-memory ET
// search) over every row -- this is what makes wide sweeps tractable on a 100M-node index.
struct EtConfig
{
    std::string label;
    float    et_theta = std::numeric_limits<float>::max();
    float    et_theta_exact = std::numeric_limits<float>::max();
    uint32_t et_conv_width = 0;
    uint32_t et_exact_patience = 1;
    uint32_t et_ref_rank = 0;
    uint32_t et_min_hops = 0;
    float    et_verify_alpha = std::numeric_limits<float>::max();
    uint32_t et_verify_patience = 1;
    bool     et_exact_led = false;
};

std::vector<EtConfig> load_grid_file(const std::string &path)
{
    std::vector<EtConfig> grid;
    std::ifstream f(path);
    if (!f) { std::cerr << "ERROR: cannot open grid_file " << path << "\n"; return grid; }
    std::string line;
    std::getline(f, line); // header, ignored (documents column order)
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tok;
        EtConfig c;
        auto next = [&]() { std::getline(ss, tok, ','); return tok; };
        c.label = next();
        c.et_theta          = next().empty() ? std::numeric_limits<float>::max() : std::stof(tok);
        c.et_theta_exact    = next().empty() ? std::numeric_limits<float>::max() : std::stof(tok);
        c.et_conv_width     = next().empty() ? 0 : (uint32_t)std::stoul(tok);
        c.et_exact_patience = next().empty() ? 1 : (uint32_t)std::stoul(tok);
        c.et_ref_rank       = next().empty() ? 0 : (uint32_t)std::stoul(tok);
        c.et_min_hops       = next().empty() ? 0 : (uint32_t)std::stoul(tok);
        c.et_verify_alpha   = next().empty() ? std::numeric_limits<float>::max() : std::stof(tok);
        c.et_verify_patience= next().empty() ? 1 : (uint32_t)std::stoul(tok);
        c.et_exact_led      = next() == "1";
        grid.push_back(c);
    }
    return grid;
}

template <typename T>
int run_analysis(const std::string &index_prefix, const std::string &query_file, const std::string &gt_file,
                 const uint64_t K, const uint64_t L, const uint64_t W, const uint32_t T_threads,
                 const float et_theta, const float et_theta_exact,
                 const uint32_t et_conv_delta,
                 const float sat_gamma, const uint32_t sat_delta,
                 const uint32_t et_conv_width, const uint32_t et_exact_patience,
                 const uint32_t et_ref_rank, const uint32_t et_min_hops,
                 const float et_verify_alpha, const uint32_t et_verify_patience,
                 const bool et_exact_led,
                 const size_t bnc_bytes,
                 const std::string &out_csv, const std::string &out_summary,
                 const std::string &grid_file)
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

    // ── Build the ET-config grid. Single-config legacy mode = a 1-row "grid" made
    // from the flat CLI et_* args, so both code paths below are unified. ──────────
    std::vector<EtConfig> grid;
    if (!grid_file.empty())
    {
        grid = load_grid_file(grid_file);
        if (grid.empty()) { std::cerr << "ERROR: empty/unreadable grid_file\n"; return -1; }
        std::cout << "Loaded " << grid.size() << " ET configs from " << grid_file << "\n";
    }
    else
    {
        EtConfig c;
        c.label = "default";
        c.et_theta = et_theta; c.et_theta_exact = et_theta_exact;
        c.et_conv_width = et_conv_width; c.et_exact_patience = et_exact_patience;
        c.et_ref_rank = et_ref_rank; c.et_min_hops = et_min_hops;
        c.et_verify_alpha = et_verify_alpha; c.et_verify_patience = et_verify_patience;
        c.et_exact_led = et_exact_led;
        grid.push_back(c);
    }
    const size_t nconf = grid.size();

    // ── Per-query buffers ────────────────────────────────────────────────
    std::vector<uint64_t> res_ids_full(K), res_ids_et(K);
    std::vector<float>    res_dists_full(K), res_dists_et(K);

    std::vector<uint32_t> total_hops_vec(nq), oracle_hops_vec(nq);
    // et_hops_vec / et_recall_correct: one row per query, one column per grid config.
    std::vector<std::vector<uint32_t>> et_hops_vec(nconf, std::vector<uint32_t>(nq));
    std::vector<std::vector<uint32_t>> et_recall_correct(nconf, std::vector<uint32_t>(nq));

    diskann::QueryStats stats_full, stats_oracle, stats_et;
    std::vector<uint32_t> oracle_ids(K);  // final result IDs cast to uint32_t

    for (size_t qi = 0; qi < nq; qi++)
    {
        const T        *q  = queries + qi * qdim_aligned;
        const uint32_t *gt = gt_ids + qi * gt_dim;

        // ── Pass 1: full search → get final result IDs and total hops (config-independent) ──
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

        // ── Pass 2: oracle replay (also config-independent) ──
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

        // ── Pass 3: ET search, once per grid config ──────────────────────────
        for (size_t ci = 0; ci < nconf; ci++)
        {
            const EtConfig &c = grid[ci];
            stats_et = diskann::QueryStats{};
            index->cached_beam_search(q, K, L, res_ids_et.data(), res_dists_et.data(), W,
                                      false, (uint32_t)0,
                                      std::numeric_limits<uint32_t>::max(),
                                      c.et_theta,
                                      std::numeric_limits<uint32_t>::max(),
                                      sat_gamma, sat_delta,
                                      c.et_theta_exact, et_conv_delta,
                                      false,
                                      &stats_et, nullptr,
                                      c.et_ref_rank, c.et_min_hops, c.et_conv_width,
                                      /*feat_log=*/nullptr,
                                      /*self_exclude_id=*/std::numeric_limits<uint32_t>::max(),
                                      c.et_verify_alpha, c.et_verify_patience,
                                      c.et_exact_led, c.et_exact_patience);

            et_hops_vec[ci][qi] = stats_et.n_beam_hops;

            uint32_t correct = 0;
            for (uint64_t k = 0; k < K; k++)
                for (uint64_t g = 0; g < K; g++)
                    if (res_ids_et[k] == gt[g]) { correct++; break; }
            et_recall_correct[ci][qi] = correct;
        }

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

    size_t oracle_found = 0;
    for (auto h : oracle_hops_vec) if (h > 0) oracle_found++;

    double mean_total  = mean_f(total_hops_vec);
    double mean_oracle = mean_f(oracle_hops_vec);
    double waste_oracle = (mean_total - mean_oracle) / mean_total * 100.0;

    std::cout << "\n=======================================================\n";
    std::cout << "Oracle Hop Analysis — L=" << L << " W=" << W << " K=" << K
              << "  (" << nconf << " ET config" << (nconf > 1 ? "s" : "") << ")\n";
    std::cout << "=======================================================\n";
    std::cout << "  Queries: " << nq << "  oracle_found: " << oracle_found
              << " (" << 100.0 * oracle_found / nq << "%)\n";
    std::cout << "  Mean total hops:   " << mean_total << "\n";
    std::cout << "  Mean oracle hops:  " << mean_oracle << "\n";
    std::cout << "  Waste fraction (oracle): " << waste_oracle << "%\n\n";

    // ── Save one combined summary CSV row per grid config ─────────────────
    {
        std::ofstream sf(out_summary);
        sf << "label,et_theta,et_theta_exact,et_conv_width,et_exact_patience,et_ref_rank,et_min_hops,"
           << "et_verify_alpha,et_verify_patience,et_exact_led,"
           << "mean_total_hops,mean_oracle_hops,mean_et_hops,waste_oracle_pct,waste_et_pct,"
           << "capture_ratio_pct,et_recall_pct\n";
        for (size_t ci = 0; ci < nconf; ci++)
        {
            const EtConfig &c = grid[ci];
            double mean_et = mean_f(et_hops_vec[ci]);
            double mean_et_recall = (double)std::accumulate(et_recall_correct[ci].begin(), et_recall_correct[ci].end(), 0ULL) / (nq * K);
            double waste_et = (mean_total - mean_et) / mean_total * 100.0;
            double capture = waste_oracle > 1e-9 ? waste_et / waste_oracle * 100.0 : 0.0;

            std::cout << "  [" << c.label << "] mean_et_hops=" << mean_et
                      << "  waste_et=" << waste_et << "%  capture=" << capture << "%"
                      << "  recall=" << mean_et_recall * 100 << "%\n";

            auto ff = [](float v) { return v == std::numeric_limits<float>::max() ? std::string("") : std::to_string(v); };
            sf << c.label << "," << ff(c.et_theta) << "," << ff(c.et_theta_exact) << ","
               << c.et_conv_width << "," << c.et_exact_patience << "," << c.et_ref_rank << "," << c.et_min_hops << ","
               << ff(c.et_verify_alpha) << "," << c.et_verify_patience << "," << (c.et_exact_led ? 1 : 0) << ","
               << mean_total << "," << mean_oracle << "," << mean_et << "," << waste_oracle << "," << waste_et << ","
               << capture << "," << mean_et_recall * 100 << "\n";
        }
        std::cout << "\nSummary saved: " << out_summary << "\n";
    }

    // ── Save per-query CSV (single-config mode only; grid mode skips this) ──
    if (nconf == 1 && !out_csv.empty())
    {
        std::ofstream qf(out_csv);
        qf << "query_id,total_hops,oracle_hops,et_hops,waste_oracle_pct,waste_et_pct,et_recall_k\n";
        for (size_t qi = 0; qi < nq; qi++)
        {
            double wo = oracle_hops_vec[qi] > 0 ?
                        100.0 * (total_hops_vec[qi] - oracle_hops_vec[qi]) / std::max(total_hops_vec[qi], 1u) : -1;
            double we = 100.0 * (total_hops_vec[qi] - et_hops_vec[0][qi]) / std::max(total_hops_vec[qi], 1u);
            qf << qi << "," << total_hops_vec[qi] << "," << oracle_hops_vec[qi] << ","
               << et_hops_vec[0][qi] << "," << wo << "," << we << "," << et_recall_correct[0][qi] << "\n";
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
    std::string data_type, index_prefix, query_file, gt_file, out_csv, out_summary, grid_file;
    uint64_t    K, L, W, T_threads, bnc_mb, sat_delta, et_conv_delta_u64;
    uint64_t    et_conv_width_u64, et_exact_patience_u64;
    uint64_t    et_ref_rank_u64, et_min_hops_u64, et_verify_patience_u64;
    bool        et_exact_led;
    float       et_theta, et_theta_exact, sat_gamma, et_verify_alpha;

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
        ("et_conv_width",  po::value<uint64_t>(&et_conv_width_u64)->default_value(0), "hybrid-exact ET: rank window M for exact-kth (0 -> k); matches search_server's --et_conv_width")
        ("et_exact_patience", po::value<uint64_t>(&et_exact_patience_u64)->default_value(1), "hybrid-exact ET: consecutive triggering hops required; matches search_server's --et_exact_patience")
        ("et_ref_rank",    po::value<uint64_t>(&et_ref_rank_u64)->default_value(0), "theta-ET reference rank (0 -> k); matches search_server's --et_ref_rank")
        ("et_min_hops",    po::value<uint64_t>(&et_min_hops_u64)->default_value(0), "ET grace period in hops (0=off); matches search_server's --et_min_hops")
        ("et_verify_alpha", po::value<float>(&et_verify_alpha)->default_value(std::numeric_limits<float>::max()), "predict-then-verify ET layer-2 exact alpha (FLT_MAX=disabled); matches search_server's --et_verify_alpha. Requires --et_theta set (layer-1 predictor) unless --et_exact_led")
        ("et_verify_patience", po::value<uint64_t>(&et_verify_patience_u64)->default_value(1), "predict-then-verify ET: consecutive agreeing hops before stopping; matches search_server's --et_verify_patience")
        ("et_exact_led",   po::bool_switch(&et_exact_led)->default_value(false), "predict-then-verify ET: lead with exact signal from prev hop, confirm with PQ look-ahead (inverse ordering)")
        ("bnc_mb",         po::value<uint64_t>(&bnc_mb)->default_value(0),    "BNC cache size in MB (0=disabled)")
        ("grid_file",      po::value<std::string>(&grid_file)->default_value(""),
                           "CSV of ET configs to sweep in one index load (columns: "
                           "label,et_theta,et_theta_exact,et_conv_width,et_exact_patience,et_ref_rank,"
                           "et_min_hops,et_verify_alpha,et_verify_patience,et_exact_led). "
                           "When set, all other --et_* flags are ignored and out_summary gets one row per config.")
        ("out_csv",        po::value<std::string>(&out_csv)->default_value(""), "per-query output CSV (single-config mode only)")
        ("out_summary",    po::value<std::string>(&out_summary)->required(),  "summary CSV");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc; return 0; }
    po::notify(vm);

    size_t bnc_bytes = (size_t)bnc_mb * 1024 * 1024;

    if (data_type == "uint8")
        return run_analysis<uint8_t>(index_prefix, query_file, gt_file, K, L, W, T_threads, et_theta, et_theta_exact, (uint32_t)et_conv_delta_u64, sat_gamma, (uint32_t)sat_delta, (uint32_t)et_conv_width_u64, (uint32_t)et_exact_patience_u64, (uint32_t)et_ref_rank_u64, (uint32_t)et_min_hops_u64, et_verify_alpha, (uint32_t)et_verify_patience_u64, et_exact_led, bnc_bytes, out_csv, out_summary, grid_file);
    else if (data_type == "float")
        return run_analysis<float>(index_prefix, query_file, gt_file, K, L, W, T_threads, et_theta, et_theta_exact, (uint32_t)et_conv_delta_u64, sat_gamma, (uint32_t)sat_delta, (uint32_t)et_conv_width_u64, (uint32_t)et_exact_patience_u64, (uint32_t)et_ref_rank_u64, (uint32_t)et_min_hops_u64, et_verify_alpha, (uint32_t)et_verify_patience_u64, et_exact_led, bnc_bytes, out_csv, out_summary, grid_file);
    else if (data_type == "int8")
        return run_analysis<int8_t>(index_prefix, query_file, gt_file, K, L, W, T_threads, et_theta, et_theta_exact, (uint32_t)et_conv_delta_u64, sat_gamma, (uint32_t)sat_delta, (uint32_t)et_conv_width_u64, (uint32_t)et_exact_patience_u64, (uint32_t)et_ref_rank_u64, (uint32_t)et_min_hops_u64, et_verify_alpha, (uint32_t)et_verify_patience_u64, et_exact_led, bnc_bytes, out_csv, out_summary, grid_file);
    else
    {
        std::cerr << "Unknown data_type: " << data_type << "\n";
        return -1;
    }
}
