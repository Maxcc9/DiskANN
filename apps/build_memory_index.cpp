// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案是一個範例應用程式，展示如何使用 DiskANN 函式庫來建立一個記憶體內的 Vamana 圖索引。
// 它會讀取一個二進位格式的資料檔案，建立索引，然後將索引儲存到磁碟上以供後續搜尋使用。

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // --- 參數定義 ---
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

    // --- 使用 boost::program_options 解析命令列參數 ---
    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
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
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // 可選參數 (Vamana 圖建立參數)
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE); // 圖的最大出度
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY); // 建立索引時的搜尋候選集大小
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA); // RobustPrune 的 alpha 參數
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);

        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

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
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // --- 將命令列參數字串轉換為對應的 Metric ENUM ---
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

    try
    {
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        // --- 索引建立核心邏輯 ---

        // 1. 從資料檔案讀取元資料 (點數量和維度)
        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

        // 2. 使用 Builder 模式設定索引的寫入參數 (Vamana 圖建立參數)
        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        // 3. 設定過濾參數 (此處未使用，但展示了 API)
        auto filter_params = diskann::IndexFilterParamsBuilder()
                                 .with_universal_label(universal_label)
                                 .with_label_file(label_file)
                                 .with_save_path_prefix(index_path_prefix)
                                 .build();

        // 4. 使用 Builder 模式組裝完整的索引設定 (IndexConfig) 
        //    這包含了度量、維度、點數量、儲存策略、資料類型、建立參數等所有資訊。
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY) // 設定資料儲存在記憶體
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY) // 設定圖儲存在記憶體
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

        // 5. 使用工廠模式 (IndexFactory) 根據設定建立索引物件實例。
        //    工廠會根據 config 自動選擇並建立正確的 Index 類型及其依賴項 (如 InMemDataStore)。
        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();

        // 6. 呼叫 build() 方法，傳入資料路徑，開始建立索引。
        //    這是最耗時的步驟。
        index->build(data_path, data_num, filter_params);

        // 7. 呼叫 save() 方法，將建立好的索引序列化到磁碟上。
        index->save(index_path_prefix.c_str());

        // 8. 釋放索引物件
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
