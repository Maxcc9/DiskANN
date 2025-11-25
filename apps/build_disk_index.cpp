// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案是一個範例應用程式，展示如何建立一個為 SSD 優化的磁碟索引 (Disk Index)。
// 這個過程比建立記憶體索引更複雜，通常包含以下步驟：
// 1. (可選) 將大型資料集分割成多個可以載入記憶體的分區 (shards)。
// 2. 對每個分區建立一個記憶體內的 Vamana 圖索引。
// 3. (可選) 訓練 PQ 碼本並將原始向量資料壓縮。
// 4. 將所有分區的圖和壓縮後的資料合併成最終的磁碟索引檔案。

#include <omp.h>
#include <boost/program_options.hpp>

#include "utils.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "index.h"
#include "partition.h"
#include "program_options_utils.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // --- 參數定義 ---
    std::string 
        // 資料向量的資料型別（必填）。接受值範例：
        //   "float"   - 32-bit 浮點向量
        //   "int8"    - 有號 8-bit 量化向量
        //   "uint8"   - 無號 8-bit 量化向量
        // 用於選擇對應 template 與壓縮/重排序支援。由命令列參數 --data_type 設定。
        data_type,

        // 距離函數（必填）。接受值範例：
        //   "l2"      - 歐氏距離
        //   "mips"    - 內積 (maximum inner product)
        //   "cosine"  - 餘弦相似度
        // 用來決定 metric 與建置 / 搜尋時的距離計算方式。由 --dist_fn 設定。
        dist_fn,

        // 輸入資料檔路徑（必填）。指向包含向量資料的檔案或資料集前置路徑。
        // 格式與具體檔案型態依專案輸入格式而定（例如二進位向量檔、.fvecs/.bin 等）。
        // 由 --data_path 設定。
        data_path,

        // 輸出索引檔案的路徑前綴（必填）。最終會以此 prefix 產生多個索引相關檔案（例如 .graph/.data 等）。
        // 由 --index_path_prefix 設定。
        index_path_prefix,

        // 預訓練 codebook 的路徑前綴（選用）。當使用 PQ/OPQ 時可指定先前訓練好的 codebook 檔案前綴，
        // 若不指定則會在建置流程中自行訓練/產生。由 --codebook_prefix 設定，預設為空字串（表示不使用）。
        codebook_prefix,

        // 標籤檔案路徑（選用）。用於啟用過濾或分群等功能時，提供每個向量的標籤資訊。
        // 檔案格式與每行/每向量標籤的具體規範請參考專案文件。由 --label_file 設定，預設為空字串（表示未使用）。
        label_file,

        // 通用標籤（選用）。當想對所有向量指定相同標籤或作為 fallback 標籤時使用：
        // 例如傳入單一字串標籤，或某種語意上的預設值。由 --universal_label 設定，預設為空字串。
        universal_label,

        // 標籤資料型別（選用）。指定 label 在索引內儲存的型別，範例：
        //   "uint"   - 使用 unsigned int（預設）
        //   "ushort" - 使用 unsigned short（較小記憶體）
        // 影響 template 特化與 IO。由 --label_type 設定，預設為 "uint"。
        label_type;
        
    uint32_t
        // num_threads: 要使用的執行緒數量。由命令列參數 --num_threads / -T 設定，預設為 omp_get_num_procs()。
        num_threads,
        // R: 建構時每個節點的最大度 (max_degree)。影響圖的稠密度與搜尋/建構效能。由 --max_degree / -R 設定 (預設 64)。
        R,
        // L: 建構時的搜尋複雜度（Lbuild）。控制在建圖階段候選節點搜尋的廣度/深度。由 --Lbuild / -L 設定 (預設 100)。
        L,
        // disk_PQ: 將向量壓縮到磁碟時每向量使用的位元組數。0 表示不壓縮。由 --PQ_disk_bytes 設定。
        disk_PQ,
        // build_PQ: 在建構流程中使用的 PQ byte 大小（若啟用）。由 --build_PQ_bytes 設定，影響建構時的壓縮/記憶體使用。
        build_PQ,
        // QD: Quantized Dimension（量化後的維度），用於特定壓縮流程。由 --QD 設定，0 表示未啟用/使用預設行為。
        QD,
        // Lf: Filtered L-build，用於有標籤或過濾情境下的建構複雜度參數（類似 L，但針對被過濾的子圖）。由 --FilteredLbuild 設定。
        Lf,
        // filter_threshold: 標籤過濾閥值 F。用於在內部切分/重建節點時限制每個節點上的最大標籤數量。由 --filter_threshold / -F 設定。
        filter_threshold;
    float B, M; // B: 搜尋時的 DRAM 預算, M: 建立時的 DRAM 預算
    bool append_reorder_data = false;
    bool use_opq = false;

    // --- 使用 boost::program_options 解析命令列參數 ---
    po::options_description desc{
        program_options_utils::make_program_description("build_disk_index", "Build a disk-based index.")};
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
        required_configs.add_options()("search_DRAM_budget,B", po::value<float>(&B)->required(),
                                       "DRAM budget in GB for searching the index to set the "
                                       "compressed level for data while search happens");
        required_configs.add_options()("build_DRAM_budget,M", po::value<float>(&M)->required(),
                                       "DRAM budget in GB for building the index");

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("QD", po::value<uint32_t>(&QD)->default_value(0),
                                       "Quantized Dimension for compression");
        optional_configs.add_options()("codebook_prefix", po::value<std::string>(&codebook_prefix)->default_value(""),
                                       "Path prefix for pre-trained codebook");
        optional_configs.add_options()("PQ_disk_bytes", po::value<uint32_t>(&disk_PQ)->default_value(0),
                                       "Number of bytes to which vectors should be compressed "
                                       "on SSD; 0 for no compression");
        optional_configs.add_options()("append_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.");
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("filter_threshold,F", po::value<uint32_t>(&filter_threshold)->default_value(0),
                                       "Threshold to break up the existing nodes to generate new graph "
                                       "internally where each node has a maximum F labels.");
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
        if (vm["append_reorder_data"].as<bool>())
            append_reorder_data = true;
        if (vm["use_opq"].as<bool>())
            use_opq = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    bool use_filters = (label_file != "") ? true : false;
    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == std::string("cosine"))
        metric = diskann::Metric::COSINE;
    else
    {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    if (append_reorder_data)
    {
        if (disk_PQ == 0)
        {
            std::cout << "Error: It is not necessary to append data for reordering "
                         "when vectors are not compressed on disk."
                      << std::endl;
            return -1;
        }
        if (data_type != std::string("float"))
        {
            std::cout << "Error: Appending data for reordering currently only "
                         "supported for float data type."
                      << std::endl;
            return -1;
        }
    }

    std::string params = std::string(std::to_string(R)) + " " + std::string(std::to_string(L)) + " " +
                         std::string(std::to_string(B)) + " " + std::string(std::to_string(M)) + " " +
                         std::string(std::to_string(num_threads)) + " " + std::string(std::to_string(disk_PQ)) + " " +
                         std::string(std::to_string(append_reorder_data)) + " " +
                         std::string(std::to_string(build_PQ)) + " " + std::string(std::to_string(QD));

    try
    {
        // --- 根據資料類型，呼叫對應的模板函式來執行索引建立 ---
        // 真正的建立邏輯被封裝在 `diskann::build_disk_index` 函式中 (位於 disk_utils.h/.cpp)。
        // 這個 main 函式主要負責解析參數並進行分派。
        if (label_file != "" && label_type == "ushort")
        {
            if (data_type == std::string("int8"))
                return diskann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                         metric, use_opq, codebook_prefix, use_filters, label_file,
                                                         universal_label, filter_threshold, Lf);
            else if (data_type == std::string("uint8"))
                return diskann::build_disk_index<uint8_t, uint16_t>(
                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                    use_filters, label_file, universal_label, filter_threshold, Lf);
            else if (data_type == std::string("float"))
                return diskann::build_disk_index<float, uint16_t>(
                    data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                    use_filters, label_file, universal_label, filter_threshold, Lf);
            else
            {
                diskann::cerr << "Error. Unsupported data type" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("int8"))
                return diskann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                         metric, use_opq, codebook_prefix, use_filters, label_file,
                                                         universal_label, filter_threshold, Lf);
            else if (data_type == std::string("uint8"))
                return diskann::build_disk_index<uint8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                          metric, use_opq, codebook_prefix, use_filters, label_file,
                                                          universal_label, filter_threshold, Lf);
            else if (data_type == std::string("float"))
                return diskann::build_disk_index<float>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                        metric, use_opq, codebook_prefix, use_filters, label_file,
                                                        universal_label, filter_threshold, Lf);
            else
            {
                diskann::cerr << "Error. Unsupported data type" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
