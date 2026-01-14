// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/**
 * dump_disk_neighbors - 從 DiskANN 索引讀取指定節點的鄰居列表
 * 
 * 功能：
 *   1. 讀取 expanded_nodes.csv 或純文字節點 ID 列表
 *   2. 從磁盤索引中讀取每個節點的鄰居信息
 *   3. 輸出 CSV 格式：node_id, degree, neighbor_pos, neighbor_id
 * 
 * 用途：
 *   - 分析最常被展開的節點（熱點節點）的鄰居分佈
 *   - 用於離線分析圖結構與搜索模式
 */

#include <boost/program_options.hpp>

#include <fstream>
#include <sstream>
#include <unordered_set>

#include "defaults.h"
#include "pq_flash_index.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

namespace po = boost::program_options;

/**
 * 從 CSV 行中解析節點 ID
 * 
 * 支援兩種格式：
 *   1. expanded_nodes.csv: L,beamwidth,query_id,order,node_id
 *   2. 純文字格式: 每行一個 node_id
 * 
 * @param line 輸入行
 * @param out_id 輸出的節點 ID
 * @return 解析成功返回 true，失敗返回 false
 */
static bool parse_node_id_from_line(const std::string &line, uint32_t &out_id)
{
    // 跳過空行
    if (line.empty())
        return false;

    // 移除前導空白字符
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    if (trimmed.empty())
        return false;
    // 跳過 CSV header 行
    if (trimmed.find("node_id") != std::string::npos)
        return false;

    // 提取最後一個逗號後的內容（對於 CSV 格式，即 node_id 欄位）
    size_t last_comma = trimmed.rfind(',');
    std::string token = (last_comma == std::string::npos) ? trimmed : trimmed.substr(last_comma + 1);
    // 移除前後空白
    token.erase(0, token.find_first_not_of(" \t\r\n"));
    token.erase(token.find_last_not_of(" \t\r\n") + 1);
    if (token.empty())
        return false;
    try
    {
        out_id = static_cast<uint32_t>(std::stoul(token));
        return true;
    }
    catch (...)
    {
        return false;
    }
}

/**
 * 從磁盤索引讀取指定節點的鄰居列表並輸出為 CSV
 * 
 * @param index_path_prefix 索引文件路徑前綴
 * @param metric 距離度量方式（L2/INNER_PRODUCT/COSINE）
 * @param node_ids 要讀取的節點 ID 列表
 * @param output_path 輸出 CSV 檔案路徑
 * @return 0 表示成功，非 0 表示失敗
 */
template <typename T, typename LabelT = uint32_t>
int dump_neighbors(const std::string &index_path_prefix, diskann::Metric metric,
                   const std::vector<uint32_t> &node_ids, const std::string &output_path)
{
    // 初始化平台相關的文件讀取器
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

    // 載入 PQ Flash 索引（單線程載入）
    diskann::PQFlashIndex<T, LabelT> index(reader, metric);
    int res = index.load(1, index_path_prefix.c_str());
    if (res != 0)
    {
        diskann::cerr << "Failed to load index: " << index_path_prefix << std::endl;
        return res;
    }

    // 分配記憶體來儲存鄰居數據
    // stride = MAX_GRAPH_DEGREE：每個節點最多可能的鄰居數
    const size_t stride = diskann::defaults::MAX_GRAPH_DEGREE;
    // coord_buffers: 暫不使用向量座標（設為 nullptr）
    std::vector<T *> coord_buffers(node_ids.size(), nullptr);
    // nbr_storage: 連續記憶體空間，儲存所有節點的鄰居 ID
    std::vector<uint32_t> nbr_storage(node_ids.size() * stride, 0);
    // nbr_buffers: 每個節點的 (degree, neighbors_ptr) 對
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers(node_ids.size());

    // 為每個節點設置鄰居緩衝區指針（指向 nbr_storage 中的對應位置）
    for (size_t i = 0; i < node_ids.size(); i++)
    {
        nbr_buffers[i] = std::make_pair(0, nbr_storage.data() + i * stride);
    }

    // 批量從磁盤讀取所有節點的鄰居信息
    auto read_status = index.read_nodes(node_ids, coord_buffers, nbr_buffers);

    // 打開輸出文件
    std::ofstream out(output_path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        diskann::cerr << "Failed to open output file: " << output_path << std::endl;
        return -1;
    }

    // 寫入 CSV header
    out << "node_id,degree,neighbor_pos,neighbor_id\n";
    // 遍歷所有節點，輸出其鄰居信息
    for (size_t i = 0; i < node_ids.size(); i++)
    {
        // 跳過讀取失敗的節點
        if (!read_status[i])
        {
            diskann::cerr << "WARN: failed to read node " << node_ids[i] << std::endl;
            continue;
        }
        // 獲取節點的實際度數（鄰居數量）
        uint32_t degree = nbr_buffers[i].first;
        // 如果度數超過緩衝區限制，截斷並發出警告
        if (degree > stride)
        {
            diskann::cerr << "WARN: degree exceeds buffer limit (" << degree << " > " << stride
                          << "), truncating" << std::endl;
            degree = static_cast<uint32_t>(stride);
        }
        // 輸出每個鄰居：node_id, degree, neighbor_pos, neighbor_id
        for (uint32_t j = 0; j < degree; j++)
        {
            out << node_ids[i] << "," << degree << "," << j << "," << nbr_buffers[i].second[j] << "\n";
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    // 命令行參數變量
    std::string data_type, dist_fn, index_path_prefix, input_nodes_path, output_path;
    uint32_t max_nodes = 0;          // 限制讀取的最大節點數（0 = 不限制）
    bool keep_duplicates = false;     // 是否保留重複的 node_id

    // 定義命令行選項
    po::options_description desc{
        program_options_utils::make_program_description("dump_disk_neighbors",
                                                        "Dump neighbor lists for given node ids.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("input_nodes", po::value<std::string>(&input_nodes_path)->required(),
                                       "Input CSV file (expanded_nodes.csv or one node_id per line)");
        required_configs.add_options()("output_path", po::value<std::string>(&output_path)->required(),
                                       "Output CSV path");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("max_nodes", po::value<uint32_t>(&max_nodes)->default_value(0),
                                       "Max unique nodes to dump (0 = all)");
        optional_configs.add_options()("keep_duplicates", po::bool_switch(&keep_duplicates)->default_value(false),
                                       "Keep duplicate node_ids in input");

        desc.add(required_configs).add(optional_configs);

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
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == std::string("cosine"))
        metric = diskann::Metric::COSINE;
    else
    {
        std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
        return -1;
    }

    // 打開輸入文件（expanded_nodes.csv 或純文字節點列表）
    std::ifstream in(input_nodes_path);
    if (!in.is_open())
    {
        std::cerr << "Failed to open input_nodes: " << input_nodes_path << std::endl;
        return -1;
    }

    // 解析輸入文件，提取節點 ID
    std::vector<uint32_t> node_ids;      // 要讀取的節點列表
    std::unordered_set<uint32_t> seen;   // 用於去重
    std::string line;
    while (std::getline(in, line))
    {
        uint32_t node_id = 0;
        if (!parse_node_id_from_line(line, node_id))
            continue;
        // 根據 keep_duplicates 參數決定是否去重
        if (!keep_duplicates)
        {
            // 僅插入尚未見過的節點
            if (seen.insert(node_id).second)
            {
                node_ids.push_back(node_id);
            }
        }
        else
        {
            // 保留所有節點（包括重複）
            node_ids.push_back(node_id);
        }
        // 如果達到最大節點數限制，提前結束
        if (max_nodes > 0 && node_ids.size() >= max_nodes)
            break;
    }

    if (node_ids.empty())
    {
        std::cerr << "No node ids parsed from input." << std::endl;
        return -1;
    }

    if (data_type == std::string("float"))
        return dump_neighbors<float>(index_path_prefix, metric, node_ids, output_path);
    if (data_type == std::string("int8"))
        return dump_neighbors<int8_t>(index_path_prefix, metric, node_ids, output_path);
    if (data_type == std::string("uint8"))
        return dump_neighbors<uint8_t>(index_path_prefix, metric, node_ids, output_path);

    std::cerr << "Unsupported data type. Use float/int8/uint8." << std::endl;
    return -1;
}
