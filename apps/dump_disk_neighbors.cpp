// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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

static bool parse_node_id_from_line(const std::string &line, uint32_t &out_id)
{
    if (line.empty())
        return false;

    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    if (trimmed.empty())
        return false;
    if (trimmed.find("node_id") != std::string::npos)
        return false;

    size_t last_comma = trimmed.rfind(',');
    std::string token = (last_comma == std::string::npos) ? trimmed : trimmed.substr(last_comma + 1);
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

template <typename T, typename LabelT = uint32_t>
int dump_neighbors(const std::string &index_path_prefix, diskann::Metric metric,
                   const std::vector<uint32_t> &node_ids, const std::string &output_path)
{
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

    diskann::PQFlashIndex<T, LabelT> index(reader, metric);
    int res = index.load(1, index_path_prefix.c_str());
    if (res != 0)
    {
        diskann::cerr << "Failed to load index: " << index_path_prefix << std::endl;
        return res;
    }

    const size_t stride = diskann::defaults::MAX_GRAPH_DEGREE;
    std::vector<T *> coord_buffers(node_ids.size(), nullptr);
    std::vector<uint32_t> nbr_storage(node_ids.size() * stride, 0);
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers(node_ids.size());

    for (size_t i = 0; i < node_ids.size(); i++)
    {
        nbr_buffers[i] = std::make_pair(0, nbr_storage.data() + i * stride);
    }

    auto read_status = index.read_nodes(node_ids, coord_buffers, nbr_buffers);

    std::ofstream out(output_path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        diskann::cerr << "Failed to open output file: " << output_path << std::endl;
        return -1;
    }

    out << "node_id,degree,neighbor_pos,neighbor_id\n";
    for (size_t i = 0; i < node_ids.size(); i++)
    {
        if (!read_status[i])
        {
            diskann::cerr << "WARN: failed to read node " << node_ids[i] << std::endl;
            continue;
        }
        uint32_t degree = nbr_buffers[i].first;
        if (degree > stride)
        {
            diskann::cerr << "WARN: degree exceeds buffer limit (" << degree << " > " << stride
                          << "), truncating" << std::endl;
            degree = static_cast<uint32_t>(stride);
        }
        for (uint32_t j = 0; j < degree; j++)
        {
            out << node_ids[i] << "," << degree << "," << j << "," << nbr_buffers[i].second[j] << "\n";
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, input_nodes_path, output_path;
    uint32_t max_nodes = 0;
    bool keep_duplicates = false;

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

    std::ifstream in(input_nodes_path);
    if (!in.is_open())
    {
        std::cerr << "Failed to open input_nodes: " << input_nodes_path << std::endl;
        return -1;
    }

    std::vector<uint32_t> node_ids;
    std::unordered_set<uint32_t> seen;
    std::string line;
    while (std::getline(in, line))
    {
        uint32_t node_id = 0;
        if (!parse_node_id_from_line(line, node_id))
            continue;
        if (!keep_duplicates)
        {
            if (seen.insert(node_id).second)
            {
                node_ids.push_back(node_id);
            }
        }
        else
        {
            node_ids.push_back(node_id);
        }
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
