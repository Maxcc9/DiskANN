// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案為 `InMemGraphStore` 類別的實作。
// 它處理在主記憶體中圖的鄰接串列的建立、儲存、載入和存取。

#include "in_mem_graph_store.h"
#include "utils.h"

namespace diskann
{
// 建構函式
InMemGraphStore::InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
    : AbstractGraphStore(total_pts, reserve_graph_degree)
{
    // 調整主向量的大小以容納所有節點
    this->resize_graph(total_pts);
    // 為每個節點的鄰居列表預留空間，以避免在建立索引時頻繁地重新分配記憶體
    for (size_t i = 0; i < total_pts; i++)
    {
        _graph[i].reserve(reserve_graph_degree);
    }
}

std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load(const std::string &index_path_prefix,
                                                             const size_t num_points)
{
    return load_impl(index_path_prefix, num_points);
}
int InMemGraphStore::store(const std::string &index_path_prefix, const size_t num_points,
                           const size_t num_frozen_points, const uint32_t start)
{
    return save_graph(index_path_prefix, num_points, num_frozen_points, start);
}

// 取得節點 i 的鄰居列表 (唯讀)
const std::vector<location_t> &InMemGraphStore::get_neighbours(const location_t i) const
{
    return _graph.at(i);
}

// 為節點 i 新增一個鄰居
void InMemGraphStore::add_neighbour(const location_t i, location_t neighbour_id)
{
    _graph[i].emplace_back(neighbour_id);
    // 更新觀察到的最大出度
    if (_max_observed_degree < _graph[i].size())
    {
        _max_observed_degree = (uint32_t)(_graph[i].size());
    }
}

// 清空節點 i 的所有鄰居
void InMemGraphStore::clear_neighbours(const location_t i)
{
    _graph[i].clear();
};

// 交換兩個節點 a 和 b 的鄰居列表
void InMemGraphStore::swap_neighbours(const location_t a, location_t b)
{
    _graph[a].swap(_graph[b]);
};

// 將節點 i 的鄰居列表設定為一個新的列表
void InMemGraphStore::set_neighbours(const location_t i, std::vector<location_t> &neighbours)
{
    _graph[i].assign(neighbours.begin(), neighbours.end());
    // 更新觀察到的最大出度
    if (_max_observed_degree < neighbours.size())
    {
        _max_observed_degree = (uint32_t)(neighbours.size());
    }
}

// 調整圖的容量以容納新的大小
size_t InMemGraphStore::resize_graph(const size_t new_size)
{
    _graph.resize(new_size);
    set_total_points(new_size);
    return _graph.size();
}

// 清空整個圖結構
void InMemGraphStore::clear_graph()
{
    _graph.clear();
}

#ifdef EXEC_ENV_OLS
std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load_impl(AlignedFileReader &reader, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;

    auto max_points = get_max_points();
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((size_t *)header.get());
    _max_observed_degree = *((uint32_t *)(header.get() + sizeof(size_t)));
    start = *((uint32_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t)));
    file_frozen_pts = *((size_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    diskann::cout << "Loading vamana graph from reader..." << std::flush;

    // If user provides more points than max_points
    // resize the _graph to the larger size.
    if (get_total_points() < expected_num_points)
    {
        diskann::cout << "resizing graph to " << expected_num_points << std::endl;
        this->resize_graph(expected_num_points);
    }

    uint32_t nodes_read = 0;
    size_t cc = 0;
    size_t graph_offset = header_size;
    while (nodes_read < expected_num_points)
    {
        uint32_t k;
        read_value(reader, k, graph_offset);
        graph_offset += sizeof(uint32_t);
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        read_array(reader, tmp.data(), k, graph_offset);
        graph_offset += k * sizeof(uint32_t);
        cc += k;
        _graph[nodes_read].swap(tmp);
        nodes_read++;
        if (nodes_read % 1000000 == 0)
        {
            diskann::cout << "." << std::flush;
        }
        if (k > _max_range_of_graph)
        {
            _max_range_of_graph = k;
        }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start
                  << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}
#endif

// 從檔案載入圖結構的實作
std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load_impl(const std::string &filename,
                                                                  size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;
    size_t file_offset = 0; 

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);

    // 讀取圖檔案的標頭，包含元資料
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&_max_observed_degree, sizeof(uint32_t));
    in.read((char *)&start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));
    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

    diskann::cout << "從圖標頭讀取, 預期檔案大小: " << expected_file_size
                  << ", 最大觀察出度: " << _max_observed_degree << ", 起始點: " << start
                  << ", 檔案中的凍結點數: " << file_frozen_pts << std::endl;

    diskann::cout << "正在載入 Vamana 圖 " << filename << "..." << std::flush;

    // 如果預期的點數量大於目前容量，則擴展圖
    if (get_total_points() < expected_num_points)
    {
        diskann::cout << "擴展圖容量至 " << expected_num_points << std::endl;
        this->resize_graph(expected_num_points);
    }

    size_t bytes_read = vamana_metadata_size;
    size_t cc = 0; // 總邊數計數器
    uint32_t nodes_read = 0;
    // 逐一讀取每個節點的鄰居列表
    while (bytes_read != expected_file_size)
    {
        uint32_t k; // 目前節點的鄰居數量
        in.read((char *)&k, sizeof(uint32_t));

        if (k == 0)
        {
            diskann::cerr << "錯誤: 發現沒有出邊的點, 點編號#" << nodes_read << std::endl;
        }

        cc += k;
        ++nodes_read;
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        in.read((char *)tmp.data(), k * sizeof(uint32_t));
        _graph[nodes_read - 1].swap(tmp);
        bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
        if (nodes_read % 10000000 == 0)
            diskann::cout << "." << std::flush;
        if (k > _max_range_of_graph)
        {
            _max_range_of_graph = k;
        }
    }

    diskann::cout << "完成。索引有 " << nodes_read << " 個節點和 " << cc << " 條出邊，起始點設定為 " << start
                  << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}

// 儲存圖結構到檔案
int InMemGraphStore::save_graph(const std::string &index_path_prefix, const size_t num_points,
                                const size_t num_frozen_points, const uint32_t start)
{
    std::ofstream out;
    open_file_to_write(out, index_path_prefix);

    size_t file_offset = 0;
    out.seekp(file_offset, out.beg);
    size_t index_size = 24; // 初始標頭大小
    uint32_t max_degree = 0;

    // 先寫入一個預估的標頭
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&_max_observed_degree, sizeof(uint32_t));
    uint32_t ep_u32 = start;
    out.write((char *)&ep_u32, sizeof(uint32_t));
    out.write((char *)&num_frozen_points, sizeof(size_t));

    // 逐一寫入每個節點的鄰居列表
    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t GK = (uint32_t)_graph[i].size(); // 鄰居數量
        out.write((char *)&GK, sizeof(uint32_t));
        out.write((char *)_graph[i].data(), GK * sizeof(uint32_t));
        max_degree = _graph[i].size() > max_degree ? (uint32_t)_graph[i].size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }

    // 回到檔案開頭，寫入最終計算出的正確檔案大小和最大出度
    out.seekp(file_offset, out.beg);
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    out.close();
    return (int)index_size;
}

size_t InMemGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

uint32_t InMemGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}

} // namespace diskann
