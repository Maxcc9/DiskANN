// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案實作了 `StaticDiskIndex` 類別的 Python 封裝。
// 這個類別允許 Python 載入一個預先建立好的、為 SSD 優化的磁碟索引 (`PQFlashIndex`)，
// 並對其執行高效能的搜尋操作。

#include "static_disk_index.h"

#include "pybind11/numpy.h"

namespace diskannpy
{

// StaticDiskIndex 的建構函式 (對應 Python 的 __init__)
template <typename DT>
StaticDiskIndex<DT>::StaticDiskIndex(const diskann::Metric metric, const std::string &index_path_prefix,
                                     const uint32_t num_threads, const size_t num_nodes_to_cache,
                                     const uint32_t cache_mechanism)
    // 建立一個平台相關的對齊檔案讀取器和一個 PQFlashIndex 物件
    : _reader(std::make_shared<PlatformSpecificAlignedFileReader>()), _index(_reader, metric)
{
    const uint32_t _num_threads = num_threads != 0 ? num_threads : omp_get_num_procs();
    
    // 呼叫底層 C++ PQFlashIndex 物件的 load 方法來載入磁碟索引檔案。
    int load_success = _index.load(_num_threads, index_path_prefix.c_str());
    if (load_success != 0)
    {
        throw std::runtime_error("index load failed.");
    }

    // 根據使用者選擇的快取機制，預先載入部分節點到記憶體中以「預熱」索引。
    if (cache_mechanism == 1)
    {
        // 策略 1: 根據樣本查詢，快取最常被訪問的「熱點」節點。
        std::string sample_file = index_path_prefix + std::string("_sample_data.bin");
        cache_sample_paths(num_nodes_to_cache, sample_file, _num_threads);
    }
    else if (cache_mechanism == 2)
    {
        // 策略 2: 快取圖的頂部幾層節點 (BFS)。
        cache_bfs_levels(num_nodes_to_cache);
    }
}

// 快取圖的頂部幾層節點 (BFS 策略)
template <typename DT> void StaticDiskIndex<DT>::cache_bfs_levels(const size_t num_nodes_to_cache)
{
    std::vector<uint32_t> node_list;
    _index.cache_bfs_levels(num_nodes_to_cache, node_list);
    _index.load_cache_list(node_list);
}

// 根據樣本查詢路徑來快取「熱點」節點
template <typename DT>
void StaticDiskIndex<DT>::cache_sample_paths(const size_t num_nodes_to_cache, const std::string &warmup_query_file,
                                             const uint32_t num_threads)
{
    if (!file_exists(warmup_query_file))
    {
        return;
    }

    std::vector<uint32_t> node_list;
    // 執行樣本查詢以產生最常訪問的節點列表
    _index.generate_cache_list_from_sample_queries(warmup_query_file, 15, 4, num_nodes_to_cache, num_threads,
                                                   node_list);
    // 載入這些節點到快取中
    _index.load_cache_list(node_list);
}

// 單一查詢搜尋
template <typename DT>
NeighborsAndDistances<StaticIdType> StaticDiskIndex<DT>::search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &query, const uint64_t knn, const uint64_t complexity,
    const uint64_t beam_width)
{
    py::array_t<StaticIdType> ids(knn);
    py::array_t<float> dists(knn);

    std::vector<uint32_t> u32_ids(knn);
    std::vector<uint64_t> u64_ids(knn);
    diskann::QueryStats stats;

    // 呼叫底層 C++ PQFlashIndex 的核心搜尋函式 `cached_beam_search`
    // complexity: 搜尋候選集大小 (L)
    // beam_width: 光束寬度，影響 I/O 數量和搜尋廣度
    _index.cached_beam_search(query.data(), knn, complexity, u64_ids.data(), dists.mutable_data(), beam_width, false,
                              &stats);

    // 將 C++ 返回的 uint64_t 結果複製到 Python 的 uint32_t NumPy 陣列中
    auto r = ids.mutable_unchecked<1>();
    for (uint64_t i = 0; i < knn; ++i)
        r(i) = (unsigned)u64_ids[i];

    return std::make_pair(ids, dists);
}

// 批次搜尋
template <typename DT>
NeighborsAndDistances<StaticIdType> StaticDiskIndex<DT>::batch_search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries, const uint64_t knn,
    const uint64_t complexity, const uint64_t beam_width, const uint32_t num_threads)
{
    py::array_t<StaticIdType> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    omp_set_num_threads(num_threads);

    std::vector<uint64_t> u64_ids(knn * num_queries);

#pragma omp parallel for schedule(dynamic, 1) default(none)                                                            \
    shared(num_queries, queries, knn, complexity, u64_ids, dists, beam_width)
    for (int64_t i = 0; i < (int64_t)num_queries; i++)
    {
        _index.cached_beam_search(queries.data(i), knn, complexity, u64_ids.data() + i * knn, dists.mutable_data(i),
                                  beam_width);
    }

    // 將結果從 1D 的 C++ 向量複製到 2D 的 Python NumPy 陣列
    auto r = ids.mutable_unchecked();
    for (uint64_t i = 0; i < num_queries; ++i)
        for (uint64_t j = 0; j < knn; ++j)
            r(i, j) = (uint32_t)u64_ids[i * knn + j];

    return std::make_pair(ids, dists);
}

// 模板實例化
template class StaticDiskIndex<float>;
template class StaticDiskIndex<uint8_t>;
template class StaticDiskIndex<int8_t>;
} // namespace diskannpy
