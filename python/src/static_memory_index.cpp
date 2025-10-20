// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案實作了 `StaticMemoryIndex` 類別的 Python 封裝。
// 這個類別允許 Python 載入一個預先建立好的靜態記憶體索引，並對其執行搜尋操作。

#include "static_memory_index.h"

#include "pybind11/numpy.h"

namespace diskannpy
{

// 輔助函式，用於建立一個為載入靜態索引而設定的 `diskann::Index` 物件。
// 它會設定 `dynamic_index = false` 和 `enable_tags = false` 等參數。
template <class DT>
diskann::Index<DT, StaticIdType, filterT> static_index_builder(const diskann::Metric m, const size_t num_points,
                                                               const size_t dimensions,
                                                               const uint32_t initial_search_complexity)
{
    if (initial_search_complexity == 0)
    {
        throw std::runtime_error("initial_search_complexity must be a positive uint32_t");
    }
    auto index_search_params = diskann::IndexSearchParams(initial_search_complexity, omp_get_num_procs());
    return diskann::Index<DT>(m, dimensions, num_points,
                              nullptr,                                                           // 索引寫入參數 (載入時不需要)
                              std::make_shared<diskann::IndexSearchParams>(index_search_params), // 索引搜尋參數
                              0,       // 凍結點數量
                              false,   // 非動態索引
                              false,   // 不啟用標籤
                              false,   // 無並行整理
                              false,   // 不使用 PQ 建立
                              0,       // PQ 區塊數
                              false);  // 不使用 OPQ
}

// StaticMemoryIndex 的建構函式 (對應 Python 的 __init__)
template <class DT>
StaticMemoryIndex<DT>::StaticMemoryIndex(const diskann::Metric m, const std::string &index_prefix,
                                         const size_t num_points, const size_t dimensions, const uint32_t num_threads,
                                         const uint32_t initial_search_complexity)
    : _index(static_index_builder<DT>(m, num_points, dimensions, initial_search_complexity))
{
    const uint32_t _num_threads = num_threads != 0 ? num_threads : omp_get_num_procs();
    // 在物件初始化時，立即呼叫底層 C++ Index 物件的 load 方法來載入索引檔案。
    _index.load(index_prefix.c_str(), _num_threads, initial_search_complexity);
}

// 單一查詢的搜尋方法
template <typename DT>
NeighborsAndDistances<StaticIdType> StaticMemoryIndex<DT>::search(
    // pybind11 會自動將傳入的 NumPy 陣列轉換為 py::array_t，無需手動轉換。
    py::array_t<DT, py::array::c_style | py::array::forcecast> &query, const uint64_t knn, const uint64_t complexity)
{
    // 建立用於儲存結果的 NumPy 陣列
    py::array_t<StaticIdType> ids(knn);
    py::array_t<float> dists(knn);
    std::vector<DT *> empty_vector;
    _index.search(query.data(), knn, complexity, ids.mutable_data(), dists.mutable_data());
    
    // 返回一個包含 ID 和距離的元組，pybind11 會將其轉換為 Python 的 tuple of numpy arrays。
    return std::make_pair(ids, dists);
}

// 帶過濾條件的單一查詢搜尋方法
template <typename DT>
NeighborsAndDistances<StaticIdType> StaticMemoryIndex<DT>::search_with_filter(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &query, const uint64_t knn, const uint64_t complexity,
    const filterT filter)
{
    py::array_t<StaticIdType> ids(knn);
    py::array_t<float> dists(knn);
    std::vector<DT *> empty_vector;
    _index.search_with_filters(query.data(), filter, knn, complexity, ids.mutable_data(), dists.mutable_data());
    return std::make_pair(ids, dists);
}

// 批次搜尋方法
template <typename DT>
NeighborsAndDistances<StaticIdType> StaticMemoryIndex<DT>::batch_search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries, const uint64_t knn,
    const uint64_t complexity, const uint32_t num_threads)
{
    const uint32_t _num_threads = num_threads != 0 ? num_threads : omp_get_num_procs();
    // 建立二維的 NumPy 陣列以儲存批次搜尋的結果
    py::array_t<StaticIdType> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});
    std::vector<DT *> empty_vector;

    omp_set_num_threads(static_cast<int32_t>(_num_threads));

    // 使用 OpenMP 將查詢分配到多個執行緒上平行執行，以提高 QPS。
#pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_queries, queries, knn, complexity, ids, dists)
    for (int64_t i = 0; i < (int64_t)num_queries; i++)
    {
        // `queries.data(i)` 和 `ids.mutable_data(i)` 用於存取 NumPy 陣列的第 i 行。
        _index.search(queries.data(i), knn, complexity, ids.mutable_data(i), dists.mutable_data(i));
    }

    return std::make_pair(ids, dists);
}

// 模板實例化
template class StaticMemoryIndex<float>;
template class StaticMemoryIndex<uint8_t>;
template class StaticMemoryIndex<int8_t>;

} // namespace diskannpy
