// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案實作了 `DynamicMemoryIndex` 類別的 Python 封裝。
// 這個類別的核心是 `diskann::Index`，但其設定為可動態新增和刪除資料點。

#include "parameters.h"
#include "dynamic_memory_index.h"

#include "pybind11/numpy.h"

namespace diskannpy
{

// 輔助函式，用於從 Python 參數建立 C++ 的 `IndexWriteParameters` 物件。
diskann::IndexWriteParameters dynamic_index_write_parameters(const uint32_t complexity, const uint32_t graph_degree,
                                                             const bool saturate_graph,
                                                             const uint32_t max_occlusion_size, const float alpha,
                                                             const uint32_t num_threads,
                                                             const uint32_t filter_complexity)
{
    return diskann::IndexWriteParametersBuilder(complexity, graph_degree)
        .with_saturate_graph(saturate_graph)
        .with_max_occlusion_size(max_occlusion_size)
        .with_alpha(alpha)
        .with_num_threads(num_threads)
        .with_filter_list_size(filter_complexity)
        .build();
}

// 輔助函式，用於建立一個為動態索引而設定的 `diskann::Index` 物件。
template <class DT>
diskann::Index<DT, DynamicIdType, filterT> dynamic_index_builder(
    const diskann::Metric m, const diskann::IndexWriteParameters &write_params, const size_t dimensions,
    const size_t max_vectors, const uint32_t initial_search_complexity, const uint32_t initial_search_threads,
    const bool concurrent_consolidation, const uint32_t num_frozen_points)
{
    const uint32_t _initial_search_threads = initial_search_threads != 0 ? initial_search_threads : omp_get_num_procs();

    auto index_search_params = diskann::IndexSearchParams(initial_search_complexity, _initial_search_threads);
    return diskann::Index<DT, DynamicIdType, filterT>(
        m, dimensions, max_vectors,
        std::make_shared<diskann::IndexWriteParameters>(write_params),     // index write params
        std::make_shared<diskann::IndexSearchParams>(index_search_params), // index_search_params
        num_frozen_points,                                                 // frozen_points
        true,                                                              // dynamic_index
        true,                                                              // enable_tags
        concurrent_consolidation,
        false,  // pq_dist_build
        0,      // num_pq_chunks
        false); // use_opq = false
}

template <class DT>
DynamicMemoryIndex<DT>::DynamicMemoryIndex(const diskann::Metric m, const size_t dimensions, const size_t max_vectors,
                                           const uint32_t complexity, const uint32_t graph_degree,
                                           const bool saturate_graph, const uint32_t max_occlusion_size,
                                           const float alpha, const uint32_t num_threads,
                                           const uint32_t filter_complexity, const uint32_t num_frozen_points,
                                           const uint32_t initial_search_complexity,
                                           const uint32_t initial_search_threads, const bool concurrent_consolidation)
    : _initial_search_complexity(initial_search_complexity != 0 ? initial_search_complexity : complexity),
      _write_parameters(dynamic_index_write_parameters(complexity, graph_degree, saturate_graph, max_occlusion_size,
                                                       alpha, num_threads, filter_complexity)),
      _index(dynamic_index_builder<DT>(m, _write_parameters, dimensions, max_vectors, _initial_search_complexity,
                                       initial_search_threads, concurrent_consolidation, num_frozen_points))
{
}

// 載入一個已儲存的索引
template <class DT> void DynamicMemoryIndex<DT>::load(const std::string &index_path)
{
    const std::string tags_file = index_path + ".tags";
    if (!file_exists(tags_file))
    {
        throw std::runtime_error("tags file not found at expected path: " + tags_file);
    }
    _index.load(index_path.c_str(), _write_parameters.num_threads, _initial_search_complexity);
}

// 插入單一向量
template <class DT>
int DynamicMemoryIndex<DT>::insert(const py::array_t<DT, py::array::c_style | py::array::forcecast> &vector,
                                   const DynamicIdType id)
{
    // 呼叫底層 C++ Index 的 insert_point 方法。
    return _index.insert_point(vector.data(), id);
}

// 批次插入多個向量
template <class DT>
py::array_t<int> DynamicMemoryIndex<DT>::batch_insert(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &vectors,
    py::array_t<DynamicIdType, py::array::c_style | py::array::forcecast> &ids, const int32_t num_inserts,
    const int num_threads)
{
    if (num_threads == 0)
        omp_set_num_threads(omp_get_num_procs());
    else
        omp_set_num_threads(num_threads);
    py::array_t<int> insert_retvals(num_inserts);

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_inserts, insert_retvals, vectors, ids)
    for (int32_t i = 0; i < num_inserts; i++)
    {
        insert_retvals.mutable_data()[i] = _index.insert_point(vectors.data(i), *(ids.data(i)));
    }

    return insert_retvals;
}

// 標記一個點為已刪除 (延遲刪除)
// 這只是一個標記，實際的資料和圖結構調整會在 `consolidate_delete` 中進行。
template <class DT> int DynamicMemoryIndex<DT>::mark_deleted(const DynamicIdType id)
{
    return this->_index.lazy_delete(id);
}

// 儲存索引的目前狀態
template <class DT> void DynamicMemoryIndex<DT>::save(const std::string &save_path, const bool compact_before_save)
{
    if (save_path.empty())
    {
        throw std::runtime_error("A save_path must be provided");
    }
    _index.save(save_path.c_str(), compact_before_save);
}

// 單一查詢搜尋
template <class DT>
NeighborsAndDistances<DynamicIdType> DynamicMemoryIndex<DT>::search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &query, const uint64_t knn, const uint64_t complexity)
{
    py::array_t<DynamicIdType> ids(knn);
    py::array_t<float> dists(knn);
    std::vector<DT *> empty_vector;
    _index.search_with_tags(query.data(), knn, complexity, ids.mutable_data(), dists.mutable_data(), empty_vector);
    return std::make_pair(ids, dists);
}

// 批次搜尋
template <class DT>
NeighborsAndDistances<DynamicIdType> DynamicMemoryIndex<DT>::batch_search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries, const uint64_t knn,
    const uint64_t complexity, const uint32_t num_threads)
{
    py::array_t<DynamicIdType> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});
    std::vector<DT *> empty_vector;

    if (num_threads == 0)
        omp_set_num_threads(omp_get_num_procs());
    else
        omp_set_num_threads(static_cast<int32_t>(num_threads));

#pragma omp parallel for schedule(dynamic, 1) default(none)                                                            \
    shared(num_queries, queries, knn, complexity, ids, dists, empty_vector)
    for (int64_t i = 0; i < (int64_t)num_queries; i++)
    {
        _index.search_with_tags(queries.data(i), knn, complexity, ids.mutable_data(i), dists.mutable_data(i),
                                empty_vector);
    }

    return std::make_pair(ids, dists);
}

// 整理刪除操作
// 實際從圖中移除被標記為刪除的點，並修復圖結構。
template <class DT> void DynamicMemoryIndex<DT>::consolidate_delete()
{
    _index.consolidate_deletes(_write_parameters);
}

template <class DT> size_t DynamicMemoryIndex<DT>::num_points()
{
    return _index.get_num_points();
}

// 模板實例化
template class DynamicMemoryIndex<float>;
template class DynamicMemoryIndex<uint8_t>;
template class DynamicMemoryIndex<int8_t>;

}; // namespace diskannpy
