// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

// 本檔案定義了 `InMemDataStore` 類別，它是 `AbstractDataStore` 的一個具體實作。
// 它負責在主記憶體中儲存和管理所有的向量資料。

#include <shared_mutex>
#include <memory>

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "tsl/sparse_map.h"
// #include "boost/dynamic_bitset.hpp"

#include "abstract_data_store.h"

#include "distance.h"
#include "natural_number_map.h"
#include "natural_number_set.h"
#include "aligned_file_reader.h"

namespace diskann
{

// 記憶體內資料儲存類別
// 管理一個大型、連續且對齊的記憶體區塊，用於存放所有向量資料。
template <typename data_t> class InMemDataStore : public AbstractDataStore<data_t>
{
  public:
    // 建構函式：分配指定容量和維度的記憶體，並取得距離函式的所有權。
    InMemDataStore(const location_t capacity, const size_t dim, std::unique_ptr<Distance<data_t>> distance_fn);
    virtual ~InMemDataStore();

    // 從檔案載入向量資料到記憶體中。
    virtual location_t load(const std::string &filename) override;
    // 將記憶體中的向量資料儲存到檔案。
    virtual size_t save(const std::string &filename, const location_t num_points) override;

    // 取得對齊後的向量維度。
    virtual size_t get_aligned_dim() const override;

    // 從一個未對齊的記憶體陣列中填充資料，會進行對齊和必要的正規化。
    virtual void populate_data(const data_t *vectors, const location_t num_pts) override;
    // 從檔案中填充資料。
    virtual void populate_data(const std::string &filename, const size_t offset) override;

    // 將指定數量的向量資料提取到一個二進位檔案中。
    virtual void extract_data_to_bin(const std::string &filename, const location_t num_pts) override;

    // 取得指定位置 i 的向量。
    virtual void get_vector(const location_t i, data_t *target) const override;
    // 設定指定位置 i 的向量。
    virtual void set_vector(const location_t i, const data_t *const vector) override;
    // 提示 CPU 預取指定位置的向量到快取中，以提高效能。
    virtual void prefetch_vector(const location_t loc) override;

    // (動態索引使用) 在記憶體中移動一批向量。
    virtual void move_vectors(const location_t old_location_start, const location_t new_location_start,
                              const location_t num_points) override;
    // (動態索引使用) 在記憶體中複製一批向量。
    virtual void copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points) override;

    // 預處理查詢向量 (例如正規化)，會呼叫內部的距離函式物件來執行。
    virtual void preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const override;

    // 計算預處理過的查詢向量與指定位置向量之間的距離。
    virtual float get_distance(const data_t *preprocessed_query, const location_t loc) const override;
    // 計算兩個指定位置向量之間的距離。
    virtual float get_distance(const location_t loc1, const location_t loc2) const override;

    // 批次計算查詢向量與一組指定位置向量之間的距離。
    virtual void get_distance(const data_t *preprocessed_query, const location_t *locations,
                              const uint32_t location_count, float *distances,
                              AbstractScratch<data_t> *scratch) const override;
    virtual void get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                              std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const override;

    // 計算資料集中的 medoid (幾何中心點)，通常用作圖的進入點。
    virtual location_t calculate_medoid() const override;

    // 取得距離函式物件的指標。
    virtual Distance<data_t> *get_dist_fn() const override;

    // 取得記憶體對齊因子。
    virtual size_t get_alignment_factor() const override;

  protected:
    // 擴展資料儲存的容量。
    virtual location_t expand(const location_t new_size) override;
    // 縮減資料儲存的容量。
    virtual location_t shrink(const location_t new_size) override;

    virtual location_t load_impl(const std::string &filename);
#ifdef EXEC_ENV_OLS
    virtual location_t load_impl(AlignedFileReader &reader);
#endif

  private:
    // 指向儲存所有向量資料的大型、連續、對齊的記憶體區塊。
    data_t *_data = nullptr;

    // 對齊後的向量維度 (為了 SIMD 優化)。
    size_t _aligned_dim;

    // 距離函式物件。將其與資料儲存放在一起可以提高效能，
    // 因為可以在內部直接計算距離，無需來回複製資料。
    std::unique_ptr<Distance<data_t>> _distance_fn;

    // 用於儲存預先計算好的向量範數，以進行優化 (例如 FAST_L2)。
    std::shared_ptr<float[]> _pre_computed_norms;
};

} // namespace diskann