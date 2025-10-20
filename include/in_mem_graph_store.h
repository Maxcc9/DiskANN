// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案定義了 `InMemGraphStore` 類別，它是 `AbstractGraphStore` 的一個具體實作。
// 它負責在主記憶體中儲存和管理圖的鄰接串列 (adjacency list) 結構。

#include "abstract_graph_store.h"

namespace diskann
{

class InMemGraphStore : public AbstractGraphStore
{
  public:
    // 建構函式：初始化圖的儲存空間，設定總點數和每個點的預留鄰居數量。
    InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree);

    // 從檔案載入圖結構。
    // 返回一個元組，包含 <讀取的節點數, 圖的起始點ID, 凍結點數量>。
    virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                        const size_t num_points) override;
    // 將圖結構儲存到檔案。
    virtual int store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                      const uint32_t start) override;

    // 取得節點 i 的鄰居列表 (唯讀)。
    virtual const std::vector<location_t> &get_neighbours(const location_t i) const override;
    // 為節點 i 新增一個鄰居。
    virtual void add_neighbour(const location_t i, location_t neighbour_id) override;
    // 清空節點 i 的所有鄰居。
    virtual void clear_neighbours(const location_t i) override;
    // 交換兩個節點 a 和 b 的鄰居列表 (用於資料壓縮)。
    virtual void swap_neighbours(const location_t a, location_t b) override;

    // 將節點 i 的鄰居列表設定為一個新的列表。
    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbors) override;

    // 調整圖的容量以容納新的大小。
    virtual size_t resize_graph(const size_t new_size) override;
    // 清空整個圖結構。
    virtual void clear_graph() override;

    // 取得圖中節點允許的最大出度 (max degree)。
    virtual size_t get_max_range_of_graph() override;
    // 取得圖中實際觀察到的最大出度。
    virtual uint32_t get_max_observed_degree() override;

  protected:
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string &filename, size_t expected_num_points);
#ifdef EXEC_ENV_OLS
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(AlignedFileReader &reader, size_t expected_num_points);
#endif

    int save_graph(const std::string &index_path_prefix, const size_t active_points, const size_t num_frozen_points,
                   const uint32_t start);

  private:
    // 圖中節點允許的最大出度。
    size_t _max_range_of_graph = 0;
    // 圖中實際觀察到的最大出度。
    uint32_t _max_observed_degree = 0;

    // 核心資料結構：一個向量的向量，`_graph[i]` 儲存了節點 i 的所有鄰居的 ID。
    std::vector<std::vector<uint32_t>> _graph;
};

} // namespace diskann
