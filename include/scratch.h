// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案定義了在搜尋和建立索引過程中使用的「暫存空間」(Scratch Space) 相關的類別。
// 為了避免在高效能要求的程式碼路徑 (hot path) 中動態配置記憶體，DiskANN 會為每個執行緒
// 預先分配一個暫存空間物件，該物件包含了執行一次查詢或插入所需的所有臨時資料結構。

#include <vector>

#include "boost_dynamic_bitset_fwd.h"
// #include "boost/dynamic_bitset.hpp"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "tsl/sparse_map.h"

#include "aligned_file_reader.h"
#include "abstract_scratch.h"
#include "neighbor.h"
#include "defaults.h"
#include "concurrent_queue.h"

namespace diskann
{
template <typename T> class PQScratch;

//
// 用於記憶體內索引搜尋的暫存空間
//
template <typename T> class InMemQueryScratch : public AbstractScratch<T>
{
  public:
    ~InMemQueryScratch();
    // 建構函式：根據搜尋和索引參數初始化所有內部資料結構的大小。
    InMemQueryScratch(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim, size_t aligned_dim,
                      size_t alignment_factor, bool init_pq_scratch = false);
    // 如果查詢時的 L (候選集大小) 大於初始化時的大小，則動態擴展相關資料結構。
    void resize_for_new_L(uint32_t new_search_l);
    // 清理暫存空間，為下一次查詢做準備。
    void clear();

    // --- 以下為各個暫存資料結構的存取介面 ---
    inline uint32_t get_L()
    {
        return _L;
    }
    inline uint32_t get_R()
    {
        return _R;
    }
    inline uint32_t get_maxc()
    {
        return _maxc;
    }
    inline T *aligned_query()
    {
        return this->_aligned_query_T;
    }
    inline PQScratch<T> *pq_scratch()
    {
        return this->_pq_scratch;
    }
    // 在 RobustPrune 中使用的擴展候選池
    inline std::vector<Neighbor> &pool()
    {
        return _pool;
    }
    // 儲存 L 個最佳候選點的優先級隊列
    inline NeighborPriorityQueue &best_l_nodes()
    {
        return _best_l_nodes;
    }
    // 在 occlude_list 中使用的遮蔽因子向量
    inline std::vector<float> &occlude_factor()
    {
        return _occlude_factor;
    }
    // 已訪問節點的集合 (robin_set 版本)
    inline tsl::robin_set<uint32_t> &inserted_into_pool_rs()
    {
        return _inserted_into_pool_rs;
    }
    // 已訪問節點的集合 (bitset 版本，點數較少時更高效)
    inline boost::dynamic_bitset<> &inserted_into_pool_bs()
    {
        return *_inserted_into_pool_bs;
    }
    // 用於批次計算距離的臨時節點 ID 緩衝區
    inline std::vector<uint32_t> &id_scratch()
    {
        return _id_scratch;
    }
    // 用於批次計算距離的臨時距離值緩衝區
    inline std::vector<float> &dist_scratch()
    {
        return _dist_scratch;
    }
    // 在處理刪除時使用的擴展節點集合
    inline tsl::robin_set<uint32_t> &expanded_nodes_set()
    {
        return _expanded_nodes_set;
    }
    inline std::vector<Neighbor> &expanded_nodes_vec()
    {
        return _expanded_nghrs_vec;
    }
    inline std::vector<uint32_t> &occlude_list_output()
    {
        return _occlude_list_output;
    }

  private:
    uint32_t _L;  // 搜尋候選列表大小
    uint32_t _R;  // 圖的最大出度
    uint32_t _maxc; // 修剪時的最大候選點數量

    std::vector<Neighbor> _pool;
    NeighborPriorityQueue _best_l_nodes;
    std::vector<float> _occlude_factor;
    tsl::robin_set<uint32_t> _inserted_into_pool_rs;
    boost::dynamic_bitset<> *_inserted_into_pool_bs;
    std::vector<uint32_t> _id_scratch;
    std::vector<float> _dist_scratch;
    tsl::robin_set<uint32_t> _expanded_nodes_set;
    std::vector<Neighbor> _expanded_nghrs_vec;
    std::vector<uint32_t> _occlude_list_output;
};

//
// 用於 SSD 索引搜尋的暫存空間
//
template <typename T> class SSDQueryScratch : public AbstractScratch<T>
{
  public:
    // 用於儲存從磁碟讀取的向量資料
    T *coord_scratch = nullptr;
    // 用於儲存從磁碟讀取的原始磁區資料
    char *sector_scratch = nullptr;
    size_t sector_idx = 0;

    // SSD 搜尋中使用的已訪問集合和候選集
    tsl::robin_set<size_t> visited;
    NeighborPriorityQueue retset;
    std::vector<Neighbor> full_retset;

    SSDQueryScratch(size_t aligned_dim, size_t visited_reserve);
    ~SSDQueryScratch();

    void reset();
};

// 代表一個 SSD 搜尋執行緒所需的全部資料，包含暫存空間和 I/O 上下文
template <typename T> class SSDThreadData
{
  public:
    SSDQueryScratch<T> scratch;
    IOContext ctx;

    SSDThreadData(size_t aligned_dim, size_t visited_reserve);
    void clear();
};

//
// RAII (資源獲取即初始化) 風格的暫存空間管理器
// 這個類別的設計是為了簡化暫存空間的生命週期管理。
//
template <typename T> class ScratchStoreManager
{
  public:
    // 在建構時，從執行緒安全的佇列中彈出一個可用的暫存空間。
    // 如果佇列為空，則會等待直到有其他執行緒歸還暫存空間。
    ScratchStoreManager(ConcurrentQueue<T *> &query_scratch) : _scratch_pool(query_scratch)
    {
        _scratch = query_scratch.pop();
        while (_scratch == nullptr)
        {
            query_scratch.wait_for_push_notify();
            _scratch = query_scratch.pop();
        }
    }
    // 取得暫存空間的指標。
    T *scratch_space()
    {
        return _scratch;
    }

    // 在解構時 (離開作用域時)，自動清理暫存空間並將其歸還到佇列中，
    // 以便其他執行緒可以重複使用。
    ~ScratchStoreManager()
    {
        _scratch->clear();
        _scratch_pool.push(_scratch);
        _scratch_pool.push_notify_all();
    }

    // 銷毀佇列中所有的暫存空間物件 (在索引解構時呼叫)。
    void destroy()
    {
        while (!_scratch_pool.empty())
        {
            auto scratch = _scratch_pool.pop();
            while (scratch == nullptr)
            {
                _scratch_pool.wait_for_push_notify();
                scratch = _scratch_pool.pop();
            }
            delete scratch;
        }
    }

  private:
    T *_scratch;
    ConcurrentQueue<T *> &_scratch_pool;
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
};
} // namespace diskann
