// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案定義了 DiskANN 最核心的記憶體內圖索引類別 `Index`。
// 這個類別實作了 Vamana 演算法，用於建立、搜尋和維護一個高效的鄰近圖。
// 它是所有其他索引類型 (如 PQFlashIndex) 的基礎。

#include "common_includes.h"

#ifdef EXEC_ENV_OLS
#include "aligned_file_reader.h"
#endif

#include "distance.h"
#include "locking.h"
#include "natural_number_map.h"
#include "natural_number_set.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "in_mem_data_store.h"
#include "in_mem_graph_store.h"
#include "abstract_index.h"

#include "quantized_distance.h"
#include "pq_data_store.h"

#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0
#define DEFAULT_MAXC 750

namespace diskann
{

// 估算索引在記憶體中的大致使用量
inline double estimate_ram_usage(size_t size, uint32_t dim, uint32_t datasize, uint32_t degree)
{
    double size_of_data = ((double)size) * ROUND_UP(dim, 8) * datasize;
    double size_of_graph = ((double)size) * degree * sizeof(uint32_t) * defaults::GRAPH_SLACK_FACTOR;
    double size_of_locks = ((double)size) * sizeof(non_recursive_mutex);
    double size_of_outer_vector = ((double)size) * sizeof(ptrdiff_t);

    return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector);
}

// T:    向量的資料型別 (例如 float, int8_t)
// TagT: 外部用來識別向量的標籤型別 (例如 uint32_t)
// LabelT: 用於過濾搜尋的標籤型別 (例如 uint32_t)
template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class Index : public AbstractIndex
{
  public:
    // 用於批次建立或載入已存在索引的建構函式。
    // 允許傳入自訂的資料和圖儲存後端。
    DISKANN_DLLEXPORT Index(const IndexConfig &index_config, std::shared_ptr<AbstractDataStore<T>> data_store,
                            std::unique_ptr<AbstractGraphStore> graph_store,
                            std::shared_ptr<AbstractDataStore<T>> pq_data_store = nullptr);

    // 用於增量索引的建構函式 (從頭建立一個可動態新增/刪除資料點的索引)。
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim, const size_t max_points,
                            const std::shared_ptr<IndexWriteParameters> index_parameters,
                            const std::shared_ptr<IndexSearchParams> index_search_params,
                            const size_t num_frozen_pts = 0, const bool dynamic_index = false,
                            const bool enable_tags = false, const bool concurrent_consolidate = false,
                            const bool pq_dist_build = false, const size_t num_pq_chunks = 0,
                            const bool use_opq = false, const bool filtered_index = false);

    DISKANN_DLLEXPORT ~Index();

    // 將圖、資料、元資料和相關的標籤儲存到檔案。
    DISKANN_DLLEXPORT void save(const char *filename, bool compact_before_save = false);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l);
#else
    // 從圖的元資料檔案區段讀取凍結點 (frozen points) 的數量。
    DISKANN_DLLEXPORT static size_t get_graph_num_frozen_points(const std::string &graph_file);

    // 從檔案載入索引。
    DISKANN_DLLEXPORT void load(const char *index_file, uint32_t num_threads, uint32_t search_l);
#endif

    // 取得索引中目前活動的資料點數量。
    DISKANN_DLLEXPORT size_t get_num_points();
    // 取得索引的最大容量。
    DISKANN_DLLEXPORT size_t get_max_points();

    // 檢查一個點的標籤是否與給定的過濾標籤有交集。
    DISKANN_DLLEXPORT bool detect_common_filters(uint32_t point_id, bool search_invocation,
                                                 const std::vector<LabelT> &incoming_labels);

    // 從檔案批次建立索引。可選擇性地傳入標籤向量。
    DISKANN_DLLEXPORT void build(const char *filename, const size_t num_points_to_load,
                                 const std::vector<TagT> &tags = std::vector<TagT>());

    // 從檔案批次建立索引。可選擇性地傳入標籤檔案。
    DISKANN_DLLEXPORT void build(const char *filename, const size_t num_points_to_load, const char *tag_filename);

    // 從記憶體中的資料陣列批次建立索引，向量必須對齊到 aligned_dim。
    DISKANN_DLLEXPORT void build(const T *data, const size_t num_points_to_load, const std::vector<TagT> &tags);

    // 根據過濾器參數建立一個帶過濾功能或不帶過濾功能的索引。
    DISKANN_DLLEXPORT void build(const std::string &data_file, const size_t num_points_to_load,
                                 IndexFilterParams &filter_params);

    // 建立帶過濾功能的索引。
    DISKANN_DLLEXPORT void build_filtered_index(const char *filename, const std::string &label_file,
                                                const size_t num_points_to_load,
                                                const std::vector<TagT> &tags = std::vector<TagT>());

    // 設定通用標籤 (universal label)，該標籤可與任何篩選條件匹配。
    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

    // 從字串標籤取得其對應的內部整數標籤。
    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &raw_label);

    // 在增量插入任何點之前，設定索引的起始點 (frozen points)。
    DISKANN_DLLEXPORT void set_start_points(const T *data, size_t data_count);

    // 將起始點設定為特定半徑球面上的隨機點。
    DISKANN_DLLEXPORT void set_start_points_at_random(T radius, uint32_t random_seed = 0);

    // 優化索引佈局，將資料與圖結構交錯儲存，以提高快取局部性，加速搜尋。
    DISKANN_DLLEXPORT void optimize_index_layout();

    // 在優化佈局上進行 FastL2 搜尋。
    DISKANN_DLLEXPORT void search_with_optimized_layout(const T *query, size_t K, size_t L, uint32_t *indices);

    // 核心搜尋函式。
    // query: 查詢向量
    // K: 要回傳的最近鄰數量
    // L: 搜尋候選列表的大小 (L >= K)，是影響速度和精度的關鍵參數
    // indices: 用於儲存結果索引的陣列
    // distances: (可選) 用於儲存結果距離的陣列
    template <typename IDType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(const T *query, const size_t K, const uint32_t L,
                                                           IDType *indices, float *distances = nullptr);

    // 帶標籤的搜尋。除了索引和距離，還會回傳對應的標籤和向量。
    DISKANN_DLLEXPORT size_t search_with_tags(const T *query, const uint64_t K, const uint32_t L, TagT *tags,
                                              float *distances, std::vector<T *> &res_vectors, bool use_filters = false,
                                              const std::string filter_label = "");

    // 帶過濾條件的搜尋。只會在符合 filter_label 的節點中進行搜尋。
    template <typename IndexType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search_with_filters(const T *query, const LabelT &filter_label,
                                                                        const size_t K, const uint32_t L,
                                                                        IndexType *indices, float *distances);

    // (動態索引) 插入一個點。如果標籤已存在或標籤為0，則會失敗。
    DISKANN_DLLEXPORT int insert_point(const T *point, const TagT tag);

    // (動態索引) 插入一個帶有過濾標籤的點。
    DISKANN_DLLEXPORT int insert_point(const T *point, const TagT tag, const std::vector<LabelT> &label);

    // (動態索引) 啟用刪除功能。在進行任何刪除操作前必須呼叫。
    DISKANN_DLLEXPORT int enable_delete();

    // (動態索引) 延遲刪除：僅將點標記為已刪除，稍後再重組圖。如果找不到標籤，返回-1。
    DISKANN_DLLEXPORT int lazy_delete(const TagT &tag);

    // (動態索引) 批次延遲刪除多個點。
    DISKANN_DLLEXPORT void lazy_delete(const std::vector<TagT> &tags, std::vector<TagT> &failed_tags);

    // (動態索引) 在一系列延遲刪除後呼叫此函式以整理索引，實際移除被標記為刪除的節點。
    // 返回整理後剩餘的活點數。
    DISKANN_DLLEXPORT consolidation_report consolidate_deletes(const IndexWriteParameters &parameters);

    // 修剪所有節點的鄰居列表，使其不超過指定的大小。
    DISKANN_DLLEXPORT void prune_all_neighbors(const uint32_t max_degree, const uint32_t max_occlusion,
                                               const float alpha);

    // 檢查索引是否已經被儲存過。
    DISKANN_DLLEXPORT bool is_index_saved();

    // (動態索引) 將凍結點重新定位到資料陣列的末尾（如果它們在刪除過程中被移動）。
    DISKANN_DLLEXPORT void reposition_frozen_point_to_end();
    DISKANN_DLLEXPORT void reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                             uint32_t num_locations);

    // 取得所有活動的 (未被刪除的) 標籤。
    DISKANN_DLLEXPORT void get_active_tags(tsl::robin_set<TagT> &active_tags);

    // 根據標籤取得對應的向量。
    DISKANN_DLLEXPORT int get_vector_by_tag(TagT &tag, T *vec);

    // 印出索引的狀態資訊。
    DISKANN_DLLEXPORT void print_status();

    // 計算並印出圖中每個 BFS 層級的節點數量。
    DISKANN_DLLEXPORT void count_nodes_at_bfs_levels();

    DISKANN_DLLEXPORT static const int METADATA_ROWS = 5;

  protected:
    // --- 抽象基底類別 AbstractIndex 的虛擬方法實作 ---
    virtual void _build(const DataType &data, const size_t num_points_to_load, TagVector &tags) override;
    virtual std::pair<uint32_t, uint32_t> _search(const DataType &query, const size_t K, const uint32_t L,
                                                  std::any &indices, float *distances = nullptr) override;
    virtual std::pair<uint32_t, uint32_t> _search_with_filters(const DataType &query,
                                                               const std::string &filter_label_raw, const size_t K,
                                                               const uint32_t L, std::any &indices,
                                                               float *distances) override;
    virtual int _insert_point(const DataType &data_point, const TagType tag) override;
    virtual int _insert_point(const DataType &data_point, const TagType tag, Labelvector &labels) override;
    virtual int _lazy_delete(const TagType &tag) override;
    virtual void _lazy_delete(TagVector &tags, TagVector &failed_tags) override;
    virtual void _get_active_tags(TagRobinSet &active_tags) override;
    virtual void _set_start_points_at_random(DataType radius, uint32_t random_seed = 0) override;
    virtual int _get_vector_by_tag(TagType &tag, DataType &vec) override;
    virtual void _search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices) override;
    virtual size_t _search_with_tags(const DataType &query, const uint64_t K, const uint32_t L, const TagType &tags,
                                     float *distances, DataVector &res_vectors, bool use_filters = false,
                                     const std::string filter_label = "") override;
    virtual void _set_universal_label(const LabelType universal_label) override;

    Index(const Index<T, TagT, LabelT> &) = delete;
    Index<T, TagT, LabelT> &operator=(const Index<T, TagT, LabelT> &) = delete;

    void build_with_data_populated(const std::vector<TagT> &tags);
    void generate_frozen_point();
    uint32_t calculate_entry_point();
    void parse_label_file(const std::string &label_file, size_t &num_pts_labels);
    std::unordered_map<std::string, LabelT> load_label_map(const std::string &map_file);
    std::vector<uint32_t> get_init_ids();

    // Vamana 搜尋演算法的核心：從初始點 `init_ids` 開始，在圖上進行貪婪搜尋，
    // 直到找到 L 個候選點的局部最小值，即候選列表不再變化為止。
    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(InMemQueryScratch<T> *scratch, const uint32_t Lindex,
                                                         const std::vector<uint32_t> &init_ids, bool use_filter,
                                                         const std::vector<LabelT> &filters, bool search_invocation);

    void search_for_point_and_prune(int location, uint32_t Lindex, std::vector<uint32_t> &pruned_list,
                                    InMemQueryScratch<T> *scratch, bool use_filter = false,
                                    uint32_t filteredLindex = 0);

    // Vamana 圖建立演算法的核心：Robust Prune。用於為節點選擇一組多樣化且穩健的鄰居。
    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
                         InMemQueryScratch<T> *scratch);
    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                         const uint32_t max_candidate_size, const float alpha, std::vector<uint32_t> &pruned_list,
                         InMemQueryScratch<T> *scratch);
    void occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha, const uint32_t degree,
                      const uint32_t maxc, std::vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
                      const tsl::robin_set<uint32_t> *const delete_set_ptr = nullptr);

    // 圖建立過程中的連接步驟：將節點 n 與其修剪後的鄰居列表 `pruned_list` 互相連接。
    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                      InMemQueryScratch<T> *scratch);
    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch);

    void link();

    // (動態索引) 為新插入的點預留一個位置。
    int reserve_location();
    // (動態索引) 釋放一個或多個位置，使其可被後續插入使用。
    size_t release_location(int location);
    size_t release_locations(const tsl::robin_set<uint32_t> &locations);

    // (動態索引) 當索引滿時，擴展索引的容量。
    void resize(size_t new_max_points);

    // (動態索引) 壓縮資料，物理上移除被刪除的點，並重新整理節點編號。
    DISKANN_DLLEXPORT void compact_data();
    DISKANN_DLLEXPORT void compact_frozen_point();

    void process_delete(const tsl::robin_set<uint32_t> &old_delete_set, size_t loc, const uint32_t range,
                        const uint32_t maxc, const float alpha, InMemQueryScratch<T> *scratch);

    void initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l, uint32_t r,
                                  uint32_t maxc, size_t dim);

    DISKANN_DLLEXPORT size_t save_graph(std::string filename);
    DISKANN_DLLEXPORT size_t save_data(std::string filename);
    DISKANN_DLLEXPORT size_t save_tags(std::string filename);
    DISKANN_DLLEXPORT size_t save_delete_list(const std::string &filename);
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT size_t load_graph(AlignedFileReader &reader, size_t expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_tags(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_delete_set(AlignedFileReader &reader);
#else
    DISKANN_DLLEXPORT size_t load_graph(const std::string filename, size_t expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(std::string filename0);
    DISKANN_DLLEXPORT size_t load_tags(const std::string tag_file_name);
    DISKANN_DLLEXPORT size_t load_delete_set(const std::string &filename);
#endif

  private:
    // --- 核心成員變數 ---
    Metric _dist_metric = diskann::L2; // 距離計算函式
    std::shared_ptr<AbstractDataStore<T>> _data_store; // 資料儲存後端
    std::unique_ptr<AbstractGraphStore> _graph_store; // 圖儲存後端
    char *_opt_graph = nullptr; // 優化後的圖結構，用於 FastL2 搜尋

    size_t _dim = 0; // 向量維度
    size_t _nd = 0; // 索引中活動點的數量
    size_t _max_points = 0; // 索引最大容量

    // _num_frozen_pts: 作為搜尋起點的「凍結點」數量，這些點對外不可見。
    size_t _num_frozen_pts = 0;
    size_t _frozen_pts_used = 0;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    //  Start point of the search. When _num_frozen_pts is greater than zero,
    //  this is the location of the first frozen point. Otherwise, this is a
    //  location of one of the points in index.
    uint32_t _start = 0;

    bool _has_built = false;
    bool _saturate_graph = false;
    bool _save_as_one_file = false; // plan to support in next version
    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _normalize_vecs = false; // Using normalied L2 for cosine.
    bool _deletes_enabled = false;

    // Filter Support

    bool _filtered_index = false;
    // Location to label is only updated during insert_point(), all other reads are protected by
    // default as a location can only be released at end of consolidate deletes
    std::vector<std::vector<LabelT>> _location_to_labels;
    tsl::robin_set<LabelT> _labels;
    std::string _labels_file;
    std::unordered_map<LabelT, uint32_t> _label_to_start_id;
    std::unordered_map<uint32_t, uint32_t> _medoid_counts;

    bool _use_universal_label = false;
    LabelT _universal_label = 0;
    uint32_t _filterIndexingQueueSize;
    std::unordered_map<std::string, LabelT> _label_map;

    // --- 索引建立參數 (Vamana 演算法參數) ---
    uint32_t _indexingQueueSize; // L: 候選列表大小
    uint32_t _indexingRange;     // R: 最大出度 (鄰居數量)
    uint32_t _indexingMaxC;      // C: 搜尋時的最大比較次數
    float _indexingAlpha;        // alpha: RobustPrune 的參數
    uint32_t _indexingThreads;   // 建立索引時的執行緒數量

    // --- 查詢時使用的暫存空間 ---
    ConcurrentQueue<InMemQueryScratch<T> *> _query_scratch;

    // Flags for PQ based distance calculation
    bool _pq_dist = false;
    bool _use_opq = false;
    size_t _num_pq_chunks = 0;
    // REFACTOR
// uint8_t *_pq_data = nullptr;
    std::shared_ptr<QuantizedDistance<T>> _pq_distance_fn = nullptr;
    std::shared_ptr<AbstractDataStore<T>> _pq_data_store = nullptr;
    bool _pq_generated = false;
    FixedChunkPQTable _pq_table;

    //
    // Data structures, locks and flags for dynamic indexing and tags
    //

    // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
    // _location_to_tag does not resolve a location, infer that it was deleted.
    tsl::sparse_map<TagT, uint32_t> _tag_to_location; // 從外部標籤到內部位置的映射
    natural_number_map<uint32_t, TagT> _location_to_tag; // 從內部位置到外部標籤的映射

    // _empty_slots: 可用的空插槽，用於新插入的點
    // _delete_set: 被延遲刪除的點的集合
    natural_number_set<uint32_t> _empty_slots;
    std::unique_ptr<tsl::robin_set<uint32_t>> _delete_set;

    bool _data_compacted = true;    // 資料是否已壓縮
    bool _is_saved = false;         // 索引是否已儲存
    bool _conc_consolidate = false; // 是否啟用並行整理

    // --- 多執行緒控制鎖 ---
    // 獲取鎖時應遵循以下順序: _update_lock -> _consolidate_lock -> _tag_lock -> _delete_lock
    std::shared_timed_mutex _update_lock;      // 存檔/載入(獨佔) vs. 搜尋/插入/刪除(共享) 的讀寫鎖
    std::shared_timed_mutex _consolidate_lock; // 確保一次只有一個整理或壓縮操作
    std::shared_timed_mutex _tag_lock;         // 標籤/位置映射相關資料結構的讀寫鎖
    std::shared_timed_mutex _delete_lock;      // _delete_set 的讀寫鎖

    // 每個節點的鎖，用於細粒度的並行控制
    std::vector<non_recursive_mutex> _locks;

    static const float INDEX_GROWTH_FACTOR;
};
} // namespace diskann
