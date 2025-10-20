// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "common_includes.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

template <typename T, typename LabelT = uint32_t> class PQFlashIndex
{
  public:
    // 建構函式
    // fileReader: 用於讀取檔案的對齊檔案讀取器
    // metric: 距離度量 (例如 L2, Inner Product)
    DISKANN_DLLEXPORT PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                                   diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    // 載入索引。它會讀取 PQ 壓縮資料、圖索引檔案，並準備好進行搜尋。
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix);
#endif

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                   const char *index_filepath, const char *pivots_filepath,
                                                   const char *compressed_filepath);
#else
    // 從不同的路徑載入索引的各個部分 (圖、PQ pivot資料、壓縮向量)。
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath);
#endif

    // 將節點列表載入到快取中，以加速搜尋。
    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                   uint64_t l_search, uint64_t beamwidth,
                                                                   uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                   std::vector<uint32_t> &node_list);
#else
    // 根據樣本查詢產生要快取的節點列表。
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                   uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                   uint32_t num_threads,
                                                                   std::vector<uint32_t> &node_list);
#endif

    // 快取圖的 BFS (廣度優先搜尋) 層級中的節點。
    DISKANN_DLLEXPORT void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                            const bool shuffle = false);

    // 快取光束搜尋 (cached beam search)。這是 SSD 索引的主要搜尋函式。
    // query: 查詢向量
    // k_search: 要回傳的最近鄰數量
    // l_search: 搜尋候選列表的大小
    // res_ids: 儲存結果的 ID
    // res_dists: 儲存結果的距離
    // beam_width: 光束寬度，影響搜尋的廣度
    // use_reorder_data: 是否使用重新排序的完整精度資料進行最終排名
    // stats: (可選) 用於收集搜尋統計資訊的結構
    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr);

    // 帶過濾條件的快取光束搜尋。
    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr);

    // 帶 I/O 限制的快取光束搜尋。
    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr);

    // 帶過濾條件和 I/O 限制的快取光束搜尋。
    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr);

    // 從字串標籤取得其對應的內部整數標籤。
    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &filter_label);

    // 範圍搜尋。
    DISKANN_DLLEXPORT uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                            const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                            std::vector<float> &distances, const uint64_t min_beam_width,
                                            QueryStats *stats = nullptr);

    // 取得資料維度。
    DISKANN_DLLEXPORT uint64_t get_data_dim();

    // 對齊檔案讀取器的共享指標。
    std::shared_ptr<AlignedFileReader> &reader;

    // 取得距離度量。
    DISKANN_DLLEXPORT diskann::Metric get_metric();

    //
    // node_ids: 要讀取的節點 ID 列表
    // coord_buffers: 指向預先配置的緩衝區的指標，座標將被複製到此處。如果為 null，則不複製。
    // nbr_buffers: 預先配置的緩衝區，用於複製鄰居節點。
    //
    // returns a vector of bool one for each node_id: true if read is success, else false
    //
    // 從 SSD 讀取節點的座標和鄰居資訊。
    DISKANN_DLLEXPORT std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);

    // 取得指定向量 ID 的 PQ 編碼向量。
    DISKANN_DLLEXPORT std::vector<std::uint8_t> get_pq_vector(std::uint64_t vid);
    // 取得索引中的點總數。
    DISKANN_DLLEXPORT uint64_t get_num_points();

  protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

  private:
    DISKANN_DLLEXPORT inline bool point_has_label(uint32_t point_id, LabelT label_id);
    std::unordered_map<std::string, LabelT> load_label_map(std::basic_istream<char> &infile);
    DISKANN_DLLEXPORT void parse_label_file(std::basic_istream<char> &infile, size_t &num_pts_labels);
    DISKANN_DLLEXPORT void get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                   uint32_t &num_total_labels);
    DISKANN_DLLEXPORT void generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                  const uint32_t nthreads);
    void reset_stream_for_reading(std::basic_istream<char> &infile);

    // sector # on disk where node_id is present with in the graph part
    // 取得節點儲存在磁碟上的哪個磁區 (sector)。
    DISKANN_DLLEXPORT uint64_t get_node_sector(uint64_t node_id);

    // ptr to start of the node
    // 從磁區緩衝區中取得指向特定節點資料起始位置的指標。
    DISKANN_DLLEXPORT char *offset_to_node(char *sector_buf, uint64_t node_id);

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    // 從節點資料緩衝區中取得指向鄰居資訊的指標。
    DISKANN_DLLEXPORT uint32_t *offset_to_node_nhood(char *node_buf);

    // returns region of `node_buf` containing [COORD(T)]
    // 從節點資料緩衝區中取得指向座標 (向量) 資料的指標。
    DISKANN_DLLEXPORT T *offset_to_node_coords(char *node_buf);

    // index info for multi-node sectors
    // (多節點磁區的索引資訊)
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // (節點 i 的鄰居資訊在磁區: [i / nnodes_per_sector])
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // (在磁區內的偏移量: [(i % nnodes_per_sector) * max_node_len])
    //
    // index info for multi-sector nodes
    // (多磁區節點的索引資訊)
    // nhood of node `i` is in sector: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)]
    // (節點 i 的鄰居資訊在磁區: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)])
    // offset in sector: [0]
    // (在磁區內的偏移量: [0])
    //
    // Common info
    // (通用資訊)
    // coords start at ofsset
    // (座標起始於偏移量)
    // #nbrs of node `i`: *(unsigned*) (offset + disk_bytes_per_point)
    // (節點 i 的鄰居數量: *(unsigned*) (offset + disk_bytes_per_point))
    // nbrs of node `i` : (unsigned*) (offset + disk_bytes_per_point + 1)
    // (節點 i 的鄰居列表: (unsigned*) (offset + disk_bytes_per_point + 1))

    // 節點在磁碟上佔用的最大長度 (bytes)
    uint64_t _max_node_len = 0;
    // 每個磁區儲存的節點數量。如果為 0，表示一個節點可能跨越多個磁區。
    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
    // 圖的最大出度 (max degree)
    uint64_t _max_degree = 0;

    // Data used for searching with re-order vectors
    // 用於重新排序向量搜尋的資料
    uint64_t _ndims_reorder_vecs = 0;
    uint64_t _reorder_data_start_sector = 0;
    uint64_t _nvecs_per_sector = 0;

    // 距離度量
    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    // 僅用於內積搜尋，以重新縮放結果值
    float _max_base_norm = 0.0f;

    // data info
    // 資料資訊
    uint64_t _num_points = 0; // 點的總數
    uint64_t _num_frozen_points = 0; // 凍結點的數量
    uint64_t _frozen_location = 0; // 凍結點的位置
    uint64_t _data_dim = 0; // 原始資料維度
    uint64_t _aligned_dim = 0; // 對齊後的資料維度
    uint64_t _disk_bytes_per_point = 0; // 每個點在磁碟上佔用的位元組數

    // 磁碟索引檔案的路徑
    std::string _disk_index_file;
    // 節點訪問計數器，用於快取策略
    std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;

    // PQ data
    // PQ (Product Quantization) 相關資料
    // _n_chunks = # of chunks ndims is split into
    // (維度被分割成的區塊數)
    // data: char * _n_chunks
    // chunk_size = chunk size of each dimension chunk
    // (每個維度區塊的大小)
    // pq_tables = float* [[2^8 * [chunk_size]] * _n_chunks]
    uint8_t *data = nullptr; // 指向 PQ 編碼資料的指標
    uint64_t _n_chunks; // PQ 的區塊數
    FixedChunkPQTable _pq_table; // PQ 碼本

    // distance comparator
    // 距離比較器
    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    // 對於非常大的資料集：我們甚至對磁碟上的索引也使用 PQ
    bool _use_disk_index_pq = false;
    uint64_t _disk_pq_n_chunks = 0;
    FixedChunkPQTable _disk_pq_table;

    // medoid/start info
    // 圖的入口點 (medoid) / 起始點資訊

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    // 圖預設有一個入口點，但也可以選擇性地擁有多個起始點
    uint32_t *_medoids = nullptr;
    // defaults to 1
    // medoid 的數量，預設為 1
    size_t _num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    // 質心資料，如果有多個質心，我們會選擇最接近的質心對應的 medoid 作為搜尋起始點
    float *_centroid_data = nullptr;

    // nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
    // 鄰居快取；nhood_Cache 中的 uint32_t 是指向 nhood_cache_buf 的偏移量
    unsigned *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

    // coord_cache; The T* in coord_cache are offsets into coord_cache_buf
    // 座標快取；coord_cache 中的 T* 是指向 coord_cache_buf 的偏移量
    T *_coord_cache_buf = nullptr;
    tsl::robin_map<uint32_t, T *> _coord_cache;

    // thread-specific scratch
    // 每個執行緒專用的暫存空間
    ConcurrentQueue<SSDThreadData<T> *> _thread_data;
    uint64_t _max_nthreads;
    bool _load_flag = false;
    // 是否計算節點訪問次數
    bool _count_visited_nodes = false;
    // 是否存在重新排序的資料
    bool _reorder_data_exists = false;
    uint64_t _reoreder_data_offset = 0;

    // filter support
    // 過濾搜尋支援
    uint32_t *_pts_to_label_offsets = nullptr;
    uint32_t *_pts_to_label_counts = nullptr;
    LabelT *_pts_to_labels = nullptr;
    std::unordered_map<LabelT, std::vector<uint32_t>> _filter_to_medoid_ids;
    bool _use_universal_label = false;
    LabelT _universal_filter_label;
    tsl::robin_set<uint32_t> _dummy_pts;
    tsl::robin_set<uint32_t> _has_dummy_pts;
    tsl::robin_map<uint32_t, uint32_t> _dummy_to_real_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _real_to_dummy_map;
    std::unordered_map<std::string, LabelT> _label_map;


#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = defaults::SECTOR_LEN;
    char *getHeaderBytes();
#endif
};
} // namespace diskann
