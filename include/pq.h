// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案定義了產品量化 (Product Quantization, PQ) 的核心資料結構與函式。
// PQ 是一種向量壓縮技術，它將高維向量分割成多個區塊 (chunks)，
// 並對每個區塊分別進行量化，從而大幅減少儲存空間並加速距離計算。

#include "utils.h"
#include "pq_common.h"

namespace diskann
{

// 固定區塊的產品量化表 (Codebook)
// 這個類別儲存了 PQ 的碼本 (也稱為 pivots 或 centroids)，並提供了使用碼本進行距離計算的方法。
class FixedChunkPQTable
{
    // PQ 碼本，一個浮點數陣列，大小為 [256 * ndims]。
    // 儲存了每個區塊的 256 個中心點。
    float *tables = nullptr; 
    uint64_t ndims = 0;      // 向量的原始維度
    uint64_t n_chunks = 0;   // 向量被分割成的區塊數量

    // --- OPQ (Optimized Product Quantization) 相關成員 ---
    bool use_rotation = false;      // 是否使用 OPQ 的旋轉矩陣
    uint32_t *chunk_offsets = nullptr; // 每個區塊在原始向量中的起始偏移量
    float *centroid = nullptr;      // 訓練資料的質心，用於將向量中心化 (zero-mean)
    float *tables_tr = nullptr;     // 轉置後的 PQ 碼本，用於優化計算
    float *rotmat_tr = nullptr;     // 轉置後的旋轉矩陣

  public:
    FixedChunkPQTable();

    virtual ~FixedChunkPQTable();

    // 從二進位檔案載入 PQ 碼本 (pivots/centroids)
#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks);
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks);
#endif

    // 取得區塊數量
    uint32_t get_num_chunks();

    // 預處理查詢向量。這是一個關鍵的優化步驟。
    // 它會對查詢向量進行中心化和旋轉 (如果使用 OPQ)，為後續的距離計算做準備。
    void preprocess_query(float *query_vec);

    // 填充區塊距離表。它會計算預處理後的查詢向量與碼本中所有中心點的距離，
    // 並將結果儲存在 dist_vec 中。後續的距離計算只需查表相加即可。
    void populate_chunk_distances(const float *query_vec, float *dist_vec);

    // 使用查表法計算 L2 距離的平方。
    // query_vec: 預處理過的查詢向量
    // base_vec: PQ 編碼後的基底向量 (資料庫中的向量)
    float l2_distance(const float *query_vec, uint8_t *base_vec);

    // 使用查表法計算內積。
    float inner_product(const float *query_vec, uint8_t *base_vec);

    // 從 PQ 編碼還原 (解壓縮) 成近似的原始向量。
    // 注意：這是一個有損壓縮過程，還原的向量不完全等於原始向量。
    void inflate_vector(uint8_t *base_vec, float *out_vec);

    // 填充區塊內積表，類似於 populate_chunk_distances，但用於內積計算。
    void populate_chunk_inner_products(const float *query_vec, float *dist_vec);
};

// --- PQ 相關的輔助函式 ---

// 從 all_coords 中聚合指定 ids 的 PQ 碼到 out 中。
void aggregate_coords(const std::vector<unsigned> &ids, const uint8_t *all_coords, const uint64_t ndims, uint8_t *out);

// 使用預先計算好的 PQ 距離表 (pq_dists)，查詢一批點 (pq_ids) 的近似距離。
void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts, const size_t pq_nchunks, const float *pq_dists,
                    std::vector<float> &dists_out);

// 上述函式的指標版本
void aggregate_coords(const unsigned *ids, const uint64_t n_ids, const uint8_t *all_coords, const uint64_t ndims,
                      uint8_t *out);

void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts, const size_t pq_nchunks, const float *pq_dists,
                    float *dists_out);

// --- 離線處理函式 (用於建立索引) ---

// 產生 PQ 碼本 (pivots)。它使用 k-means 演算法在訓練資料上為每個區塊找出 256 個中心點。
DISKANN_DLLEXPORT int generate_pq_pivots(const float *const train_data, size_t num_train, unsigned dim,
                                         unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
                                         std::string pq_pivots_path, bool make_zero_mean = false);

// 產生 OPQ 碼本。除了 k-means，它還會學習一個旋轉矩陣，以最小化量化誤差。
DISKANN_DLLEXPORT int generate_opq_pivots(const float *train_data, size_t num_train, unsigned dim, unsigned num_centers,
                                          unsigned num_pq_chunks, std::string opq_pivots_path,
                                          bool make_zero_mean = false);

// 簡化版的 PQ 碼本產生函式，將結果直接存入 vector 中。
DISKANN_DLLEXPORT int generate_pq_pivots_simplified(const float *train_data, size_t num_train, size_t dim,
                                                    size_t num_pq_chunks, std::vector<float> &pivot_data_vector);

// 使用產生的碼本 (pivots) 將原始資料集轉換為 PQ 編碼的壓縮檔。
template <typename T>
int generate_pq_data_from_pivots(const std::string &data_file, unsigned num_centers, unsigned num_pq_chunks,
                                 const std::string &pq_pivots_path, const std::string &pq_compressed_vectors_path,
                                 bool use_opq = false);

// 簡化版的 PQ 資料產生函式，處理記憶體中的資料並將結果存入 vector。
DISKANN_DLLEXPORT int generate_pq_data_from_pivots_simplified(const float *data, const size_t num,
                                                              const float *pivot_data, const size_t pivots_num,
                                                              const size_t dim, const size_t num_pq_chunks,
                                                              std::vector<uint8_t> &pq);

// 為磁碟索引產生量化資料 (通常使用較低維度的 PQ)。
template <typename T>
void generate_disk_quantized_data(const std::string &data_file_to_use, const std::string &disk_pq_pivots_path,
                                  const std::string &disk_pq_compressed_vectors_path,
                                  const diskann::Metric compareMetric, const double p_val, size_t &disk_pq_dims);

// 產生量化資料的通用函式。
template <typename T>
void generate_quantized_data(const std::string &data_file_to_use, const std::string &pq_pivots_path,
                             const std::string &pq_compressed_vectors_path, const diskann::Metric compareMetric,
                             const double p_val, const uint64_t num_pq_chunks, const bool use_opq,
                             const std::string &codebook_prefix = "");
} // namespace diskann
