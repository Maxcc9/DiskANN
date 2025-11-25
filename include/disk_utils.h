// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案宣告了建立和處理磁碟索引所需的高階輔助函式。
// 這些函式協調了整個複雜的建立流程，包括資料分割、圖建立、PQ壓縮和檔案合併等。

#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "cached_io.h"
#include "common_includes.h"

#include "utils.h"
#include "windows_customizations.h"

namespace diskann
{
// 用於索引預熱 (warmup) 的最大採樣點數。
// 說明：在建立或載入索引時會抽樣一部分向量來做 warmup（例如載入到快取、訓練參數或估算分布）。
//       為避免抽樣時佔用過多記憶體或耗費過長時間，上限設為 100000。
// 影響：數值越大，warmup 的樣本代表性越好，但耗時與記憶體需求也越高。
// 調整建議：小型資料集或受限記憶體可減少；若資料非常大且可接受較長 warmup，可以提高。
const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;

// 用於訓練 PQ 碼本的資料採樣比例（0..1）。
// 說明：訓練 Product Quantization (PQ) codebook 時並不使用全部資料，而是用全資料的此比例來隨機抽樣訓練集。
// 影響：較高比例 -> 更具代表性的 codebook，但訓練時間與記憶體使用增加；較低比例 -> 快速但可能品質下降。
// 調整建議：預設 0.1（10%）通常在大多數資料上平衡品質與成本；對於非常大或重複性高的資料，可降低；對於維度高或分布複雜的資料，考慮提高。
const double PQ_TRAINING_SET_FRACTION = 0.1;

// 預留給快取節點的記憶體空間（以 GB 為單位）。
// 說明：用於估算在 RAM 中快取多少節點以加速查詢（例如把部分圖節點或重排資料放入快取），這是以 GB 為單位的目標空間量。
// 影響：數值越大，可快取更多節點、降低 SSD I/O，但佔用更多主記憶體；需與系統總記憶體協調。
// 調整建議：在有大量 RAM 且期望低延遲的環境可提高；記憶體受限時應降低。
const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;

// 當索引或需要快取的資料小於此閾值（GB）時，才考慮啟用/增加快取策略。
// 說明：此閾值用來決定在多大體積的索引上才值得啟用快取機制；判斷條件為 (index_size_GB < THRESHOLD_FOR_CACHING_IN_GB)。
// 影響：閾值越小 -> 更少索引會滿足條件（因此越不容易啟用快取）；閾值越大 -> 更多情況下會啟用快取（可能導致過度使用 RAM）。
// 調整建議：把閾值設太低會使即便是中小型索引也不啟用快取；把閾值設太高則可能在過多情況下啟用快取，耗盡記憶體。預設 1.0 GB 是保守值，依部署 RAM 與索引典型大小調整。
const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;

// 快取節點數量上限（整數個節點）。
// 說明：在計算可快取節點數時會以此作為上限（即使依 SPACE_FOR_CACHED_NODES_IN_GB 計算結果更大，也不會超過此值）。
// 影響：避免在某些配置下計算出極大快取數導致記憶體耗盡；為保險上限。
// 調整建議：若節點表示非常小且系統 RAM 充足，可提高；對記憶體敏感部署請降低。
const uint32_t NUM_NODES_TO_CACHE = 250000;

// warmup / 試驗搜尋時使用的 L（beam width / 探索寬度）預設值。
// 說明：在進行預熱搜尋或做 beamwidth/參數調整（如自動找 beamwidth）時，使用的 L 值以此為基準。
// 影響：較高的 L 會使預熱搜尋覆蓋更多節點、結果更穩定但耗時較久；較低的 L 更快但可能代表性不足。
// 調整建議：如果要更精準地評估搜尋行為可提高；用於快速 warmup 時可降低。
const uint32_t WARMUP_L = 20;

// KMeans 重複次數（用於 PQ 碼本訓練時多次隨機初始化以避免局部最小值）。
// 說明：在執行 k-means（或類似聚類）訓練 codebook 時會執行多次不同初始中心，取最佳結果。次數越多，找到較佳聚類解的機率越高。
// 影響：增加此值會提高碼本品質的機率，但訓練時間按次數等比例增加。
// 調整建議：12 是常見的折衷；對於要求最高碼本品質且能接受更長訓練時間的情況可增加。
const uint32_t NUM_KMEANS_REPS = 12;

template <typename T, typename LabelT> class PQFlashIndex;

DISKANN_DLLEXPORT double get_memory_budget(const std::string &mem_budget_str);
DISKANN_DLLEXPORT double get_memory_budget(double search_ram_budget_in_gb);
DISKANN_DLLEXPORT void add_new_file_to_single_index(std::string index_file, std::string new_file);

DISKANN_DLLEXPORT size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

DISKANN_DLLEXPORT void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs);

#ifdef EXEC_ENV_OLS
template <typename T>
DISKANN_DLLEXPORT T *load_warmup(MemoryMappedFiles &files, const std::string &cache_warmup_file, uint64_t &warmup_num,
                                 uint64_t warmup_dim, uint64_t warmup_aligned_dim);
#else
template <typename T>
DISKANN_DLLEXPORT T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num, uint64_t warmup_dim,
                                 uint64_t warmup_aligned_dim);
#endif

DISKANN_DLLEXPORT int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix,
                                   const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                                   const uint64_t nshards, uint32_t max_degree, const std::string &output_vamana,
                                   const std::string &medoids_file, bool use_filters = false,
                                   const std::string &labels_to_medoids_file = std::string(""));

DISKANN_DLLEXPORT void extract_shard_labels(const std::string &in_label_file, const std::string &shard_ids_bin,
                                            const std::string &shard_label_file);

template <typename T>
DISKANN_DLLEXPORT std::string preprocess_base_file(const std::string &infile, const std::string &indexPrefix,
                                                   diskann::Metric &distMetric);

template <typename T, typename LabelT = uint32_t>
DISKANN_DLLEXPORT int build_merged_vamana_index(std::string base_file, diskann::Metric _compareMetric, uint32_t L,
                                                uint32_t R, double sampling_rate, double ram_budget,
                                                std::string mem_index_path, std::string medoids_file,
                                                std::string centroids_file, size_t build_pq_bytes, bool use_opq,
                                                uint32_t num_threads, bool use_filters = false,
                                                const std::string &label_file = std::string(""),
                                                const std::string &labels_to_medoids_file = std::string(""),
                                                const std::string &universal_label = "", const uint32_t Lf = 0);

// 自動優化光束寬度 (beamwidth)。
// 透過在一小部分調校樣本上執行搜尋，自動找到一個在速度和精度之間取得良好平衡的 beamwidth 值。
template <typename T, typename LabelT>
DISKANN_DLLEXPORT uint32_t optimize_beamwidth(std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> &_pFlashIndex,
                                              T *tuning_sample, uint64_t tuning_sample_num,
                                              uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
                                              uint32_t start_bw = 2);

// 建立磁碟索引的主進入點函式，由 `apps/build_disk_index.cpp` 呼叫。
// 這個函式是整個磁碟索引建立流程的總指揮，它接收所有參數並呼叫其他輔助函式來完成：
// 1. (如果需要) 分割資料
// 2. 建立 Vamana 圖 (build_merged_vamana_index)
// 3. 訓練 PQ 碼本
// 4. 使用訓練好的碼本壓縮原始資料
// 5. 組裝最終的磁碟索引檔案佈局 (create_disk_layout)
template <typename T, typename LabelT = uint32_t>
DISKANN_DLLEXPORT int build_disk_index(
    const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
    diskann::Metric _compareMetric, bool use_opq = false,
    const std::string &codebook_prefix = "", // default is empty for no codebook pass in
    bool use_filters = false,
    const std::string &label_file = std::string(""), // default is empty string for no label_file
    const std::string &universal_label = "", const uint32_t filter_threshold = 0,
    const uint32_t Lf = 0); // default is empty string for no universal label

// 建立最終的磁碟索引檔案佈局。
// 這個函式會將 Vamana 圖檔案、壓縮後的向量資料檔案、以及可選的全精度重排序資料檔案，
// 按照為了優化 I/O 模式而設計的特定佈局，合併成一個最終的索引檔案，以供 PQFlashIndex 載入。
template <typename T>
DISKANN_DLLEXPORT void create_disk_layout(const std::string base_file, const std::string mem_index_file,
                                          const std::string output_file,
                                          const std::string reorder_data_file = std::string(""));

} // namespace diskann
