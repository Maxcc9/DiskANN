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
// 用於索引預熱 (warmup) 的最大採樣點數
const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
// 用於訓練 PQ 碼本的資料採樣比例
const double PQ_TRAINING_SET_FRACTION = 0.1;
const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
const uint32_t NUM_NODES_TO_CACHE = 250000;
const uint32_t WARMUP_L = 20;
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
