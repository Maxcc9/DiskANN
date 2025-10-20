// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案宣告了與資料分割 (partitioning) 相關的函式。
// 這些函式主要用於 k-means 叢集演算法，是離線建立 PQ 碼本 (pivots) 的關鍵步驟。
// 整個流程包括：對資料進行採樣、執行 k-means 找到中心點 (pivots)、然後根據中心點將完整資料集分割成多個分區 (shards)。

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"

#include "windows_customizations.h"

// 從一個大型資料檔案中隨機抽樣一部分資料，並存成新的檔案。
// 這通常用於為 k-means 演算法準備訓練資料。
template <typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate);

// 從資料檔案中隨機抽樣，並將結果載入到記憶體中的 `sampled_data` 緩衝區。
template <typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims);

// 從一個已在記憶體中的資料陣列 `inputdata` 中隨機抽樣。
template <typename T>
void gen_random_slice(const T *inputdata, size_t npts, size_t ndims, double p_val, float *&sampled_data,
                      size_t &slice_size);

// 給定一組測試資料和中心點 (pivots)，估算每個中心點 (叢集) 會包含多少測試資料點。
int estimate_cluster_sizes(float *test_data_float, size_t num_test, float *pivots, const size_t num_centers,
                           const size_t dim, const size_t k_base, std::vector<size_t> &cluster_sizes);

// 將完整的資料集根據給定的中心點 (pivots) 分割成多個叢集 (分區)。
// 對於每個中心點，會產生一個新的二進位檔案，其中包含所有最接近該中心點的資料向量。
template <typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path);

// 功能與 `shard_data_into_clusters` 類似，但不寫入實際的向量資料，
// 而是為每個叢集產生一個只包含資料點 ID 的檔案。
template <typename T>
int shard_data_into_clusters_only_ids(const std::string data_file, float *pivots, const size_t num_centers,
                                      const size_t dim, const size_t k_base, std::string prefix_path);

// 給定一個包含資料點 ID 的檔案和原始的完整資料檔案，
// 產生一個新的資料檔案，其中只包含那些 ID 對應的向量資料。
template <typename T>
int retrieve_shard_data_from_ids(const std::string data_file, std::string idmap_filename, std::string data_filename);

// 執行資料分割的主函式。
// 它會協調整個流程：採樣 -> k-means -> 分割資料。
template <typename T>
int partition(const std::string data_file, const float sampling_rate, size_t num_centers, size_t max_k_means_reps,
              const std::string prefix_path, size_t k_base);

// 根據給定的記憶體預算來執行資料分割。
// 這個函式會自動計算出在給定記憶體限制下最優的叢集數量，然後再執行分割。
template <typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base);
