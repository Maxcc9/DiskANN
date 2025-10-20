// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案宣告了專案中使用的數學工具函式，
// 特別是 k-means 叢集演算法的核心元件。

#include "common_includes.h"
#include "utils.h"

namespace math_utils
{

// 計算兩個浮點數向量的 L2 距離平方
float calc_distance(float *vec_1, float *vec_2, size_t dim);

// 預先計算一批向量的 L2 範數平方值。這在 k-means 中是個優化，
// 因為點的範數在迭代過程中不會改變。
void compute_vecs_l2sq(float *vecs_l2sq, float *data, const size_t num_points, const size_t dim);

// (未使用) 使用一個隨機旋轉矩陣來旋轉資料。
void rotate_data_randomly(float *data, size_t num_points, size_t dim, float *rot_mat, float *&new_mat,
                          bool transpose_rot = false);

// k-means 的「分配步驟」(Assignment Step) 的核心實作。
// 計算一小批資料點到所有中心點的距離，並找出最近的 k 個中心點。
void compute_closest_centers_in_block(const float *const data, const size_t num_points, const size_t dim,
                                      const float *const centers, const size_t num_centers,
                                      const float *const docs_l2sq, const float *const centers_l2sq,
                                      uint32_t *center_index, float *const dist_matrix, size_t k = 1);

// `compute_closest_centers_in_block` 的包裝函式，處理大規模資料。
void compute_closest_centers(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers,
                             size_t k, uint32_t *closest_centers_ivf, std::vector<size_t> *inverted_index = NULL,
                             float *pts_norms_squared = NULL);

// 計算殘差向量 (data_point - closest_center)。
// 這是多階段量化 (如 Residual Quantization) 的關鍵步驟。
void process_residuals(float *data_load, size_t num_points, size_t dim, float *cur_pivot_data, size_t num_centers,
                       uint32_t *closest_centers, bool to_subtract);

} // namespace math_utils

// k-means 演算法的命名空間
namespace kmeans
{

// 執行一次 Lloyd's 演算法的迭代。
// 包含「分配步驟」(將點分配到最近的中心) 和「更新步驟」(重新計算每個叢集的質心)。
float lloyds_iter(float *data, size_t num_points, size_t dim, float *centers, size_t num_centers, float *docs_l2sq,
                  std::vector<size_t> *closest_docs, uint32_t *&closest_center);

// 執行完整的 Lloyd's k-means 演算法。
// 它會重複呼叫 `lloyds_iter` 直到達到最大迭代次數或收斂為止。
float run_lloyds(float *data, size_t num_points, size_t dim, float *centers, const size_t num_centers,
                 const size_t max_reps, std::vector<size_t> *closest_docs, uint32_t *closest_center);

// 隨機選擇 k 個點作為初始中心點 (pivots)。
void selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers);

// 使用 k-means++ 演算法來選擇初始中心點。
// k-means++ 能選擇出分佈更均勻的初始中心點，有助於演算法更快、更穩定地收斂。
// 其核心思想是：下一個中心點應該盡量遠離已經選擇的中心點。
void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers);
} // namespace kmeans
