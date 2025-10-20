# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# 本檔案定義了 `StaticDiskIndex` 類別，這是使用者與靜態磁碟索引互動的主要 Python 介面。
# 它封裝了底層的 C++ `PQFlashIndex` 綁定，提供了參數驗證和更易於使用的 API。

import os
import warnings
from typing import Optional

import numpy as np

from . import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatch,
    VectorDType,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_2d,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_index_prefix,
    _valid_metric,
)

__ALL__ = ["StaticDiskIndex"]


class StaticDiskIndex:
    """
    一個不可變的、以磁碟為後端的 DiskANN 索引。
    """

    def __init__(
        self,
        index_directory: str,
        num_threads: int,
        num_nodes_to_cache: int,
        cache_mechanism: int = 1,
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
        index_prefix: str = "ann",
    ):
        """
        建構函式，用於載入一個已存在的磁碟索引。

        ### 參數
        - **index_directory**: 包含索引檔案的目錄。
        - **num_threads**: 搜尋時使用的執行緒數。
        - **num_nodes_to_cache**: 要快取到記憶體中的節點數量。
        - **cache_mechanism**: 快取策略。1=根據樣本查詢快取熱點，2=快取圖的頂層節點。
        - ... (其他元資料參數)
        """
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)
        
        # --- 載入元資料 ---
        vector_dtype, metric, _, _ = _ensure_index_metadata(
            index_prefix_path,
            vector_dtype,
            distance_metric,
            1,  # 在此上下文中 max_vectors 不重要
            dimensions,
        )
        dap_metric = _valid_metric(metric)

        # --- 參數驗證 ---
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(num_nodes_to_cache, "num_nodes_to_cache")

        self._vector_dtype = vector_dtype
        
        # --- 根據資料類型，選擇對應的底層 C++ 原生類別 ---
        if vector_dtype == np.uint8:
            _index = _native_dap.StaticDiskUInt8Index
        elif vector_dtype == np.int8:
            _index = _native_dap.StaticDiskInt8Index
        else:
            _index = _native_dap.StaticDiskFloatIndex
        
        # --- 實例化並載入 C++ 索引物件 ---
        # 這裡會呼叫 C++ 封裝層的建構函式，進而觸發底層 C++ `PQFlashIndex` 的 `load` 方法，
        # 並根據 cache_mechanism 執行預熱。
        self._index = _index(
            distance_metric=dap_metric,
            index_path_prefix=index_prefix_path,
            num_threads=num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
            cache_mechanism=cache_mechanism,
        )

    def search(
        self, query: VectorLike, k_neighbors: int, complexity: int, beam_width: int = 2
    ) -> QueryResponse:
        """
        對單一查詢向量執行搜尋。

        ### 參數
        - **query**: 一維 NumPy 陣列。
        - **k_neighbors**: 要返回的鄰居數量。
        - **complexity**: 搜尋候選集大小 (L)。
        - **beam_width**: 光束寬度，控制每次迭代發出的 I/O 請求數量，是 SSD 索引的關鍵效能參數。
        """
        # --- 搜尋前的參數驗證 ---
        _query = _castable_dtype_or_raise(query, expected=self._vector_dtype)
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(beam_width, "beam_width")

        if k_neighbors > complexity:
            warnings.warn(
                f"{k_neighbors=} asked for, but {complexity=} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        # --- 呼叫底層 C++ 搜尋函式 ---
        neighbors, distances = self._index.search(
            query=_query,
            knn=k_neighbors,
            complexity=complexity,
            beam_width=beam_width,
        )
        return QueryResponse(identifiers=neighbors, distances=distances)

    def batch_search(
        self,
        queries: VectorLikeBatch,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
        beam_width: int = 2,
    ) -> QueryResponseBatch:
        """
        對一批查詢向量執行平行的批次搜尋。
        """
        # --- 批次搜尋前的參數驗證 ---
        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert_2d(_queries, "queries")
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_positive_uint32(beam_width, "beam_width")

        if k_neighbors > complexity:
            warnings.warn(
                f"{k_neighbors=} asked for, but {complexity=} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        num_queries, dim = _queries.shape
        neighbors, distances = self._index.batch_search(
            queries=_queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            beam_width=beam_width,
            num_threads=num_threads,
        )
        return QueryResponseBatch(identifiers=neighbors, distances=distances)
