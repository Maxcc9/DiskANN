# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# 本檔案定義了 `StaticMemoryIndex` 類別，這是使用者與靜態記憶體索引互動的主要 Python 介面。
# 它封裝了底層的 C++ 綁定，提供了參數驗證和更易於使用的 API。

import json
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
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_index_prefix,
    _valid_metric,
)

__ALL__ = ["StaticMemoryIndex"]


class StaticMemoryIndex:
    """
    一個不可變的、完全載入到記憶體中的 DiskANN 索引。
    """

    def __init__(
        self,
        index_directory: str,
        num_threads: int,
        initial_search_complexity: int,
        index_prefix: str = "ann",
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
        enable_filters: bool = False
    ):
        """
        建構函式，用於載入一個已存在的靜態記憶體索引。

        ### 參數
        - **index_directory**: 包含索引檔案的目錄。
        - **num_threads**: 搜尋時使用的執行緒數。
        - **initial_search_complexity**: 初始化時為搜尋分配的候選集大小。
        - ... (其他參數)
        """
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)
        self._labels_map = {}
        self._labels_metadata = {}
        if enable_filters:
            try:
                with open(f"{index_prefix_path}_labels_map.txt", "r") as labels_map_if:
                    for line in labels_map_if:
                        (key, val) = line.split("\t")
                        self._labels_map[key] = int(val)
                with open(f"{index_prefix_path}_label_metadata.json", "r") as labels_metadata_if:
                    self._labels_metadata = json.load(labels_metadata_if)
            except: # noqa: E722
                # exceptions are basically presumed to be either file not found or file not formatted correctly
                raise RuntimeException("Filter labels file was unable to be processed.")
        vector_dtype, metric, num_points, dims = _ensure_index_metadata(
            index_prefix_path,
            vector_dtype,
            distance_metric,
            1,  # 在此上下文中 max_vectors 不重要
            dimensions,
        )
        dap_metric = _valid_metric(metric)

        # --- 參數驗證 ---
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_positive_uint32(
            initial_search_complexity, "initial_search_complexity"
        )

        self._vector_dtype = vector_dtype
        self._dimensions = dims

        # --- 根據資料類型，選擇對應的底層 C++ 原生類別 ---
        if vector_dtype == np.uint8:
            _index = _native_dap.StaticMemoryUInt8Index
        elif vector_dtype == np.int8:
            _index = _native_dap.StaticMemoryInt8Index
        else:
            _index = _native_dap.StaticMemoryFloatIndex

        # --- 實例化並載入 C++ 索引物件 ---
        # 這裡會呼叫 C++ 封裝層的建構函式，進而觸發底層 C++ 索引的 `load` 方法。
        self._index = _index(
            distance_metric=dap_metric,
            num_points=num_points,
            dimensions=dims,
            index_path=index_prefix_path,
            num_threads=num_threads,
            initial_search_complexity=initial_search_complexity,
        )

    def search(
            self, query: VectorLike, k_neighbors: int, complexity: int, filter_label: str = ""
    ) -> QueryResponse:
        """
        對單一查詢向量執行搜尋。

        ### 參數
        - **query**: 一維 NumPy 陣列，其維度和 dtype 必須與索引匹配。
        - **k_neighbors**: 要返回的鄰居數量。
        - **complexity**: 搜尋時使用的候選集大小 (L)。L 越大，召回率越高，但延遲也越高。
        - **filter_label**: (可選) 用於過濾搜尋的標籤字串。
        """
        # --- 搜尋前的參數驗證 ---
        if filter_label != "":
            if len(self._labels_map) == 0:
                raise ValueError(
                    f"A filter label of {filter_label} was provided, but this class was not initialized with filters "
                    "enabled, e.g. StaticDiskMemory(..., enable_filters=True)"
                )
            if filter_label not in self._labels_map:
                raise ValueError(
                    f"A filter label of {filter_label} was provided, but the external(str)->internal(np.uint32) labels map "
                    f"does not include that label."
                )
            k_neighbors = min(k_neighbors, self._labels_metadata[filter_label])
        _query = _castable_dtype_or_raise(query, expected=self._vector_dtype)
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert(
            _query.shape[0] == self._dimensions,
            f"query vector must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_query.shape[0]}",
            )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_nonnegative_uint32(complexity, "complexity")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        # --- 呼叫底層 C++ 搜尋函式 ---
        if filter_label == "":
            neighbors, distances = self._index.search(query=_query, knn=k_neighbors, complexity=complexity)
        else:
            filter = self._labels_map[filter_label]
            neighbors, distances = self._index.search_with_filter(
                query=query,
                knn=k_neighbors,
                complexity=complexity,
                filter=filter
            )
        
        # --- 將結果封裝成更易於使用的 NamedTuple 並返回 ---
        return QueryResponse(identifiers=neighbors, distances=distances)


    def batch_search(
        self,
        queries: VectorLikeBatch,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
    ) -> QueryResponseBatch:
        """
        對一批查詢向量執行批次搜尋。這種方式比逐一搜尋效率更高，因為它在底層利用了多執行緒平行處理。

        ### 參數
        - **queries**: 二維 NumPy 陣列，(查詢數量, 維度)。
        - ... (其他參數與 search 類似)
        """

        # --- 批次搜尋前的參數驗證 ---
        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert(len(_queries.shape) == 2, "queries must must be 2-d np array")
        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_queries.shape[1]}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        num_queries, dim = _queries.shape
        neighbors, distances = self._index.batch_search(
            queries=_queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
        )
        
        # --- 將結果封裝成 NamedTuple 並返回 ---
        return QueryResponseBatch(identifiers=neighbors, distances=distances)
