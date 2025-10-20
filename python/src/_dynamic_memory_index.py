# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# 本檔案定義了 `DynamicMemoryIndex` 類別，這是使用者與動態記憶體索引互動的主要 Python 介面。
# 它封裝了底層的 C++ 綁定，提供了建立、搜尋、插入和刪除向量的功能。

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from . import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatch,
    VectorDType,
    VectorIdentifier,
    VectorIdentifierBatch,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_2d,
    _assert_dtype,
    _assert_existing_directory,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_index_prefix,
    _valid_metric,
    _write_index_metadata,
)
from ._diskannpy import defaults

__ALL__ = ["DynamicMemoryIndex"]


class DynamicMemoryIndex:
    """
    一個可變的、完全載入到記憶體中的 DiskANN 索引。
    與靜態索引不同，它支援向量的插入和刪除。
    """

    @classmethod
    def from_file(
        cls,
        index_directory: str,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True,
        index_prefix: str = "ann",
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
    ) -> "DynamicMemoryIndex":
        """
        一個類別方法，用於從磁碟載入一個先前儲存的動態索引。
        這是載入索引的建議方式。
        """
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)

        # 動態索引必須有 .tags 檔案
        tags_file = index_prefix_path + ".tags"
        _assert(
            Path(tags_file).exists(),
            f"The file {tags_file} does not exist in {index_directory}",
        )
        
        # 讀取或確認元資料
        vector_dtype, dap_metric, num_vectors, dimensions = _ensure_index_metadata(
            index_prefix_path, vector_dtype, distance_metric, max_vectors, dimensions, warn_size_exceeded=True
        )

        # 步驟 1: 呼叫 __init__ 建立一個空的、但已設定好參數的索引物件。
        index = cls(
            distance_metric=dap_metric,  # type: ignore
            vector_dtype=vector_dtype,
            dimensions=dimensions,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation,
        )
        index._index.load(index_prefix_path)
        index._num_vectors = num_vectors  # current number of vectors loaded
        return index

    def __init__(
        self,
        distance_metric: DistanceMetric,
        vector_dtype: VectorDType,
        dimensions: int,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True,
    ):
        """
        The `diskannpy.DynamicMemoryIndex` represents our python API into a mutable DiskANN memory index.

        This constructor is used to create a new, empty index. If you wish to load a previously saved index from disk,
        please use the `diskannpy.DynamicMemoryIndex.from_file` classmethod instead.

        ### Parameters
        - **distance_metric**: A `str`, strictly one of {"l2", "mips", "cosine"}. `l2` and `cosine` are supported for all 3
          vector dtypes, but `mips` is only available for single precision floats.
        - **vector_dtype**: One of {`np.float32`, `np.int8`, `np.uint8`}. The dtype of the vectors this index will
          be storing.
        - **dimensions**: The vector dimensionality of this index. All new vectors inserted must be the same
          dimensionality.
        - **max_vectors**: Capacity of the data store including space for future insertions
        - **graph_degree**: Graph degree (a.k.a. `R`) is the maximum degree allowed for a node in the index's graph
          structure. This degree will be pruned throughout the course of the index build, but it will never grow beyond
          this value. Higher `graph_degree` values require longer index build times, but may result in an index showing
          excellent recall and latency characteristics.
        - **saturate_graph**: If True, the adjacency list of each node will be saturated with neighbors to have exactly
          `graph_degree` neighbors. If False, each node will have between 1 and `graph_degree` neighbors.
        - **max_occlusion_size**: The maximum number of points that can be considered by occlude_list function.
        - **alpha**: The alpha parameter (>=1) is used to control the nature and number of points that are added to the
          graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) to convergence, but probably
          more distance comparisons compared to a lower alpha value.
        - **num_threads**: Number of threads to use when creating this index. `0` indicates we should use all available
          logical processors.
        - **filter_complexity**: Complexity to use when using filters. Default is 0.
        - **num_frozen_points**: Number of points to freeze. Default is 1.
        - **initial_search_complexity**: Should be set to the most common `complexity` expected to be used during the
          life of this `diskannpy.DynamicMemoryIndex` object. The working scratch memory allocated is based off of
          `initial_search_complexity` * `search_threads`. Note that it may be resized if a `search` or `batch_search`
          operation requests a space larger than can be accommodated by these values.
        - **search_threads**: Should be set to the most common `num_threads` expected to be used during the
          life of this `diskannpy.DynamicMemoryIndex` object. The working scratch memory allocated is based off of
          `initial_search_complexity` * `search_threads`. Note that it may be resized if a `batch_search`
          operation requests a space larger than can be accommodated by these values.
        - **concurrent_consolidation**: This flag dictates whether consolidation can be run alongside inserts and
          deletes, or whether the index is locked down to changes while consolidation is ongoing.

        """
        self._num_vectors = 0
        self._removed_num_vectors = 0
        dap_metric = _valid_metric(distance_metric)
        self._dap_metric = dap_metric
        _assert_dtype(vector_dtype)
        _assert_is_positive_uint32(dimensions, "dimensions")

        self._vector_dtype = vector_dtype
        self._dimensions = dimensions

        _assert_is_positive_uint32(max_vectors, "max_vectors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(graph_degree, "graph_degree")
        _assert(
            alpha >= 1,
            "alpha must be >= 1, and realistically should be kept between [1.0, 2.0)",
        )
        _assert_is_nonnegative_uint32(max_occlusion_size, "max_occlusion_size")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
        _assert_is_nonnegative_uint32(num_frozen_points, "num_frozen_points")
        _assert_is_nonnegative_uint32(
            initial_search_complexity, "initial_search_complexity"
        )
        _assert_is_nonnegative_uint32(search_threads, "search_threads")

        self._max_vectors = max_vectors
        self._complexity = complexity
        self._graph_degree = graph_degree

        if vector_dtype == np.uint8:
            _index = _native_dap.DynamicMemoryUInt8Index
        elif vector_dtype == np.int8:
            _index = _native_dap.DynamicMemoryInt8Index
        else:
            _index = _native_dap.DynamicMemoryFloatIndex

        # --- 實例化 C++ 索引物件 ---
        # 這裡會呼叫 C++ 封裝層的建構函式，進而建立一個已設定好動態索引參數的 `diskann::Index` 物件。
        self._index = _index(
            distance_metric=dap_metric,
            dimensions=dimensions,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation,
        )
        self._points_deleted = False

    def search(
        self, query: VectorLike, k_neighbors: int, complexity: int
    ) -> QueryResponse:
        """
        Searches the index by a single query vector.

        ### Parameters
        - **query**: 1d numpy array of the same dimensionality and dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        """
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
        neighbors, distances = self._index.search(query=_query, knn=k_neighbors, complexity=complexity)
        return QueryResponse(identifiers=neighbors, distances=distances)

    def batch_search(
        self,
        queries: VectorLikeBatch,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
    ) -> QueryResponseBatch:
        """
        Searches the index by a batch of query vectors.

        This search is parallelized and far more efficient than searching for each vector individually.

        ### Parameters
        - **queries**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of queries intended to search for in parallel. Dtype must match dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        """
        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert_2d(_queries, "queries")
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

        num_queries, dim = queries.shape
        neighbors, distances = self._index.batch_search(
            queries=_queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
        )
        return QueryResponseBatch(identifiers=neighbors, distances=distances)

    def save(self, save_path: str, index_prefix: str = "ann"):
        """
        Saves this index to file.

        ### Parameters
        - **save_path**: The path to save these index files to.
        - **index_prefix**: The prefix of the index files. Defaults to "ann".
        """
        if save_path == "":
            raise ValueError("save_path cannot be empty")
        if index_prefix == "":
            raise ValueError("index_prefix cannot be empty")

        index_prefix = index_prefix.format(complexity=self._complexity, graph_degree=self._graph_degree)
        _assert_existing_directory(save_path, "save_path")
        save_path = os.path.join(save_path, index_prefix)
        if self._points_deleted is True:
            warnings.warn(
                "DynamicMemoryIndex.save() currently requires DynamicMemoryIndex.consolidate_delete() to be called "
                "prior to save when items have been marked for deletion. This is being done automatically now, though"
                "it will increase the time it takes to save; on large sets of data it can take a substantial amount of "
                "time. In the future, we will implement a faster save with unconsolidated deletes, but for now this is "
                "required."
            )
            self._index.consolidate_delete()
        self._index.save(
            save_path=save_path, compact_before_save=True
        )  # we do not yet support uncompacted saves
        _write_index_metadata(
            save_path,
            self._vector_dtype,
            self._dap_metric,
            self._index.num_points(),
            self._dimensions,
        )

    def insert(self, vector: VectorLike, vector_id: VectorIdentifier):
        """
        Inserts a single vector into the index with the provided vector_id.

        If this insertion will overrun the `max_vectors` count boundaries of this index, `consolidate_delete()` will
        be executed automatically.

        ### Parameters
        - **vector**: The vector to insert. Note that dtype must match.
        - **vector_id**: The vector_id to use for this vector.
        """
        _vector = _castable_dtype_or_raise(vector, expected=self._vector_dtype)
        _assert(len(vector.shape) == 1, "insert vector must be 1-d")
        _assert_is_positive_uint32(vector_id, "vector_id")
        if self._num_vectors + 1 > self._max_vectors:
            if self._removed_num_vectors > 0:
                warnings.warn(f"Inserting this vector would overrun the max_vectors={self._max_vectors} specified at index "
                              f"construction. We are attempting to consolidate_delete() to make space.")
                self.consolidate_delete()
            else:
                raise RuntimeError(f"Inserting this vector would overrun the max_vectors={self._max_vectors} specified "
                                   f"at index construction. Unable to make space by consolidating deletions. The insert"
                                   f"operation has failed.")
        status = self._index.insert(_vector, np.uint32(vector_id))
        if status == 0:
            self._num_vectors += 1
        else:
            raise RuntimeError(
                f"Insert was unable to complete successfully; error code returned from diskann C++ lib: {status}"
            )


    def batch_insert(
        self,
        vectors: VectorLikeBatch,
        vector_ids: VectorIdentifierBatch,
        num_threads: int = 0,
    ):
        """
        Inserts a batch of vectors into the index with the provided vector_ids.

        If this batch insertion will overrun the `max_vectors` count boundaries of this index, `consolidate_delete()`
        will be executed automatically.

        ### Parameters
        - **vectors**: The 2d numpy array of vectors to insert.
        - **vector_ids**: The 1d array of vector ids to use. This array must have the same number of elements as
            the vectors array has rows. The dtype of vector_ids must be `np.uint32`
        - **num_threads**: Number of threads to use when inserting into this index. (>= 0), 0 = num_threads in system
        """
        _query = _castable_dtype_or_raise(vectors, expected=self._vector_dtype)
        _assert(len(vectors.shape) == 2, "vectors must be a 2-d array")
        _assert(
            vectors.shape[0] == vector_ids.shape[0],
            "Number of vectors must be equal to number of ids",
        )
        _vectors = vectors.astype(dtype=self._vector_dtype, casting="safe", copy=False)
        _vector_ids = vector_ids.astype(dtype=np.uint32, casting="safe", copy=False)

        if self._num_vectors + _vector_ids.shape[0] > self._max_vectors:
            if self._max_vectors + self._removed_num_vectors >= _vector_ids.shape[0]:
                warnings.warn(f"Inserting these vectors, count={_vector_ids.shape[0]} would overrun the "
                              f"max_vectors={self._max_vectors} specified at index construction. We are attempting to "
                              f"consolidate_delete() to make space.")
                self.consolidate_delete()
            else:
                raise RuntimeError(f"Inserting these vectors count={_vector_ids.shape[0]} would overrun the "
                                   f"max_vectors={self._max_vectors} specified at index construction. Unable to make "
                                   f"space by consolidating deletions. The batch insert operation has failed.")

        statuses = self._index.batch_insert(
            _vectors, _vector_ids, _vector_ids.shape[0], num_threads
        )
        successes = []
        failures = []
        for i in range(0, len(statuses)):
            if statuses[i] == 0:
                successes.append(i)
            else:
                failures.append(i)
        self._num_vectors += len(successes)
        if len(failures) == 0:
            return
        failed_ids = vector_ids[failures]
        raise RuntimeError(
            f"During batch insert, the following vector_ids were unable to be inserted into the index: {failed_ids}. "
            f"{len(successes)} were successfully inserted"
        )


    def mark_deleted(self, vector_id: VectorIdentifier):
        """
        將向量標記為已刪除 (軟刪除)。
        該向量不會再出現在搜尋結果中，但其實體仍存在於索引結構中。
        """
        _assert_is_positive_uint32(vector_id, "vector_id")
        self._points_deleted = True
        self._removed_num_vectors += 1
        # we do not decrement self._num_vectors until consolidate_delete
        self._index.mark_deleted(np.uint32(vector_id))

    def consolidate_delete(self):
        """
        整理索引，實際地從圖結構中移除所有被標記為刪除的點 (硬刪除)。
        這是一個耗時的操作。
        """
        self._index.consolidate_delete()
        self._points_deleted = False
        self._num_vectors -= self._removed_num_vectors
        self._removed_num_vectors = 0
