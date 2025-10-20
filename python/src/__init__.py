# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
# Documentation Overview
`diskannpy` is mostly structured around 2 distinct processes: [Index Builder Functions](#index-builders) and [Search Classes](#search-classes)
...
"""

# 本檔案是 `diskann` Python 套件的主進入點。
# 它定義了套件的公開 API，透過從內部模組匯入特定的類別和函式，
# 並使用 `__all__` 來控制哪些名稱是公開的。

from typing import Any, Literal, NamedTuple, Type, Union

import numpy as np
from numpy import typing as npt

# --- 型別別名定義 ---
# 為了提高程式碼可讀性和靜態型別檢查的準確性，這裡定義了一系列的型別別名。

DistanceMetric = Literal["l2", "mips", "cosine"]
""" 距離度量的型別別名，可為 {"l2", "mips", "cosine"} 中的一種 """
VectorDType = Union[Type[np.float32], Type[np.int8], Type[np.uint8]]
""" 向量資料型別的型別別名，可為 {`numpy.float32`, `numpy.int8`, `numpy.uint8`} 中的一種 """
VectorLike = npt.NDArray[VectorDType]
""" 可被視為向量的型別別名 (即一個 NumPy 陣列) """
VectorLikeBatch = npt.NDArray[VectorDType]
""" 一批向量的型別別名 """
VectorIdentifier = np.uint32
""" 向量識別碼的型別別名，通常是 uint32 整數 """
VectorIdentifierBatch = npt.NDArray[np.uint32]
""" 一批向量識別碼的型別別名 """


# --- 標準化的搜尋回應型別 ---
# 使用 NamedTuple 可以讓搜尋結果更具可讀性，而不僅僅是一個原始的元組。

class QueryResponse(NamedTuple):
    """
    單一查詢的回應，包含兩個一維陣列：識別碼和距離。
    """

    identifiers: npt.NDArray[VectorIdentifier]
    """ 向量識別碼的一維 NumPy 陣列 """
    distances: npt.NDArray[np.float32]
    """ 對應的距離值的一維 NumPy 陣列 """


class QueryResponseBatch(NamedTuple):
    """
    批次查詢的回應，包含兩個二維陣列：識別碼和距離。
    """

    identifiers: npt.NDArray[VectorIdentifier]
    """ 向量識別碼的二維 NumPy 陣列，(查詢數量, k) """
    distances: np.ndarray[np.float32]
    """ 對應的距離值的二維 NumPy 陣列，(查詢數量, k) """


# --- 從內部模組匯入，以建立公開 API ---
# 這是一個常見的 Python 模式，將實作細節放在底線開頭的內部模組中，
# 然後在 __init__.py 中匯入並公開一個乾淨的 API。

from . import defaults
from ._builder import build_disk_index, build_memory_index
from ._common import valid_dtype
from ._dynamic_memory_index import DynamicMemoryIndex
from ._files import (
    Metadata,
    tags_from_file,
    tags_to_file,
    vectors_from_file,
    vectors_metadata_from_file,
    vectors_to_file,
)
from ._static_disk_index import StaticDiskIndex
from ._static_memory_index import StaticMemoryIndex

# `__all__` 列表明確定義了當使用者執行 `from diskann import *` 時，
# 哪些名稱會被匯入。這有助於避免汙染使用者的命名空間。
__all__ = [
    "build_disk_index",
    "build_memory_index",
    "StaticDiskIndex",
    "StaticMemoryIndex",
    "DynamicMemoryIndex",
    "defaults",
    "DistanceMetric",
    "VectorDType",
    "QueryResponse",
    "QueryResponseBatch",
    "VectorIdentifier",
    "VectorIdentifierBatch",
    "VectorLike",
    "VectorLikeBatch",
    "Metadata",
    "vectors_metadata_from_file",
    "vectors_to_file",
    "vectors_from_file",
    "tags_to_file",
    "tags_from_file",
    "valid_dtype",
]
