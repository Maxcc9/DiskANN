# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# 本檔案提供了用於處理 DiskANN 特定二進位檔案格式的輔助函式。
# 這些函式負責將 NumPy 陣列序列化為檔案，以及從檔案反序列化回 NumPy 陣列。

import warnings
from typing import BinaryIO, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from . import VectorDType, VectorIdentifierBatch, VectorLikeBatch
from ._common import _assert, _assert_2d, _assert_dtype, _assert_existing_file


class Metadata(NamedTuple):
    """一個簡單的具名元組，用於儲存從二進位檔案中讀取的元資料。"""

    num_vectors: int
    """ 檔案中的向量數量。 """
    dimensions: int
    """ 檔案中向量的維度。 """


def vectors_metadata_from_file(vector_file: str) -> Metadata:
    """
    從 DiskANN 二進位向量檔案中快速讀取元資料，而無需讀取整個檔案。
    它只讀取檔案的前 8 個位元組，這 8 個位元組儲存了兩個 int32 整數：點的數量和維度。
    """
    _assert_existing_file(vector_file, "vector_file")
    points, dims = np.fromfile(file=vector_file, dtype=np.int32, count=2)
    return Metadata(points, dims)


def _write_bin(data: np.ndarray, file_handler: BinaryIO):
    """內部輔助函式，將一個 NumPy 陣列寫入一個二進位檔案流。"""
    if len(data.shape) == 1:
        # 對於一維陣列，維度寫為 (n, 1)
        _ = file_handler.write(np.array([data.shape[0], 1], dtype=np.int32).tobytes())
    else:
        # 對於二維陣列，寫入其形狀 (n, d)
        _ = file_handler.write(np.array(data.shape, dtype=np.int32).tobytes())
    # 寫入 NumPy 陣列的原始位元組資料
    _ = file_handler.write(data.tobytes())


def vectors_to_file(vector_file: str, vectors: VectorLikeBatch) -> None:
    """
    將一個 NumPy 向量陣列寫入 DiskANN 的二進位向量檔案格式。

    ### 參數
    - **vector_file**: 要寫入的檔案路徑。
    - **vectors**: 一個二維的 NumPy 陣列，dtype 必須是支援的類型。
    """
    _assert_dtype(vectors.dtype)
    _assert_2d(vectors, "vectors")
    with open(vector_file, "wb") as fh:
        _write_bin(vectors, fh)


def vectors_from_file(
    vector_file: str,
    dtype: VectorDType,
    use_memmap: bool = False,
    mode: Literal["r", "r+"] = "r"
) -> npt.NDArray[VectorDType]:
    """
    從 DiskANN 的二進位向量檔案中讀取向量。

    ### 參數
    - **vector_file**: 要讀取的檔案路徑。
    - **dtype**: 檔案中向量的資料類型。
    - **use_memmap**: 如果為 True，則返回一個記憶體映射 (memory-mapped) 的 NumPy 陣列。
      這對於不希望一次性將整個大檔案載入到 RAM 的情況非常有用。
    - **mode**: 記憶體映射的模式，'r' (唯讀) 或 'r+' (讀寫)。
    """
    assert mode in ["r", "r+"]
    points, dims = vectors_metadata_from_file(vector_file)
    if not use_memmap:
        # 將整個檔案讀入記憶體
        return np.fromfile(file=vector_file, dtype=dtype, offset=8).reshape(points, dims)
    else:
        # 建立一個記憶體映射陣列
        return np.memmap(vector_file, dtype=dtype, mode=mode, offset=8, shape=(points, dims), order='C')


def tags_to_file(tags_file: str, tags: VectorIdentifierBatch) -> None:
    """
    將標籤 (tags) 寫入 DiskANN 的二進位標籤檔案。
    格式與向量檔案相同，但維度固定為 1。
    """
    _assert(np.can_cast(tags.dtype, np.uint32), "valid tags must be uint32")
    _assert(
        len(tags.shape) == 1 or tags.shape[1] == 1,
        "tags must be 1d or 2d with 1 column",
    )
    if len(tags.shape) == 2:
        warnings.warn(
            "Tags in 2d with one column will be reshaped and copied to a new array. "
            "It is more efficient for you to reshape without copying first."
        )
        tags = tags.reshape(tags.shape[0], copy=True)
    with open(tags_file, "wb") as fh:
        _write_bin(tags.astype(np.uint32), fh)


def tags_from_file(tags_file: str) -> VectorIdentifierBatch:
    """
    從 DiskANN 的二進位標籤檔案中讀取標籤。
    """
    _assert_existing_file(tags_file, "tags_file")
    points, dims = vectors_metadata_from_file(
        tags_file
    )  # 標籤檔案也包含相同的元資料格式
    return np.fromfile(file=tags_file, dtype=np.uint32, offset=8).reshape(points)
