// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案為 `InMemDataStore` 類別的實作。
// 它處理在主記憶體中對齊、儲存、載入和存取向量資料的所有細節。

#include <memory>
#include "abstract_scratch.h"
#include "in_mem_data_store.h"

#include "utils.h"

namespace diskann
{

// 建構函式
template <typename data_t>
InMemDataStore<data_t>::InMemDataStore(const location_t num_points, const size_t dim,
                                       std::unique_ptr<Distance<data_t>> distance_fn)
    : AbstractDataStore<data_t>(num_points, dim), _distance_fn(std::move(distance_fn))
{
    // 根據距離函式所需的對齊方式，計算對齊後的維度
    _aligned_dim = ROUND_UP(dim, _distance_fn->get_required_alignment());
    // 分配對齊的記憶體區塊以儲存所有向量
    alloc_aligned(((void **)&_data), this->_capacity * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    // 將記憶體初始化為 0
    std::memset(_data, 0, this->_capacity * _aligned_dim * sizeof(data_t));
}

// 解構函式
template <typename data_t> InMemDataStore<data_t>::~InMemDataStore()
{
    if (_data != nullptr)
    {
        // 釋放對齊的記憶體
        aligned_free(this->_data);
    }
}

template <typename data_t> size_t InMemDataStore<data_t>::get_aligned_dim() const
{
    return _aligned_dim;
}

template <typename data_t> size_t InMemDataStore<data_t>::get_alignment_factor() const
{
    return _distance_fn->get_required_alignment();
}

template <typename data_t> location_t InMemDataStore<data_t>::load(const std::string &filename)
{
    return load_impl(filename);
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t InMemDataStore<data_t>::load_impl(AlignedFileReader &reader)
{
    size_t file_dim, file_num_points;

    diskann::get_bin_metadata(reader, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }
    copy_aligned_data_from_file<data_t>(reader, _data, file_num_points, file_dim, _aligned_dim);

    return (location_t)file_num_points;
}
#endif

// 從二進位檔案載入資料
template <typename data_t> location_t InMemDataStore<data_t>::load_impl(const std::string &filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "錯誤: 資料檔案 " << filename << " 不存在。" << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    // 讀取檔案的元資料 (點數量和維度)
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        // 維度不匹配錯誤
        std::stringstream stream;
        stream << "錯誤: 要求載入 " << this->_dim << " 維，但檔案是 " << file_dim << " 維。" << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // 如果檔案中的點數量超過目前容量，則擴展容量
    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }

    // 從檔案中複製對齊的資料到記憶體緩衝區
    copy_aligned_data_from_file<data_t>(filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);

    return (location_t)file_num_points;
}

// 將資料儲存到二進位檔案
template <typename data_t> size_t InMemDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    // 僅儲存原始維度的資料，忽略為了對齊而填充的部分
    return save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

// 從記憶體陣列填充資料
template <typename data_t> void InMemDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    memset(_data, 0, _aligned_dim * sizeof(data_t) * num_pts);
    // 逐一複製向量，從原始維度複製到對齊後的維度
    for (location_t i = 0; i < num_pts; i++)
    {
        std::memmove(_data + i * _aligned_dim, vectors + i * this->_dim, this->_dim * sizeof(data_t));
    }

    // 如果距離函式需要，對所有基底資料進行預處理 (例如正規化)
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, this->_aligned_dim, num_pts);
    }
}

// 從檔案填充資料
template <typename data_t> void InMemDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    size_t npts, ndim;
    copy_aligned_data_from_file(filename.c_str(), _data, npts, ndim, _aligned_dim, offset);

    if ((location_t)npts > this->capacity())
    {
        std::stringstream ss;
        ss << "Number of points in the file: " << filename
           << " is greater than the capacity of data store: " << this->capacity()
           << ". Must invoke resize before calling populate_data()" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if ((location_t)ndim != this->get_dims())
    {
        std::stringstream ss;
        ss << "Number of dimensions of a point in the file: " << filename
           << " is not equal to dimensions of data store: " << this->capacity() << "." << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, this->_aligned_dim, this->capacity());
    }
}

template <typename data_t>
void InMemDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_points)
{
    save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

// 取得指定位置的向量
template <typename data_t> void InMemDataStore<data_t>::get_vector(const location_t i, data_t *dest) const
{
    memcpy(dest, _data + i * _aligned_dim, this->_dim * sizeof(data_t));
}

// 設定指定位置的向量
template <typename data_t> void InMemDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    size_t offset_in_data = loc * _aligned_dim;
    // 先將對齊後的空間清零
    memset(_data + offset_in_data, 0, _aligned_dim * sizeof(data_t));
    // 複製原始維度的資料
    memcpy(_data + offset_in_data, vector, this->_dim * sizeof(data_t));
    // 如果需要，對這個新設定的向量進行預處理
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data + offset_in_data, _aligned_dim, 1);
    }
}

// 預取向量到快取
template <typename data_t> void InMemDataStore<data_t>::prefetch_vector(const location_t loc)
{
    diskann::prefetch_vector((const char *)_data + _aligned_dim * (size_t)loc * sizeof(data_t),
                             sizeof(data_t) * _aligned_dim);
}

// 預處理查詢向量
template <typename data_t>
void InMemDataStore<data_t>::preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const
{
    if (query_scratch != nullptr)
    {
        memcpy(query_scratch->aligned_query_T(), query, sizeof(data_t) * this->get_dims());
    }
    else
    {
        std::stringstream ss;
        ss << "In InMemDataStore::preprocess_query: Query scratch is null";
        diskann::cerr << ss.str() << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
}

// 計算查詢向量與指定位置向量的距離
template <typename data_t> float InMemDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    return _distance_fn->compare(query, _data + _aligned_dim * loc, (uint32_t)_aligned_dim);
}

// 批次計算距離
template <typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *query, const location_t *locations,
                                          const uint32_t location_count, float *distances,
                                          AbstractScratch<data_t> *scratch_space) const
{
    for (location_t i = 0; i < location_count; i++)
    {
        distances[i] = _distance_fn->compare(query, _data + locations[i] * _aligned_dim, (uint32_t)this->_aligned_dim);
    }
}

// 計算兩個儲存中向量的距離
template <typename data_t>
float InMemDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    return _distance_fn->compare(_data + loc1 * _aligned_dim, _data + loc2 * _aligned_dim,
                                 (uint32_t)this->_aligned_dim);
}

template <typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                          std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const
{
    for (int i = 0; i < ids.size(); i++)
    {
        distances[i] =
            _distance_fn->compare(preprocessed_query, _data + ids[i] * _aligned_dim, (uint32_t)this->_aligned_dim);
    }
}

template <typename data_t> location_t InMemDataStore<data_t>::expand(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size < this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    // 在 Linux 上，重新分配、複製、然後釋放舊記憶體
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _data, this->capacity() * _aligned_dim * sizeof(data_t));
    aligned_free(_data);
    _data = new_data;
#else
    // 在 Windows 上，使用 realloc_aligned
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

// 縮減資料儲存容量
template <typename data_t> location_t InMemDataStore<data_t>::shrink(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size > this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    memcpy(new_data, _data, new_size * _aligned_dim * sizeof(data_t));
    aligned_free(_data);
    _data = new_data;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

// 移動向量區塊 (用於資料壓縮)
template <typename data_t>
void InMemDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                          const location_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    /*    // Update pointers to the moved nodes. Note: the computation is correct
       even
        // when new_location_start < old_location_start given the C++ uint32_t
        // integer arithmetic rules.
        const uint32_t location_delta = new_location_start - old_location_start;
    */
    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    if (new_location_start < old_location_start)
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }

    // Use memmove to handle overlapping ranges.
    copy_vectors(old_location_start, new_location_start, num_locations);
    memset(_data + _aligned_dim * mem_clear_loc_start, 0,
           sizeof(data_t) * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
}

// 複製向量區塊
template <typename data_t>
void InMemDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
                                          const location_t num_points)
{
    assert(from_loc < this->_capacity);
    assert(to_loc < this->_capacity);
    assert(num_points < this->_capacity);
    memmove(_data + _aligned_dim * to_loc, _data + _aligned_dim * from_loc, num_points * _aligned_dim * sizeof(data_t));
}

// 計算資料集的 medoid (幾何中心點)
// 演算法：1. 計算所有點的質心 (平均向量)。 2. 找到距離質心最近的那個點。
template <typename data_t> location_t InMemDataStore<data_t>::calculate_medoid() const
{
    // 分配並初始化質心向量
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    // 累加所有向量的座標
    for (size_t i = 0; i < this->capacity(); i++)
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += (float)_data[i * _aligned_dim + j];

    // 計算平均值
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)this->capacity();

    // 計算每個點到質心的 L2 距離平方
    float *distances = new float[this->capacity()];
    for (int64_t i = 0; i < (int64_t)this->capacity(); i++)
    {
        float &dist = distances[i];
        const data_t *cur_vec = _data + (i * (size_t)_aligned_dim);
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (float)cur_vec[j]) * (center[j] - (float)cur_vec[j]);
            dist += diff;
        }
    }
    
    // 找到距離最小的點的索引
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < this->capacity(); i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

template <typename data_t> Distance<data_t> *InMemDataStore<data_t>::get_dist_fn() const
{
    return this->_distance_fn.get();
}

// 模板實例化
template DISKANN_DLLEXPORT class InMemDataStore<float>;
template DISKANN_DLLEXPORT class InMemDataStore<int8_t>;
template DISKANN_DLLEXPORT class InMemDataStore<uint8_t>;

} // namespace diskann