#pragma once
#include "windows_customizations.h"
#include <cstring>

// 本檔案定義了各種距離/相似度度量的介面與實作。
// 這是所有鄰近搜尋演算法的基礎。

namespace diskann
{
// 定義支援的距離度量
enum Metric
{
    L2 = 0,            // L2 距離，即歐幾里得距離的平方
    INNER_PRODUCT = 1, // 內積。注意：為了用於最小化搜尋，通常會返回負內積
    COSINE = 2,        // 餘弦相似度
    FAST_L2 = 3        // 快速 L2 計算，通常用於預先計算了範數的特殊情況
};

// 距離計算的抽象基底類別
template <typename T> class Distance
{
  public:
    DISKANN_DLLEXPORT Distance(diskann::Metric dist_metric) : _distance_metric(dist_metric)
    {
    }

    // 核心比較函式，計算兩個向量 a 和 b 之間的距離。這是一個純虛擬函式，必須由子類別實作。
    DISKANN_DLLEXPORT virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

    // 僅用於位元組 (byte) 類型的 餘弦/內積 計算，需要傳入預先計算好的範數。
    DISKANN_DLLEXPORT virtual float compare(const T *a, const T *b, const float normA, const float normB,
                                            uint32_t length) const;

    // 對於某些度量 (如內積)，為了轉換成 L2 搜尋，可能會在正規化過程中增加維度。
    // 這個函式返回正規化後的維度。
    DISKANN_DLLEXPORT virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const;

    // 取得目前的度量類型。
    DISKANN_DLLEXPORT virtual diskann::Metric get_metric() const;

    // 指示是否需要在建立索引前對基底資料進行預處理 (例如，正規化)。
    DISKANN_DLLEXPORT virtual bool preprocessing_required() const;

    // Check the preprocessing_required() function before calling this.
    // Clients can call the function like this:
    //
    //  if (metric->preprocessing_required()){
    //     T* normalized_data_batch;
    //      Split data into batches of batch_size and for each, call:
    //       metric->preprocess_base_points(data_batch, batch_size);
    //
    //  TODO: This does not take into account the case for SSD inner product
    //  where the dimensions change after normalization.
    DISKANN_DLLEXPORT virtual void preprocess_base_points(T *original_data, const size_t orig_dim,
                                                          const size_t num_points);

    // 對單一的查詢向量進行預處理。
    DISKANN_DLLEXPORT virtual void preprocess_query(const T *query_vec, const size_t query_dim, T *scratch_query);

    // 取得為了 SIMD 優化所需的記憶體對齊位元組數。
    DISKANN_DLLEXPORT virtual size_t get_required_alignment() const;

    DISKANN_DLLEXPORT virtual ~Distance() = default;

  protected:
    diskann::Metric _distance_metric;
    size_t _alignment_factor = 8;
};

// --- 以下為各種距離度量的具體實作 ---

// 餘弦距離 (int8_t 版本)
class DistanceCosineInt8 : public Distance<int8_t>
{
  public:
    DistanceCosineInt8() : Distance<int8_t>(diskann::Metric::COSINE)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
};

// L2 距離 (int8_t 版本)
class DistanceL2Int8 : public Distance<int8_t>
{
  public:
    DistanceL2Int8() : Distance<int8_t>(diskann::Metric::L2)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
};

// 使用 AVX2 指令集優化的 L2 距離 (int8_t 版本)
class AVXDistanceL2Int8 : public Distance<int8_t>
{
  public:
    AVXDistanceL2Int8() : Distance<int8_t>(diskann::Metric::L2)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
};

// 餘弦距離 (float 版本)
class DistanceCosineFloat : public Distance<float>
{
  public:
    DistanceCosineFloat() : Distance<float>(diskann::Metric::COSINE)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

// L2 距離 (float 版本)
class DistanceL2Float : public Distance<float>
{
  public:
    DistanceL2Float() : Distance<float>(diskann::Metric::L2)
    {
    }

#ifdef _WINDOWS
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t size) const;
#else
    // 標記為熱點函式以供編譯器優化
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));
#endif
};

// 使用 AVX 指令集優化的 L2 距離 (float 版本)
class AVXDistanceL2Float : public Distance<float>
{
  public:
    AVXDistanceL2Float() : Distance<float>(diskann::Metric::L2)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

// 慢速、通用的 L2 距離實作，主要用於比對或無 SIMD 優化時的備用方案
template <typename T> class SlowDistanceL2 : public Distance<T>
{
  public:
    SlowDistanceL2() : Distance<T>(diskann::Metric::L2)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const T *a, const T *b, uint32_t length) const;
};

// 慢速 餘弦距離 (uint8_t 版本)
class SlowDistanceCosineUInt8 : public Distance<uint8_t>
{
  public:
    SlowDistanceCosineUInt8() : Distance<uint8_t>(diskann::Metric::COSINE)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
};

// L2 距離 (uint8_t 版本)
class DistanceL2UInt8 : public Distance<uint8_t>
{
  public:
    DistanceL2UInt8() : Distance<uint8_t>(diskann::Metric::L2)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
};

// 內積距離 (通用範本版本)
template <typename T> class DistanceInnerProduct : public Distance<T>
{
  public:
    DistanceInnerProduct() : Distance<T>(diskann::Metric::INNER_PRODUCT)
    {
    }

    DistanceInnerProduct(diskann::Metric metric) : Distance<T>(metric)
    {
    }
    inline float inner_product(const T *a, const T *b, unsigned size) const;

    // 為了將最大內積問題轉換為最小距離問題，返回內積的相反數。
    inline float compare(const T *a, const T *b, unsigned size) const
    {
        float result = inner_product(a, b, size);
        //      if (result < 0)
        //      return std::numeric_limits<float>::max();
        //      else
        return -result;
    }
};

// 快速 L2 距離，通常用於已預先計算範數的情況
template <typename T> class DistanceFastL2 : public DistanceInnerProduct<T>
{
  public:
    DistanceFastL2() : DistanceInnerProduct<T>(diskann::Metric::FAST_L2)
    {
    }
    float norm(const T *a, unsigned size) const;
    float compare(const T *a, const T *b, float norm, unsigned size) const;
};

// 使用 AVX 指令集優化的內積 (float 版本)
class AVXDistanceInnerProductFloat : public Distance<float>
{
  public:
    AVXDistanceInnerProductFloat() : Distance<float>(diskann::Metric::INNER_PRODUCT)
    {
    }
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const;
};

// 使用 AVX 指令集優化的正規化 코사인距離 (float 版本)
// 透過先將向量正規化，然後計算內積來實現 코사인相似度，這是一種常見的優化技巧。
class AVXNormalizedCosineDistanceFloat : public Distance<float>
{
  private:
    AVXDistanceInnerProductFloat _innerProduct;

  protected:
    void normalize_and_copy(const float *a, uint32_t length, float *a_norm) const;

  public:
    AVXNormalizedCosineDistanceFloat() : Distance<float>(diskann::Metric::COSINE)
    {
    }
    // 餘弦距離 = 1 - 餘弦相似度 = 1 - (正規化後的內積)
    DISKANN_DLLEXPORT virtual float compare(const float *a, const float *b, uint32_t length) const
    {
        return 1.0f + _innerProduct.compare(a, b, length);
    }
    DISKANN_DLLEXPORT virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const override;

    DISKANN_DLLEXPORT virtual bool preprocessing_required() const;

    DISKANN_DLLEXPORT virtual void preprocess_base_points(float *original_data, const size_t orig_dim,
                                                          const size_t num_points) override;

    DISKANN_DLLEXPORT virtual void preprocess_query(const float *query_vec, const size_t query_dim,
                                                    float *scratch_query_vector) override;
};

// 工廠函式：根據傳入的度量類型 m，返回對應的 Distance 物件實例。
template <typename T> Distance<T> *get_distance_function(Metric m);

} // namespace diskann
