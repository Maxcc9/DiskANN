// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案提供了兩個輔助類別 `cached_ifstream` 和 `cached_ofstream`，
// 用於實現帶有快取的循序檔案讀寫。透過在記憶體中快取一塊資料，
// 可以將多次小的 I/O 操作合併成一次大的 I/O 操作，從而減少系統呼叫次數，提高效率。

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "logger.h"
#include "ann_exception.h"

// 帶快取的循序檔案讀取器
class cached_ifstream
{
  public:
    cached_ifstream()
    {
    }
    cached_ifstream(const std::string &filename, uint64_t cacheSize) : cache_size(cacheSize), cur_off(0)
    {
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        this->open(filename, cache_size);
    }
    ~cached_ifstream()
    {
        delete[] cache_buf;
        reader.close();
    }

    // 開啟檔案，分配快取，並預讀第一塊資料到快取中
    void open(const std::string &filename, uint64_t cacheSize)
    {
        this->cur_off = 0;

        try
        {
            reader.open(filename, std::ios::binary | std::ios::ate);
            fsize = reader.tellg();
            reader.seekg(0, std::ios::beg);
            assert(reader.is_open());
            assert(cacheSize > 0);
            cacheSize = (std::min)(cacheSize, fsize);
            this->cache_size = cacheSize;
            cache_buf = new char[cacheSize];
            reader.read(cache_buf, cacheSize);
            diskann::cout << "Opened: " << filename.c_str() << ", size: " << fsize << ", cache_size: " << cacheSize
                          << std::endl;
        }
        catch (std::system_error &e)
        {
            throw diskann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    size_t get_file_size()
    {
        return fsize;
    }

    // 從檔案/快取中讀取 n_bytes 位元組到 read_buf
    void read(char *read_buf, uint64_t n_bytes)
    {
        assert(cache_buf != nullptr);
        assert(read_buf != nullptr);

        if (n_bytes <= (cache_size - cur_off))
        {
            // 情況 1: 快取中剩餘的資料足以滿足本次讀取請求
            // 直接從快取記憶體中複製資料，不需存取磁碟
            memcpy(read_buf, cache_buf + cur_off, n_bytes);
            cur_off += n_bytes;
        }
        else
        {
            // 情況 2: 快取中的資料不足以滿足本次讀取請求
            // 先將快取中剩餘的資料複製到 read_buf
            uint64_t cached_bytes = cache_size - cur_off;
            if (n_bytes - cached_bytes > fsize - reader.tellg())
            {
                std::stringstream stream;
                stream << "Reading beyond end of file" << std::endl;
                stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes << " fsize: " << fsize
                       << " current pos:" << reader.tellg() << std::endl;
                diskann::cout << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            memcpy(read_buf, cache_buf + cur_off, cached_bytes);

            // 直接從磁碟讀取剩餘所需的資料
            reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
            // reset cur off
            cur_off = cache_size;

            // 嘗試為下一次讀取重新填充快取
            uint64_t size_left = fsize - reader.tellg();

            if (size_left >= cache_size)
            {
                reader.read(cache_buf, cache_size);
                cur_off = 0; // 重設快取偏移量
            }
            // note that if size_left < cache_size, then cur_off = cache_size,
            // so subsequent reads will all be directly from file
        }
    }

  private:
    std::ifstream reader;     // 底層的檔案讀取流
    uint64_t cache_size = 0;  // 快取大小 (位元組)
    char *cache_buf = nullptr; // 指向快取記憶體的指標
    uint64_t cur_off = 0;     // 目前在快取中的讀取位置偏移量
    uint64_t fsize = 0;       // 檔案總大小
};

// 帶快取的循序檔案寫入器
class cached_ofstream
{
  public:
    cached_ofstream(const std::string &filename, uint64_t cache_size) : cache_size(cache_size), cur_off(0)
    {
        writer.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            writer.open(filename, std::ios::binary);
            assert(writer.is_open());
            assert(cache_size > 0);
            cache_buf = new char[cache_size];
            diskann::cout << "Opened: " << filename.c_str() << ", cache_size: " << cache_size << std::endl;
        }
        catch (std::system_error &e)
        {
            throw diskann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    ~cached_ofstream()
    {
        this->close();
    }

    // 關閉檔案，在關閉前會確保所有在快取中的資料都被寫入磁碟
    void close()
    {
        // dump any remaining data in memory
        if (cur_off > 0)
        {
            this->flush_cache();
        }

        if (cache_buf != nullptr)
        {
            delete[] cache_buf;
            cache_buf = nullptr;
        }

        if (writer.is_open())
            writer.close();
        diskann::cout << "Finished writing " << fsize << "B" << std::endl;
    }

    size_t get_file_size()
    {
        return fsize;
    }

    // 將 write_buf 中的 n_bytes 位元組寫入檔案/快取
    void write(char *write_buf, uint64_t n_bytes)
    {
        assert(cache_buf != nullptr);
        if (n_bytes <= (cache_size - cur_off))
        {
            // 情況 1: 快取有足夠的剩餘空間容納這次寫入的資料
            // 直接將資料複製到快取中，不存取磁碟
            memcpy(cache_buf + cur_off, write_buf, n_bytes);
            cur_off += n_bytes;
        }
        else
        {
            // 情況 2: 快取空間不足
            // 先將快取中現有的資料刷入磁碟
            writer.write(cache_buf, cur_off);
            fsize += cur_off;
            // 然後直接將本次要寫入的資料也寫入磁碟
            writer.write(write_buf, n_bytes);
            fsize += n_bytes;
            // 清空快取並重設偏移量
            memset(cache_buf, 0, cache_size);
            cur_off = 0;
        }
    }

    // 強制將快取中的資料寫入磁碟
    void flush_cache()
    {
        assert(cache_buf != nullptr);
        writer.write(cache_buf, cur_off);
        fsize += cur_off;
        memset(cache_buf, 0, cache_size);
        cur_off = 0;
    }

    void reset()
    {
        flush_cache();
        writer.seekp(0);
    }

  private:
    std::ofstream writer;     // 底層的檔案寫入流
    uint64_t cache_size = 0;  // 快取大小 (位元組)
    char *cache_buf = nullptr; // 指向快取記憶體的指標
    uint64_t cur_off = 0;     // 目前在快取中的寫入位置偏移量
    uint64_t fsize = 0;       // 已寫入的檔案大小
};