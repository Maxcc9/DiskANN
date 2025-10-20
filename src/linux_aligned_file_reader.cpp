// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案是 `AlignedFileReader` 在 Linux 平台上的具體實作。
// 它使用 libaio (Linux Asynchronous I/O) 函式庫來實現高效能、非同步、
// 對齊的檔案讀取，這對於最大化 SSD 效能至關重要。

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024 // 一次最多提交的 I/O 事件數量

namespace
{
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

// 執行一批 I/O 請求的輔助函式
void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
{
#ifdef DEBUG
    for (auto &req : read_reqs)
    {
        assert(IS_ALIGNED(req.len, 512));
        // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
        assert(IS_ALIGNED(req.offset, 512));
        assert(IS_ALIGNED(req.buf, 512));
        // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
        std::vector<iocb_t *> cbs(n_ops, nullptr);
        std::vector<io_event_t> evts(n_ops);
        std::vector<struct iocb> cb(n_ops);
        
        // 步驟 1: 準備 I/O 控制區塊 (iocb)
        // 為每個讀取請求填充一個 iocb 結構。
        for (uint64_t j = 0; j < n_ops; j++)
        {
            io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len,
                          read_reqs[j + iter * MAX_EVENTS].offset);
        }

        for (uint64_t i = 0; i < n_ops; i++)
        {
            cbs[i] = cb.data() + i;
        }

        uint64_t n_tries = 0;
        while (n_tries <= n_retries)
        {
            // 步驟 2: 提交 I/O 請求
            // 使用 io_submit() 將一批請求一次性提交給核心，這是 libaio 的關鍵效能優勢。
            int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                          << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
            }
            else
            {
                // 步驟 3: 等待 I/O 完成
                // 使用 io_getevents() 阻塞等待，直到所有已提交的請求都完成。
                ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
                if (ret != (int64_t)n_ops)
                {
                    std::cerr << "io_getevents() failed; returned " << ret << ", expected=" << n_ops
                              << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                    exit(-1);
                }
                else
                {
                    break; // 所有請求成功完成
                }
            }
        }
    }
}
} // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                          << std::endl;
            }
        }
    }
}

io_context_t &LinuxAlignedFileReader::get_ctx()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        std::cerr << "錯誤的執行緒存取；返回 -1 作為 io_context_t" << std::endl;
        return this->bad_ctx;
    }
    else
    {
        return ctx_map[std::this_thread::get_id()];
    }
}

// 為目前執行緒註冊一個 I/O 上下文
void LinuxAlignedFileReader::register_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(my_id) != ctx_map.end())
    {
        std::cerr << "來自同一個執行緒的多個 register_thread 呼叫" << std::endl;
        return;
    }
    io_context_t ctx = 0;
    // 呼叫 io_setup 為此執行緒建立一個 libaio 上下文
    int ret = io_setup(MAX_EVENTS, &ctx);
    if (ret != 0)
    {
        lk.unlock();
        if (ret == -EAGAIN)
        {
            std::cerr << "io_setup() failed with EAGAIN: Consider increasing /proc/sys/fs/aio-max-nr" << std::endl;
        }
        else
        {
            std::cerr << "io_setup() failed; returned " << ret << ": " << ::strerror(-ret) << std::endl;
        }
    }
    else
    {
        diskann::cout << "為執行緒 ID: " << my_id << " 分配上下文: " << ctx << std::endl;
        ctx_map[my_id] = ctx;
    }
    lk.unlock();
}

// 取消註冊目前執行緒的 I/O 上下文
void LinuxAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());

    lk.unlock();
    io_context_t ctx = this->get_ctx();
    // 銷毀 libaio 上下文
    io_destroy(ctx);
    lk.lock();
    ctx_map.erase(my_id);
    std::cerr << "已從執行緒 ID: " << my_id << " 回收上下文" << std::endl;
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
        io_context_t ctx = x.value();
        io_destroy(ctx);
    }
    ctx_map.clear();
}

// 開啟檔案
void LinuxAlignedFileReader::open(const std::string &fname)
{
    // O_DIRECT 是關鍵旗標，它會繞過作業系統的頁面快取 (page cache)，
    // 進行直接 I/O。這避免了用索引資料污染快取，並減少了 CPU 開銷，
    // 對於最大化 SSD 輸送量至關重要。
    int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
    this->file_desc = ::open(fname.c_str(), flags);
    assert(this->file_desc != -1);
    std::cerr << "已開啟檔案 : " << fname << std::endl;
}

// 關閉檔案
void LinuxAlignedFileReader::close()
{
    ::fcntl(this->file_desc, F_GETFD);
    ::close(this->file_desc);
}

// 讀取一批請求
void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, io_context_t &ctx, bool async)
{
    if (async == true)
    {
        diskann::cout << "Linux 上目前不支援非同步。" << std::endl;
    }
    assert(this->file_desc != -1);
    // 呼叫輔助函式來執行 I/O
    execute_io(ctx, this->file_desc, read_reqs);
}
