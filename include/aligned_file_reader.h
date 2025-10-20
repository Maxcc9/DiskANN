// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案定義了 `AlignedFileReader` 類別，它是一個抽象基底類別，
// 為高效能的對齊 I/O 操作提供了統一的介面。
// 為了最大化 SSD 的讀取輸送量，通常需要使用直接 I/O (Direct I/O)，
// 這要求讀取的位址、長度和記憶體緩衝區都對齊到磁區大小 (通常是 512 或 4096 位元組)。
// 這個類別就是為了封裝這種複雜性。

#define MAX_IO_DEPTH 128 // 最大 I/O 請求佇列深度

#include <vector>
#include <atomic>

#ifndef _WINDOWS
// 在 Linux 上，使用 libaio 進行非同步 I/O
#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>
typedef io_context_t IOContext; // IOContext 直接對應到 libaio 的 io_context_t
#else
// 在 Windows 上，使用 I/O 完成埠 (IOCP) 進行非同步 I/O
#include <Windows.h>
#include <minwinbase.h>

#ifndef USE_BING_INFRA
// 標準 Windows 環境下的 I/O 上下文
struct IOContext
{
    HANDLE fhandle = NULL; // 檔案控制代碼
    HANDLE iocp = NULL;    // I/O 完成埠控制代碼
    std::vector<OVERLAPPED> reqs; // 重疊 I/O 請求結構
};
#else
#include "IDiskPriorityIO.h"
#include <atomic>
// TODO: Caller code is very callous about copying IOContext objects
// all over the place. MUST verify that it won't cause leaks/logical
// errors.
// Because of such callous copying, we have to use ptr->atomic instead
// of atomic, as atomic is not copyable.
struct IOContext
{
    enum Status
    {
        READ_WAIT = 0,
        READ_SUCCESS,
        READ_FAILED,
        PROCESS_COMPLETE
    };

    std::shared_ptr<ANNIndex::IDiskPriorityIO> m_pDiskIO = nullptr;
    std::shared_ptr<std::vector<ANNIndex::AsyncReadRequest>> m_pRequests;
    std::shared_ptr<std::vector<Status>> m_pRequestsStatus;

    // waitonaddress on this memory to wait for IO completion signal
    // reader should signal this memory after IO completion
    // TODO: WindowsAlignedFileReader can be modified to take advantage of this
    //   and can largely share code with the file reader for Bing.
    mutable volatile long m_completeCount = 0;

    IOContext()
        : m_pRequestsStatus(new std::vector<Status>()), m_pRequests(new std::vector<ANNIndex::AsyncReadRequest>())
    {
        (*m_pRequestsStatus).reserve(MAX_IO_DEPTH);
        (*m_pRequests).reserve(MAX_IO_DEPTH);
    }
};
#endif

#endif

#include <malloc.h>
#include <cstdio>
#include <mutex>
#include <thread>
#include "tsl/robin_map.h"
#include "utils.h"

// 代表一個單一的、對齊的讀取請求
struct AlignedRead
{
    uint64_t offset; // 從檔案的哪個位移開始讀取
    uint64_t len;    // 讀取多長
    void *buf;       // 讀取到哪個緩衝區

    AlignedRead() : offset(0), len(0), buf(nullptr)
    {
    }

    AlignedRead(uint64_t offset, uint64_t len, void *buf) : offset(offset), len(len), buf(buf)
    {
        // 斷言確保 offset, len, buf 都滿足 512 位元組對齊，這是直接 I/O 的要求
        assert(IS_512_ALIGNED(offset));
        assert(IS_512_ALIGNED(len));
        assert(IS_512_ALIGNED(buf));
        // assert(malloc_usable_size(buf) >= len);
    }
};

// 對齊檔案讀取器的抽象基底類別
class AlignedFileReader
{
  protected:
    // 儲存每個執行緒 ID 到其對應 I/O 上下文的映射，以支援多執行緒讀取
    tsl::robin_map<std::thread::id, IOContext> ctx_map;
    std::mutex ctx_mut; // 保護 ctx_map 的互斥鎖

  public:
    // 取得目前執行緒的 I/O 上下文。純虛擬函式。
    virtual IOContext &get_ctx() = 0;

    virtual ~AlignedFileReader(){};

    // 為目前執行緒註冊一個 I/O 上下文。純虛擬函式。
    virtual void register_thread() = 0;
    // 取消註冊目前執行緒的 I/O 上下文。純虛擬函式。
    virtual void deregister_thread() = 0;
    virtual void deregister_all_threads() = 0;

    // 開啟和關閉檔案。純虛擬函式。
    virtual void open(const std::string &fname) = 0;
    virtual void close() = 0;

    // 核心讀取函式：處理一批對齊的讀取請求。
    // 這是一個阻塞呼叫，但在內部會以非同步方式提交所有 I/O 請求以最大化輸送量。
    // 純虛擬函式。
    virtual void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) = 0;

#ifdef USE_BING_INFRA
    // 等待一批請求中的某一個完成 (Bing 架構專用)。純虛擬函式。
    virtual void wait(IOContext &ctx, int &completedIndex) = 0;
#endif
};
