// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// 本檔案定義了 DiskANN 函式庫中使用的自訂異常類別，
// 用於提供比標準異常更豐富的錯誤資訊。

#include <string>
#include <stdexcept>
#include <system_error>
#include <cstdint>
#include "windows_customizations.h"

#ifndef _WINDOWS
// 在非 Windows 平台上，將 __FUNCSIG__ 宏定義為 __PRETTY_FUNCTION__，
// 以取得包含函式簽名的詳細資訊，用於偵錯。
#define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace diskann
{

// DiskANN 的基礎異常類別，繼承自 std::runtime_error。
class ANNException : public std::runtime_error
{
  public:
    // 基本建構函式，包含錯誤訊息和錯誤碼。
    DISKANN_DLLEXPORT ANNException(const std::string &message, int errorCode);
    // 詳細建構函式，額外包含函式簽名、檔案名稱和行號，方便偵錯。
    DISKANN_DLLEXPORT ANNException(const std::string &message, int errorCode, const std::string &funcSig,
                                   const std::string &fileName, uint32_t lineNum);

  private:
    int _errorCode;
};

// 專門用於檔案操作的異常類別，繼承自 ANNException。
class FileException : public ANNException
{
  public:
    // 建構函式，接收檔名和一個 std::system_error 物件，以提供更具體的 I/O 錯誤資訊。
    DISKANN_DLLEXPORT FileException(const std::string &filename, std::system_error &e, const std::string &funcSig,
                                    const std::string &fileName, uint32_t lineNum);
};
} // namespace diskann
