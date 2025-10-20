// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// 本檔案為自訂異常類別的實作檔案。
// 主要負責格式化錯誤訊息，將來源檔案、行號、函式簽名等偵錯資訊加入到異常訊息中。

#include "ann_exception.h"
#include <sstream>
#include <string>

namespace diskann
{
// ANNException 的基本建構函式
ANNException::ANNException(const std::string &message, int errorCode)
    : std::runtime_error(message), _errorCode(errorCode)
{
}

// 輔助函式，用於將鍵值對格式化為 "[鍵: 值]" 的字串
std::string package_string(const std::string &item_name, const std::string &item_val)
{
    return std::string("[") + item_name + ": " + std::string(item_val) + std::string("]");
}

// ANNException 的詳細建構函式
// 它使用委派建構函式 (delegating constructor) 來呼叫基本建構函式。
// 主要功能是將函式簽名、檔案名稱、行號等上下文資訊格式化後，加到原始錯誤訊息的前面。
ANNException::ANNException(const std::string &message, int errorCode, const std::string &funcSig,
                           const std::string &fileName, uint32_t lineNum)
    : ANNException(package_string(std::string("FUNC"), funcSig) + package_string(std::string("FILE"), fileName) +
                       package_string(std::string("LINE"), std::to_string(lineNum)) + "  " + message,
                   errorCode)
{
}

// FileException 的建構函式
// 同樣使用委派建構函式，呼叫 ANNException 的詳細建構函式。
// 它會建立一個針對檔案錯誤的特定訊息，包含檔名、系統錯誤碼和系統錯誤訊息，
// 然後將這些資訊連同上下文一起傳遞給基底類別。
FileException::FileException(const std::string &filename, std::system_error &e, const std::string &funcSig,
                             const std::string &fileName, uint32_t lineNum)
    : ANNException(std::string(" 開啟檔案 '") + filename + std::string("' 時發生錯誤, 錯誤碼: ") +
                       std::to_string(e.code().value()) + "  " + e.code().message(),
                   e.code().value(), funcSig, fileName, lineNum)
{
}

} // namespace diskann
