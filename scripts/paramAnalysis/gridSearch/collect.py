#!/usr/bin/env python3
"""
這個script取得所有在./outputFiles/search/內所有資料夾內的所有 _summary_stats.csv 檔案，
並將它們彙總成一個大的CSV檔案，方便後續分析。

用法：
  python collect.py [output_file]

參數：
  output_file  彙總結果輸出檔案，預設 ./outputFiles/analyze/collected_stats_{timestamp}.csv
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd


def find_summary_files(search_dir):
    """遞迴查找所有 *_summary_stats.csv 檔案"""
    pattern = os.path.join(search_dir, "**", "*_summary_stats.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def extract_index_info(file_path):
    """從檔案路徑提取 index 名稱"""
    # 路徑格式: ./outputFiles/search/{index_name}/{result_file}
    parts = Path(file_path).parts
    if len(parts) >= 3:
        return parts[-2]  # 返回倒數第二個部分（資料夾名稱）
    return "unknown"


def collect_summary_stats(search_dir, output_file):
    """
    蒐集所有 summary_stats.csv 並彙總到一個檔案
    
    Args:
        search_dir: search 輸出目錄
        output_file: 彙總輸出檔案路徑
    
    Returns:
        成功處理的檔案數量
    """
    # 查找所有 summary_stats.csv 檔案
    summary_files = find_summary_files(search_dir)
    
    if not summary_files:
        print(f"警告: 在 {search_dir} 內找不到任何 *_summary_stats.csv 檔案", file=sys.stderr)
        return 0
    
    print(f"找到 {len(summary_files)} 個 summary_stats.csv 檔案")
    
    all_data = []
    row_id = 1
    
    for summary_file in summary_files:
        try:
            # 讀取 CSV 檔案
            df = pd.read_csv(summary_file)
            
            # 添加 id 列（在最前面）
            ids = list(range(row_id, row_id + len(df)))
            df.insert(0, "id", ids)
            row_id += len(df)
            
            all_data.append(df)
            index_name = extract_index_info(summary_file)
            print(f"  ✓ 已讀取: {summary_file} (index: {index_name}, 行數: {len(df)})")
            
        except Exception as e:
            print(f"  ✗ 讀取失敗: {summary_file} - {e}", file=sys.stderr)
            continue
    
    if not all_data:
        print("錯誤: 沒有成功讀取任何檔案", file=sys.stderr)
        return 0
    
    # 合併所有資料
    print(f"\n正在合併 {len(all_data)} 個資料框...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"合併後總列數: {len(combined_df)}")
    print(f"列名: {list(combined_df.columns)}")
    
    # 儲存到輸出檔案
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ 彙總完成，已儲存至: {output_file}")
    print(f"  總行數: {len(combined_df)}")
    print(f"  總列數: {len(combined_df.columns)}")
    
    return len(all_data)


def main():
    parser = argparse.ArgumentParser(
        description="蒐集 search 結果中的 summary_stats.csv 並彙總到單一檔案"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="彙總結果輸出檔案 (預設: ./outputFiles/analyze/collected_stats_{timestamp}.csv)"
    )
    parser.add_argument(
        "-d", "--search-dir",
        default="./outputFiles/search",
        help="search 輸出目錄 (預設: ./outputFiles/search)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="顯示詳細資訊"
    )
    
    args = parser.parse_args()
    
    # 轉為絕對路徑
    search_dir = os.path.abspath(args.search_dir)
    
    # 生成預設輸出檔案名稱（帶時間戳）
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.abspath(f"./outputFiles/analyze/collected_stats_{timestamp}.csv")
    else:
        output_file = os.path.abspath(args.output)
    
    # 檢查輸入目錄
    if not os.path.isdir(search_dir):
        print(f"錯誤: search 目錄不存在或不是目錄: {search_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 建立輸出目錄
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"搜尋目錄: {search_dir}")
    print(f"輸出檔案: {output_file}")
    print("-" * 60)
    
    # 執行蒐集
    count = collect_summary_stats(search_dir, output_file)
    
    if count == 0:
        sys.exit(1)
    
    # 顯示統計資訊
    print("\n" + "=" * 60)
    print("統計資訊:")
    print("=" * 60)
    try:
        df = pd.read_csv(output_file)
        print(f"總列數: {len(df)}")
        print(f"ID 範圍: {df['id'].min()} - {df['id'].max()}")
        print(f"列名: {list(df.columns)[:15]}...")
        print("\n前 10 行:")
        print(df.head(10).to_string())
    except Exception as e:
        print(f"無法讀取輸出檔案: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()