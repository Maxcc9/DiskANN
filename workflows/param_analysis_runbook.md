# Param Analysis Runbook

本文件提供從 `gen_build_configs.py` 到 `run_all_notebooks.py` 的完整實驗流程與指令範本。
請依序執行，確保每次實驗輸出在獨立資料夾內（建議使用 `EXPERIMENT_TAG`）。

## 0) 前置條件

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
```

## 1) 產生 build 參數

用途：建立建置索引的參數組合（R/L 等），供批次建置使用。

```bash
python gen_build_configs.py
```

輸出：
- `inputFiles/build_configs.csv`

## 2) 批次建置 index

用途：依 `build_configs.csv` 批次建置索引，輸出到獨立實驗資料夾。

```bash
EXPERIMENT_TAG=exp01 NUM_THREADS=$(nproc) bash build_batch.sh --build-csv ./inputFiles/build_configs.csv
EXPERIMENT_TAG=exp01 NUM_THREADS=$(nproc) bash build_batch.sh --build-csv ./inputFiles/build_configs.csv --dataset sift
```

輸出：
- `outputFiles/build/exp01/`

## 3) 產生 search 參數

用途：建立搜尋參數組合（W/L/K/cache/threads）。

```bash
python gen_search_configs.py --dataset_size 1000000 --max_cores $(nproc)
```

輸出：
- `inputFiles/search_configs.csv`

## 4) 批次搜尋

用途：依 `search_configs.csv` 對所有 index 進行搜尋，產生 summary / expanded nodes / iostat 等原始結果。

```bash
EXPERIMENT_TAG=exp01 bash search_batch.sh --search-csv ./inputFiles/search_configs.csv
EXPERIMENT_TAG=exp01 bash search_batch.sh --search-csv ./inputFiles/search_configs.csv --dataset sift
```

如需 iostat 與 expanded nodes：

```bash
EXPERIMENT_TAG=exp01 \
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 \
ENABLE_EXPANDED_NODES=1 EXPANDED_NODES_LIMIT=0 \
bash search_batch.sh --search-csv ./inputFiles/search_configs.csv
```

輸出：
- `outputFiles/search/exp01/`

## 5) 產生鄰居資訊（必做）

用途：將 expanded_nodes 轉為鄰居列表，提供後續 cold/hot 結構分析所需資訊。

### 6.1 全部 expanded_nodes 轉鄰居表

```bash
EXPERIMENT_TAG=exp01 bash dump_all_neighbors.sh
```

### 6.2 全部 expanded_nodes 的 Top-K

```bash
EXPERIMENT_TAG=exp01 TOPK=10 bash dump_all_topk_neighbors.sh
```

輸出位置：
- `outputFiles/search/exp01/` 內對應的 `*_neighbors.csv` / `*_topk{K}_*`

## 6) 彙總統計

用途：彙總 summary_stats / node_counts / topk 資訊，產出分析用的整合 CSV。

```bash
EXPERIMENT_TAG=exp01 python collect.py
```

輸出：
- `outputFiles/analyze/exp01/collected_stats_exp01_<timestamp>.csv`
- `outputFiles/analyze/exp01/collected_topk_exp01_<timestamp>.csv`

## 7) 執行分析（00~06 notebooks）

用途：依研究計畫自動產出圖表與報表（QC、tradeoff、bottleneck、graph、worst-case 等）。

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch/analysis
REPORT_PREFIX=exp01 ./run_all_notebooks.py
```

輸出：
- `outputFiles/analyze/exp01/figures/`
- `outputFiles/analyze/exp01/tables/`
- `outputFiles/analyze/exp01/summary.md`

## 8) 常用變數整理

- `EXPERIMENT_TAG`：每次實驗的子資料夾名稱（建議必填）
- `REPORT_PREFIX`：分析報告資料夾名稱
- `TOPK`：Top‑K 節點數
- `ENABLE_IOSTAT` / `IOSTAT_INTERVAL`：iostat 記錄
- `ENABLE_EXPANDED_NODES` / `EXPANDED_NODES_LIMIT`：展開節點記錄

## 9) 重跑新實驗範例

```bash
EXPERIMENT_TAG=exp02 bash build_batch.sh --build-csv ./inputFiles/build_configs.csv
EXPERIMENT_TAG=exp02 bash search_batch.sh --search-csv ./inputFiles/search_configs.csv
EXPERIMENT_TAG=exp02 python collect.py
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch/analysis
REPORT_PREFIX=exp02 ./run_all_notebooks.py
```
