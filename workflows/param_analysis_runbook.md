# 參數分析完整流程

> 從產生參數到分析輸出的一條龍流程，確保每次實驗都有獨立的輸出資料夾（建議設定 `EXPERIMENT_TAG`）。

## A. 前置準備：建置與 siftsmall 範例

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev

export DISKANN_ROOT="$(pwd)"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all -- -j4

# 下載 siftsmall
mkdir -p data/siftsmall && cd data/siftsmall
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar xzf siftsmall.tar.gz

# 下載 sift1M
mkdir -p data/sift && cd data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz

# 將資料集轉檔
cd "$DISKANN_ROOT"

# siftsmall
build/apps/utils/fvecs_to_bin float data/siftsmall/siftsmall/siftsmall_base.fvecs data/siftsmall/siftsmall_base.bin

build/apps/utils/fvecs_to_bin float data/siftsmall/siftsmall/siftsmall_query.fvecs data/siftsmall/siftsmall_query.bin

build/apps/utils/ivecs_to_bin data/siftsmall/siftsmall/siftsmall_groundtruth.ivecs data/siftsmall/siftsmall_groundtruth.bin

# sift
build/apps/utils/fvecs_to_bin float data/sift/sift/sift_base.fvecs data/sift/sift_base.bin

build/apps/utils/fvecs_to_bin float data/sift/sift/sift_query.fvecs data/sift/sift_query.bin

build/apps/utils/ivecs_to_bin data/sift/sift/sift_groundtruth.ivecs data/sift/sift_groundtruth.bin

# 若沒有 ground truth，可用此方式產生
build/apps/utils/compute_groundtruth \
  --data_type float --dist_fn l2 \
  --base_file data/siftsmall/siftsmall_base.bin \
  --query_file data/siftsmall/siftsmall_query.bin \
  --gt_file data/siftsmall/siftsmall_gt_100.bin --K 100
```

## B. 進入腳本目錄

```bash
cd "$DISKANN_ROOT/scripts/paramAnalysis/gridSearch"
```

## C. 流程總覽

1. 產生 build 參數  
2. 批次建置 index  
3. 產生 search 參數  
4. 批次搜尋（可選擇啟用 iostat/expanded nodes）  
5. 產生鄰居資訊（必做）  
6. 彙總統計  
7. 執行分析 notebooks  

以下各步驟預設 `EXPERIMENT_TAG=siftsmall01`，請視需求調整。

### 1) 產生 build 參數

用途：建立建置索引的參數組合（R/L 等），供批次建置使用。

```bash
python gen_build_configs.py
```

輸出：`inputFiles/build_configs.csv`

### 2) 批次建置 index

用途：依 `build_configs.csv` 批次建置索引，輸出到獨立實驗資料夾。

```bash
EXPERIMENT_TAG=siftsmall01 NUM_THREADS=$(nproc) bash build_batch.sh --build-csv ./inputFiles/build_configs.csv

EXPERIMENT_TAG=sift01 NUM_THREADS=$(nproc) bash build_batch.sh --build-csv ./inputFiles/build_configs.csv --dataset sift
```

輸出：`outputFiles/build/siftsmall01/`

### 3) 產生 search 參數

用途：建立搜尋參數組合（W/L/K/cache/threads）。

```bash
python gen_search_configs.py --dataset_size 10000 --max_cores $(nproc)
```

輸出：`inputFiles/search_configs.csv`

### 4) 批次搜尋

用途：依 `search_configs.csv` 對所有 index 進行搜尋，產生 summary / expanded nodes / iostat 等原始結果。

```bash
EXPERIMENT_TAG=siftsmall01 bash search_batch.sh --search-csv ./inputFiles/search_configs.csv

EXPERIMENT_TAG=sift01 bash search_batch.sh --search-csv ./inputFiles/search_configs.csv --dataset sift
```

啟用 iostat 與 expanded nodes：

```bash
EXPERIMENT_TAG=siftsmall01 \
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 \
ENABLE_EXPANDED_NODES=1 EXPANDED_NODES_LIMIT=0 \
COOLDOWN_TEMP_C=60 COOLDOWN_CHECK_INTERVAL=15 TEMP_DEVICE=/dev/nvme0 \
NVME_USE_SUDO=0 \
bash search_batch.sh --search-csv ./inputFiles/search_configs.csv --max-parallel 1
```

輸出：`outputFiles/search/siftsmall01/`

### 5) 產生鄰居資訊（必做）

用途：將 expanded_nodes 轉為鄰居列表，供冷／熱節點結構分析。

```bash
EXPERIMENT_TAG=siftsmall01 bash dump_all_neighbors.sh

EXPERIMENT_TAG=siftsmall01 TOPK=200 bash dump_all_topk_neighbors.sh
```

輸出：`outputFiles/search/siftsmall01/*_neighbors.csv`、`*_topk{K}_*`

### 6) 彙總統計

用途：彙總 summary_stats / node_counts / topk 資訊，產出分析用整合 CSV。

```bash
EXPERIMENT_TAG=siftsmall01 python collect.py
```

輸出：
- `outputFiles/analyze/siftsmall01/collected_stats_siftsmall01_<timestamp>.csv`
- `outputFiles/analyze/siftsmall01/collected_topk_siftsmall01_<timestamp>.csv`

### 7) 執行分析（00~06 notebooks）

用途：依研究計畫自動產出圖表與報表（QC、tradeoff、bottleneck、graph、worst-case 等）。

```bash
cd "$DISKANN_ROOT/scripts/paramAnalysis/gridSearch/analysis"
REPORT_PREFIX=siftsmall01 ./run_all_notebooks.py
```

輸出：
- `outputFiles/analyze/siftsmall01/figures/`
- `outputFiles/analyze/siftsmall01/tables/`
- `outputFiles/analyze/siftsmall01/summary.md`（含 00~06 重點摘要）

## D. 常用變數

- `EXPERIMENT_TAG`：實驗輸出子資料夾（建議必填）
- `REPORT_PREFIX`：分析報告資料夾名稱
- `TOPK`：Top‑K 節點數
- `ENABLE_IOSTAT` / `IOSTAT_INTERVAL`：iostat 記錄控制
- `ENABLE_EXPANDED_NODES` / `EXPANDED_NODES_LIMIT`：展開節點記錄控制
- `FILTER_SEARCH_K`：分析階段只保留指定 K（預設 `10`）
- `PLOT_MAX_POINTS`：圖表下採樣上限
- `PLOT_LOG_LATENCY`：延遲圖使用 log 軸（`1`/`0`）
- `QC_RECALL_THRESHOLD` / `QC_RECALL_PCTL`：QC 低召回門檻（固定/分位數）
- `QC_OUTLIER_Z`：QC robust z 門檻
- `SHAP_MAX_SAMPLES`：SHAP 抽樣上限
- `MODEL_TEST_SIZE` / `MODEL_RANDOM_STATE`：surrogate 模型切分與隨機種子
- `WORSTCASE_PCTL` / `WORSTCASE_MIN_COUNT`：worstcase 定義與分組下限
- `BOTTLENECK_SHARE_THRESHOLD`：瓶頸分類門檻

## E. 範例：重跑新實驗

```bash
EXPERIMENT_TAG=exp02 bash build_batch.sh --build-csv ./inputFiles/build_configs.csv

EXPERIMENT_TAG=exp02 bash search_batch.sh --search-csv ./inputFiles/search_configs.csv

EXPERIMENT_TAG=exp02 python collect.py
cd "$DISKANN_ROOT/scripts/paramAnalysis/gridSearch/analysis"

REPORT_PREFIX=exp02 ./run_all_notebooks.py
```
