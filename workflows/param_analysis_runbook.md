# Grid Search 參數建置與搜尋流程

從產生參數到批次搜尋並輸出統合 csv 檔的流程，確保每次實驗都有獨立的輸出資料夾（建議設定 `EXPERIMENT_TAG`）。後續可透過輸出的 collect_all_xxx.csv 進行完整分析。

## 重要：EXPERIMENT_TAG 機制

`EXPERIMENT_TAG` 環境變數用於：
  1. **配置文件隔離**：`gen_build_configs.py` 和 `gen_search_configs.py` 在 `inputFiles/{EXPERIMENT_TAG}/` 下產生各自的 CSV
  2. **輸出資料夾隔離**：`build_batch.sh` 和 `search_batch.sh` 將結果輸出至 `outputFiles/{build|search}/{EXPERIMENT_TAG}/`
  3. **批次腳本自動配對**：若設定 `EXPERIMENT_TAG`，批次腳本會自動從對應的 `inputFiles/{EXPERIMENT_TAG}/` 讀取配置，無需額外指定 `--build-csv` 或 `--search-csv`（但仍可覆寫）
  4. **DATASET 自動推斷**：若未明確設定 `--dataset`，批次腳本會使用 `EXPERIMENT_TAG` 值作為 DATASET（用於推斷資料集路徑）
  5. **TEMP_DEVICE 自動檢測**：在啟用降溫控制（`COOLDOWN_TEMP_C`）時，會自動檢測當前資料夾所在的 NVMe 設備，無需手動指定

## A. 前置準備：建置與 siftsmall 範例

```bash
# 安裝必要套件
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev

# 設定專案根目錄
export DISKANN_ROOT="$(pwd)"

# 建置專案
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all -- -j4

# 下載 siftsmall 資料集
mkdir -p data/siftsmall && cd data/siftsmall
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar xzf siftsmall.tar.gz

# 下載 sift1M 資料集
mkdir -p data/sift && cd data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz

# 回根目錄
cd "$DISKANN_ROOT"

# siftsmall 轉檔
build/apps/utils/fvecs_to_bin float data/siftsmall/siftsmall/siftsmall_base.fvecs data/siftsmall/siftsmall_base.bin

build/apps/utils/fvecs_to_bin float data/siftsmall/siftsmall/siftsmall_query.fvecs data/siftsmall/siftsmall_query.bin

build/apps/utils/ivecs_to_bin data/siftsmall/siftsmall/siftsmall_groundtruth.ivecs data/siftsmall/siftsmall_groundtruth.bin

# sift 轉檔
build/apps/utils/fvecs_to_bin float data/sift/sift/sift_base.fvecs data/sift/sift_base.bin

build/apps/utils/fvecs_to_bin float data/sift/sift/sift_query.fvecs data/sift/sift_query.bin

build/apps/utils/ivecs_to_bin data/sift/sift/sift_groundtruth.ivecs data/sift/sift_groundtruth.bin

# 若資料集沒有 ground truth，可用此方式產生
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

用途：建立建置索引的參數組合（R/L 等），供批次建置使用。依 `EXPERIMENT_TAG` 分隔配置檔案。

```bash
EXPERIMENT_TAG=siftsmall01 python gen_build_configs.py

EXPERIMENT_TAG=sift01 python gen_build_configs.py
```

輸出：`inputFiles/{EXPERIMENT_TAG}/build_configs.csv`

### 2) 批次建置 index

用途：依 `inputFiles/{EXPERIMENT_TAG}/build_configs.csv` 批次建置索引，輸出到獨立實驗資料夾。設定 `EXPERIMENT_TAG` 即自動配對配置檔案與推斷 DATASET。

```bash
EXPERIMENT_TAG=siftsmall01 NUM_THREADS=$(nproc) bash build_batch.sh --clean

EXPERIMENT_TAG=sift01 NUM_THREADS=$(nproc) bash build_batch.sh --clean
```

若需覆寫 DATASET（使用不同的資料集名稱）：

```bash
EXPERIMENT_TAG=sift01_test NUM_THREADS=$(nproc) bash build_batch.sh --dataset sift --clean
```

輸出：`outputFiles/build/siftsmall01/`

### 3) 產生 search 參數

用途：建立搜尋參數組合（W/L/K/cache/threads）。依 `EXPERIMENT_TAG` 分隔配置檔案。

```bash
EXPERIMENT_TAG=siftsmall01 python gen_search_configs.py --dataset_size 10000 --max_cores $(nproc)

EXPERIMENT_TAG=sift01 python gen_search_configs.py --dataset_size 1000000 --max_cores $(nproc)
```

輸出：`inputFiles/{EXPERIMENT_TAG}/search_configs.csv`

### 4) 批次搜尋

用途：依 `inputFiles/{EXPERIMENT_TAG}/search_configs.csv` 對所有 index 進行搜尋，產生 summary / expanded nodes / iostat 等原始結果。設定 `EXPERIMENT_TAG` 即自動配對配置檔案與推斷 DATASET。

```bash
EXPERIMENT_TAG=siftsmall01 bash search_batch.sh --clean

EXPERIMENT_TAG=sift01 bash search_batch.sh --clean
```

若需覆寫 DATASET：

```bash
EXPERIMENT_TAG=sift01_test bash search_batch.sh --dataset sift --clean
```

啟用 iostat 與 expanded nodes（設定 `EXPERIMENT_TAG` 自動配對配置及推斷設備）：

```bash
EXPERIMENT_TAG=siftsmall01 \
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 \
ENABLE_PIDSTAT=1 PIDSTAT_INTERVAL=1 \
ENABLE_WA_LOG=1 WA_INTERVAL=1 \
ENABLE_THREAD_TIMELINE=1 \
ENABLE_READ_TRACE=1 \
ENABLE_EXPANDED_NODES=1 EXPANDED_NODES_LIMIT=0 \
COOLDOWN_TEMP_C=60 COOLDOWN_CHECK_INTERVAL=15 \
NVME_USE_SUDO=0 \
bash search_batch.sh --max-parallel 1 --clean
```

說明：
- `ENABLE_PIDSTAT` 需要 `pidstat`（通常在 `sysstat` 套件內），輸出 per-thread CPU/IO/等待統計
- `ENABLE_WA_LOG` 需要 `mpstat`（同屬 `sysstat`），輸出 per-CPU `wa` (IO wait)
- `ENABLE_THREAD_TIMELINE` 會輸出 `*_thread_timeline.csv`，含 `os_tid` 與時間窗，供對齊 OS 指標

或手動指定設備（覆寫自動推斷）：

```bash
EXPERIMENT_TAG=siftsmall01 \
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 \
ENABLE_PIDSTAT=1 PIDSTAT_INTERVAL=1 \
ENABLE_WA_LOG=1 WA_INTERVAL=1 \
ENABLE_THREAD_TIMELINE=1 \
ENABLE_READ_TRACE=1 \
ENABLE_EXPANDED_NODES=1 EXPANDED_NODES_LIMIT=0 \
COOLDOWN_TEMP_C=60 COOLDOWN_CHECK_INTERVAL=15 \ TEMP_DEVICE=/dev/nvme1 \
NVME_USE_SUDO=0 \
bash search_batch.sh --max-parallel 1 --clean
```

輸出：`outputFiles/search/siftsmall01/<index_tag>/<S..._W..._L..._K..._cache..._T...>/`

其中（啟用對應開關時）會包含：
- `*_summary_stats.csv`
- `*_expanded_nodes.csv`
- `*_iostat.log`
- `*_pidstat.log`
- `*_wa.log`
- `*_thread_timeline.csv`
- `*_read_trace.csv`
- `*_read_trace_hot_nodes_<W>ms_top<N>.csv`
- `*_read_trace_window_<W>ms_node_stats.csv`
- `*_read_trace_window_<W>ms_stats.csv`

read trace 相關環境變數（彙總時使用）：
- `READ_TRACE_WINDOWS_MS=10,50,100` 控制多個時間窗（預設 50）
- `READ_TRACE_TOPK=100` 產生熱點節點清單的 Top-K（預設 100）
- `READ_TRACE_NODE_STATS=1` 產出每個 node 的時間窗統計檔（預設 1）
- `READ_TRACE_WINDOW_STATS=1` 產出每個時間窗統計檔（預設 1）

### 5) 產生鄰居資訊（必做）

用途：將 `*_expanded_nodes.csv` 轉為鄰居列表，供冷／熱節點結構分析。  
注意：此步驟依賴 `ENABLE_EXPANDED_NODES=1` 產生的檔案，若第 4 步未啟用 expanded nodes，這一步會找不到輸入檔。

```bash
# 轉出全部 expanded nodes 的鄰居
EXPERIMENT_TAG=siftsmall01 bash dump_all_neighbors.sh

# Top‑K 熱點節點的鄰居（從 expanded nodes 的頻次統計；每次 run 僅處理一個 TOPK）
EXPERIMENT_TAG=siftsmall01 TOPK=1000 bash dump_topk_neighbors.sh
```

輸出：`outputFiles/search/siftsmall01/*_neighbors.csv`、`*_topk{K}_neighbors.csv`

### 6) 彙總統計

用途：彙總 summary_stats / node_counts / topk 資訊，產出分析用整合 CSV。

```bash
EXPERIMENT_TAG=siftsmall01 python collect.py
```

輸出：
- `outputFiles/analyze/siftsmall01/collected_all_siftsmall01_<timestamp>.csv`

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
- `ENABLE_SUMMARY_STATS`：是否輸出 summary stats（`1`/`0`，預設 `1`）
- `ENABLE_PER_QUERY_STATS`：是否輸出 per-query stats（`1`/`0`，預設 `0`）

## E. 範例：重跑新實驗

```bash
EXPERIMENT_TAG=sift02 python gen_build_configs.py
EXPERIMENT_TAG=sift02 bash build_batch.sh --dataset sift --clean

EXPERIMENT_TAG=sift02 python gen_search_configs.py --dataset_size 1000000 --max_cores $(nproc)
EXPERIMENT_TAG=sift02 \
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 \
ENABLE_EXPANDED_NODES=1 EXPANDED_NODES_LIMIT=0 \
COOLDOWN_TEMP_C=60 COOLDOWN_CHECK_INTERVAL=15 \
NVME_USE_SUDO=0 \
bash search_batch.sh --max-parallel 1 --clean

EXPERIMENT_TAG=sift02 TOPK=100000 bash dump_topk_neighbors.sh
EXPERIMENT_TAG=sift02 python collect.py

cd "$DISKANN_ROOT/scripts/paramAnalysis/gridSearch/analysis"
REPORT_PREFIX=sift02 ./run_all_notebooks.py
```
## F. 常見問題與最佳實踐

### Q: build_batch.sh 與 search_batch.sh 的 DATASET 是否一定要指定？

A: 不一定。若設定 `EXPERIMENT_TAG`，批次腳本會自動以 `EXPERIMENT_TAG` 值作為 DATASET，用於推斷資料集路徑（如 `data/{EXPERIMENT_TAG}/{EXPERIMENT_TAG}_base.bin`）。
- 若 `EXPERIMENT_TAG=sift01`，會自動搜尋 `data/sift01/sift01_base.bin`
- 若實際資料集名稱不同，才需指定 `--dataset sift` 來覆寫

### Q: TEMP_DEVICE 要怎樣指定？

A: 自動推斷優先順序：
1. 環境變數 `TEMP_DEVICE`（若非預設值 `/dev/nvme0`）
2. 從當前 `OUTPUT_DIR` 所在的 NVMe 設備自動檢測
3. 回退到預設 `/dev/nvme0`

在啟用降溫控制（`COOLDOWN_TEMP_C`）時，自動推斷功能會啟動。若環境中只有一個 NVMe，通常無需手動指定。

### Q: 如何在不設定 EXPERIMENT_TAG 的情況下運行？

A: 不設定 `EXPERIMENT_TAG` 時，配置文件將使用預設位置：
- Build 配置：`inputFiles/build_configs.csv`
- Search 配置：`inputFiles/search_configs.csv`
- Build 輸出：`outputFiles/build/`
- Search 輸出：`outputFiles/search/`

但**強烈建議總是設定 `EXPERIMENT_TAG`**，避免不同實驗混淆。

### Q: 重複使用相同 EXPERIMENT_TAG 會發生什麼？

A: **預設情況下會混合舊數據**。若重複運行：
```bash
EXPERIMENT_TAG=sift01 bash build_batch.sh
EXPERIMENT_TAG=sift01 bash build_batch.sh  # ← 新索引與舊索引混在一起
```

**解決方案**：使用 `--clean` 參數清除舊數據：
```bash
# 方式 1：命令行參數
EXPERIMENT_TAG=sift01 bash build_batch.sh --clean
EXPERIMENT_TAG=sift01 bash search_batch.sh --clean

# 方式 2：環境變數
EXPERIMENT_TAG=sift01 CLEAN=1 bash build_batch.sh
EXPERIMENT_TAG=sift01 CLEAN=1 bash search_batch.sh

# 方式 3：手動清除
rm -rf ./outputFiles/build/sift01
rm -rf ./outputFiles/search/sift01
```
