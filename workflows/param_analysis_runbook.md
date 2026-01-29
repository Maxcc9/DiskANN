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

## B.1 設定免密碼執行系統操作（首次執行時需要）

**重要**：`search_batch.sh` 預設會在每次搜尋前執行以下操作，均需要 sudo 權限：
1. 清除系統快取（`CACHE_CLEAR_ENABLED=1`）- 確保測量真實 SSD I/O 效能（cold cache）
2. 讀取 NVMe 溫度（用於溫度控制）- 可能需要 sudo

為了實現完全自動化（不需要每次輸入密碼），需要配置 sudoers 規則：

```bash
# 執行安裝腳本（只需執行一次）
bash setup_sudoers.sh
```

**這會做什麼？**
- 產生 sudoers 規則檔案（只允許特定命令免密碼）：
  - `sync`：檔案系統同步
  - `tee /proc/sys/vm/drop_caches`：清除系統快取
  - `nvme smart-log`：讀取 NVMe 溫度
- 驗證語法（避免系統 sudo 功能損壞）
- 詢問確認後安裝到 `/etc/sudoers.d/diskann-drop-caches`（需要輸入**一次**密碼）
- 測試驗證是否成功

**安全性**：
- ✅ 只允許特定命令免密碼（最小權限原則）
- ✅ 不影響其他 sudo 命令（如 `rm`、`apt` 等仍需密碼）
- ✅ 可隨時移除：`sudo rm /etc/sudoers.d/diskann-drop-caches`

**改進**（相比舊版）：
- ✅ 自動嘗試讀取 NVMe 溫度（不需要設置 `NVME_USE_SUDO`）
- ✅ 完全自動化溫度檢測（無需手動指定 sudo 參數）
- ✅ 簡化使用方式

**替代方案**（如果不需要測量 cold cache 效能）：
```bash
# 關閉清除快取功能
CACHE_CLEAR_ENABLED=0 bash search_batch.sh ...

# 或關閉溫度控制
COOLDOWN_TEMP_C=0 bash search_batch.sh ...
```

## B.2 在 Docker 容器中執行實驗（共用電腦環境）

如果需要在與其他人共用的電腦上透過 Docker 執行實驗，使用 `docker_run_search.sh` 腳本：

**步驟 1**：主機上配置一次 sudoers（見 B.1）

**步驟 2**：執行 Docker 運行腳本
```bash
# 先確保 Docker 鏡像已建置
docker build -t diskann:dev .

# 執行實驗（會自動在主機清除快取，然後在容器內執行）
bash docker_run_search.sh diskann:dev bash search_batch.sh --repeat-count 3 --max-parallel 1
```

**或者設定環境變數傳遞參數**：
```bash
EXPERIMENT_TAG=sift01_test \
ENABLE_IOSTAT=1 ENABLE_PIDSTAT=1 \
bash docker_run_search.sh diskann:dev bash search_batch.sh --repeat-count 3 --max-parallel 1 --clean
```

**優勢**：
- ✅ 自動分解 `--repeat-count`，每次搜尋前清除主機快取（cold cache）
- ✅ 容器內無需 sudo 權限
- ✅ 無需 `--privileged`（安全性更高）
- ✅ 不影響主機其他程式
- ✅ 每次執行都是冷啟動（科學準確）
- ✅ 實驗完全隔離，輸出保存至主機

**重複執行的工作原理**（`--repeat-count 3` 範例）：
```
docker_run_search.sh 自動分解為 3 次執行：

【重複 1/3】
  ├─ 主機清除快取 → cold cache ✓
  └─ 容器執行搜尋

【重複 2/3】
  ├─ 主機清除快取 → cold cache ✓
  └─ 容器執行搜尋

【重複 3/3】
  ├─ 主機清除快取 → cold cache ✓
  └─ 容器執行搜尋
```

**與直接執行的差異**：
- 直接執行（主機）：主機直接清除快取，search_batch.sh 在容器內應用清除邏輯
- Docker 執行：docker_run_search.sh 在主機清除快取，容器內禁用清除（因為無法執行 sudo）
- **結果相同**：每次搜尋都測量 cold cache 效能

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

**每組參數多次重複執行**（用於統計穩定性，每次搜尋前自動清除快取並檢測溫度）：

```bash
# 3 次重複
EXPERIMENT_TAG=siftsmall01 \
bash search_batch.sh --repeat-count 3 --clean

# 5 次重複，自訂溫度控制
EXPERIMENT_TAG=siftsmall01 \
REPEAT_COUNT=5 COOLDOWN_TEMP_C=50 COOLDOWN_CHECK_INTERVAL=10 \
bash search_batch.sh --clean
```

說明：
- `--repeat-count N` 或 `REPEAT_COUNT=N`：每組參數重複執行 N 次（預設 1 次）
- `CACHE_CLEAR_ENABLED=1`：每次搜尋前自動清除系統快取（預設啟用）
- `COOLDOWN_TEMP_C=60`：搜尋前要求 NVMe 溫度低於 60°C，否則等待並重新檢測
- `COOLDOWN_CHECK_INTERVAL=15`：溫度檢測間隔 15 秒
- `TEMP_DEVICE`：留空時自動推斷（根據 OUTPUT_DIR 所在分區檢測 NVMe 設備）
- 多次重複或啟用快取清除時自動強制單工模式（`--max-parallel 1`）確保快取清除有效性
- 輸出文件名自動附加重複索引：`..._T{threads}_1`、`..._T{threads}_2`、`..._T{threads}_3`

**每次搜尋的執行流程：**
1. **清除快取**：執行 `drop_caches` 清除系統記憶體快取（避免上次搜尋的影響）
2. **溫度檢測**：檢測 NVMe 溫度，若高於閾值則等待指定時間後重試
3. **搜尋執行**：溫度合格後開始搜尋並記錄 iostat/pidstat/mpstat 等 metrics

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
bash search_batch.sh --repeat-count 3 --max-parallel 1 --clean
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
COOLDOWN_TEMP_C=60 COOLDOWN_CHECK_INTERVAL=15 \
TEMP_DEVICE=/dev/nvme1 \
bash search_batch.sh --repeat-count 3 --max-parallel 1 --clean
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
- `READ_TRACE_WINDOWS_MS=0.5,1,2,5` 控制多個時間窗（預設 0.5）
  小數視窗在欄位名會轉成 p（例：0.5ms -> ms0p5）
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

**預設用法**（包含相關性分析）：
```bash
EXPERIMENT_TAG=siftsmall01 \
READ_TRACE_WINDOWS_MS=0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50 \
python collect.py --cleanup --workers $(nproc)
```

**跳過相關性分析**（減少中間檔案 & CSV 欄位）：
```bash
EXPERIMENT_TAG=siftsmall01 \
READ_TRACE_WINDOWS_MS=0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50 \
READ_TRACE_WINDOW_STATS=0 python collect.py --workers $(nproc)
```
說明：相關性指標（385 欄）需要中間統計檔案計算。設定 `READ_TRACE_WINDOW_STATS=0` 會跳過中間檔案生成與相關性計算，加快速度並減少磁碟使用。

**分析完後清理中間檔案**：
```bash
python collect.py --cleanup --workers $(nproc)
```

輸出：
- `outputFiles/analyze/siftsmall01/collected_all_siftsmall01_<timestamp>.csv`

### 6.1) 去重（多次重複實驗時使用）

用途：當使用 `--repeat-count` 進行多次重複實驗時，`collect.py` 會包含所有重複執行的結果。`median_dedupe.py` 用於去除重複，保留中位數 QPS 的執行結果，避免極端值影響分析。

**何時需要？**
- 使用 `--repeat-count N`（N > 1）執行 search_batch.sh
- 希望每組參數只保留一次「代表性」結果

**使用方式：**
```bash
# 假設 collect.py 產生了包含多次重複的 CSV
python median_dedupe.py \
  -i outputFiles/analyze/siftsmall01/collected_all_siftsmall01_20260129_120000.csv
```

**自訂選項：**
```bash
# 使用不同的分組欄位
python median_dedupe.py -i input.csv -o output.csv \
  --key-cols "build_R,build_L,search_K,search_L,search_W,search_T"

# 使用不同的效能指標（例如延遲而非 QPS）
python median_dedupe.py -i input.csv -o output.csv \
  --qps-col "latency_p99_us"
```

說明：
- 預設按 `build_R`, `build_L`, `search_K`, `search_L`, `search_W`, `search_T`, `actual_cached_nodes` 分組
- 每組參數保留 QPS 中位數的那次執行結果
- 中位數比平均值更穩健，不受極端值影響

輸出：
- 去重後的 CSV，每組參數只有一行

### 7) 執行分析（00~07 notebooks）

用途：依研究計畫自動產出圖表與報表（QC、tradeoff、bottleneck、graph、worst-case、tail latency 等）。

```bash
cd "$DISKANN_ROOT/scripts/paramAnalysis/gridSearch/analysis"
REPORT_PREFIX=siftsmall01 ./run_all_notebooks.py
```

輸出：
- `outputFiles/analyze/siftsmall01/figures/`
- `outputFiles/analyze/siftsmall01/tables/`
- `outputFiles/analyze/siftsmall01/summary.md`（含 00~07 重點摘要）

## D. 常用變數

- `EXPERIMENT_TAG`：實驗輸出子資料夾（建議必填）
- `REPORT_PREFIX`：分析報告資料夾名稱
- `TOPK`：Top‑K 節點數
- `REPEAT_COUNT`：每組參數重複執行次數（預設 `1`）
- `CACHE_CLEAR_ENABLED`：每次搜尋前是否清除系統快取（`1`/`0`，預設 `1`）
- `COOLDOWN_TEMP_C`：NVMe 溫度閾值（°C），搜尋前要求溫度低於此值（預設 `60`）
- `COOLDOWN_CHECK_INTERVAL`：溫度檢測間隔秒數（預設 `15`）
- `TEMP_DEVICE`：指定 NVMe 設備，留空自動推斷（預設自動）
- `ENABLE_IOSTAT` / `IOSTAT_INTERVAL`：iostat 記錄控制
- `ENABLE_PIDSTAT` / `PIDSTAT_INTERVAL`：pidstat 記錄控制（需 sysstat 套件）
- `ENABLE_WA_LOG` / `WA_INTERVAL`：mpstat 記錄控制（需 sysstat 套件）
- `ENABLE_EXPANDED_NODES` / `EXPANDED_NODES_LIMIT`：展開節點記錄控制
- `FILTER_SEARCH_K`：分析階段只保留指定 K（預設 `10`）
- `ENABLE_SUMMARY_STATS`：是否輸出 summary stats（`1`/`0`，預設 `1`）
- `ENABLE_PER_QUERY_STATS`：是否輸出 per-query stats（`1`/`0`，預設 `0`）

## D.1 參數總表（build_batch.sh / search_batch.sh / collect.py）

> 下列清單以腳本內 `usage()` 與 CLI 參數定義為準。

### build_batch.sh

**命令列參數**
- `--build-csv PATH`
- `--dataset NAME`
- `--max-parallel N`
- `--clean`

**環境變數**
- `EXPERIMENT_TAG`
- `DATA_FILE`
- `OUTPUT_DIR`
- `DATA_TYPE`
- `DIST_FN`
- `BUILD_B`
- `BUILD_M`
- `PQ_DISK_BYTES`
- `BUILD_PQ_BYTES`
- `NUM_THREADS`
- `APPEND_PARAMS`（`1` 使用 `-A` 自動附加參數到檔名前綴）
- `EXTRA_ARGS`
- `CLEAN`（同 `--clean`）
- `DRY_RUN`

### search_batch.sh

**命令列參數**
- `--search-csv PATH`
- `--dataset NAME`
- `--max-parallel N`
- `--repeat-count N`
- `--clean`

**環境變數**
實驗與路徑：
- `EXPERIMENT_TAG`
- `BUILD_DIR`
- `OUTPUT_DIR`
- `DATASET`

搜尋參數：
- `DATA_TYPE`
- `DIST_FN`
- `QUERY_FILE`
- `GT_FILE`
- `SEARCH_IO_LIMIT`
- `THREAD_OVERRIDE`
- `K_OVERRIDE`
- `EXTRA_ARGS`
- `SLEEP_SECONDS`

快取與溫度控制：
- `CACHE_CLEAR_ENABLED`：每次搜尋前是否清除系統快取（`1`/`0`，預設 `1`）
- `COOLDOWN_TEMP_C`：NVMe 溫度閾值（°C），搜尋前要求溫度低於此值（預設 `60`）
- `COOLDOWN_CHECK_INTERVAL`：溫度檢測間隔秒數（預設 `15`）
- `TEMP_DEVICE`：指定 NVMe 設備，留空自動推斷（預設自動）

監測開關：
- `ENABLE_IOSTAT` / `IOSTAT_INTERVAL` / `IOSTAT_DEVICE` / `IOSTAT_DATA_PATH`
- `ENABLE_PIDSTAT` / `PIDSTAT_INTERVAL`
- `ENABLE_WA_LOG` / `WA_INTERVAL`
- `ENABLE_THREAD_TIMELINE`
- `ENABLE_READ_TRACE`
- `ENABLE_EXPANDED_NODES` / `EXPANDED_NODES_LIMIT`
- `ENABLE_SUMMARY_STATS`
- `ENABLE_PER_QUERY_STATS`

診斷模式：
- `DRY_RUN`

### collect.py

**命令列參數**
- `-o, --output PATH`
- `-d, --search-dir PATH`
- `-v, --verbose`
- `--workers N` ：用於平行化的 worker 數量（預設：1，或由 `COLLECT_WORKERS` 環境變數決定）
- `--cleanup`：運行完成後清理所有中間統計檔案（`*_read_trace_window_*ms_stats.csv`, `*_node_stats.csv`）

**環境變數**
- `EXPERIMENT_TAG`（當 `--search-dir` 為預設值時，自動附加）
- `COLLECT_WORKERS`（可作為 `--workers` 的替代）
- `READ_TRACE_WINDOWS_MS`：逗號分隔的時間窗大小（毫秒），預設 `0.5`
  - 例：`0.01,0.02,0.05,0.1,0.5,1,2,5,10,20,50`
  - 小數視窗在欄位名會轉成 `p`（例：0.5ms → `ms0p5`）
- `READ_TRACE_NODE_STATS`：是否產生每個 node 的時間窗統計檔（`1`/`0`，預設 `1`）
- `READ_TRACE_WINDOW_STATS`：是否產生每個時間窗統計檔並計算相關性指標（`1`/`0`，預設 `1`）
  - 若設定為 `0`，會跳過 385 個相關性欄位的計算，減少中間檔案生成與最終 CSV 大小

## E. 範例：重跑新實驗

```bash
EXPERIMENT_TAG=sift02 python gen_build_configs.py
EXPERIMENT_TAG=sift02 bash build_batch.sh --dataset sift --clean

EXPERIMENT_TAG=sift02 python gen_search_configs.py --dataset_size 1000000 --max_cores $(nproc)
EXPERIMENT_TAG=sift02 \
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 \
ENABLE_EXPANDED_NODES=1 EXPANDED_NODES_LIMIT=0 \
COOLDOWN_TEMP_C=60 COOLDOWN_CHECK_INTERVAL=15 \
bash search_batch.sh --repeat-count 3 --max-parallel 1 --clean

EXPERIMENT_TAG=sift02 TOPK=100000 bash dump_topk_neighbors.sh
EXPERIMENT_TAG=sift02 python collect.py

cd "$DISKANN_ROOT/scripts/paramAnalysis/gridSearch/analysis"
REPORT_PREFIX=sift02 ./run_all_notebooks.py
```
## F. 常見問題與最佳實踐

### Q: 如何進行多次重複實驗以確保統計穩定性？

A: 使用 `--repeat-count` 或 `REPEAT_COUNT` 環境變數，每次搜尋都會自動清除快取並進行溫度檢測：

```bash
# 每組參數重複 3 次（每次搜尋前自動清除快取並檢測 NVMe 溫度）
EXPERIMENT_TAG=sift01 bash search_batch.sh --repeat-count 3 --clean

# 自訂溫度控制參數
EXPERIMENT_TAG=sift01 \
COOLDOWN_TEMP_C=50 COOLDOWN_CHECK_INTERVAL=10 \
bash search_batch.sh --repeat-count 5 --clean
```

說明：
- 多次重複或啟用快取清除時自動強制單工模式確保快取清除有效性
- 輸出文件名自動附加重複索引：`..._T{threads}_1`、`..._T{threads}_2`、`..._T{threads}_3`
- 每次搜尋前執行：
  1. **清除快取**：`drop_caches` 清除系統記憶體
  2. **溫度檢測**：檢測 NVMe 溫度，若高於閾值則等待後重試
  3. **搜尋執行**：溫度合格後執行實際搜尋

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

### Q: 多次重複時如何彙總統計？

A: `collect.py` 會自動掃描所有重複執行的結果並彙總：

```bash
EXPERIMENT_TAG=sift01 \
READ_TRACE_WINDOWS_MS=0.01,0.02,0.05,0.1,0.5,1,2,5,10,20,50 \
python collect.py
```

輸出 CSV 會包含所有重複執行的統計數據（mean/std/iqr 等），每組參數對應多行（分別代表各次重複）。后续分析 notebooks 會自動處理多次重複的彙總。

### Q: collect.py 的 --cleanup 選項做什麼？

A: 清理彙總完成後的中間統計檔案（用於減少磁碟佔用）：

```bash
# 執行彙總並清理中間檔案
python collect.py --cleanup --workers $(nproc)
```

會移除以下類型的檔案：
- `*_read_trace_window_<W>ms_stats.csv`：時間窗統計（用於相關性計算）
- `*_node_stats.csv`：節點統計

對最終輸出 CSV 無影響（相關性指標已在彙總時計算），只是清除磁碟臨時檔案。

### Q: 如何跳過相關性指標計算以加快彙總速度？

A: 設定環境變數 `READ_TRACE_WINDOW_STATS=0`：

```bash
# 快速彙總（不含 385 個相關性欄位）
READ_TRACE_WINDOW_STATS=0 python collect.py --workers $(nproc)

# 快速彙總且同時清理其他中間檔案
READ_TRACE_WINDOW_STATS=0 READ_TRACE_NODE_STATS=0 python collect.py --cleanup --workers $(nproc)
```

效果：
- 減少 CSV 欄數（從 3683 降至 3298）
- 跳過中間統計檔案生成（減少磁碟 I/O）
- 彙總速度更快

### Q: 每執行一次 search_batch.sh，記憶體用量就會上升，為什麼？

A: 原因可能是背景監測進程（iostat/pidstat/mpstat）沒有完全清理。最新版已改進進程管理：

**改進項目：**
- 添加 `cleanup_bg_processes()` 函數，確保所有背景進程被正確終止
- 使用 `SIGTERM` 先溫和終止，再用 `SIGKILL` 強制終止
- 添加 EXIT 陷阱確保腳本結束時所有進程都被清理

**若仍有問題，可手動清理：**
```bash
# 終止所有監測進程
pkill -f "iostat|pidstat|mpstat"

# 清除檔案系統快取（需要 sudo）
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

**預防措施：**
- 禁用不需要的監測工具可減少進程開銷：
  ```bash
  ENABLE_IOSTAT=0 ENABLE_PIDSTAT=0 ENABLE_WA_LOG=0 bash search_batch.sh ...
  ```
- 記憶體上升但不會下降是正常的（系統快取），`drop_caches` 後會釋放
