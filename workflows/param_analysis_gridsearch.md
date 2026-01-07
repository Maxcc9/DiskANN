# 參數網格搜尋腳本參數索引

> 本文件僅列出 `scripts/paramAnalysis/gridSearch/` 的腳本參數。完整流程請見 `workflows/param_analysis_runbook.md`。

## 0) 共用說明

- 建議先設定 `DISKANN_ROOT`，避免硬編碼路徑。
- 任何腳本若支援 `EXPERIMENT_TAG`，會在輸出目錄下建立獨立子資料夾。

## 1) gen_build_configs.py

用途：產生 `inputFiles/build_configs.csv`。  
參數：目前無需額外參數。

## 2) gen_search_configs.py

用途：產生 `inputFiles/search_configs.csv`。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | `--dataset_size` | 否 |  | 資料集大小 |
| CLI | `--max_cores` | 否 |  | 最大核心數 |

## 3) build_batch.sh

用途：批次建置 index。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | BUILD_CSV | 否 | `./inputFiles/build_configs.csv` | build 參數 CSV |
| CLI | DATASET | 否 | `siftsmall` | 資料集名稱 |
| CLI | MAX_PARALLEL | 否 | `1` | 平行建置數 |
| CLI | `--build-csv` | 否 |  | build 參數 CSV |
| CLI | `--dataset` | 否 |  | 覆寫資料集名稱 |
| CLI | `--max-parallel` | 否 |  | 覆寫平行建置數 |
| ENV | DATA_FILE | 否 | `${DISKANN_ROOT}/data/${DATASET}/${DATASET}_base.bin` | 資料檔路徑 |
| ENV | OUTPUT_DIR | 否 | `./outputFiles/build` | 索引輸出資料夾 |
| ENV | EXPERIMENT_TAG | 否 | 空 | 追加到 OUTPUT_DIR 形成每次實驗的獨立資料夾 |
| ENV | DATA_TYPE | 否 | `float` | `build_disk_index --data_type` |
| ENV | DIST_FN | 否 | `l2` | `build_disk_index --dist_fn` |
| ENV | BUILD_B | 否 | `30` | `-B` 搜尋 DRAM budget |
| ENV | BUILD_M | 否 | `30` | `-M` 建置 DRAM budget |
| ENV | PQ_DISK_BYTES | 否 | `0` | `--PQ_disk_bytes` |
| ENV | BUILD_PQ_BYTES | 否 | `0` | `--build_PQ_bytes` |
| ENV | NUM_THREADS | 否 | `8` | `--num_threads` |
| ENV | APPEND_PARAMS | 否 | `1` | `1` 代表使用 `-A` 自動附加參數到 index prefix |
| ENV | EXTRA_ARGS | 否 | 空 | 追加給 `build_disk_index` 的額外參數 |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |

## 4) search_batch.sh

用途：批次搜尋（可啟用 iostat / expanded nodes）。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | SEARCH_CSV | 否 | `./inputFiles/search_configs.csv` | 搜尋參數 CSV |
| CLI | DATASET | 否 | 自動解析 | 從 index 檔名解析，或手動覆寫 |
| CLI | MAX_PARALLEL | 否 | `4` | 平行搜尋數（`ENABLE_IOSTAT=1` 時會強制為 1） |
| CLI | `--search-csv` | 否 |  | 搜尋參數 CSV |
| CLI | `--dataset` | 否 |  | 覆寫資料集名稱 |
| CLI | `--max-parallel` | 否 |  | 覆寫平行搜尋數 |
| ENV | BUILD_DIR | 否 | `./outputFiles/build` | 索引輸入資料夾 |
| ENV | OUTPUT_DIR | 否 | `./outputFiles/search` | 搜尋結果輸出資料夾 |
| ENV | EXPERIMENT_TAG | 否 | 空 | 追加到 OUTPUT_DIR；若 BUILD_DIR 使用預設值則同步追加 |
| ENV | DATA_TYPE | 否 | `float` | `search_disk_index --data_type` |
| ENV | DIST_FN | 否 | `l2` | `search_disk_index --dist_fn` |
| ENV | QUERY_FILE | 否 | `${DATASET}_query.bin` | 查詢檔路徑 |
| ENV | GT_FILE | 否 | `${DATASET}_groundtruth.bin` | Ground truth 路徑 |
| ENV | SEARCH_IO_LIMIT | 否 | 空 | `--search_io_limit` |
| ENV | THREAD_OVERRIDE | 否 | 空 | 覆寫 `search_thread` |
| ENV | EXTRA_ARGS | 否 | 空 | 追加給 `search_disk_index` 的參數 |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |
| ENV | ENABLE_IOSTAT | 否 | `0` | `1` 啟用每筆搜尋 iostat |
| ENV | IOSTAT_INTERVAL | 否 | `1` | iostat 取樣秒數 |
| ENV | IOSTAT_DEVICE | 否 | 空 | 指定裝置（如 `/dev/nvme0n1`） |
| ENV | IOSTAT_DATA_PATH | 否 | 空 | 用檔案路徑推估裝置 |
| ENV | ENABLE_EXPANDED_NODES | 否 | `0` | `1` 會為每筆樣本輸出 expanded nodes CSV |
| ENV | EXPANDED_NODES_LIMIT | 否 | `0` | 每個 query 最多記錄展開節點數（0 = unlimited） |
| ENV | K_OVERRIDE | 否 | 空 | 覆寫 CSV 的 `search_K` |

## 5) collect.py

用途：彙總統計與 topk 結果。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | `-o` | 否 | 空 | 指定輸出檔案 |
| ENV | EXPERIMENT_TAG | 否 | 空 | 只收集特定實驗資料夾 |

## 6) dump_disk_neighbors

用途：由 expanded nodes 轉出鄰居清單。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | `--data_type` | 是 |  | `float` / `int8` / `uint8` |
| CLI | `--dist_fn` | 是 |  | `l2` / `mips` / `cosine` |
| CLI | `--index_path_prefix` | 是 |  | index prefix（不含 `_disk.index`） |
| CLI | `--input_nodes` | 是 |  | 節點清單或 `expanded_nodes.csv` |
| CLI | `--output_path` | 是 |  | 輸出 CSV |
| CLI | `--max_nodes` | 否 | `0` | 限制 unique node 數量（0 = all） |
| CLI | `--keep_duplicates` | 否 | `false` | 保留重複 node_id |

## 7) dump_all_neighbors.sh

用途：批次 dump 所有 `*_expanded_nodes.csv` 的鄰居清單。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | search_dir | 否 | `./outputFiles/search` | 搜尋輸出目錄 |
| ENV | BUILD_DIR | 否 | `./outputFiles/build` | 索引輸入資料夾 |
| ENV | EXPERIMENT_TAG | 否 | 空 | 追加到預設 SEARCH_DIR/BUILD_DIR |
| ENV | DATA_TYPE | 否 | `float` | `dump_disk_neighbors --data_type` |
| ENV | DIST_FN | 否 | `l2` | `dump_disk_neighbors --dist_fn` |
| ENV | MAX_NODES | 否 | `0` | 限制 unique node 數量（0 = all） |
| ENV | KEEP_DUPLICATES | 否 | `0` | `1` 保留重複 node_id |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |

## 8) dump_topk_neighbors.sh

用途：對單一 expanded nodes 檔案做 Top‑K 鄰居輸出。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | expanded_nodes_csv | 是 |  | `*_expanded_nodes.csv` 路徑 |
| ENV | TOPK | 否 | `10` | Top‑K node 數量 |
| ENV | OUTPUT_DIR | 否 | same as input | 輸出資料夾 |
| ENV | BUILD_DIR | 否 | `./outputFiles/build` | 索引輸入資料夾 |
| ENV | EXPERIMENT_TAG | 否 | 空 | 追加到預設 BUILD_DIR |
| ENV | DATA_TYPE | 否 | `float` | `dump_disk_neighbors --data_type` |
| ENV | DIST_FN | 否 | `l2` | `dump_disk_neighbors --dist_fn` |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |

## 9) dump_all_topk_neighbors.sh

用途：批次對所有 expanded nodes 檔案做 Top‑K 鄰居輸出。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | search_dir | 否 | `./outputFiles/search` | 搜尋輸出目錄 |
| ENV | TOPK | 否 | `10` | Top‑K node 數量 |
| ENV | OUTPUT_DIR | 否 | 空 | 若設定，所有輸出集中到此資料夾 |
| ENV | BUILD_DIR | 否 | `./outputFiles/build` | 索引輸入資料夾 |
| ENV | EXPERIMENT_TAG | 否 | 空 | 追加到預設 SEARCH_DIR/BUILD_DIR |
| ENV | DATA_TYPE | 否 | `float` | `dump_disk_neighbors --data_type` |
| ENV | DIST_FN | 否 | `l2` | `dump_disk_neighbors --dist_fn` |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |

## 10) measure_queue_depth.sh

用途：包住任意命令並記錄 iostat。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | `<command>` | 是 |  | 要執行的命令（字串） |
| ENV | IO_INTERVAL | 否 | `1` | iostat 取樣秒數 |
| ENV | OUTPUT_DIR | 否 | `./outputFiles/analyze` | iostat log 輸出資料夾 |
| ENV | DEVICE | 否 | 空 | 指定裝置（如 `/dev/nvme0n1`） |
| ENV | DATA_PATH | 否 | 空 | 用檔案路徑推估裝置 |

## 11) analysis/run_all_notebooks.py

用途：執行 00~06 notebooks 並產生 `summary.md`。

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| ENV | REPORT_PREFIX | 否 | `analysis_reports` | 報告輸出資料夾 |
| ENV | COLLECT_PREFIX | 否 | `REPORT_PREFIX` | collect 輸入資料夾 |
| ENV | FILTER_SEARCH_K | 否 | `10` | 僅分析指定 K |
| ENV | PLOT_MAX_POINTS | 否 | `20000` | 圖表下採樣上限 |
| ENV | PLOT_LOG_LATENCY | 否 | `1` | 延遲圖使用 log 軸 |
| ENV | QC_RECALL_THRESHOLD | 否 | `0.7` | QC 低召回門檻 |
| ENV | QC_RECALL_PCTL | 否 | `0` | QC 低召回分位數 |
| ENV | QC_OUTLIER_Z | 否 | `4.0` | QC robust z 門檻 |
| ENV | SHAP_MAX_SAMPLES | 否 | `2000` | SHAP 抽樣上限 |
| ENV | MODEL_TEST_SIZE | 否 | `0.2` | surrogate 切分比例 |
| ENV | MODEL_RANDOM_STATE | 否 | `42` | surrogate 隨機種子 |
| ENV | WORSTCASE_PCTL | 否 | `0.95` | worstcase 分位數 |
| ENV | WORSTCASE_MIN_COUNT | 否 | `10` | worstcase 分組下限 |
| ENV | WORSTCASE_MAX_SAMPLES | 否 | `200` | worstcase 反事實抽樣上限 |
| ENV | BOTTLENECK_SHARE_THRESHOLD | 否 | `0.5` | 瓶頸分類門檻 |
