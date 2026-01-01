# Grid Search Param Analysis Workflow

This document summarizes the end-to-end flow for grid-search builds, searches, and offline analysis.

## 1) Generate Build Configs

Generate build parameter combinations:

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
python gen_build_configs.py
```

Outputs:
- `scripts/paramAnalysis/gridSearch/inputFiles/build_configs.csv`

## 2) Batch Build Indexes

Build indexes from `build_configs.csv`:

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
bash build_batch.sh
```

Optional overrides:

```bash
BUILD_B=0.2 BUILD_M=1 NUM_THREADS=8 bash build_batch.sh
```

Outputs:
- `scripts/paramAnalysis/gridSearch/outputFiles/build/*.index` and related files

## 3) Generate Search Configs

Generate search parameter combinations:

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
python gen_search_configs.py --dataset_size 50000 --max_cores 16
```

Outputs:
- `scripts/paramAnalysis/gridSearch/inputFiles/search_configs.csv`

## 4) Batch Search (per-combination iostat supported)

Run batch search:

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
bash search_batch.sh
```

Enable per-combination iostat logging:

```bash
ENABLE_IOSTAT=1 IOSTAT_INTERVAL=1 bash search_batch.sh
```

Specify device:

```bash
ENABLE_IOSTAT=1 IOSTAT_DEVICE=/dev/nvme0n1 bash search_batch.sh
```

Record expanded nodes (for offline hot/cold analysis):

```bash
EXTRA_ARGS="--record_expanded_nodes --expanded_nodes_path ./outputFiles/search/expanded_nodes.csv --expanded_nodes_limit 0" \
  bash search_batch.sh
```

Outputs:
- Search results and logs under `scripts/paramAnalysis/gridSearch/outputFiles/search/`
- Per-combination iostat logs: `*_iostat.log` next to the matching `*_summary_stats.csv`
- Optional: `expanded_nodes.csv`

## 5) Collect Summary Stats

Aggregate all `*_summary_stats.csv` into one CSV:

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
python collect.py
```

Optional output:

```bash
python collect.py -o ./outputFiles/analyze/collected_stats.csv
```

Outputs:
- `scripts/paramAnalysis/gridSearch/outputFiles/analyze/collected_stats_*.csv`

## 6) Dump Neighbor Lists (Offline Analysis)

Build the tool:

```bash
cmake --build /home/gt/research/DiskANN/build --target dump_disk_neighbors -- -j
```

Dump neighbor lists from expanded nodes:

```bash
/home/gt/research/DiskANN/build/apps/dump_disk_neighbors \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch/outputFiles/build/siftsmall_R64_L128_B0.2_M1 \
  --input_nodes /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch/outputFiles/search/expanded_nodes.csv \
  --output_path /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch/outputFiles/analyze/neighbors_dump.csv
```

Outputs:
- Neighbor edge list: `node_id,degree,neighbor_pos,neighbor_id`

## Optional: External iostat Wrapper

If you want to measure arbitrary commands (not only search_batch):

```bash
cd /home/gt/research/DiskANN/scripts/paramAnalysis/gridSearch
bash measure_queue_depth.sh "bash search_batch.sh"
```

## Reference: Input Parameters

### build_batch.sh (`scripts/paramAnalysis/gridSearch/build_batch.sh`)

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | BUILD_CSV | 否 | `./inputFiles/build_configs.csv` | build 參數 CSV（含表頭 `build_id,build_R,build_L`） |
| CLI | DATASET | 否 | `siftsmall` | 資料集名稱，用於預設資料路徑 |
| CLI | MAX_PARALLEL | 否 | `2` | 平行建置數 |
| ENV | DATA_FILE | 否 | `${DISKANN_ROOT}/data/${DATASET}/${DATASET}_base.bin` | 資料檔路徑 |
| ENV | OUTPUT_DIR | 否 | `./outputFiles/build` | 索引輸出資料夾 |
| ENV | DATA_TYPE | 否 | `float` | `build_disk_index --data_type` |
| ENV | DIST_FN | 否 | `l2` | `build_disk_index --dist_fn` |
| ENV | BUILD_B | 否 | `0.2` | `-B` 搜尋 DRAM budget |
| ENV | BUILD_M | 否 | `1` | `-M` 建置 DRAM budget |
| ENV | PQ_DISK_BYTES | 否 | `0` | `--PQ_disk_bytes` |
| ENV | BUILD_PQ_BYTES | 否 | `0` | `--build_PQ_bytes` |
| ENV | NUM_THREADS | 否 | `8` | `--num_threads` |
| ENV | APPEND_PARAMS | 否 | `1` | `1` 代表使用 `-A` 自動附加參數到 index prefix |
| ENV | EXTRA_ARGS | 否 | 空 | 追加給 `build_disk_index` 的額外參數 |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |

### search_batch.sh (`scripts/paramAnalysis/gridSearch/search_batch.sh`)

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | SEARCH_CSV | 否 | `./inputFiles/search_configs.csv` | 搜尋參數 CSV（含表頭） |
| CLI | DATASET | 否 | 自動解析 | 從 index 檔名解析，或手動覆寫 |
| CLI | MAX_PARALLEL | 否 | `4` | 平行搜尋數（`ENABLE_IOSTAT=1` 時會強制為 1） |
| ENV | BUILD_DIR | 否 | `./outputFiles/build` | 索引輸入資料夾 |
| ENV | OUTPUT_DIR | 否 | `./outputFiles/search` | 搜尋結果輸出資料夾 |
| ENV | DATA_TYPE | 否 | `float` | `search_disk_index --data_type` |
| ENV | DIST_FN | 否 | `l2` | `search_disk_index --dist_fn` |
| ENV | QUERY_FILE | 否 | `${DATASET}_query.bin` | 查詢檔路徑 |
| ENV | GT_FILE | 否 | `${DATASET}_groundtruth.bin` | Ground truth 路徑 |
| ENV | SEARCH_IO_LIMIT | 否 | 空 | `--search_io_limit` |
| ENV | THREAD_OVERRIDE | 否 | 空 | 覆寫 `search_thread` |
| ENV | APPEND_SEARCH_PARAMS | 否 | `1` | 使用 `-A` 附加搜尋參數到結果路徑 |
| ENV | EXTRA_ARGS | 否 | 空 | 追加給 `search_disk_index` 的參數（可用於 expanded nodes） |
| ENV | DRY_RUN | 否 | `0` | `1` 僅輸出指令不執行 |
| ENV | ENABLE_IOSTAT | 否 | `0` | `1` 啟用每筆搜尋 iostat |
| ENV | IOSTAT_INTERVAL | 否 | `1` | iostat 取樣秒數 |
| ENV | IOSTAT_DEVICE | 否 | 空 | 指定裝置（如 `/dev/nvme0n1`） |
| ENV | IOSTAT_DATA_PATH | 否 | 空 | 用檔案路徑推估裝置 |
| ENV | TOPK | 否 | `10` | 目前未使用（保留變數） |
| ENV | K_OVERRIDE | 否 | 空 | 覆寫 CSV 的 `search_K` |

補充說明：`EXTRA_ARGS`
- 用途：將額外 CLI 參數原樣傳給 `search_disk_index`，避免每次新增/調整 CLI 選項都要改 `search_batch.sh`。
- 典型用法：啟用擴充功能或實驗性參數，例如：
  ```bash
  EXTRA_ARGS="--record_expanded_nodes --expanded_nodes_path ./outputFiles/search/expanded_nodes.csv --expanded_nodes_limit 0" \
    bash search_batch.sh
  ```

為何把重要設定放在 `EXTRA_ARGS`
- `search_disk_index` 參數多、變動頻繁；集中在 `EXTRA_ARGS` 可維持腳本穩定、避免頻繁改動。
- 方便實驗：臨時測試新旗標或不同組合時，無需改腳本或 CSV。
- 降低耦合：`search_batch.sh` 專注負責流程與批次控制，功能細節交給 `search_disk_index`。

### dump_disk_neighbors (`apps/dump_disk_neighbors.cpp`)

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | `--data_type` | 是 |  | `float` / `int8` / `uint8` |
| CLI | `--dist_fn` | 是 |  | `l2` / `mips` / `cosine` |
| CLI | `--index_path_prefix` | 是 |  | index prefix（不含 `_disk.index`） |
| CLI | `--input_nodes` | 是 |  | 節點清單或 `expanded_nodes.csv` |
| CLI | `--output_path` | 是 |  | 輸出 CSV |
| CLI | `--max_nodes` | 否 | `0` | 限制 unique node 數量（0 = all） |
| CLI | `--keep_duplicates` | 否 | `false` | 保留重複 node_id |

### measure_queue_depth.sh (`scripts/paramAnalysis/gridSearch/measure_queue_depth.sh`)

| 參數類型 | 名稱 | 必填 | 預設 | 說明 |
|---|---|---|---|---|
| CLI | `<command>` | 是 |  | 要執行的命令（字串） |
| ENV | IO_INTERVAL | 否 | `1` | iostat 取樣秒數 |
| ENV | OUTPUT_DIR | 否 | `./outputFiles/analyze` | iostat log 輸出資料夾 |
| ENV | DEVICE | 否 | 空 | 指定裝置（如 `/dev/nvme0n1`） |
| ENV | DATA_PATH | 否 | 空 | 用檔案路徑推估裝置 |
