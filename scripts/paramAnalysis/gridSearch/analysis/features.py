欲分析變量:
# build_R/build_L: 建索引時的圖參數（R = 最大出邊數, L = 建圖搜尋列表長度）
# search_K: 搜尋回傳的 Top-K
# search_L: 搜尋候選列表長度（通常決定 recall 與成本）
# search_W: beam width（每步擴展的寬度）
# search_T: 搜尋執行緒數
# actual_cached_nodes: 實際快取到記憶體的節點數
build_R	build_L	search_K search_L search_W search_T actual_cached_nodes

以下是感覺可以做分析的資料:
# qps: 每秒查詢數（吞吐量）
qps 

# recall_*: 召回率的統計（對每個 query 的 recall）
recall_mean
recall_std
recall_iqr
recall_cv
recall_p0~p100

# hop_*: hops 統計（搜尋步數的分佈）
hop_mean
hop_std
hop_iqr
hop_cv
hop_p0~p100


# compares_*: 距離比較次數分佈（反映計算量）
compares_mean
compares_std
compares_iqr
compares_cv
compares_p0~p100

# cache_hit_rate_*: cache 命中率分佈（命中 / (IO + 命中)）
cache_hit_rate_mean
cache_hit_rate_std
cache_hit_rate_iqr
cache_hit_rate_cv
cache_hit_rate_p0~p100

# latency_us_*: 查詢延遲分佈（微秒）
latency_us_mean
latency_us_std
latency_us_iqr
latency_us_cv
latency_us_p0~p100

# io_us_*: IO 時間分佈（微秒）
io_us_mean
io_us_std
io_us_iqr
io_us_cv
io_us_p0~p100

# ios_*: IO 次數分佈
ios_mean
ios_std
ios_iqr
ios_cv
ios_p0~p100

# frontier_queue_depth_mean_*: frontier IO batch size 的分佈（平均佇列深度）
frontier_queue_depth_mean_mean
frontier_queue_depth_mean_std
frontier_queue_depth_mean_iqr
frontier_queue_depth_mean_cv
frontier_queue_depth_mean_p0~p100

# reorder_queue_depth_max_*: reorder 階段 IO batch 最大深度分佈
reorder_queue_depth_max_mean
reorder_queue_depth_max_std
reorder_queue_depth_max_iqr
reorder_queue_depth_max_cv
reorder_queue_depth_max_p0~p100

# thread_util_*: 執行緒利用率分佈（0~1）
thread_util_mean
thread_util_std
thread_util_iqr
thread_util_cv
thread_util_p0~p100


# visited_node_count_*: 每個 query 造訪到的 unique 節點數分佈
visited_node_count_mean
visited_node_count_std
visited_node_count_iqr
visited_node_count_cv
visited_node_count_p0~p100

# expanded_node_out_degree_*: 被展開節點的平均出度分佈（圖結構特性）
expanded_node_out_degree_mean
expanded_node_out_degree_std
expanded_node_out_degree_iqr
expanded_node_out_degree_cv
expanded_node_out_degree_p0~p100

# expanded_nodes_*: 由 expanded_nodes.csv 彙總的全域展開統計
expanded_nodes_total
expanded_nodes_unique
expanded_nodes_revisit_ratio
expanded_node_hottest_count
expanded_node_top1_share
expanded_node_top10_share
expanded_node_top100_share
expanded_node_top1000_share
expanded_node_top10000_share

# iostat_%rrqm_*: 讀取合併比例：有多少讀取請求在送到裝置前被合併了
iostat_%rrqm_mean
iostat_%rrqm_std
iostat_%rrqm_iqr
iostat_%rrqm_cv
iostat_%rrqm_p0~p100

# iostat_%util_*: 裝置忙碌程度
iostat_%util_mean
iostat_%util_std
iostat_%util_iqr
iostat_%util_cv
iostat_%util_p0~p100

# iostat_%wrqm_*: 寫入合併比率
iostat_%wrqm_mean
iostat_%wrqm_std
iostat_%wrqm_iqr
iostat_%wrqm_cv
iostat_%wrqm_p0~p100

# iostat_aqu-sz_*: 平均佇列深度（重要 queue 指標）
iostat_aqu-sz_mean
iostat_aqu-sz_std
iostat_aqu-sz_iqr
iostat_aqu-sz_cv
iostat_aqu-sz_p0~p100

# iostat_r/s_*: 每秒讀取次數
iostat_r/s_mean
iostat_r/s_std
iostat_r/s_iqr
iostat_r/s_cv
iostat_r/s_p0~p100


# iostat_r_await_*: 讀取平均等待時間
iostat_r_await_mean
iostat_r_await_std
iostat_r_await_iqr
iostat_r_await_cv
iostat_r_await_p0~p100

# iostat_rareq-sz_*: 讀取請求平均大小
iostat_rareq-sz_mean
iostat_rareq-sz_std
iostat_rareq-sz_iqr
iostat_rareq-sz_cv
iostat_rareq-sz_p0~p100

# iostat_rkB/s_*: 每秒讀取 KB 數
iostat_rkB/s_mean
iostat_rkB/s_std
iostat_rkB/s_iqr
iostat_rkB/s_cv
iostat_rkB/s_p0~p100

# iostat_rrqm/s_*: 每秒讀取合併次數
iostat_rrqm/s_mean
iostat_rrqm/s_std
iostat_rrqm/s_iqr
iostat_rrqm/s_cv
iostat_rrqm/s_p0~p100

# iostat_w/s_*: 每秒寫入次數
iostat_w/s_mean
iostat_w/s_std
iostat_w/s_iqr
iostat_w/s_cv
iostat_w/s_p0~p100

# iostat_w_await_*: 寫入平均等待時間
iostat_w_await_mean
iostat_w_await_std
iostat_w_await_iqr
iostat_w_await_cv
iostat_w_await_p0~p100

# iostat_wareq-sz_*: 寫入請求平均大小
iostat_wareq-sz_mean
iostat_wareq-sz_std
iostat_wareq-sz_iqr
iostat_wareq-sz_cv
iostat_wareq-sz_p0~p100

# iostat_wkB/s_*: 每秒寫入 KB 數
iostat_wkB/s_mean
iostat_wkB/s_std
iostat_wkB/s_iqr
iostat_wkB/s_cv
iostat_wkB/s_p0~p100

# iostat_wrqm/s_*: 每秒寫入合併次數
iostat_wrqm/s_mean
iostat_wrqm/s_std
iostat_wrqm/s_iqr
iostat_wrqm/s_cv
iostat_wrqm/s_p0~p100

# topk_*: Top-K 熱點節點鄰居統計（靜態圖結構）
topk_expanded_neighbor_count
topk_expanded_unique_neighbors_count
topk_expanded_degree_mean
topk_expanded_coverage_ratio