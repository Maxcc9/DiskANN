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

# frontier_queue_depth_mean_*: 單次迭代實際發出 SSD 讀取的節點數量分佈（平均佇列深度）
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

# pidstat_*: per-thread OS 統計（需要 ENABLE_PIDSTAT）
# pidstat_thread_count: 觀測到的 thread 數量
pidstat_thread_count

# pidstat_%usr_*: user space CPU 使用率分佈
pidstat_%usr_mean
pidstat_%usr_std
pidstat_%usr_iqr
pidstat_%usr_cv
pidstat_%usr_p0~p100

# pidstat_%system_*: kernel space CPU 使用率分佈
pidstat_%system_mean
pidstat_%system_std
pidstat_%system_iqr
pidstat_%system_cv
pidstat_%system_p0~p100

# pidstat_%wait_*: I/O wait 比例分佈（thread 等待磁碟/IO）
pidstat_%wait_mean
pidstat_%wait_std
pidstat_%wait_iqr
pidstat_%wait_cv
pidstat_%wait_p0~p100

# pidstat_%CPU_*: thread 總 CPU 使用率分佈
pidstat_%CPU_mean
pidstat_%CPU_std
pidstat_%CPU_iqr
pidstat_%CPU_cv
pidstat_%CPU_p0~p100

# pidstat_minflt/s_*: 每秒 minor page fault 分佈
pidstat_minflt/s_mean
pidstat_minflt/s_std
pidstat_minflt/s_iqr
pidstat_minflt/s_cv
pidstat_minflt/s_p0~p100

# pidstat_majflt/s_*: 每秒 major page fault 分佈
pidstat_majflt/s_mean
pidstat_majflt/s_std
pidstat_majflt/s_iqr
pidstat_majflt/s_cv
pidstat_majflt/s_p0~p100

# pidstat_VSZ_*: 虛擬記憶體大小分佈（行程層級）
pidstat_VSZ_mean
pidstat_VSZ_std
pidstat_VSZ_iqr
pidstat_VSZ_cv
pidstat_VSZ_p0~p100

# pidstat_RSS_*: 常駐記憶體大小分佈（行程層級）
pidstat_RSS_mean
pidstat_RSS_std
pidstat_RSS_iqr
pidstat_RSS_cv
pidstat_RSS_p0~p100

# pidstat_%MEM_*: 記憶體佔比（行程層級）
pidstat_%MEM_mean
pidstat_%MEM_std
pidstat_%MEM_iqr
pidstat_%MEM_cv
pidstat_%MEM_p0~p100

# pidstat_kB_rd/s_*: 讀取吞吐量分佈
pidstat_kB_rd/s_mean
pidstat_kB_rd/s_std
pidstat_kB_rd/s_iqr
pidstat_kB_rd/s_cv
pidstat_kB_rd/s_p0~p100

# pidstat_kB_wr/s_*: 寫入吞吐量分佈
pidstat_kB_wr/s_mean
pidstat_kB_wr/s_std
pidstat_kB_wr/s_iqr
pidstat_kB_wr/s_cv
pidstat_kB_wr/s_p0~p100

# pidstat_kB_ccwr/s_*: 寫入 page cache 吞吐量分佈
pidstat_kB_ccwr/s_mean
pidstat_kB_ccwr/s_std
pidstat_kB_ccwr/s_iqr
pidstat_kB_ccwr/s_cv
pidstat_kB_ccwr/s_p0~p100

# pidstat_iodelay_*: IO 延遲統計（平台依賴）
pidstat_iodelay_mean
pidstat_iodelay_std
pidstat_iodelay_iqr
pidstat_iodelay_cv
pidstat_iodelay_p0~p100

# pidstat_cswch/s_*: 自願 context switch 次數分佈
pidstat_cswch/s_mean
pidstat_cswch/s_std
pidstat_cswch/s_iqr
pidstat_cswch/s_cv
pidstat_cswch/s_p0~p100

# pidstat_nvcswch/s_*: 非自願 context switch 次數分佈
pidstat_nvcswch/s_mean
pidstat_nvcswch/s_std
pidstat_nvcswch/s_iqr
pidstat_nvcswch/s_cv
pidstat_nvcswch/s_p0~p100

# wa_*: mpstat 觀測的 %iowait（需要 ENABLE_WA_LOG）
wa_%iowait_mean
wa_%iowait_std
wa_%iowait_iqr
wa_%iowait_cv
wa_%iowait_p0~p100

# thread_timeline_*: per-query wall time 統計（需要 ENABLE_THREAD_TIMELINE）
# thread_timeline_duration_us_*: 每個 query 的 wall time 分佈
thread_timeline_duration_us_mean
thread_timeline_duration_us_std
thread_timeline_duration_us_iqr
thread_timeline_duration_us_cv
thread_timeline_duration_us_p0~p100

# thread_timeline_os_tid_unique: 觀測到的 OS tid 數量
thread_timeline_os_tid_unique

# thread_timeline_thread_id_unique: 觀測到的 OpenMP thread id 數量
thread_timeline_thread_id_unique

# thread_timeline_window_*: 每個時間窗的 latency 抽樣（需要 ENABLE_THREAD_TIMELINE + ENABLE_READ_TRACE）
# thread_timeline_window_ms_list: latency window 清單（逗號分隔，與 read_trace_window_ms_list 對齊）
thread_timeline_window_ms_list

# read_trace_window_corr_*: read_trace 時間窗統計 vs latency 時間窗統計的 Pearson 相關係數
#   計算方式：以 window_id 對齊後，對每一對欄位計算 Pearson 相關係數
read_trace_window_corr_repeat_ratio_vs_latency_mean_us_ms{W}
read_trace_window_corr_repeat_ratio_vs_latency_p50_us_ms{W}
read_trace_window_corr_repeat_ratio_vs_latency_p95_us_ms{W}
read_trace_window_corr_repeat_ratio_vs_latency_p99_us_ms{W}
read_trace_window_corr_repeat_ratio_vs_latency_p100_us_ms{W}
read_trace_window_corr_repeat_multi_thread_ratio_vs_latency_mean_us_ms{W}
read_trace_window_corr_repeat_multi_thread_ratio_vs_latency_p50_us_ms{W}
read_trace_window_corr_repeat_multi_thread_ratio_vs_latency_p95_us_ms{W}
read_trace_window_corr_repeat_multi_thread_ratio_vs_latency_p99_us_ms{W}
read_trace_window_corr_repeat_multi_thread_ratio_vs_latency_p100_us_ms{W}
read_trace_window_corr_max_node_reads_ratio_vs_latency_mean_us_ms{W}
read_trace_window_corr_max_node_reads_ratio_vs_latency_p50_us_ms{W}
read_trace_window_corr_max_node_reads_ratio_vs_latency_p95_us_ms{W}
read_trace_window_corr_max_node_reads_ratio_vs_latency_p99_us_ms{W}
read_trace_window_corr_max_node_reads_ratio_vs_latency_p100_us_ms{W}
read_trace_window_corr_max_same_thread_reads_ratio_vs_latency_mean_us_ms{W}
read_trace_window_corr_max_same_thread_reads_ratio_vs_latency_p50_us_ms{W}
read_trace_window_corr_max_same_thread_reads_ratio_vs_latency_p95_us_ms{W}
read_trace_window_corr_max_same_thread_reads_ratio_vs_latency_p99_us_ms{W}
read_trace_window_corr_max_same_thread_reads_ratio_vs_latency_p100_us_ms{W}
read_trace_window_corr_max_multi_thread_reads_ratio_vs_latency_mean_us_ms{W}
read_trace_window_corr_max_multi_thread_reads_ratio_vs_latency_p50_us_ms{W}
read_trace_window_corr_max_multi_thread_reads_ratio_vs_latency_p95_us_ms{W}
read_trace_window_corr_max_multi_thread_reads_ratio_vs_latency_p99_us_ms{W}
read_trace_window_corr_max_multi_thread_reads_ratio_vs_latency_p100_us_ms{W}
read_trace_window_corr_max_unique_threads_vs_latency_mean_us_ms{W}
read_trace_window_corr_max_unique_threads_vs_latency_p50_us_ms{W}
read_trace_window_corr_max_unique_threads_vs_latency_p95_us_ms{W}
read_trace_window_corr_max_unique_threads_vs_latency_p99_us_ms{W}
read_trace_window_corr_max_unique_threads_vs_latency_p100_us_ms{W}
read_trace_window_corr_total_reads_vs_latency_mean_us_ms{W}
read_trace_window_corr_total_reads_vs_latency_p50_us_ms{W}
read_trace_window_corr_total_reads_vs_latency_p95_us_ms{W}
read_trace_window_corr_total_reads_vs_latency_p99_us_ms{W}
read_trace_window_corr_total_reads_vs_latency_p100_us_ms{W}

# read_trace_*: SSD node read 事件統計（需要 ENABLE_READ_TRACE）
# read_trace_total_reads: 總讀取事件數（含 cache hit），等於 read_trace.csv 的總列數
read_trace_total_reads

# read_trace_unique_nodes: 有被讀取的 unique node 數（node_id 去重數量）
read_trace_unique_nodes

# read_trace_cache_hits: cache hit 事件數（is_cache_hit=1 的筆數）
read_trace_cache_hits

# read_trace_disk_reads: SSD 讀取事件數（is_cache_hit=0 的筆數）
read_trace_disk_reads

# read_trace_cache_hit_ratio: cache hit 佔比 = read_trace_cache_hits / read_trace_total_reads
read_trace_cache_hit_ratio

# read_trace_disk_read_ratio: SSD 讀取佔比 = read_trace_disk_reads / read_trace_total_reads
read_trace_disk_read_ratio

# read_trace_window_ms_list: 記錄視窗清單（逗號分隔，例如 0.5,1,2,5）
#   小數視窗在欄位名會轉成 p（例：0.5ms -> ms0p5）
read_trace_window_ms_list

# read_trace_repeat_reads_ms{W}: 以時間窗為單位的重複讀取事件數（含 cache hit）
#   計算方式：對每個時間窗，repeat = sum(max(node_count-1, 0))，再跨窗加總
read_trace_repeat_reads_ms{W}

# read_trace_repeat_ratio_ms{W}: 重複讀取比例 = read_trace_repeat_reads_ms{W} / read_trace_total_reads
read_trace_repeat_ratio_ms{W}

# read_trace_repeat_multi_thread_reads_ms{W}: 多 thread 重複事件數（含 cache hit）
#   計算方式：對每個時間窗，sum(node_count) 其中 node_count>1 且該窗內 threads>=2
read_trace_repeat_multi_thread_reads_ms{W}

# read_trace_repeat_multi_thread_ratio_ms{W}: 多 thread 重複比例 =
#   read_trace_repeat_multi_thread_reads_ms{W} / read_trace_total_reads
read_trace_repeat_multi_thread_ratio_ms{W}

# read_trace_max_unique_threads_ms{W}_*: 每個時間窗內「最大不同 thread 數」的分佈
#   先對每個時間窗取 max(unique_threads_per_node)，再做 mean/std/iqr/cv/p0~p100
read_trace_max_unique_threads_ms{W}_mean
read_trace_max_unique_threads_ms{W}_std
read_trace_max_unique_threads_ms{W}_iqr
read_trace_max_unique_threads_ms{W}_cv
read_trace_max_unique_threads_ms{W}_p0~p100

# read_trace_repeat_*_disk_ms{W}: 只針對 SSD read（is_cache_hit=0）的重複讀取統計
#   與 read_trace_repeat_*_ms{W} 相同算法，但只計 SSD read
read_trace_repeat_reads_disk_ms{W}
read_trace_repeat_ratio_disk_ms{W}
read_trace_repeat_multi_thread_reads_disk_ms{W}
read_trace_repeat_multi_thread_ratio_disk_ms{W}

# read_trace_node_window_reads_ms{W}_*: 每個時間窗的「最大單一 node 讀取次數」分佈
#   計算方式：每個時間窗取 max(node_reads) -> 對所有窗做統計
read_trace_node_window_reads_ms{W}_mean
read_trace_node_window_reads_ms{W}_std
read_trace_node_window_reads_ms{W}_iqr
read_trace_node_window_reads_ms{W}_cv
read_trace_node_window_reads_ms{W}_p0~p100

# read_trace_node_window_reads_ratio_ms{W}_*: 每個時間窗的「max(node_reads)/window_total」分佈
read_trace_node_window_reads_ratio_ms{W}_mean
read_trace_node_window_reads_ratio_ms{W}_std
read_trace_node_window_reads_ratio_ms{W}_iqr
read_trace_node_window_reads_ratio_ms{W}_cv
read_trace_node_window_reads_ratio_ms{W}_p0~p100

# read_trace_node_same_thread_reads_ms{W}_*: 每個時間窗的「max(per-thread node_reads)」分佈
read_trace_node_same_thread_reads_ms{W}_mean
read_trace_node_same_thread_reads_ms{W}_std
read_trace_node_same_thread_reads_ms{W}_iqr
read_trace_node_same_thread_reads_ms{W}_cv
read_trace_node_same_thread_reads_ms{W}_p0~p100

# read_trace_node_same_thread_reads_ratio_ms{W}_*: 每個時間窗的「max(per-thread node_reads)/window_total」分佈
read_trace_node_same_thread_reads_ratio_ms{W}_mean
read_trace_node_same_thread_reads_ratio_ms{W}_std
read_trace_node_same_thread_reads_ratio_ms{W}_iqr
read_trace_node_same_thread_reads_ratio_ms{W}_cv
read_trace_node_same_thread_reads_ratio_ms{W}_p0~p100

# read_trace_node_multi_thread_reads_ms{W}_*: 每個時間窗的「max(node_reads) among nodes with >=2 threads」分佈
read_trace_node_multi_thread_reads_ms{W}_mean
read_trace_node_multi_thread_reads_ms{W}_std
read_trace_node_multi_thread_reads_ms{W}_iqr
read_trace_node_multi_thread_reads_ms{W}_cv
read_trace_node_multi_thread_reads_ms{W}_p0~p100

# read_trace_node_multi_thread_reads_ratio_ms{W}_*: 每個時間窗的「max(node_reads with >=2 threads)/window_total」分佈
read_trace_node_multi_thread_reads_ratio_ms{W}_mean
read_trace_node_multi_thread_reads_ratio_ms{W}_std
read_trace_node_multi_thread_reads_ratio_ms{W}_iqr
read_trace_node_multi_thread_reads_ratio_ms{W}_cv
read_trace_node_multi_thread_reads_ratio_ms{W}_p0~p100

# read_trace_window_node_reads_ms{W}_*: 「以時間窗為單位」統計 node 讀取次數分佈
#   計算方式：對每個時間窗 W，計算 node_counts（node_id -> 次數），
#   再把該窗所有次數值加入全域列表，最後對列表做 mean/std/iqr/cv/p0~p100。
read_trace_window_node_reads_ms{W}_mean
read_trace_window_node_reads_ms{W}_std
read_trace_window_node_reads_ms{W}_iqr
read_trace_window_node_reads_ms{W}_cv
read_trace_window_node_reads_ms{W}_p0~p100

# read_trace_window_node_read_ratio_ms{W}_*: 每個時間窗內「node_count/window_total」列表的分佈
read_trace_window_node_read_ratio_ms{W}_mean
read_trace_window_node_read_ratio_ms{W}_std
read_trace_window_node_read_ratio_ms{W}_iqr
read_trace_window_node_read_ratio_ms{W}_cv
read_trace_window_node_read_ratio_ms{W}_p0~p100

# read_trace_window_node_threads_ms{W}_*: 每個時間窗內「node 被多少 thread 存取」列表的分佈
#   範例：窗內 A={t1,t2}, B={t3} -> 追加 {2,1}
read_trace_window_node_threads_ms{W}_mean
read_trace_window_node_threads_ms{W}_std
read_trace_window_node_threads_ms{W}_iqr
read_trace_window_node_threads_ms{W}_cv
read_trace_window_node_threads_ms{W}_p0~p100

# read_trace_window_node_thread_ratio_ms{W}_*: 每個時間窗內「node threads / 該窗總 threads」列表分佈
#   範例：窗內 threads=3 -> 追加 {2/3,1/3}
read_trace_window_node_thread_ratio_ms{W}_mean
read_trace_window_node_thread_ratio_ms{W}_std
read_trace_window_node_thread_ratio_ms{W}_iqr
read_trace_window_node_thread_ratio_ms{W}_cv
read_trace_window_node_thread_ratio_ms{W}_p0~p100

# read_trace_hot_nodes_*: 熱點節點貢獻度（依 hot window W 與 TOPK，彙總 *_read_trace_hot_nodes_<W>ms_top<N>.csv）
read_trace_hot_nodes_topk
# read_trace_hot_nodes_read_share: Top-K 熱點節點讀取事件數總和 / read_trace_total_reads
read_trace_hot_nodes_read_share
# read_trace_hot_nodes_repeat_mt_share: Top-K 熱點節點的多 thread 重複事件數總和 / read_trace_total_reads
read_trace_hot_nodes_repeat_mt_share

# topk_*: Top-K 熱點節點鄰居統計（靜態圖結構）
topk_expanded_neighbor_count
topk_expanded_unique_neighbors_count
topk_expanded_degree_mean
topk_expanded_coverage_ratio
