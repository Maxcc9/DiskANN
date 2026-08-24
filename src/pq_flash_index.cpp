// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include "tsl/robin_set.h"
#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, diskann::Metric m)
    : reader(fileReader), metric(m), _thread_data(nullptr)
{
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
    if (_full_nhood_preload_buf != nullptr)
    {
        delete[] _full_nhood_preload_buf;
    }
}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT> inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    reader->read(read_reqs, ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif

        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

        if (coord_buffers[i] != nullptr)
        {
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
        }

        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    size_t num_cached_nodes = node_list.size();

    // Allocate space for neighborhood cache
    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
        auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    diskann::cout << "..done." << std::endl;
}

// Pre-seed the bounded (dynamic) cache with BFS entry-region nodes. Reads the
// nodes itself (independent of the separate static cache) and inserts them into
// _bounded_cache with neighbours + coords, exactly as the search hot-path would.
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::seed_bounded_cache_bfs(uint64_t num_seed_nodes)
{
    if (!_bounded_cache.enabled() || num_seed_nodes == 0)
        return;

    std::vector<uint32_t> node_list;
    cache_bfs_levels(num_seed_nodes, node_list);
    const size_t n = node_list.size();
    diskann::cout << "[BNC seed] seeding " << n << " BFS nodes into bounded cache..." << std::flush;

    // Large batch so each read_nodes() coalesces many aligned disk reads (was 8 →
    // ~7.5M round-trips for 60M nodes; 1024 matches cache_bfs_levels, ~6x faster).
    const size_t BLOCK_SIZE = 1024;
    const size_t num_blocks = DIV_ROUND_UP(n, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        const size_t start_idx = block * BLOCK_SIZE;
        const size_t end_idx = (std::min)(n, (block + 1) * BLOCK_SIZE);

        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        std::vector<std::unique_ptr<T[]>> coord_owned;
        std::vector<std::unique_ptr<uint32_t[]>> nbr_owned;
        for (size_t i = start_idx; i < end_idx; i++)
        {
            nodes_to_read.push_back(node_list[i]);
            coord_owned.emplace_back(new T[_aligned_dim]);
            nbr_owned.emplace_back(new uint32_t[_max_degree + 1]);
            coord_buffers.push_back(coord_owned.back().get());
            nbr_buffers.emplace_back(0, nbr_owned.back().get());
        }

        auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i])
                _bounded_cache.insert(nodes_to_read[i], nbr_buffers[i].second, nbr_buffers[i].first,
                                      coord_buffers[i]);
        }
    }
    diskann::cout << "done." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    if (num_nodes_to_cache >= this->_num_points)
    {
        // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
        node_list.resize(this->_num_points);
        for (uint32_t i = 0; i < this->_num_points; ++i)
        {
            node_list[i] = i;
        }
        return;
    }

    this->_count_visited_nodes = true;
    this->_node_visit_counter.clear();
    this->_node_visit_counter.resize(this->_num_points);
    for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    {
        this->_node_visit_counter[i].first = i;
        this->_node_visit_counter[i].second = 0;
    }

    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

    bool filtered_search = false;
    std::vector<LabelT> random_query_filters(sample_num);
    if (_filter_to_medoid_ids.size() != 0)
    {
        filtered_search = true;
        generate_random_labels(random_query_filters, (uint32_t)sample_num, nthreads);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        auto &label_for_search = random_query_filters[i];
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                           tmp_result_dists.data() + i, beamwidth, filtered_search, label_for_search,
                           std::numeric_limits<float>::max(), std::numeric_limits<uint32_t>::max(), 1.0f, 0, false);
    }

    std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    num_nodes_to_cache = std::min(num_nodes_to_cache, this->_node_visit_counter.size());
    node_list.reserve(num_nodes_to_cache);
    for (uint64_t i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->_node_visit_counter[i].first);
    }
    this->_count_visited_nodes = false;

    diskann::aligned_free(samples);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    // Cap at total node count (no artificial 10% limit — let BFS cache fill to budget).
    uint64_t tenp_nodes = this->_num_points;
    if (num_nodes_to_cache > tenp_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                      << "(total nodes:" << this->_num_points << ")" << std::endl;
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }

    // Per-medoid BFS: distribute budget evenly so every shard region is represented equally.
    // Each medoid gets budget = N / num_medoids; remainder goes to the last medoid.
    // global_seen is shared across all BFS runs to prevent double-counting overlapping regions.
    // Single-medoid BFS (DiskANN-style): the full budget is taken from one entry
    // point. The multi-medoid variant is disabled: for clustered medoids (e.g.
    // spacev's 5 near-co-located medoids) it gave no extra coverage AND skipped
    // "already claimed" medoids via the shared global_seen, loading only ~1/N of
    // num_nodes_to_cache. Single medoid loads the full requested budget.
    uint64_t n_entries = 1;
    uint64_t budget_each = num_nodes_to_cache;
    uint64_t budget_last = num_nodes_to_cache;

    diskann::cout << "Caching " << num_nodes_to_cache << " nodes across " << n_entries
                  << " medoid(s) (budget_per_medoid=" << budget_each << ")" << std::endl;

    tsl::robin_set<uint32_t> global_seen;
    node_list.clear();
    node_list.reserve(num_nodes_to_cache);

    for (uint64_t m = 0; m < n_entries; m++)
    {
        uint32_t entry_node = _medoids[m];
        uint64_t budget = (m == n_entries - 1) ? budget_last : budget_each;

        if (global_seen.count(entry_node))
        {
            diskann::cout << "Medoid " << m << " (node " << entry_node
                          << ") already claimed, skipping." << std::endl;
            continue;
        }

        diskann::cout << "Medoid " << m << " (node " << entry_node
                      << "), budget=" << budget << std::endl;

        std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
        cur_level  = std::make_unique<tsl::robin_set<uint32_t>>();
        prev_level = std::make_unique<tsl::robin_set<uint32_t>>();
        cur_level->insert(entry_node);

        tsl::robin_set<uint32_t> local_set;
        uint64_t lvl = 1;
        uint64_t prev_local_size = 0;

        while ((local_set.size() + cur_level->size() < budget) && !cur_level->empty())
        {
            std::swap(prev_level, cur_level);
            cur_level->clear();

            std::vector<uint32_t> nodes_to_expand;
            for (const uint32_t &id : *prev_level)
            {
                if (global_seen.count(id) || local_set.count(id))
                    continue;
                local_set.insert(id);
                nodes_to_expand.push_back(id);
            }

            if (shuffle)
                std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
            else
                std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

            diskann::cout << "  Level " << lvl << std::flush;
            bool finish_flag = false;
            uint64_t BLOCK_SIZE = 1024;
            uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);

            for (size_t block = 0; block < nblocks && !finish_flag; block++)
            {
                diskann::cout << "." << std::flush;
                size_t blk_start = block * BLOCK_SIZE;
                size_t blk_end   = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

                std::vector<uint32_t> nodes_to_read;
                std::vector<T *> coord_buffers(blk_end - blk_start, nullptr);
                std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

                for (size_t cur_pt = blk_start; cur_pt < blk_end; cur_pt++)
                {
                    nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                    nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
                }

                auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

                for (uint32_t i = 0; i < read_status.size(); i++)
                {
                    if (!read_status[i])
                    {
                        delete[] nbr_buffers[i].second;
                        continue;
                    }
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        if (!global_seen.count(nbrs[j]) && !local_set.count(nbrs[j]))
                            cur_level->insert(nbrs[j]);
                        if (local_set.size() + cur_level->size() >= budget)
                            finish_flag = true;
                    }
                    delete[] nbr_buffers[i].second;
                }
            }

            diskann::cout << " #nodes: " << local_set.size() - prev_local_size
                          << ", total: " << local_set.size() << std::endl;
            prev_local_size = local_set.size();
            lvl++;
        }

        // Drain remaining cur_level frontier into local_set up to budget.
        for (const uint32_t &id : *cur_level)
        {
            if (local_set.size() >= budget)
                break;
            if (!global_seen.count(id))
                local_set.insert(id);
        }

        // Commit this medoid's nodes to the shared global state.
        for (const uint32_t &id : local_set)
        {
            global_seen.insert(id);
            node_list.push_back(id);
        }

        diskann::cout << "Medoid " << m << " done: " << local_set.size()
                      << " nodes (running total: " << node_list.size() << ")" << std::endl;
    }

    diskann::cout << "cache_bfs_levels done. Total cached: " << node_list.size() << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < _data_dim; i++)
                    _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                _disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (_centroid_data + cur_m * _aligned_dim));
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix)
{
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#else
    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#endif
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->_disk_index_file = _disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint8_t>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file))
    {
        FileContent &content_labels = files.getContent(labels_file);
        std::stringstream infile(std::string((const char *)content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file))
    {
        std::ifstream infile(labels_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + labels_file, -1);
        }
#endif
        parse_label_file(infile, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->_num_points);

#ifndef EXEC_ENV_OLS
        infile.close();
#endif

#ifdef EXEC_ENV_OLS
        FileContent &content_labels_map = files.getContent(labels_map_file);
        std::stringstream map_reader(std::string((const char *)content_labels_map._content, content_labels_map._size));
#else
        std::ifstream map_reader(labels_map_file);
#endif
        _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
        map_reader.close();
#endif

#ifdef EXEC_ENV_OLS
        if (files.fileExists(labels_to_medoids))
        {
            FileContent &content_labels_to_meoids = files.getContent(labels_to_medoids);
            std::stringstream medoid_stream(
                std::string((const char *)content_labels_to_meoids._content, content_labels_to_meoids._size));
#else
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
#endif
            std::string line, token;

            _filter_to_medoid_ids.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    uint32_t cnt = 0;
                    std::vector<uint32_t> medoids;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = (LabelT)std::stoul(token);
                        else
                            medoids.push_back((uint32_t)stoul(token));
                        cnt++;
                    }
                    _filter_to_medoid_ids[label].swap(medoids);
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
        if (files.fileExists(univ_label_file))
        {
            FileContent &content_univ_label = files.getContent(univ_label_file);
            std::stringstream universal_label_reader(
                std::string((const char *)content_univ_label._content, content_univ_label._size));
#else
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
#endif
            std::string univ_label;
            universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
            universal_label_reader.close();
#endif
            LabelT label_as_num = (LabelT)std::stoul(univ_label);
            set_universal_label(label_as_num);
        }

#ifdef EXEC_ENV_OLS
        if (files.fileExists(dummy_map_file))
        {
            FileContent &content_dummy_map = files.getContent(dummy_map_file);
            std::stringstream dummy_map_stream(
                std::string((const char *)content_dummy_map._content, content_dummy_map._size));
#else
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
#endif
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t dummy_id;
                uint32_t real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (uint32_t)stoul(token);
                    else
                        real_id = (uint32_t)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<uint32_t>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
#ifndef EXEC_ENV_OLS
            dummy_map_stream.close();
#endif
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
#ifdef EXEC_ENV_OLS
    if (files.fileExists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
    if (file_exists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
        _disk_bytes_per_point =
            _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << _disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(_disk_index_file);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);
#endif

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != _num_points)
    {
        diskann::cout << "Mismatch in #points for compressed data file and disk "
                         "index file: "
                      << disk_nnodes << " vs " << _num_points << std::endl;
        return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, _max_node_len);
    READ_U64(index_metadata, _nnodes_per_sector);
    _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

    if (_max_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->_num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->_num_frozen_points == 1)
        this->_frozen_location = file_frozen_id;
    if (this->_num_frozen_points == 1)
    {
        diskann::cout << " Detected frozen point in index at location " << this->_frozen_location
                      << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->_reorder_data_exists);
    if (this->_reorder_data_exists)
    {
        if (this->_use_disk_index_pq == false)
        {
            throw ANNException("Reordering is designed for used with disk PQ "
                               "compression option",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->_reorder_data_start_sector);
        READ_U64(index_metadata, this->_ndims_reorder_vecs);
        READ_U64(index_metadata, this->_nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << _nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << _max_node_len;
    diskann::cout << ", max node degree: " << _max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    std::string index_fname(_disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, norm_file, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(files, norm_val, dumr, dumc);
#else
    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
#endif
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }
    diskann::cout << "done.." << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// enable_neighbor_cache(): Phase-1 full preload
// Scans the entire disk.index sector-by-sector and populates _nhood_cache
// with every node's adjacency list.  After this call the beam-search loop
// will hit _nhood_cache for every traversal node and issue zero graph I/Os.
// ---------------------------------------------------------------------------
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::enable_neighbor_cache()
{
    if (!_load_flag)
    {
        diskann::cout << "[NeighborCache] Index not loaded yet; cannot preload neighbors." << std::endl;
        return;
    }

    diskann::cout << "[NeighborCache] Initialising full-preload neighbor cache for "
                  << _num_points << " nodes (max_degree=" << _max_degree << ") ..." << std::flush;

    // Initialise the NeighborCache data structure.
    _neighbor_cache.init(static_cast<uint32_t>(_num_points), static_cast<uint32_t>(_max_degree));

    // Allocate a single sector-sized read buffer (reused across reads).
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // We read one "logical sector group" at a time (either a sector shared by
    // multiple nodes, or one or more sectors belonging to a single large node).
    // For multi-node-per-sector layout (nnodes_per_sector > 0) we iterate over
    // graph sectors 1 … ceil(num_points / nnodes_per_sector).
    // For multi-sector-per-node layout we iterate over each node individually.

    char *sector_buf = nullptr;
    alloc_aligned(reinterpret_cast<void **>(&sector_buf),
                  num_sectors_per_node * defaults::SECTOR_LEN,
                  defaults::SECTOR_LEN);

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto thread_data = manager.scratch_space();
    IOContext &ctx   = thread_data->ctx;

    // Also allocate the backing buffer for _nhood_cache (if not already done).
    // We use a separate large flat buffer owned by this function and stored as a
    // class member so that the pointers in _nhood_cache remain valid.
    //
    // Design note: _nhood_cache_buf is already used by load_cache_list() for the
    // BFS warm-up cache.  Rather than reusing that buffer (whose layout is
    // arbitrary), we allocate a fresh contiguous block sized for ALL nodes and
    // insert entries directly into _nhood_cache, overwriting any prior entries
    // for the same node_id.  This is safe because _nhood_cache is only read
    // during beam_search (single-writer at startup time).

    // Allocate the full-preload nhood buffer: num_points × (max_degree+1) uint32_t
    // (+1 because the DiskANN sector layout stores [nnbrs][id0][id1]…)
    const size_t buf_elems = static_cast<size_t>(_num_points) * (_max_degree + 1);
    uint32_t *full_nhood_buf = new uint32_t[buf_elems];
    memset(full_nhood_buf, 0, buf_elems * sizeof(uint32_t));

    uint32_t nodes_populated = 0;

    if (_nnodes_per_sector > 0)
    {
        // Multi-node-per-sector layout.
        const uint64_t total_graph_sectors = DIV_ROUND_UP(_num_points, _nnodes_per_sector);
        for (uint64_t sec = 0; sec < total_graph_sectors; ++sec)
        {
            // Sector numbering: graph sector 0 is at disk offset SECTOR_LEN
            // (sector 0 = header).
            AlignedRead req;
            req.offset = (1 + sec) * defaults::SECTOR_LEN;
            req.len    = defaults::SECTOR_LEN;
            req.buf    = sector_buf;

            std::vector<AlignedRead> reqs = {req};
            reader->read(reqs, ctx);

            const uint64_t nodes_in_sec = std::min(static_cast<uint64_t>(_nnodes_per_sector),
                                                    _num_points - sec * _nnodes_per_sector);
            for (uint64_t local = 0; local < nodes_in_sec; ++local)
            {
                const uint32_t node_id = static_cast<uint32_t>(sec * _nnodes_per_sector + local);
                char *node_buf         = offset_to_node(sector_buf, node_id);
                uint32_t *nhood_ptr    = offset_to_node_nhood(node_buf);
                const uint32_t nnbrs   = *nhood_ptr;
                const uint32_t *nbrs   = nhood_ptr + 1;

                // Copy into full_nhood_buf.
                uint32_t *dst = full_nhood_buf + static_cast<size_t>(node_id) * (_max_degree + 1);
                *dst          = nnbrs;
                const uint32_t copy_n = (nnbrs <= _max_degree) ? nnbrs : static_cast<uint32_t>(_max_degree);
                memcpy(dst + 1, nbrs, copy_n * sizeof(uint32_t));

                // Insert into _nhood_cache (points to the slot we just wrote).
                _nhood_cache[node_id] = std::make_pair(copy_n, dst + 1);

                // Also record in NeighborCache for future lookup().
                _neighbor_cache.insert(node_id, dst + 1, copy_n);
                ++nodes_populated;
            }
        }
    }
    else
    {
        // Multi-sector-per-node layout.
        for (uint64_t node_id = 0; node_id < _num_points; ++node_id)
        {
            AlignedRead req;
            req.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
            req.len    = num_sectors_per_node * defaults::SECTOR_LEN;
            req.buf    = sector_buf;

            std::vector<AlignedRead> reqs = {req};
            reader->read(reqs, ctx);

            char *node_buf      = offset_to_node(sector_buf, node_id);
            uint32_t *nhood_ptr = offset_to_node_nhood(node_buf);
            const uint32_t nnbrs = *nhood_ptr;
            const uint32_t *nbrs = nhood_ptr + 1;

            uint32_t *dst = full_nhood_buf + static_cast<size_t>(node_id) * (_max_degree + 1);
            *dst          = nnbrs;
            const uint32_t copy_n = (nnbrs <= _max_degree) ? nnbrs : static_cast<uint32_t>(_max_degree);
            memcpy(dst + 1, nbrs, copy_n * sizeof(uint32_t));

            _nhood_cache[node_id] = std::make_pair(copy_n, dst + 1);
            _neighbor_cache.insert(node_id, dst + 1, copy_n);
            ++nodes_populated;
        }
    }

    // Hand ownership of the buffer to the appropriate member so that the
    // destructor will free it.
    // Case 1: load_cache_list() was never called  → _nhood_cache_buf is null;
    //         we own it via _nhood_cache_buf (destructor already deletes it).
    // Case 2: load_cache_list() was called first   → _nhood_cache_buf is taken;
    //         store in _full_nhood_preload_buf (destructor deletes it too).
    //         The old BFS-cache entries in _nhood_cache have been overwritten
    //         for every node_id, so _nhood_cache_buf's memory is now unused but
    //         not leaked (it is still freed by the destructor).
    if (_nhood_cache_buf == nullptr)
    {
        _nhood_cache_buf = full_nhood_buf;
    }
    else
    {
        _full_nhood_preload_buf = full_nhood_buf;
    }

    aligned_free(sector_buf);

    const size_t mb = _neighbor_cache.size_bytes() / (1024 * 1024);
    diskann::cout << " done. Populated " << nodes_populated << " nodes."
                  << " Approx DRAM used: " << mb << " MB." << std::endl;
}

// ---------------------------------------------------------------------------
// init_bounded_neighbor_cache(): Phase-3 on-demand bounded cache setup.
// Initialises the BoundedNeighborCache data structure with the given byte
// budget.  Actual node data is inserted lazily during cached_beam_search().
// ---------------------------------------------------------------------------
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::init_bounded_neighbor_cache(size_t capacity_bytes)
{
    if (!_load_flag)
    {
        diskann::cout << "[BoundedCache] Index not loaded yet; cannot initialise." << std::endl;
        return;
    }
    _bounded_cache.init(capacity_bytes, static_cast<uint32_t>(_max_degree), _disk_bytes_per_point,
                        static_cast<uint32_t>(_max_nthreads));
    const size_t nodes = _bounded_cache.total_capacity_nodes();
    const double pct   = (_num_points > 0)
                         ? 100.0 * static_cast<double>(nodes) / static_cast<double>(_num_points)
                         : 0.0;
    diskann::cout << "[BoundedCache] capacity=" << nodes << " nodes ("
                  << pct << "% of " << _num_points << "), "
                  << capacity_bytes / (1024.0 * 1024.0 * 1024.0) << " GB, "
                  << _bounded_cache.num_shards() << " shards (from " << _max_nthreads
                  << " worker threads)" << std::endl;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(std::shared_ptr<AlignedFileReader> &reader, IOContext &ctx, size_t size,
                             int &completedIndex)
{
    if ((*ctx.m_pRequests)[0].m_callback)
    {
        bool waitsRemaining = false;
        long completeCount = ctx.m_completeCount;
        do
        {
            for (int i = 0; i < size; i++)
            {
                auto ithStatus = (*ctx.m_pRequestsStatus)[i];
                if (ithStatus == IOContext::Status::READ_SUCCESS)
                {
                    completedIndex = i;
                    return true;
                }
                else if (ithStatus == IOContext::Status::READ_WAIT)
                {
                    waitsRemaining = true;
                }
            }

            // if we didn't find one in READ_SUCCESS, wait for one to complete.
            if (waitsRemaining)
            {
                WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
                // this assumes the knowledge of the reader behavior (implicit
                // contract). need better factoring?
            }
        } while (waitsRemaining);

        completedIndex = -1;
        return false;
    }
    else
    {
        reader->wait(ctx, completedIndex);
        return completedIndex != -1;
    }
}
#endif

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::load_entry_candidates(const std::string &file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in.is_open())
        throw ANNException("Could not open entry_candidates file: " + file, -1, __FUNCSIG__, __FILE__, __LINE__);
    uint32_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(uint32_t));
    _entry_candidates.resize(count);
    in.read(reinterpret_cast<char *>(_entry_candidates.data()), (std::streamsize)count * sizeof(uint32_t));
    diskann::cout << "[EntryRouter] loaded " << _entry_candidates.size() << " entry candidates from " << file
                  << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const float et_theta, const uint32_t hop_budget,
                                                 const float et_sat_gamma, const uint32_t et_sat_delta,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       et_theta, hop_budget, et_sat_gamma, et_sat_delta, use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const float et_theta, const uint32_t hop_budget,
                                                 const float et_sat_gamma, const uint32_t et_sat_delta,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), et_theta, hop_budget, et_sat_gamma, et_sat_delta,
                       std::numeric_limits<float>::max(), 0, use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const float et_theta,
                                                 const uint32_t hop_budget, const float et_sat_gamma,
                                                 const uint32_t et_sat_delta, const bool use_reorder_data,
                                                 QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       et_theta, hop_budget, et_sat_gamma, et_sat_delta,
                       std::numeric_limits<float>::max(), 0, use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const float et_theta,
                                                 const uint32_t hop_budget, const float et_sat_gamma,
                                                 const uint32_t et_sat_delta, const float et_theta_exact,
                                                 const uint32_t et_conv_delta,
                                                 const bool use_reorder_data,
                                                 QueryStats *stats, const uint32_t *oracle_gt_ids,
                                                 const uint32_t et_ref_rank, const uint32_t et_min_hops,
                                                 const uint32_t et_conv_width, std::vector<float> *feat_log,
                                                 const uint32_t self_exclude_id, const float et_verify_alpha,
                                                 const uint32_t et_verify_patience, const bool et_exact_led,
                                                 const uint32_t et_exact_patience, const float et_exact_beta)
{
    // Predict-then-verify ET: θ-ET (PQ) is layer-1 predictor; this hop's minimum
    // EXACT distance vs the conv_width-th best EXACT distance is layer-2 verifier.
    const bool et_verify_on = (et_verify_alpha < std::numeric_limits<float>::max());

    // Patience convergence window AND layer-2 verify reference rank.
    //   conv_topk holds the conv_width smallest EXACT distances seen so far;
    //   conv_topk.top() is the conv_width-th best. Widening conv_width (e.g. 40)
    //   moves the verify reference to a farther rank → larger stop threshold →
    //   more conservative ET → higher recall. Decoupled from k_search (recall@k):
    //   measure recall@10 while letting ET watch top-40. Default = k_search.
    const uint32_t conv_width = (et_conv_width > 0) ? et_conv_width : (uint32_t)k_search;

    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__,
                           __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // normalization step. for cosine, we simply normalize the query
    // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
    // convert MIPS to L2 search
    if (metric == diskann::Metric::INNER_PRODUCT || metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> PQ chunk centers distances
    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                            float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
    };
    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m, (uint32_t)_aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else
    {
        if (_filter_to_medoid_ids.find(filter_label) != _filter_to_medoid_ids.end())
        {
            const auto &medoid_ids = _filter_to_medoid_ids[filter_label];
            for (uint64_t cur_m = 0; cur_m < medoid_ids.size(); cur_m++)
            {
                // for filtered index, we dont store global centroid data as for unfiltered index, so we use PQ distance
                // as approximation to decide closest medoid matching the query filter.
                compute_dists(&medoid_ids[cur_m], 1, dist_scratch);
                float cur_expanded_dist = dist_scratch[0];
                if (cur_expanded_dist < best_dist)
                {
                    best_medoid = medoid_ids[cur_m];
                    best_dist = cur_expanded_dist;
                }
            }
        }
        else
        {
            throw ANNException("Cannot find medoid for specified filter.", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    // T4 query-adaptive entry router: replace the global medoid with the nearest
    // entry candidate (by PQ distance). Candidates are processed in batches that
    // fit the PQ scratch (sized for MAX_GRAPH_DEGREE), tracking the global min.
    if (!use_filter && !_entry_candidates.empty())
    {
        const size_t BATCH = 256;
        for (size_t off = 0; off < _entry_candidates.size(); off += BATCH)
        {
            const uint32_t cnt = (uint32_t)std::min(BATCH, _entry_candidates.size() - off);
            compute_dists(_entry_candidates.data() + off, cnt, dist_scratch);
            for (uint32_t j = 0; j < cnt; j++)
            {
                if (dist_scratch[j] < best_dist)
                {
                    best_dist = dist_scratch[j];
                    best_medoid = _entry_candidates[off + j];
                }
            }
        }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    // For feature-dump (learned-ET): track previous-hop top-k / top-40 id sets to
    // detect membership churn. Only used when feat_log != nullptr.
    std::vector<uint32_t> feat_prev_topk, feat_prev_top40;
    // Running PQ-vs-exact residual stats over all expanded nodes (group B features).
    std::vector<std::pair<uint32_t, float>> beam_pq; // id -> PQ dist for current beam
    double pq_resid_sum = 0.0, pq_resid_sqsum = 0.0;
    float pq_resid_max = 0.0f;
    uint64_t pq_resid_n = 0;

    // Nodes for which we already hold the exact L2 distance (computed from
    // the disk sector during the search phase).  These can be skipped in the
    // Phase-3 re-rank to avoid redundant disk reads at high concurrency.
    tsl::robin_set<uint32_t> exact_dist_nodes;

    // Top-k saturation state: track prev top-k IDs to detect convergence
    std::vector<uint32_t> sat_prev_ids(k_search, std::numeric_limits<uint32_t>::max());
    uint32_t sat_count = 0;

    // Exact Convergence ET: max-heap (worst of current top-K on top), tracks full_retset top-K
    // by exact distance. If top-K IDs unchanged for et_conv_delta consecutive hops → terminate.
    // Works because Direction 3 gives full_retset exact distances for all BNC-hit nodes.
    std::priority_queue<std::pair<float, uint32_t>> conv_topk; // max-heap by distance
    uint32_t et_conv_streak = 0;
    bool hop_topk_changed = false;

    // Predict-then-verify ET state (reset every hop).
    //   theta_pred_stop : layer-1 θ-ET (PQ) flagged "stop" this hop.
    //   hop_min_exact   : min EXACT distance among nodes expanded THIS hop
    //                     (PQ-fallback nodes excluded — strict exact-vs-exact at layer-2).
    float hop_min_exact = std::numeric_limits<float>::max();
    bool theta_pred_stop = false;
    // Patience: both layers must agree for this many CONSECUTIVE hops before we
    // actually stop. Any hop that fails to fire (θ-ET silent OR exact veto) resets it.
    uint32_t verify_streak = 0;

    // Dual-rail same-scale ET (new primary ET): consecutive-hop patience counter for
    // the joint (PQ-divergence AND exact-convergence) gate. Reset on any non-firing hop.
    uint32_t exact_streak = 0;
    // Global best (smallest) EXACT distance over all hops so far (the 1st-place exact),
    // used as the convergence anchor by the exact rail. Updated at each hop's end.
    float global_best_exact = std::numeric_limits<float>::max();

    // Exact-led mode: carry the PREVIOUS hop's min exact distance (persists across
    // hops, NOT reset each hop). Lets the hop-top gate use already-paid-for exact
    // info to decide BEFORE issuing this hop's beam (no verify-beam cost).
    float prev_hop_min_exact = std::numeric_limits<float>::max();

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        // Reset predict-then-verify per-hop state before any gate runs.
        hop_min_exact = std::numeric_limits<float>::max();
        theta_pred_stop = false;

        // ── Exact-led ET (hop top, decides BEFORE issuing this hop's beam) ─────────
        // Inverts predict-then-verify: lead with the FREE & reliable exact signal
        // (previous hop found nothing within α of the k-th best exact), then confirm
        // with the PQ look-ahead (immediate frontier also θ× far). Both must hold for
        // `patience` consecutive hops. Uses prev_hop_min_exact → no verify-beam cost,
        // and requires OBSERVED exact non-improvement → more recall-safe (good for
        // truncation-sensitive datasets like deep100m).
        if (et_verify_on && et_exact_led)
        {
            bool fired = false;
            if (hops >= et_min_hops && prev_hop_min_exact < std::numeric_limits<float>::max() &&
                (uint32_t)conv_topk.size() >= conv_width)
            {
                const float kth_exact = conv_topk.top().first;
                if (kth_exact > 1e-9f && prev_hop_min_exact > kth_exact * et_verify_alpha)
                {
                    // exact gate held → PQ look-ahead on the immediate frontier
                    const uint32_t rr = (et_ref_rank > 0) ? et_ref_rank : (uint32_t)k_search;
                    if (et_theta < std::numeric_limits<float>::max() && retset.size() >= rr)
                    {
                        const float best_unexp_pq = retset.peek_unexpanded_dist();
                        const float ref_pq = retset[rr - 1].distance;
                        if (ref_pq > 1e-9f && best_unexp_pq > ref_pq * et_theta)
                            fired = true;
                    }
                }
            }
            if (fired)
            {
                if (++verify_streak >= std::max(1u, et_verify_patience))
                    break; // stop before issuing this hop's beam
            }
            else
                verify_streak = 0;
        }

        // θ-ET with configurable reference rank (et_ref_rank).
        //   ref_rank = 0 → defaults to k_search (classic: compare vs k-th best).
        //   ref_rank > k → widen the "attention window" toward L; ET becomes more
        //     conservative (stop only when the closest unexpanded node is θ× farther
        //     than the ref_rank-th candidate). With θ=1 this means "all of the top
        //     ref_rank candidates are already expanded".
        // Decouples the search-pool width (L) from the termination depth (ref_rank).
        // et_min_hops: grace period. ET only activates AFTER a query has run this
        // many hops, so the bulk of queries (which converge within the profiled
        // P50/P75 hop count) run untouched and keep full recall; only the long-
        // running tail queries become subject to ET, capping P99. 0 = no grace.
        if (!et_exact_led && et_theta < std::numeric_limits<float>::max() && hops >= et_min_hops)
        {
            const uint32_t rr = (et_ref_rank > 0) ? et_ref_rank : (uint32_t)k_search;
            if (retset.size() >= rr)
            {
                const float best_unexp_pq = retset.peek_unexpanded_dist();
                const float ref_pq = retset[rr - 1].distance;
                if (ref_pq > 1e-9f && best_unexp_pq > ref_pq * et_theta)
                {
                    if (!et_verify_on)
                        break; // classic θ-ET: PQ predictor stops immediately.
                    else
                        // Predict-then-verify: defer. Grant this hop so the layer-2
                        // exact verifier (hop bottom) can confirm or veto the stop.
                        theta_pred_stop = true;
                }
            }
        }

        // ── Cross-scale exact ET (primary mechanism) ─────────────────────────────
        // Before issuing this hop's SSD I/O, a single cross-scale check:
        //   D_pq_cand  = best UNexpanded node's PQ distance (predicted, not yet read)
        //   D_exact_k2 = conv_width-th (= k×2) best EXACT distance held in the candidate
        //                list from prior hops (conv_topk.top()).
        //   gate (γ = et_theta_exact): D_pq_cand > D_exact_k2 × γ
        // PQ underestimates true distance, so using the EXACT k×2-th as the reliable
        // per-hop anchor and requiring the next PQ candidate to exceed it by γ is a
        // conservative "this hop won't improve recall" signal — and the exact anchor is
        // checked PER HOP against the actual next candidate (this per-hop exact anchor is
        // what makes it stronger than a same-scale convergence threshold). On trigger we
        // bump a patience counter; only after et_exact_patience CONSECUTIVE triggering
        // hops do we stop — BEFORE issuing this hop's I/O. Any non-firing hop resets it.
        // conv_width (et_conv_width, set to k×2) decouples the ET reference rank from k.
        // (et_exact_beta is reserved/unused in this mechanism.)
        if (et_theta_exact < std::numeric_limits<float>::max() && hops >= et_min_hops &&
            (uint32_t)conv_topk.size() >= conv_width)
        {
            const float best_unexp_pq = retset.peek_unexpanded_dist();
            const float d_exact_k2 = conv_topk.top().first; // conv_width-th (=k×2) best EXACT
            if (d_exact_k2 > 1e-9f && best_unexp_pq > d_exact_k2 * et_theta_exact)
            {
                if (++exact_streak >= std::max(1u, et_exact_patience))
                    break; // stop before issuing this hop's SSD I/O
            }
            else
                exact_streak = 0;
        }

        if (hops >= hop_budget)
        {
            break;
        }

        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        hop_topk_changed = false;
        beam_pq.clear();
        // find new beam
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            if (feat_log != nullptr) beam_pq.push_back({nbr.id, nbr.distance}); // PQ dist for residual
            auto iter = _nhood_cache.find(nbr.id);
            if (iter != _nhood_cache.end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else if (_bounded_cache.enabled())
            {
                const auto *entry = _bounded_cache.lookup(nbr.id);
                if (entry != nullptr)
                {
                    // Bounded cache hit: reuse the same pair<uint64_t, uint32_t*> format
                    // as _nhood_cache so the processing loop below is unchanged.
                    cached_nhoods.push_back(
                        std::make_pair(nbr.id,
                                       std::make_pair(static_cast<uint64_t>(entry->degree),
                                                      entry->neighbors)));
                    if (stats != nullptr)
                    {
                        stats->n_cache_hits++;
                    }
                }
                else
                {
                    frontier.push_back(nbr.id);
                }
            }
            else
            {
                frontier.push_back(nbr.id);
            }
            if (this->_count_visited_nodes)
            {
                reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
            }
        }

        // read nhoods of frontier ids
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            _global_io_count.fetch_add(frontier.size(), std::memory_order_relaxed); // T5: count SSD reads
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = id;
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * defaults::SECTOR_LEN,
                                                num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                num_ios++;
            }
            io_timer.reset();
#ifdef USE_BING_INFRA
            reader->read(frontier_read_reqs, ctx,
                         true); // asynhronous reader for Bing.
#else
            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
#endif
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            auto global_cache_iter = _coord_cache.find(cached_nhood.first);
            uint32_t node_id = (uint32_t)cached_nhood.first;
            float cur_expanded_dist;
            bool cur_is_exact = false; // true iff cur_expanded_dist is a true exact L2/IP dist
            if (global_cache_iter != _coord_cache.end())
            {
                // Coords available in _coord_cache (Phase-1 BFS preload).
                // Compute exact distance unconditionally: accurate kth_exact for ET,
                // and reduces post-loop re-rank work (node is added to exact_dist_nodes).
                T *node_fp_coords_copy = global_cache_iter->second;
                if (!_use_disk_index_pq)
                {
                    cur_expanded_dist =
                        _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
                }
                else
                {
                    if (metric == diskann::Metric::INNER_PRODUCT)
                        cur_expanded_dist =
                            _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance(
                            query_float, (uint8_t *)node_fp_coords_copy);
                }
                cur_is_exact = true; // Phase-1 coord_cache → exact distance
                if (_bounded_cache.enabled())
                    exact_dist_nodes.insert(node_id);
            }
            else if (_bounded_cache.enabled() && _bounded_cache.has_coords() &&
                     _bounded_cache.get_coords(node_id, data_buf))
            {
                // BNC-hit node: coords stored at original SSD-read time.
                // Compute exact distance now so full_retset has accurate distances
                // for kth_exact ET, and to skip this node in the post-loop re-rank.
                if (!_use_disk_index_pq)
                    cur_expanded_dist =
                        _dist_cmp->compare(aligned_query_T, (T *)data_buf, (uint32_t)_aligned_dim);
                else
                    cur_expanded_dist = (metric == diskann::Metric::INNER_PRODUCT)
                                            ? _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf)
                                            : _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
                cur_is_exact = true; // BNC-stored coords → exact distance
                exact_dist_nodes.insert(node_id);
            }
            else
            {
                // Fallback: no exact coords available (nhood_cache hit without coords).
                // Use PQ distance. cur_is_exact stays false → excluded from layer-2 verify
                // and from conv_topk (which must hold exact distances only).
                compute_dists(&node_id, 1, dist_scratch);
                cur_expanded_dist = dist_scratch[0];
            }
            full_retset.push_back(Neighbor(node_id, cur_expanded_dist));
            if (feat_log != nullptr)
                for (auto &bp : beam_pq)
                    if (bp.first == node_id)
                    { float rr = bp.second - cur_expanded_dist; pq_resid_sum += rr;
                      pq_resid_sqsum += (double)rr * rr; if (rr > pq_resid_max) pq_resid_max = rr;
                      pq_resid_n++; break; }
            // Layer-2 verifier accumulator + exact-only conv_topk maintenance.
            if (cur_is_exact)
            {
                if (cur_expanded_dist < hop_min_exact)
                    hop_min_exact = cur_expanded_dist;
                if (et_conv_delta > 0 || et_theta_exact < std::numeric_limits<float>::max() || et_verify_on)
                {
                    if ((uint32_t)conv_topk.size() < conv_width)
                    { conv_topk.push({cur_expanded_dist, node_id}); hop_topk_changed = true; }
                    else if (cur_expanded_dist < conv_topk.top().first)
                    { conv_topk.pop(); conv_topk.push({cur_expanded_dist, node_id}); hop_topk_changed = true; }
                }
            }

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            // compute node_nbrs <-> query dists in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }
#ifdef USE_BING_INFRA
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        // If we issued read requests and if a read is complete or there are
        // reads in wait state, then enter the while loop.
        while (requestCount > 0 && getNextCompletedRequest(reader, ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
        for (auto &frontier_nhood : frontier_nhoods)
        {
#endif
            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            uint32_t *node_nbrs_raw = node_buf + 1;

            // Extract coords before inserting into cache (insert needs them).
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);

            // Phase-3: populate bounded cache with neighbor list + raw coords.
            // Coords stored here eliminate re-rank disk reads for cache-hit nodes.
            if (_bounded_cache.enabled())
            {
                _bounded_cache.insert(frontier_nhood.first, node_nbrs_raw,
                                      static_cast<uint32_t>(nnbrs), node_fp_coords);
            }
            memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
            exact_dist_nodes.insert(static_cast<uint32_t>(frontier_nhood.first));
            // Frontier (disk-read) nodes are always exact → feed layer-2 verifier.
            if (cur_expanded_dist < hop_min_exact)
                hop_min_exact = cur_expanded_dist;
            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            if (feat_log != nullptr)
                for (auto &bp : beam_pq)
                    if (bp.first == (uint32_t)frontier_nhood.first)
                    { float rr = bp.second - cur_expanded_dist; pq_resid_sum += rr;
                      pq_resid_sqsum += (double)rr * rr; if (rr > pq_resid_max) pq_resid_max = rr;
                      pq_resid_n++; break; }
            if (et_conv_delta > 0 || et_theta_exact < std::numeric_limits<float>::max() || et_verify_on)
            {
                if ((uint32_t)conv_topk.size() < conv_width)
                { conv_topk.push({cur_expanded_dist, (uint32_t)frontier_nhood.first}); hop_topk_changed = true; }
                else if (cur_expanded_dist < conv_topk.top().first)
                { conv_topk.pop(); conv_topk.push({cur_expanded_dist, (uint32_t)frontier_nhood.first}); hop_topk_changed = true; }
            }
            uint32_t *node_nbrs = (node_buf + 1);
            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }

        // ── Predict-then-verify ET, layer-2 (exact) ──────────────────────────────
        // Fires only when layer-1 θ-ET (PQ) flagged stop THIS hop (theta_pred_stop).
        // Confirmation is strict exact-vs-exact: the closest node expanded this hop
        // (hop_min_exact) must still be α× beyond the k-th best EXACT distance
        // (conv_topk.top()). If so, divergence is real → stop. Otherwise θ-ET was a
        // PQ false-trigger and the search continues, preserving recall on hard queries.
        // Guards: kth_exact valid (conv_topk has ≥ k exact entries), hop produced an
        // exact node (hop_min_exact < FLT_MAX), grace period satisfied.
        if (et_verify_on && !et_exact_led && hops >= et_min_hops)
        {
            bool fired_this_hop = false;
            if (theta_pred_stop && (uint32_t)conv_topk.size() >= conv_width &&
                hop_min_exact < std::numeric_limits<float>::max())
            {
                const float kth_exact = conv_topk.top().first; // O(1): max-heap top = conv_width-th best exact
                if (kth_exact > 1e-9f && hop_min_exact > kth_exact * et_verify_alpha)
                    fired_this_hop = true;
            }

            if (fired_this_hop)
            {
                // Patience ≥ 1: require this many consecutive agreeing hops to stop.
                if (++verify_streak >= std::max(1u, et_verify_patience))
                {
                    hops++;
                    break;
                }
            }
            else
            {
                verify_streak = 0; // non-consecutive → reset
            }
        }

        // Top-k saturation check (Patience style): terminate if top-k IDs stable for et_sat_delta hops.
        //
        // Guard: only count a hop as "stable" when ALL top-k candidates are already expanded.
        // Without this guard, unexpanded cache-hit nodes (PQ lower-bound distances) flood
        // retset[0..k-1] early and their IDs appear stable before the search has actually
        // explored their neighbourhoods — causing premature termination in a local cached pocket.
        if (et_sat_delta > 0 && retset.size() >= k_search)
        {
            // Check that every one of the top-k candidates has been expanded.
            bool all_expanded = true;
            for (size_t i = 0; i < k_search; i++)
            {
                if (!retset[i].expanded) { all_expanded = false; break; }
            }

            if (all_expanded)
            {
                uint32_t overlap = 0;
                for (size_t i = 0; i < k_search; i++)
                {
                    uint32_t curr = retset[i].id;
                    for (size_t j = 0; j < k_search; j++)
                    {
                        if (curr == sat_prev_ids[j]) { overlap++; break; }
                    }
                }
                if ((float)overlap / (float)k_search >= et_sat_gamma)
                    sat_count++;
                else
                    sat_count = 0;
                for (size_t i = 0; i < k_search; i++)
                    sat_prev_ids[i] = retset[i].id;
                if (sat_count >= et_sat_delta)
                {
                    hops++;
                    break;
                }
            }
            else
            {
                // Some top-k are unexpanded; reset counter so we only count
                // consecutive stable hops where top-k is truly settled.
                sat_count = 0;
            }
        }

        // Exact Convergence ET: terminate if full_retset top-K (by exact distance) unchanged for
        // et_conv_delta consecutive hops. Unlike Saturation ET, this monitors the actual output
        // structure (full_retset) not the PQ-ordered traversal queue (retset).
        if (et_conv_delta > 0 && (uint32_t)conv_topk.size() >= conv_width)
        {
            if (hop_topk_changed)
                et_conv_streak = 0;
            else if (++et_conv_streak >= et_conv_delta)
            {
                hops++;
                break;
            }
        }

        // Oracle hop: first hop where full_retset's top-K by EXACT distance matches oracle_gt_ids.
        // full_retset holds all expanded nodes with their exact distances (SSD-read or BNC-resolved).
        // This is the true earliest-stop oracle: the search could return the same final result here.
        // O(N log N) per hop for the sort — analysis-only, never used in production.
        if (oracle_gt_ids != nullptr && stats != nullptr && full_retset.size() >= k_search &&
            (feat_log != nullptr || stats->oracle_hops == 0))
        {
            // Sort a copy to find top-K by exact distance.
            std::vector<Neighbor> sorted_all = full_retset;
            std::sort(sorted_all.begin(), sorted_all.end());
            // Leave-one-out: drop the query's own node (base-vector-as-query self-match).
            std::vector<Neighbor> sorted_frt;
            sorted_frt.reserve(sorted_all.size());
            for (auto &nb : sorted_all)
                if (nb.id != self_exclude_id) sorted_frt.push_back(nb);
          if (sorted_frt.size() >= k_search)
          {
            uint32_t found = 0;
            for (uint64_t g = 0; g < k_search; g++)
                for (uint64_t r = 0; r < k_search; r++)
                    if (sorted_frt[r].id == oracle_gt_ids[g]) { found++; break; }
            if (found == k_search && stats->oracle_hops == 0)
                stats->oracle_hops = hops + 1;

            // Per-hop raw feature dump for the learned-ET predictor (analysis only).
            // 14 raw values; engineered features (ratios/derivatives/windows) derived offline.
            if (feat_log != nullptr)
            {
                const uint32_t sz = (uint32_t)sorted_frt.size();
                const uint32_t m = std::min<uint32_t>(40, sz);
                const float dk_exact = sorted_frt[k_search - 1].distance;
                // threat scan over retset: unexpanded entries whose PQ dist could beat k-th exact
                uint32_t n_threats = 0, n_unexp = 0, n_expanded = 0;
                for (uint32_t r = 0; r < (uint32_t)retset.size(); r++)
                {
                    if (retset[r].expanded) { n_expanded++; }
                    else { n_unexp++; if (retset[r].distance < dk_exact) n_threats++; }
                }
                // top-k / top-40 id-set churn vs previous hop
                std::vector<uint32_t> cur_topk(k_search), cur_top40(m);
                for (uint32_t r = 0; r < (uint32_t)k_search; r++) cur_topk[r] = sorted_frt[r].id;
                for (uint32_t r = 0; r < m; r++) cur_top40[r] = sorted_frt[r].id;
                std::vector<uint32_t> sk = cur_topk, s40 = cur_top40;
                std::sort(sk.begin(), sk.end()); std::sort(s40.begin(), s40.end());
                float topk_changed = (sk != feat_prev_topk) ? 1.0f : 0.0f;
                float top40_changed = (s40 != feat_prev_top40) ? 1.0f : 0.0f;
                feat_prev_topk = sk; feat_prev_top40 = s40;

                feat_log->push_back((float)hops);                                   // 0 hop
                feat_log->push_back(sorted_frt[0].distance);                        // 1 d1 exact
                feat_log->push_back(dk_exact);                                      // 2 dk exact
                feat_log->push_back(sz > k_search ? sorted_frt[k_search].distance : dk_exact);        // 3 d(k+1)
                feat_log->push_back(sz >= 2*k_search ? sorted_frt[2*k_search-1].distance : dk_exact); // 4 d(2k)
                feat_log->push_back(sorted_frt[m - 1].distance);                    // 5 d40 exact
                feat_log->push_back(retset[k_search - 1].distance);                 // 6 dk pq
                feat_log->push_back(retset.peek_unexpanded_dist());                 // 7 best unexp pq
                feat_log->push_back((float)n_threats);                             // 8 #threats (unexp pq<dk_exact)
                feat_log->push_back((float)n_unexp);                              // 9 #unexpanded
                feat_log->push_back((float)n_expanded);                          // 10 #expanded
                feat_log->push_back(topk_changed);                              // 11 top-k id churn flag
                feat_log->push_back(top40_changed);                            // 12 top-40 id churn flag
                // group B: PQ-vs-exact residual stats over all expanded nodes so far
                const double rmean = pq_resid_n ? pq_resid_sum / pq_resid_n : 0.0;
                const double rvar = pq_resid_n ? std::max(0.0, pq_resid_sqsum / pq_resid_n - rmean * rmean) : 0.0;
                feat_log->push_back((float)rmean);                            // 13 PQ residual mean
                feat_log->push_back((float)std::sqrt(rvar));                  // 14 PQ residual std
                feat_log->push_back(pq_resid_max);                           // 15 PQ residual max
                feat_log->push_back((float)found);                          // 16 found (label src)
            }
          }
        }

        // Carry this hop's min exact to the next hop's exact-led gate (no reset).
        prev_hop_min_exact = hop_min_exact;
        // Update the global best (1st-place) exact distance — the exact rail's anchor.
        if (hop_min_exact < global_best_exact)
            global_best_exact = hop_min_exact;
        hops++;
    }

    if (stats != nullptr)
        stats->n_beam_hops = hops;

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data)
    {
        if (!(this->_reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does "
                               "not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        std::vector<AlignedRead> vec_read_reqs;

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER, full_retset.end());

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            // MULTISECTORFIX
            vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * defaults::SECTOR_LEN,
                                       defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }

        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(vec_read_reqs, ctx, true); // async reader windows.
#else
        reader->read(vec_read_reqs, ctx); // synchronous IO linux
#endif
        if (stats != nullptr)
        {
            stats->io_us += io_timer.elapsed();
        }

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            auto id = full_retset[i].id;
            // MULTISECTORFIX
            auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
            full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *)location, (uint32_t)this->_data_dim);
        }

        std::sort(full_retset.begin(), full_retset.end());
    }

    // Phase-3: re-rank top candidates with exact distances.
    // full_retset holds uniform PQ distances (set during traversal); this phase overwrites
    // them with exact distances. Nodes in _coord_cache are resolved in RAM; all others
    // require a disk read.
    //
    // Re-rank window is capped at l_search (not MAX_N_SECTOR_READS=512) to avoid
    // I/O bursts at high concurrency: with C=32, a 512-read re-rank burst per thread
    // causes thundering-herd P99 spikes.  l_search is the natural candidate-list size;
    // re-ranking l_search nodes preserves recall while reducing burst reads by ~5-10×.
    if (_bounded_cache.enabled())
    {
        const size_t rerank_n = std::min(full_retset.size(), l_search);
        if (full_retset.size() > rerank_n)
            full_retset.erase(full_retset.begin() + rerank_n, full_retset.end());

        // Pass 1: resolve _coord_cache hits in RAM (no disk IO).
        std::vector<size_t> disk_slots;
        disk_slots.reserve(rerank_n);
        for (size_t i = 0; i < rerank_n; ++i)
        {
            const uint32_t nid = full_retset[i].id;
            // SSD-read nodes already carry exact distances computed during the
            // search phase; no disk re-read needed.
            if (exact_dist_nodes.count(nid))
                continue;
            auto coord_it = _coord_cache.find(nid);
            if (coord_it != _coord_cache.end())
            {
                // Phase-1 BFS cache: exact dist from RAM.
                if (!_use_disk_index_pq)
                    full_retset[i].distance =
                        _dist_cmp->compare(aligned_query_T, coord_it->second, (uint32_t)_aligned_dim);
                else
                    full_retset[i].distance =
                        (metric == diskann::Metric::INNER_PRODUCT)
                            ? _disk_pq_table.inner_product(query_float, (uint8_t *)coord_it->second)
                            : _disk_pq_table.l2_distance(query_float, (uint8_t *)coord_it->second);
            }
            else if (_bounded_cache.has_coords() &&
                     _bounded_cache.get_coords(nid, data_buf))
            {
                // Phase-3 bounded cache: coords stored at insert time → exact dist from RAM.
                // Eliminates the thundering-herd disk burst at high concurrency.
                if (!_use_disk_index_pq)
                    full_retset[i].distance =
                        _dist_cmp->compare(aligned_query_T, (T *)data_buf, (uint32_t)_aligned_dim);
                else
                    full_retset[i].distance =
                        (metric == diskann::Metric::INNER_PRODUCT)
                            ? _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf)
                            : _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
            else
            {
                disk_slots.push_back(i);
            }
        }

        // Pass 2: batch disk read for nodes not resolvable from any RAM cache.
        // With coord storage enabled, this path is taken only for nodes that
        // were never SSD-read by this query and are not yet in _bounded_cache
        // (i.e., they were cached but evicted, or coord storage is disabled).
        // disk_slots[j] → full_retset index; sector_scratch slot j is reused for each.
        std::vector<AlignedRead> rerank_reqs;
        rerank_reqs.reserve(disk_slots.size());
        for (size_t j = 0; j < disk_slots.size(); ++j)
        {
            rerank_reqs.emplace_back(
                get_node_sector((size_t)full_retset[disk_slots[j]].id) * defaults::SECTOR_LEN,
                num_sectors_per_node * defaults::SECTOR_LEN,
                sector_scratch + j * num_sectors_per_node * defaults::SECTOR_LEN);
            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }
        if (!rerank_reqs.empty())
        {
            io_timer.reset();
            reader->read(rerank_reqs, ctx);
            if (stats != nullptr)
                stats->io_us += (float)io_timer.elapsed();

            for (size_t j = 0; j < disk_slots.size(); ++j)
            {
                size_t i = disk_slots[j];
                char *node_disk_buf = offset_to_node(
                    sector_scratch + j * (size_t)num_sectors_per_node * defaults::SECTOR_LEN,
                    full_retset[i].id);
                T *node_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_coords, _disk_bytes_per_point);
                if (!_use_disk_index_pq)
                    full_retset[i].distance =
                        _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                else
                    full_retset[i].distance =
                        (metric == diskann::Metric::INNER_PRODUCT)
                            ? _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf)
                            : _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
        }
        std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values. Leave-one-out: skip the query's own node (self-match)
    // so base-vector-as-query results match a real out-of-sample query.
    for (uint64_t i = 0, si = 0; i < k_search; i++, si++)
    {
        while (si < full_retset.size() && full_retset[si].id == self_exclude_id) si++;
        if (si >= full_retset.size()) break;
        indices[i] = full_retset[si].id;
        auto key = (uint32_t)indices[i];
        if (_dummy_pts.find(key) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[key];
        }

        if (distances != nullptr)
        {
            distances[i] = full_retset[si].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    distances[i] *= (_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
    }
}

// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                               const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                               std::vector<float> &distances, const uint64_t min_beam_width,
                                               QueryStats *stats)
{
    uint32_t res_count = 0;

    bool stop_flag = false;

    uint32_t l_search = (uint32_t)min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw,
                                 std::numeric_limits<float>::max(),    // et_theta: disabled
                                 std::numeric_limits<uint32_t>::max(), // hop_budget
                                 1.0f, 0,                              // sat_gamma, sat_delta
                                 false, stats);                        // use_reorder, stats
        for (uint32_t i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (uint32_t)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
