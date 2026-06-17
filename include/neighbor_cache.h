// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// NeighborCache: Phase 1 full-preload of all neighbor IDs into DRAM.
// When enabled, every node's adjacency list is read once at index-load time and
// stored in a flat contiguous array.  During beam search the existing
// _nhood_cache lookup path is reused, so the traversal issues zero disk I/Os
// and only the final re-rank reads touch the disk.
//
// Usage:
//   NeighborCache nc;
//   nc.init(num_nodes, max_degree);
//   // ... fill via nc.insert() for each node ...
//   // then call nc.populate_nhood_cache() to fill the tsl::robin_map
//
// BoundedNeighborCache: Phase 3 on-demand LRU (CLOCK) cache.
// Bounded by a user-specified memory budget.  Nodes are inserted on disk-read
// and evicted via the CLOCK algorithm when capacity is full.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "tsl/robin_map.h"

namespace diskann
{

// One entry per node stored inside NeighborCache's flat buffer.
struct NeighborEntry
{
    uint32_t degree;      // actual neighbour count (<= max_degree)
    uint32_t *neighbors;  // pointer into the flat storage array
};

class NeighborCache
{
  public:
    NeighborCache() = default;

    // Allocate flat storage for num_nodes * max_degree uint32_t values.
    // Must be called before any insert().
    void init(uint32_t num_nodes, uint32_t max_degree)
    {
        num_nodes_  = num_nodes;
        max_degree_ = max_degree;
        enabled_    = true;

        // flat storage: node i's neighbors live at data_[i * max_degree_]
        data_.assign(static_cast<size_t>(num_nodes_) * max_degree_, 0u);
        degrees_.assign(num_nodes_, 0u);
        populated_.assign(num_nodes_, false);

        // Pre-build the NeighborEntry array so that lookup() is a single
        // array index with no pointer arithmetic at query time.
        entries_.resize(num_nodes_);
        for (uint32_t i = 0; i < num_nodes_; ++i)
        {
            entries_[i].degree    = 0;
            entries_[i].neighbors = data_.data() + static_cast<size_t>(i) * max_degree_;
        }
    }

    // Record the neighbours of node_id parsed from a sector buffer.
    // degree must be <= max_degree_.
    void insert(uint32_t node_id, const uint32_t *neighbors, uint32_t degree)
    {
        if (!enabled_ || node_id >= num_nodes_)
            return;
        uint32_t d = (degree <= max_degree_) ? degree : max_degree_;
        std::memcpy(entries_[node_id].neighbors, neighbors, d * sizeof(uint32_t));
        entries_[node_id].degree = d;
        degrees_[node_id]        = d;
        populated_[node_id]      = true;
    }

    // Look up node_id.  Returns pointer to the entry (degree + neighbor ptr)
    // or nullptr on miss (node not yet inserted or cache disabled).
    const NeighborEntry *lookup(uint32_t node_id) const
    {
        if (!enabled_ || node_id >= num_nodes_ || !populated_[node_id])
            return nullptr;
        return &entries_[node_id];
    }

    bool   enabled()    const { return enabled_; }
    size_t num_nodes()  const { return num_nodes_; }

    // Approximate DRAM consumption in bytes.
    size_t size_bytes() const
    {
        return data_.size() * sizeof(uint32_t)
             + degrees_.size() * sizeof(uint32_t)
             + populated_.size() * sizeof(bool)
             + entries_.size() * sizeof(NeighborEntry);
    }

  private:
    bool     enabled_    = false;
    uint32_t num_nodes_  = 0;
    uint32_t max_degree_ = 0;

    std::vector<uint32_t>      data_;       // size: num_nodes_ * max_degree_
    std::vector<uint32_t>      degrees_;    // per-node actual degree (redundant with entries_, kept for clarity)
    std::vector<bool>          populated_;  // true once insert() has been called for that node
    std::vector<NeighborEntry> entries_;    // fast lookup table; entries_[i].neighbors points into data_
};

// ---------------------------------------------------------------------------
// BoundedNeighborCache – Phase 3 on-demand, memory-bounded CLOCK cache.
//
// Capacity is expressed in bytes at init() time.  Internally the cache is
// sharded across NUM_SHARDS independent shards to reduce lock contention.
// Each shard owns a flat data array of size (capacity_per_shard × max_degree)
// uint32_t values; entries are addressed by slot index.
//
// CLOCK eviction is used instead of strict LRU to avoid the per-access cost
// of a doubly-linked list.  A ref bit per slot is set on lookup() and cleared
// (given a second chance) by find_victim(); the first slot whose ref bit is
// already false becomes the eviction target.
//
// Thread safety:
//   lookup() holds a shared_lock → multiple concurrent readers per shard.
//   insert() holds a unique_lock → exclusive write per shard.
//   find_victim() is called only from inside insert() under the unique_lock.
// ---------------------------------------------------------------------------
class BoundedNeighborCache
{
  public:
    struct Entry
    {
        uint32_t  degree;
        uint32_t *neighbors; // points into the shard's data array
        uint8_t  *coords;    // points into the shard's coord_buf; nullptr if coord storage disabled
    };

    BoundedNeighborCache() = default;

    // Allocate sharded storage.
    // capacity_bytes : total bytes available (split between neighbor IDs and optional coords).
    // max_degree     : graph R value (max neighbours per node).
    // coord_bytes    : bytes per vector (e.g. dim * sizeof(T)); 0 = disable coord storage.
    //
    // When coord_bytes > 0, each slot stores both neighbor IDs and raw coordinates.
    // The effective node capacity is reduced accordingly:
    //   capacity_per_shard = capacity_bytes / (max_degree*4 + coord_bytes) / NUM_SHARDS
    void init(size_t capacity_bytes, uint32_t max_degree, size_t coord_bytes = 0)
    {
        if (capacity_bytes == 0 || max_degree == 0)
            return;

        max_degree_  = max_degree;
        coord_bytes_ = coord_bytes;

        const size_t nbr_bytes_per_slot   = static_cast<size_t>(max_degree) * sizeof(uint32_t);
        const size_t total_bytes_per_slot = nbr_bytes_per_slot + coord_bytes_;
        const size_t total_slots          = capacity_bytes / total_bytes_per_slot;
        capacity_per_shard_               = static_cast<uint32_t>(
            std::max<size_t>(1, total_slots / NUM_SHARDS));

        for (auto &shard : shards_)
        {
            shard.capacity = capacity_per_shard_;
            shard.size     = 0;
            shard.data     = std::make_unique<uint32_t[]>(
                static_cast<size_t>(capacity_per_shard_) * max_degree_);
            shard.node_ids = std::make_unique<uint32_t[]>(capacity_per_shard_);
            shard.ref_bits = std::make_unique<std::atomic<bool>[]>(capacity_per_shard_);
            shard.entries  = std::make_unique<Entry[]>(capacity_per_shard_);

            if (coord_bytes_ > 0)
                shard.coord_buf = std::make_unique<uint8_t[]>(
                    static_cast<size_t>(capacity_per_shard_) * coord_bytes_);

            for (uint32_t s = 0; s < capacity_per_shard_; ++s)
            {
                shard.node_ids[s] = UINT32_MAX; // sentinel: empty
                shard.ref_bits[s].store(false, std::memory_order_relaxed);
                shard.entries[s].degree    = 0;
                shard.entries[s].neighbors =
                    shard.data.get() + static_cast<size_t>(s) * max_degree_;
                shard.entries[s].coords =
                    (coord_bytes_ > 0)
                        ? shard.coord_buf.get() + static_cast<size_t>(s) * coord_bytes_
                        : nullptr;
            }
            shard.clock_hand = 0;
            // Pre-size the hash index to avoid any rehash during insert().
            shard.index.reserve(capacity_per_shard_);
        }
    }

    // Return true if the cache was initialised with a non-zero capacity.
    bool enabled() const { return capacity_per_shard_ > 0; }

    // Check presence without setting ref_bit (used for I/O-aware ET check).
    bool contains(uint32_t node_id) const
    {
        if (capacity_per_shard_ == 0)
            return false;
        const uint32_t shard_idx = node_id % NUM_SHARDS;
        const Shard    &shard    = shards_[shard_idx];
        std::shared_lock<std::shared_mutex> lock(shard.mu);
        return shard.index.find(node_id) != shard.index.end();
    }

    // Total cache slots across all shards.
    size_t total_capacity_nodes() const
    {
        return static_cast<size_t>(capacity_per_shard_) * NUM_SHARDS;
    }

    // Lookup node_id.  Returns a pointer to the Entry on hit (and sets the
    // ref bit), nullptr on miss.  Caller must not retain the pointer across
    // any subsequent insert() call on the same shard.
    const Entry *lookup(uint32_t node_id)
    {
        if (capacity_per_shard_ == 0)
            return nullptr;

        const uint32_t shard_idx = node_id % NUM_SHARDS;
        Shard         &shard     = shards_[shard_idx];

        std::shared_lock<std::shared_mutex> lock(shard.mu);
        auto it = shard.index.find(node_id);
        if (it == shard.index.end())
            return nullptr;

        const uint32_t slot = it->second;
        shard.ref_bits[slot].store(true, std::memory_order_relaxed);
        return &shard.entries[slot];
    }

    // Insert node_id with its neighbor list (and optionally raw coordinates).
    // No-op if already present.  Evicts via CLOCK when capacity full.
    //
    // coords : pointer to coord_bytes_ bytes of raw vector data; ignored when
    //          coord storage is disabled (coord_bytes_ == 0) or coords == nullptr.
    //
    // Uses try_to_lock: if contended, skips the insert rather than blocking
    // lookup() callers.  The cache fills from non-contended inserts / warmup.
    void insert(uint32_t node_id, const uint32_t *neighbors, uint32_t degree,
                const void *coords = nullptr)
    {
        if (capacity_per_shard_ == 0)
            return;

        const uint32_t shard_idx = node_id % NUM_SHARDS;
        Shard         &shard     = shards_[shard_idx];

        std::unique_lock<std::shared_mutex> lock(shard.mu, std::try_to_lock);
        if (!lock.owns_lock())
            return; // skip insert when contended; no blocking on the hot path

        // Double-check: another thread may have inserted while we waited.
        if (shard.index.find(node_id) != shard.index.end())
            return;

        uint32_t slot;
        if (shard.size < shard.capacity)
        {
            slot = shard.size++;
        }
        else
        {
            slot = find_victim(shard);
            shard.index.erase(shard.node_ids[slot]);
        }

        // Write neighbor data.
        const uint32_t copy_n = (degree <= max_degree_) ? degree : max_degree_;
        std::memcpy(shard.entries[slot].neighbors, neighbors, copy_n * sizeof(uint32_t));
        shard.entries[slot].degree = copy_n;

        // Write coordinate data if storage is enabled and caller provided coords.
        if (coord_bytes_ > 0 && coords != nullptr)
            std::memcpy(shard.entries[slot].coords, coords, coord_bytes_);

        shard.node_ids[slot] = node_id;
        shard.ref_bits[slot].store(false, std::memory_order_relaxed);
        shard.index[node_id] = slot;
    }

    // Copy stored coordinates for node_id into out_buf (which must be at least
    // coord_bytes_ bytes).  Returns true on success, false on miss or when
    // coord storage is disabled.  Thread-safe: coords are copied under
    // shared_lock so eviction cannot race with the caller's read.
    bool get_coords(uint32_t node_id, void *out_buf) const
    {
        if (capacity_per_shard_ == 0 || coord_bytes_ == 0)
            return false;

        const uint32_t shard_idx = node_id % NUM_SHARDS;
        const Shard   &shard     = shards_[shard_idx];

        std::shared_lock<std::shared_mutex> lock(shard.mu);
        auto it = shard.index.find(node_id);
        if (it == shard.index.end())
            return false;

        const uint32_t slot = it->second;
        shard.ref_bits[slot].store(true, std::memory_order_relaxed);
        std::memcpy(out_buf, shard.entries[slot].coords, coord_bytes_);
        return true;
    }

    bool has_coords() const { return coord_bytes_ > 0; }

  private:
    static constexpr uint32_t NUM_SHARDS = 256;

    struct Shard
    {
        std::unique_ptr<uint32_t[]>              data;       // [capacity × max_degree]
        std::unique_ptr<uint8_t[]>               coord_buf;  // [capacity × coord_bytes_]; null if disabled
        std::unique_ptr<uint32_t[]>              node_ids;   // slot → node_id
        std::unique_ptr<std::atomic<bool>[]>     ref_bits;   // CLOCK ref bit per slot
        std::unique_ptr<Entry[]>                 entries;    // pre-built entry table
        tsl::robin_map<uint32_t, uint32_t>       index;      // node_id → slot
        mutable std::shared_mutex                mu;
        uint32_t                                 size{0};
        uint32_t                                 capacity{0};
        uint32_t                                 clock_hand{0}; // only touched under unique_lock
    };

    // Find a victim slot using the CLOCK algorithm.
    // Must be called with shard.mu held exclusively (clock_hand is not atomic).
    static uint32_t find_victim(Shard &shard)
    {
        while (true)
        {
            const uint32_t hand = shard.clock_hand++ % shard.capacity;
            // exchange(false): if was true → give second chance, keep scanning;
            //                  if was false → evict this slot.
            const bool was_ref =
                shard.ref_bits[hand].exchange(false, std::memory_order_relaxed);
            if (!was_ref)
                return hand;
        }
    }

    std::array<Shard, NUM_SHARDS> shards_;
    uint32_t max_degree_{0};
    uint32_t capacity_per_shard_{0};
    size_t   coord_bytes_{0};
};

} // namespace diskann
