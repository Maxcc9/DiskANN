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
    };

    BoundedNeighborCache() = default;

    // Allocate sharded storage.
    // capacity_bytes : total bytes available for neighbor ID data.
    // max_degree     : graph R value (max neighbours per node).
    void init(size_t capacity_bytes, uint32_t max_degree)
    {
        if (capacity_bytes == 0 || max_degree == 0)
            return;

        max_degree_           = max_degree;
        // Each slot stores max_degree uint32_t values (neighbor IDs).
        const size_t bytes_per_slot = static_cast<size_t>(max_degree) * sizeof(uint32_t);
        const size_t total_slots    = capacity_bytes / bytes_per_slot;
        capacity_per_shard_         = static_cast<uint32_t>(
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
            for (uint32_t s = 0; s < capacity_per_shard_; ++s)
            {
                shard.node_ids[s]        = UINT32_MAX; // sentinel: empty
                shard.ref_bits[s].store(false, std::memory_order_relaxed);
                shard.entries[s].degree    = 0;
                shard.entries[s].neighbors =
                    shard.data.get() + static_cast<size_t>(s) * max_degree_;
            }
            shard.clock_hand.store(0, std::memory_order_relaxed);
        }
    }

    // Return true if the cache was initialised with a non-zero capacity.
    bool enabled() const { return capacity_per_shard_ > 0; }

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

    // Insert node_id with its neighbor list.  No-op if already present
    // (another thread beat us to it).  Evicts via CLOCK when capacity full.
    void insert(uint32_t node_id, const uint32_t *neighbors, uint32_t degree)
    {
        if (capacity_per_shard_ == 0)
            return;

        const uint32_t shard_idx = node_id % NUM_SHARDS;
        Shard         &shard     = shards_[shard_idx];

        std::unique_lock<std::shared_mutex> lock(shard.mu);

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
            // Remove the evicted node_id from the hash index.
            shard.index.erase(shard.node_ids[slot]);
        }

        // Write neighbor data.
        const uint32_t copy_n =
            (degree <= max_degree_) ? degree : max_degree_;
        std::memcpy(shard.entries[slot].neighbors,
                    neighbors,
                    copy_n * sizeof(uint32_t));
        shard.entries[slot].degree = copy_n;
        shard.node_ids[slot]       = node_id;
        // Newly inserted entries start with ref bit = false so they are
        // evictable on the very next CLOCK sweep if nothing accesses them.
        shard.ref_bits[slot].store(false, std::memory_order_relaxed);

        shard.index[node_id] = slot;
    }

  private:
    static constexpr uint32_t NUM_SHARDS = 256;

    struct Shard
    {
        std::unique_ptr<uint32_t[]>              data;       // [capacity × max_degree]
        std::unique_ptr<uint32_t[]>              node_ids;   // slot → node_id
        std::unique_ptr<std::atomic<bool>[]>     ref_bits;   // CLOCK ref bit per slot
        std::unique_ptr<Entry[]>                 entries;    // pre-built entry table
        tsl::robin_map<uint32_t, uint32_t>       index;      // node_id → slot
        mutable std::shared_mutex                mu;
        uint32_t                                 size{0};
        uint32_t                                 capacity{0};
        std::atomic<uint32_t>                    clock_hand{0};
    };

    // Find a victim slot using the CLOCK algorithm.
    // Must be called with shard.mu held exclusively.
    static uint32_t find_victim(Shard &shard)
    {
        while (true)
        {
            const uint32_t hand =
                shard.clock_hand.fetch_add(1, std::memory_order_relaxed)
                % shard.capacity;
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
};

} // namespace diskann
