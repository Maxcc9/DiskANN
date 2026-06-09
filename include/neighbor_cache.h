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

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

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

} // namespace diskann
