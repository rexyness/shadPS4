// SPDX-License-Identifier: GPL-2.0-or-later
// LBP3 Recompiled, milestone 5 measurement: where the command processor's recording time and the
// GPU's time go in a frame (the "first frame of a new scene" stalls, perf/intro_stalls.md).
//
// CPU side: cumulative counters (count, microseconds, bytes) for the buffer cache's uploads and
// allocations, the staging pool, the texture cache's image creation, refreshes (uploads),
// detiling and expansions, VMA allocations, resource binding and descriptor writes, and draws.
// They are counted only when LBP3_FRAME_SPLIT or LBP3_FIRST_FRAME is set (otherwise each site is
// one branch on a cached bool); the runtime snapshots them per frame (gpu/frame_split.cpp).
//
// GPU side (LBP3_FIRST_FRAME=1 only): timestamp pairs around the recorded buffer copies, image
// uploads and detile dispatches, read back and host-reset after each full GPU wait
// (CollectGpu, called by the runtime right after the frame's Finish), summed per kind.
// Measurement only: nothing here changes what is recorded except the timestamp writes.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace Lbp3Perf {

enum Kind : int {
    BufSync,      // BufferCache::SynchronizeMemory with data to upload: memcpy to staging + copy
    Staging,      // StagingBufferPool::Request (may wait for the GPU, may create a block)
    StagingNew,   //   ... of which created a new block or large buffer (count only)
    Resident,     // BufferCache::EnsureResident: device memory allocated for sparse arenas
    ArenaNew,     // BufferCache::GetArena created a new sparse arena (count, us)
    ImgNew,       // TextureCache::FindImage created and registered a new image
    ImgAlloc,     // UniqueImage::Create (vmaCreateImage): count, us, bytes
    BufAlloc,     // UniqueBuffer::Create (vmaCreateBuffer): count, us, bytes
    VmaMem,       // VMA's vkAllocateMemory calls (device memory callback): count, bytes
    ImgRefresh,   // TextureCache::RefreshImage that uploads: buffer sync + detile + copy recording
    Detile,       // TileManager::DetileImage of a tiled image (recording)
    Expand,       // TextureCache::ExpandImage
    ResBind,      // Rasterizer::BindResources (all buffer/texture lookups and syncs for a draw)
    DescBind,     // Pipeline::BindResources (descriptor writes / pushes)
    Draws,        // draws (count only)
    Dispatches,   // compute dispatches (count only)
    GpuCopy,      // GPU: buffer copies (Runtime::CopyBuffer), pairs and microseconds
    GpuImgUpload, // GPU: buffer-to-image uploads (Runtime::UploadImage)
    GpuDetile,    // GPU: detile dispatches
    GpuOverflow,  // timestamp pairs not written (pool full before a collect)
    ImgFree,      // UniqueImage destroy (vmaDestroyImage): count, us
    VmaFree,      // VMA's vkFreeMemory calls (device memory callback): count, bytes
    GpuImgCopy,   // GPU: image-to-image copies (Runtime::CopyImage, e.g. ExpandImage)
    Memcpy,       // guest memory -> staging copies (CopySparseMemory) in buffer and image uploads
    ZeroCopy,     //   ... 4 KB chunks of those copies that were all zero (LBP3_FIRST_FRAME only): bytes
    kKinds
};

struct Counter {
    std::atomic<uint64_t> n{0}, us{0}, bytes{0};
};

extern Counter g_counters[kKinds];
extern const bool g_on;      // LBP3_FRAME_SPLIT or LBP3_FIRST_FRAME
extern const bool g_gpuOn;   // LBP3_FIRST_FRAME

inline uint64_t NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void Add(Kind k, uint64_t us, uint64_t bytes = 0) {
    if (!g_on) return;
    g_counters[k].n.fetch_add(1, std::memory_order_relaxed);
    g_counters[k].us.fetch_add(us, std::memory_order_relaxed);
    g_counters[k].bytes.fetch_add(bytes, std::memory_order_relaxed);
}

// Times a scope into kind k (when on). bytes may be set before the scope ends; a scope with
// `counted` cleared at the end adds nothing (e.g. a sync that had nothing to upload).
struct Scope {
    Kind k;
    uint64_t t0;
    uint64_t bytes = 0;
    bool counted = true;
    explicit Scope(Kind kind) : k(kind), t0(g_on ? NowUs() : 0) {}
    ~Scope() {
        if (g_on && counted) Add(k, NowUs() - t0, bytes);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// GPU timestamps. Begin writes the first timestamp of a pair into cmdbuf and returns its query
// index (or -1: off, unsupported, or the pool is full); End writes the second.
void InitGpu(const Vulkan::Instance& instance);
int GpuBegin(vk::CommandBuffer cmdbuf);
void GpuEnd(vk::CommandBuffer cmdbuf, int query, Kind k);
// After a full GPU wait: reads the written pairs, adds them to their kinds, resets the pool.
void CollectGpu();

// VMA device memory callback (vk_instance.cpp): counts vkAllocateMemory calls made by VMA.
void NoteVmaAllocate(uint64_t bytes);
void NoteVmaFree(uint64_t bytes);

} // namespace Lbp3Perf
