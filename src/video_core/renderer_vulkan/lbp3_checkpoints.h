// SPDX-FileCopyrightText: Copyright 2025 LBP3 Recompiled
// SPDX-License-Identifier: GPL-2.0-or-later
//
// LBP3 Recompiled: GPU checkpoints to name the work that was running when the driver reset
// the GPU (Windows TDR). LBP3_GPU_CHECKPOINTS=1 enables VK_NV_device_diagnostic_checkpoints
// and puts a marker before every draw, dispatch and display readback. When the device is
// lost, OnDeviceLost logs which markers last reached the top and the bottom of the pipe,
// with the work recorded around them, then ends the process instead of waiting forever.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan::Lbp3Checkpoints {

enum class Kind : u32 {
    Draw, DrawIndexed, DrawIndirect, Dispatch, DispatchIndirect, Readback, SubmitEnd,
    CopyBuffer, FillBuffer, Resolve, DepthStencilCopy
};

inline const char* KindName(Kind k) {
    switch (k) {
    case Kind::Draw: return "draw";
    case Kind::DrawIndexed: return "draw-indexed";
    case Kind::DrawIndirect: return "draw-indirect";
    case Kind::Dispatch: return "dispatch";
    case Kind::DispatchIndirect: return "dispatch-indirect";
    case Kind::Readback: return "display-readback";
    case Kind::SubmitEnd: return "submit-end";
    case Kind::CopyBuffer: return "copy-buffer";
    case Kind::FillBuffer: return "fill-buffer";
    case Kind::Resolve: return "resolve";
    case Kind::DepthStencilCopy: return "depth-stencil-copy";
    }
    return "?";
}

struct Record {
    u64 seq;
    u64 tick;    // the scheduler tick (submission) the marker was recorded into
    u64 h0, h1;  // vs/ps shader hashes, the cs hash, or the readback's guest address
    u32 a, b, c; // counts: indices/instances, group dims, readback size
    Kind kind;
};

inline constexpr size_t kRing = size_t{1} << 16;
inline std::array<Record, kRing> g_ring{};
inline std::atomic<u64> g_seq{0};
inline bool g_on = false;  // set when the device is created with the extension

// LBP3_SKIP_SHADERS=<hex>,<hex>...: drop every draw or dispatch that uses one of these
// shader program hashes (a stopgap test: does the hang go away without that work?).
inline bool ShouldSkip(u64 h0, u64 h1 = 0) {
    static const std::vector<u64> skip = [] {
        std::vector<u64> v;
        if (const char* s = std::getenv("LBP3_SKIP_SHADERS")) {
            for (const char* p = s; *p;) {
                char* end = nullptr;
                const u64 h = std::strtoull(p, &end, 16);
                if (end == p) break;
                v.push_back(h);
                p = *end == ',' ? end + 1 : end;
            }
            std::fprintf(stderr, "[gpu] skipping work with %zu shader hash(es)\n", v.size());
        }
        return v;
    }();
    if (skip.empty()) return false;
    for (const u64 h : skip) {
        if ((h0 && h == h0) || (h1 && h == h1)) {
            static std::atomic<u32> n{0};
            if (n++ < 20) std::fprintf(stderr, "[gpu] skipped work with shader %016llx\n", (unsigned long long)h);
            return true;
        }
    }
    return false;
}

inline bool Requested() {
    static const bool requested = std::getenv("LBP3_GPU_CHECKPOINTS") != nullptr;
    return requested;
}

inline void Mark(vk::CommandBuffer cmd, Kind kind, u64 tick, u64 h0 = 0, u64 h1 = 0, u32 a = 0,
                 u32 b = 0, u32 c = 0) {
    if (!g_on) {
        return;
    }
    const u64 seq = g_seq.fetch_add(1, std::memory_order_relaxed) + 1;  // 0 would be a null marker
    g_ring[seq % kRing] = Record{seq, tick, h0, h1, a, b, c, kind};
    cmd.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(seq)));
}

inline void PrintRecord(u64 seq, const char* tag) {
    const Record& r = g_ring[seq % kRing];
    if (r.seq != seq) {
        std::fprintf(stderr, "[gpu]   %s #%llu (overwritten)\n", tag, (unsigned long long)seq);
        return;
    }
    std::fprintf(stderr, "[gpu]   %s #%llu tick %llu %s h0=%016llx h1=%016llx a=%u b=%u c=%u\n", tag,
                 (unsigned long long)seq, (unsigned long long)r.tick, KindName(r.kind),
                 (unsigned long long)r.h0, (unsigned long long)r.h1, r.a, r.b, r.c);
}

// Logs the markers that last reached the top and the bottom of the pipe, with the work
// recorded around them. The data is only defined once the device is lost.
inline void PrintCheckpoints(vk::Queue queue) {
    if (!g_on) {
        std::fprintf(stderr, "[gpu] checkpoints off: run with LBP3_GPU_CHECKPOINTS=1 to name the work\n");
        return;
    }
    u32 count = 0;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(static_cast<VkQueue>(queue), &count, nullptr);
    std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
    if (count) {
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(static_cast<VkQueue>(queue), &count, data.data());
    }
    u64 top = 0, bottom = 0;
    for (u32 i = 0; i < count; ++i) {
        const u64 seq = static_cast<u64>(reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker));
        std::fprintf(stderr, "[gpu] checkpoint: stage %s marker #%llu\n",
                     vk::to_string(vk::PipelineStageFlagBits(data[i].stage)).c_str(),
                     (unsigned long long)seq);
        if (data[i].stage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT && seq > top) top = seq;
        if (data[i].stage == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT && seq > bottom) bottom = seq;
    }
    // Each marker sits just before its work. Work before the last marker that reached the
    // bottom of the pipe has finished; work from there up to the last marker that reached
    // the top of the pipe was in flight when the GPU hung.
    const u64 from = bottom > 4 ? bottom - 4 : 1;
    const u64 to = std::max(top, bottom) + 4;
    std::fprintf(stderr, "[gpu] in flight: from #%llu (bottom of pipe) to #%llu (top of pipe)\n",
                 (unsigned long long)bottom, (unsigned long long)top);
    for (u64 s = from; s <= to && s <= g_seq.load() && s - from < 400; ++s) {
        PrintRecord(s, s < bottom ? "done     " : s <= top ? "IN FLIGHT" : "not begun");
    }
    // The whole submission the hung work belongs to, one line per distinct piece of work.
    const u64 last = g_seq.load();
    if (top && g_ring[top % kRing].seq == top) {
        const u64 tick = g_ring[top % kRing].tick;
        struct Group { Kind kind; u64 h0, h1; u32 count; u64 first, lastSeq; };
        std::vector<Group> groups;
        for (u64 s = last > kRing ? last - kRing + 1 : 1; s <= last; ++s) {
            const Record& r = g_ring[s % kRing];
            if (r.seq != s || r.tick != tick) continue;
            auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& g) {
                return g.kind == r.kind && g.h0 == r.h0 && g.h1 == r.h1;
            });
            if (it == groups.end()) groups.push_back({r.kind, r.h0, r.h1, 1, s, s});
            else { ++it->count; it->lastSeq = s; }
        }
        std::fprintf(stderr, "[gpu] submission tick %llu: %zu distinct pieces of work\n", (unsigned long long)tick,
                     groups.size());
        for (const auto& g : groups) {
            std::fprintf(stderr, "[gpu]   %-17s h0=%016llx h1=%016llx x%u  #%llu..#%llu%s\n", KindName(g.kind),
                         (unsigned long long)g.h0, (unsigned long long)g.h1, g.count,
                         (unsigned long long)g.first, (unsigned long long)g.lastSeq,
                         g.first <= top && top <= g.lastSeq ? "  <- top of pipe" : "");
        }
    }
    std::fflush(stderr);
}

// Called once the device is lost. Never returns.
[[noreturn]] inline void OnDeviceLost(vk::Queue queue, const char* where, u64 tick) {
    static std::atomic<bool> once{false};
    if (once.exchange(true)) {
        for (;;) {  // another thread is reporting; it ends the process
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    std::fprintf(stderr,
                 "[gpu] Vulkan device lost in %s (tick %llu): the driver reset the GPU "
                 "(Windows TDR). Last marker recorded: #%llu\n",
                 where, (unsigned long long)tick, (unsigned long long)g_seq.load());
    PrintCheckpoints(queue);
    std::fflush(stderr);
    std::_Exit(3);
}

// A wait that has gone on for seconds without a lost device: report once and keep waiting.
inline void OnLongWait(vk::Queue queue, const char* where, u64 tick, int seconds, int result) {
    static std::atomic<bool> once{false};
    if (once.exchange(true)) {
        return;
    }
    std::fprintf(stderr,
                 "[gpu] %s has waited %d s for tick %llu (last result %d); last marker recorded: #%llu\n",
                 where, seconds, (unsigned long long)tick, result, (unsigned long long)g_seq.load());
    PrintCheckpoints(queue);
}

} // namespace Vulkan::Lbp3Checkpoints
