// SPDX-License-Identifier: GPL-2.0-or-later
// LBP3 Recompiled, milestone 5 measurement (see lbp3_perf.h).
#include "video_core/lbp3_perf.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "video_core/renderer_vulkan/vk_instance.h"

namespace Lbp3Perf {

Counter g_counters[kKinds];
const bool g_gpuOn = std::getenv("LBP3_FIRST_FRAME") != nullptr;
const bool g_on = g_gpuOn || std::getenv("LBP3_FRAME_SPLIT") != nullptr;

namespace {

constexpr uint32_t kQueries = 8192;  // 4096 pairs between two collects
std::mutex g_mutex;
vk::Device g_device{};
vk::QueryPool g_pool{};
double g_periodNs = 1.0;
uint32_t g_next = 0;  // next free query; everything below it was written in recorded commands
std::vector<Kind> g_pairKind(kQueries / 2, GpuCopy);

} // namespace

void NoteVmaAllocate(uint64_t bytes) {
    if (!g_on) return;
    g_counters[VmaMem].n.fetch_add(1, std::memory_order_relaxed);
    g_counters[VmaMem].bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteVmaFree(uint64_t bytes) {
    if (!g_on) return;
    g_counters[VmaFree].n.fetch_add(1, std::memory_order_relaxed);
    g_counters[VmaFree].bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void InitGpu(const Vulkan::Instance& instance) {
    if (!g_gpuOn) return;
    std::lock_guard lk(g_mutex);
    if (g_pool) return;
    if (!instance.IsHostQueryResetSupported()) {
        std::fprintf(stderr, "[first-frame] the device has no host query reset: GPU timestamps off\n");
        return;
    }
    g_device = instance.GetDevice();
    g_periodNs = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    const vk::QueryPoolCreateInfo ci{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = kQueries,
    };
    const auto [result, pool] = g_device.createQueryPool(ci);
    if (result != vk::Result::eSuccess) {
        std::fprintf(stderr, "[first-frame] couldn't create the timestamp pool: GPU timestamps off\n");
        return;
    }
    g_pool = pool;
    g_device.resetQueryPool(g_pool, 0, kQueries);
    std::fprintf(stderr, "[first-frame] GPU timestamps on (%u queries, %.3f ns per tick)\n", kQueries,
                 g_periodNs);
}

int GpuBegin(vk::CommandBuffer cmdbuf) {
    if (!g_gpuOn) return -1;
    std::lock_guard lk(g_mutex);
    if (!g_pool) return -1;
    if (g_next + 2 > kQueries) {
        g_counters[GpuOverflow].n.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    const int q = static_cast<int>(g_next);
    g_next += 2;
    cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_pool, static_cast<uint32_t>(q));
    return q;
}

void GpuEnd(vk::CommandBuffer cmdbuf, int query, Kind k) {
    if (query < 0) return;
    std::lock_guard lk(g_mutex);
    cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_pool, static_cast<uint32_t>(query + 1));
    g_pairKind[static_cast<size_t>(query) / 2] = k;
}

void CollectGpu() {
    if (!g_gpuOn) return;
    std::lock_guard lk(g_mutex);
    if (!g_pool || g_next == 0) return;
    std::vector<uint64_t> ticks(g_next, 0);
    // Called right after a full GPU wait, so every written query is available.
    const vk::Result r = g_device.getQueryPoolResults(
        g_pool, 0, g_next, ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (r == vk::Result::eSuccess) {
        for (uint32_t q = 0; q + 1 < g_next; q += 2) {
            if (!ticks[q] || ticks[q + 1] < ticks[q]) continue;
            const double us = double(ticks[q + 1] - ticks[q]) * g_periodNs / 1000.0;
            Counter& c = g_counters[g_pairKind[q / 2]];
            c.n.fetch_add(1, std::memory_order_relaxed);
            c.us.fetch_add(static_cast<uint64_t>(us + 0.5), std::memory_order_relaxed);
        }
    }
    g_device.resetQueryPool(g_pool, 0, g_next);
    g_next = 0;
}

} // namespace Lbp3Perf
