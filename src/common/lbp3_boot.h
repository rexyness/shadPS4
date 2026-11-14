// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// LBP3 Recompiled: how far the boot's pipeline preload has got, for the runtime's loading
// screen (runtime/src/video/boot_screen.cpp). Written by the PipelineCache constructor and
// WarmUp on the thread that creates the rasterizer, read by the presenter thread. Counts only;
// nothing waits on them.

#include <atomic>

#include "common/types.h"

namespace Lbp3Boot {

enum Phase : u32 {
    kNotStarted = 0, // the rasterizer isn't being made yet
    kDriverCache = 1, // reading the driver's pipeline cache file
    kReading = 2,     // reading the cached pipelines (done/total = keys)
    kBuilding = 3,    // building them on the worker threads (done/total = pipelines)
    kDone = 4,        // the preload is over (or there was no cache)
};

inline std::atomic<u32> g_phase{kNotStarted};
inline std::atomic<u32> g_done{0};
inline std::atomic<u32> g_total{0};

inline void Enter(Phase phase, u32 total = 0) {
    g_done.store(0, std::memory_order_relaxed);
    g_total.store(total, std::memory_order_relaxed);
    g_phase.store(phase, std::memory_order_release);
}

inline void Step() {
    g_done.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Lbp3Boot
