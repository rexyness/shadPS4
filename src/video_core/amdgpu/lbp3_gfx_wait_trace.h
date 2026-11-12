// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// LBP3 Recompiled (milestone 5 measurement, lbp3-pc perf/gfx_wait.md). Included by
// liverpool.cpp only. Measurement only: nothing here changes what the command processor does.
//
// LBP3_GFX_WAIT_TRACE=1 writes analysis\gfx_wait_trace.csv, one row per event. t is the steady
// clock in microseconds (frame_split.csv's submit_ms is the same clock in ms); flip is the number
// of flip packets the command processor has passed. The first column says what the row is:
//
//   F  t flip                                   a flip packet (the file is flushed here)
//   GB t flip kind addr value ref mask func buf off_end size top
//      c_state c_addr c_value c_ref c_func c_sat c_pending known fault_n_cum fault_us_cum
//                                               the graphics queue blocks on a wait.
//                                               kind wrm/sem; buf: top-level buffer index in
//                                               the frame; off_end: dwords from the packet to
//                                               its buffer's end; top: 1 top-level, 0 indirect.
//                                               c_*: the compute queue at that moment (c_state
//                                               1 parked on a wait; c_sat 1 if that wait's
//                                               condition already holds; c_pending: compute
//                                               submits queued, the parked one included).
//                                               known: the address is a learned wait address;
//                                               *_cum: guest-thread write-tracking faults and
//                                               handler time so far (diff two rows for a span)
//   GE t flip addr value dur polls max_gap gres cres cres_n cadv cpkts cdisp pc sat first
//      writer writer_t c_state c_addr fault_n fault_us fault_cp_n prot_cp_n prot_cp_us
//      prot_n prot_us vq_us vp_us vq_cp_us vp_cp_us
//                                               the graphics wait ended. polls: tests after
//                                               the first; max_gap: longest time between two
//                                               tests (us); during the wait, time this thread
//                                               spent in graphics resumes (gres), compute
//                                               resumes (cres, cres_n of them; cadv: those that
//                                               ran at least one packet), compute packets and
//                                               dispatches parsed, ProcessCommands (pc);
//                                               sat: when the watcher thread first saw the
//                                               condition hold (-1: never), first: its first
//                                               look (sat == first: already true then, so sat
//                                               is an upper bound); writer: the last command-
//                                               stream write to the address during the wait,
//                                               or cpu if none (a guest CPU thread wrote it);
//                                               write-tracking page faults taken by guest
//                                               threads during the wait and the time inside
//                                               the page manager's handler (fault_n,
//                                               fault_us), faults on this thread
//                                               (fault_cp_n), and the page manager's Protect
//                                               calls on this thread (prot_cp_n, prot_cp_us)
//                                               and on other threads (prot_n, prot_us), and the
//                                               VirtualQuery / VirtualProtect system calls
//                                               inside them (runtime AddressSpace::Protect;
//                                               vq/vp other threads, vq_cp/vp_cp this thread)
//   CB t flip kind addr value ref mask func known g_blocked     the compute queue blocks
//   CE t flip addr value dur polls max_gap gres pc sat first writer writer_t
//   W  t flip src addr data buf off_end         a command-stream label write (buf/off_end: the
//                                               graphics packet's place, -1 for compute)
//   I  t flip buf off_end fault_n_cum fault_us_cum   a graphics EOP raised its interrupt
//   CI t flip pipe                              a compute RELEASE_MEM raised its interrupt
//   CS t flip tid dwords                        a compute submit arrived (guest thread id)
//   GS t flip tid dwords                        a graphics buffer was handed over
//   L / X t flip addr gfx                       EndBlockedWait learned / forgot an address
//
// A watcher thread spins on the address of each blocked wait (one graphics, one compute) to
// time when the guest satisfied it, independently of when this thread next tested it. It only
// spins while a watched wait is unsatisfied (one core, at most for the length of the wait).

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <thread>

#include <immintrin.h>

#include "common/types.h"

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
#endif

namespace AmdGpu::Lbp3Trace {

inline bool On() {
    static const bool on = [] {
        const char* v = std::getenv("LBP3_GFX_WAIT_TRACE");
        return v && std::string_view{v} != "0";
    }();
    return on;
}

inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline long long Us(s64 ns) {
    return static_cast<long long>(ns / 1000);
}
inline double DUs(s64 ns) {
    return static_cast<double>(ns) / 1000.0;
}

inline u32 ThreadId() {
#ifdef _WIN32
    return static_cast<u32>(GetCurrentThreadId());
#else
    return 0;
#endif
}

// ---- the file ----
inline std::mutex file_mutex;
inline FILE* file = nullptr;
inline bool file_tried = false;

inline void Row(const char* fmt, ...) {
    std::scoped_lock lk{file_mutex};
    if (!file) {
        if (file_tried) return;
        file_tried = true;
        file = std::fopen("analysis\\gfx_wait_trace.csv", "w");
        if (!file) return;
        std::setvbuf(file, nullptr, _IOFBF, 1 << 20);
        std::fputs("# LBP3_GFX_WAIT_TRACE rows: see shadPS4 src/video_core/amdgpu/"
                   "lbp3_gfx_wait_trace.h\n",
                   file);
        std::fprintf(stderr, "[gfxwait] trace on: analysis\\gfx_wait_trace.csv\n");
    }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(file, fmt, ap);
    va_end(ap);
}

inline void Flush() {
    std::scoped_lock lk{file_mutex};
    if (file) std::fflush(file);
}

inline std::atomic<u32> flips{0};

// ---- command-processor thread only ----
// Cumulative times and counts, read as differences across a wait.
inline s64 acc_gres_ns = 0;   // inside graphics task resumes (main loop)
inline s64 acc_cres_ns = 0;   // inside compute task resumes
inline u64 acc_cres_n = 0;
inline s64 acc_cadv_ns = 0;   // compute resumes that parsed at least one packet
inline s64 acc_pc_ns = 0;     // the main loop's ProcessCommands
inline u64 compute_packets = 0;
inline u64 compute_dispatches = 0;

// Write tracking (page_manager.cpp): faults and Protect calls, split by whether they ran on
// the command-processor thread.
inline std::atomic<u64> fault_n{0}, fault_ns{0}, fault_cp_n{0}, fault_cp_ns{0};
inline std::atomic<u64> protect_n{0}, protect_ns{0}, protect_cp_n{0}, protect_cp_ns{0};
inline std::atomic<u64> vq_ns{0}, vp_ns{0}, vq_cp_ns{0}, vp_cp_ns{0};
inline std::atomic<u32> cp_tid{0};  // the command-processor thread (set when it starts)

// Where the graphics packet that writes a label sits (set around those packets).
inline int ctx_buf = -1;
inline int ctx_off_end = -1;

// The trace's own record of command-stream label writes (the runtime's 64-entry history drives
// the learn/forget rule and stays untouched).
struct TraceWrite {
    const char* source;
    u64 address;
    u64 data;
    s64 ns;
};
inline std::array<TraceWrite, 4096> writes{};
inline u32 write_seq = 0;

inline void NoteWrite(const char* source, u64 address, u64 data) {
    const s64 now = NowNs();
    writes[write_seq++ % writes.size()] = {source, address, data, now};
    const bool gfx = source[0] == 'g';
    Row("W,%lld,%u,%s,%#llx,%#llx,%d,%d\n", Us(now), flips.load(), source,
        static_cast<unsigned long long>(address), static_cast<unsigned long long>(data),
        gfx ? ctx_buf : -1, gfx ? ctx_off_end : -1);
}

// The latest write overlapping the 8 bytes at address since write number `since`.
inline const TraceWrite* LatestWrite(u64 address, u32 since) {
    const u32 now = write_seq;
    const u32 n = now - since > writes.size() ? static_cast<u32>(writes.size()) : now - since;
    for (u32 i = 0; i < n; ++i) {
        const TraceWrite& w = writes[(now - 1 - i) % writes.size()];
        if (w.address + 7 >= address && w.address <= address + 7) return &w;
    }
    return nullptr;
}

// ---- the watcher ----
struct WatchSlot {
    std::atomic<u32> gen{0};
    std::atomic<u64> addr{0};  // 0: not watching
    std::atomic<u32> kind{0};  // 0 WAIT_REG_MEM, 1 MEM_SEMAPHORE
    std::atomic<u32> ref{0}, mask{0}, func{0};
    std::atomic<u32> first_gen{0}, sat_gen{0};
    std::atomic<s64> first_ns{0}, sat_ns{0};
};
inline std::array<WatchSlot, 2> slots{};  // 0 graphics, 1 compute
inline std::mutex watch_mutex;
inline std::condition_variable watch_cv;
inline std::once_flag watch_started;

inline bool Satisfied(u32 kind, u64 addr, u32 ref, u32 mask, u32 func) {
    if (kind == 1) return *reinterpret_cast<const volatile u64*>(addr) > 0;
    const u32 v = *reinterpret_cast<const volatile u32*>(addr) & mask;
    switch (func) {
    case 0: return true;
    case 1: return v < ref;
    case 2: return v <= ref;
    case 3: return v == ref;
    case 4: return v != ref;
    case 5: return v >= ref;
    case 6: return v > ref;
    default: return false;
    }
}

inline bool AnyUnsatisfied() {
    for (auto& s : slots) {
        if (s.addr.load(std::memory_order_acquire) &&
            s.sat_gen.load(std::memory_order_relaxed) != s.gen.load(std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

inline void WatcherMain() {
    for (;;) {
        {
            std::unique_lock lk{watch_mutex};
            watch_cv.wait(lk, [] { return AnyUnsatisfied(); });
        }
        for (bool pending = true; pending;) {
            pending = false;
            for (auto& s : slots) {
                const u32 g = s.gen.load(std::memory_order_acquire);
                const u64 a = s.addr.load(std::memory_order_acquire);
                if (!a || s.sat_gen.load(std::memory_order_relaxed) == g) continue;
                const u32 kind = s.kind, ref = s.ref, mask = s.mask, func = s.func;
                if (s.gen.load(std::memory_order_acquire) != g) {
                    pending = true;  // re-armed while reading: look again
                    continue;
                }
                const bool sat = Satisfied(kind, a, ref, mask, func);
                const s64 now = NowNs();
                if (s.first_gen.load(std::memory_order_relaxed) != g) {
                    s.first_ns.store(now, std::memory_order_relaxed);
                    s.first_gen.store(g, std::memory_order_release);
                }
                if (sat) {
                    s.sat_ns.store(now, std::memory_order_relaxed);
                    s.sat_gen.store(g, std::memory_order_release);
                } else {
                    pending = true;
                }
            }
            if (pending) _mm_pause();
        }
    }
}

// Arms slot i; returns the generation to read back with Disarm.
inline u32 Arm(int i, u32 kind, u64 addr, u32 ref, u32 mask, u32 func) {
    std::call_once(watch_started, [] { std::thread{WatcherMain}.detach(); });
    auto& s = slots[i];
    s.addr.store(0, std::memory_order_release);
    s.kind = kind;
    s.ref = ref;
    s.mask = mask;
    s.func = func;
    const u32 g = s.gen.fetch_add(1, std::memory_order_acq_rel) + 1;
    s.addr.store(addr, std::memory_order_release);
    {
        std::scoped_lock lk{watch_mutex};
    }
    watch_cv.notify_one();
    return g;
}

struct WatchResult {
    s64 sat_ns = -1;
    s64 first_ns = -1;
};
inline WatchResult Disarm(int i, u32 g) {
    auto& s = slots[i];
    s.addr.store(0, std::memory_order_release);
    WatchResult r;
    if (s.first_gen.load(std::memory_order_acquire) == g) r.first_ns = s.first_ns.load();
    if (s.sat_gen.load(std::memory_order_acquire) == g) r.sat_ns = s.sat_ns.load();
    return r;
}

// ---- one blocked wait (lives in the coroutine frame across yields) ----
struct Wait {
    bool on = false;
    int slot = 0;
    u64 addr = 0;
    s64 begin_ns = 0, last_ns = 0, max_gap_ns = 0;
    u32 polls = 0;
    u32 write_since = 0;
    u32 gen = 0;
    s64 gres0 = 0, cres0 = 0, cadv0 = 0, pc0 = 0;
    u64 cres_n0 = 0, cpk0 = 0, cdisp0 = 0;
    u64 fault_n0 = 0, fault_ns0 = 0, fault_cp_n0 = 0, prot_cp_n0 = 0, prot_cp_ns0 = 0;
    u64 prot_n0 = 0, prot_ns0 = 0, vq0 = 0, vp0 = 0, vq_cp0 = 0, vp_cp0 = 0;

    void Begin(int slot_, u32 kind, u64 addr_, u32 ref, u32 mask, u32 func) {
        on = true;
        slot = slot_;
        addr = addr_;
        begin_ns = last_ns = NowNs();
        write_since = write_seq;
        gres0 = acc_gres_ns;
        cres0 = acc_cres_ns;
        cadv0 = acc_cadv_ns;
        pc0 = acc_pc_ns;
        cres_n0 = acc_cres_n;
        cpk0 = compute_packets;
        cdisp0 = compute_dispatches;
        fault_n0 = fault_n.load();
        fault_ns0 = fault_ns.load();
        fault_cp_n0 = fault_cp_n.load();
        prot_cp_n0 = protect_cp_n.load();
        prot_cp_ns0 = protect_cp_ns.load();
        prot_n0 = protect_n.load();
        prot_ns0 = protect_ns.load();
        vq0 = vq_ns.load();
        vp0 = vp_ns.load();
        vq_cp0 = vq_cp_ns.load();
        vp_cp0 = vp_cp_ns.load();
        if (addr) gen = Arm(slot, kind, addr, ref, mask, func);
    }
    void Poll() {
        const s64 now = NowNs();
        if (now - last_ns > max_gap_ns) max_gap_ns = now - last_ns;
        last_ns = now;
        ++polls;
    }
};

// The compute queue's current blocked wait (command-processor thread).
struct ComputeState {
    bool parked = false;
    u32 kind = 0;
    u64 addr = 0;
    u32 ref = 0, mask = 0, func = 0;
};
inline ComputeState compute{};

inline u64 ReadValue(u32 kind, u64 addr) {
    if (!addr) return 0;
    return kind == 1 ? *reinterpret_cast<const volatile u64*>(addr)
                     : *reinterpret_cast<const volatile u32*>(addr);
}

} // namespace AmdGpu::Lbp3Trace
