// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

namespace Lbp3Trace {
struct Wait;  // lbp3_gfx_wait_trace.h
}

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    u32 index_buffer_num_indices{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0; });
    }

    // Like WaitGpuIdle, but also returns while the graphics queue waits on memory the guest
    // may only write after its submit returns (a synchronous submit would deadlock there).
    // A short wait (e.g. on async compute) is waited out, so the guest's frames stay in step
    // with the GPU; only a queue blocked for a grace period releases the guest early. A wait
    // address that once needed that release (the guest writes it after the submit) releases
    // the guest at once from then on, so such frames don't pay the grace period every time.
    // LBP3 Recompiled (milestone 5 measurement): how a WaitGpuIdleOrBlocked or
    // WaitGfxIdleOrBlocked call returned, so the runtime can count releases per call site.
    enum WaitResult : int { WaitResultIdle = 0, WaitReleasedKnown = 1, WaitReleasedGrace = 2 };

    int WaitGpuIdleOrBlocked() noexcept {
        using namespace std::chrono_literals;
        std::unique_lock lk{submit_mutex};
        std::chrono::steady_clock::time_point blocked_since{};
        while (num_submits != 0) {
            if (queues_blocked_on_wait > 0) {
                const auto now = std::chrono::steady_clock::now();
                const u64 addr = last_blocked_addr;
                const bool known = addr && std::find(guest_written_waits.begin(), guest_written_waits.end(),
                                                     addr) != guest_written_waits.end();
                if (known) {
                    ++early_releases;
                    return WaitReleasedKnown;
                }
                if (blocked_since == std::chrono::steady_clock::time_point{}) {
                    blocked_since = now;
                } else if (now - blocked_since >= 20ms) {
                    ++grace_releases;  // EndBlockedWait learns the address if the guest wrote it
                    ++early_releases;
                    return WaitReleasedGrace;
                }
            } else {
                blocked_since = {};
            }
            submit_cv.wait_for(lk, 1ms);
        }
        return WaitResultIdle;
    }

    // LBP3 Recompiled: the same release rules as WaitGpuIdleOrBlocked, but for the graphics
    // queue alone. A graphics submit returns once the graphics queue has run what it was
    // given or blocks on a wait; a blocked compute queue no longer lets it return before the
    // graphics queue has even started (the game then raced ahead and reset the flag the new
    // command buffer's first WAIT_REG_MEM checks: ticket 10's save deadlock).
    int WaitGfxIdleOrBlocked() noexcept {
        using namespace std::chrono_literals;
        std::unique_lock lk{submit_mutex};
        std::chrono::steady_clock::time_point blocked_since{};
        while (num_gfx_submits != 0) {
            if (gfx_blocked_on_wait > 0) {
                const auto now = std::chrono::steady_clock::now();
                const u64 addr = gfx_last_blocked_addr;
                const bool known = addr && std::find(guest_written_waits.begin(), guest_written_waits.end(),
                                                     addr) != guest_written_waits.end();
                if (known) {
                    ++early_releases;
                    return WaitReleasedKnown;
                }
                if (blocked_since == std::chrono::steady_clock::time_point{}) {
                    blocked_since = now;
                } else if (now - blocked_since >= 20ms) {
                    ++grace_releases;
                    ++early_releases;
                    return WaitReleasedGrace;
                }
            } else {
                blocked_since = {};
            }
            submit_cv.wait_for(lk, 1ms);
        }
        return WaitResultIdle;
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{64};

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid;
    }
#endif

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    // top_level: false for an indirect buffer (LBP3 Recompiled: only a top-level buffer's own
    // patched flip NOP is a flip; see the PatchedFlip case).
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb, bool top_level = true);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    void ProcessCommands();
    void Process(std::stop_token stoken);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::atomic<u32> queues_blocked_on_wait{};
    std::atomic<u32> num_gfx_submits{};        // graphics submissions not finished yet
    std::atomic<u32> gfx_blocked_on_wait{};    // the graphics queue is blocked on a wait
    std::atomic<u64> gfx_last_blocked_addr{};
    std::atomic<u64> last_blocked_addr{};      // memory address of the latest blocked wait (0: register)
    std::vector<u64> guest_written_waits;      // wait addresses that needed an early release

    std::atomic<u32> grace_releases{};          // guest releases after the grace period

    // A queue starts waiting on memory (addr) or a register (0) that isn't ready yet.
    // Returns the state EndBlockedWait needs.
    struct BlockedWait {
        u64 addr;
        u32 label_writes;    // label write count at the start
        u32 grace_releases;  // grace_releases at the start
        bool gfx;            // on the graphics queue
        std::chrono::steady_clock::time_point start;  // LBP3 measurement: when it began
    };
    BlockedWait BeginBlockedWait(u64 addr, bool gfx = false);
    // The wait ended. If the guest had to be released to end it and no command stream
    // wrote the address meanwhile, the guest's CPU wrote it: remember the address. If a
    // command stream wrote it, forget it (a GPU-side label must be waited out).
    void EndBlockedWait(const BlockedWait& wait);

    // LBP3_GFX_WAIT_TRACE (lbp3_gfx_wait_trace.h, measurement only): the GB/GE and CB/CE rows.
    Lbp3Trace::Wait TraceGfxWaitBegin(u32 kind, u64 addr, u32 ref, u32 mask, u32 func, int buf,
                                      int off_end, size_t size, bool top);
    void TraceGfxWaitEnd(const Lbp3Trace::Wait& wait, u32 kind);
    Lbp3Trace::Wait TraceComputeWaitBegin(u32 kind, u64 addr, u32 ref, u32 mask, u32 func);
    void TraceComputeWaitEnd(const Lbp3Trace::Wait& wait, u32 kind);

public:
    std::atomic<u32> early_releases{};  // guest released from a submit while the GPU was blocked
    // LBP3 Recompiled (milestone 5 measurement): time the graphics queue spent blocked on
    // WAIT_REG_MEM / MEM_SEMAPHORE waits (from BeginBlockedWait to EndBlockedWait), the number
    // of such waits, and the time the current graphics wait began (0 = none), for the runtime's
    // per-frame split.
    std::atomic<u64> lbp3_gfx_blocked_ns{};
    std::atomic<u32> lbp3_gfx_blocked_waits{};
    std::atomic<s64> lbp3_gfx_blocked_since_ns{};
    std::atomic<u32> learned_waits{};   // wait addresses added to guest_written_waits
    // LBP3 Recompiled (perf/early_flip.md): top-level graphics buffers started since the last
    // accepted flip, for naming the buffer a stale flip marker was found in (command-processor
    // thread only).
    u32 lbp3_gfx_buffers_since_flip = 0;

private:
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
#ifdef __linux__
    u32 gpu_tid;
#endif
    s32 curr_qid{-1};
};

} // namespace AmdGpu
