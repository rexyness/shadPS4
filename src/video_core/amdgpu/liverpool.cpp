// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string_view>
#include <tuple>

#include <boost/preprocessor/stringize.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/lbp3_gfx_wait_trace.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

// Recent label writes from command streams, printed when a WAIT_REG_MEM gets stuck.
struct LabelWrite {
    const char* source;
    u64 address;
    u64 data;
};
static std::array<LabelWrite, 64> recent_label_writes{};
static std::atomic<u32> recent_label_write_count{};

static void NoteLabelWrite(const char* source, u64 address, u64 data) {
    recent_label_writes[recent_label_write_count++ % recent_label_writes.size()] = {source,
                                                                                    address, data};
    if (Lbp3Trace::On()) Lbp3Trace::NoteWrite(source, address, data);
}

// LBP3_GFX_WAIT_TRACE: where the current graphics packet sits, for the trace's W and I rows.
#define LBP3_TRACE_GFX_CTX()                                                                       \
    if (Lbp3Trace::On()) {                                                                         \
        Lbp3Trace::ctx_buf = static_cast<int>(lbp3_buffer_index);                                  \
        Lbp3Trace::ctx_off_end =                                                                   \
            static_cast<int>(lbp3_dcb_end - reinterpret_cast<const u32*>(header));                 \
    }

// Whether a command stream wrote the 8 bytes at address since write number `since`.
static bool LabelWrittenSince(u64 address, u32 since) {
    const u32 now = recent_label_write_count;
    if (now - since > recent_label_writes.size()) return true;  // history overrun: assume so
    for (u32 i = since; i != now; ++i) {
        const LabelWrite& w = recent_label_writes[i % recent_label_writes.size()];
        if (w.address >= address - 7 && w.address <= address + 7) return true;
    }
    return false;
}

Liverpool::BlockedWait Liverpool::BeginBlockedWait(u64 addr, bool gfx) {
    std::scoped_lock lk{submit_mutex};
    last_blocked_addr = addr;
    ++queues_blocked_on_wait;
    const auto start = std::chrono::steady_clock::now();
    if (gfx) {
        gfx_last_blocked_addr = addr;
        ++gfx_blocked_on_wait;
        lbp3_gfx_blocked_since_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    }
    submit_cv.notify_all();
    return {addr, recent_label_write_count.load(), grace_releases.load(), gfx, start};
}

void Liverpool::EndBlockedWait(const BlockedWait& wait) {
    int lbp3_trace_event = 0;  // LBP3_GFX_WAIT_TRACE: 1 learned, -1 forgotten
    {
        std::scoped_lock lk{submit_mutex};
        --queues_blocked_on_wait;
        if (wait.gfx) {
            --gfx_blocked_on_wait;
            lbp3_gfx_blocked_ns += static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait.start).count());
            ++lbp3_gfx_blocked_waits;
            lbp3_gfx_blocked_since_ns = 0;
        }
        if (!wait.addr) return;
        const auto known = std::find(guest_written_waits.begin(), guest_written_waits.end(), wait.addr);
        if (LabelWrittenSince(wait.addr, wait.label_writes)) {
            if (known != guest_written_waits.end()) {
                guest_written_waits.erase(known);
                lbp3_trace_event = -1;
            }
        } else if (grace_releases != wait.grace_releases && known == guest_written_waits.end() &&
                   guest_written_waits.size() < 256) {
            guest_written_waits.push_back(wait.addr);
            ++learned_waits;
            lbp3_trace_event = 1;
        }
    }
    if (lbp3_trace_event && Lbp3Trace::On()) {
        Lbp3Trace::Row("%s,%lld,%u,%#llx,%d\n", lbp3_trace_event > 0 ? "L" : "X",
                       Lbp3Trace::Us(Lbp3Trace::NowNs()), Lbp3Trace::flips.load(),
                       static_cast<unsigned long long>(wait.addr), wait.gfx ? 1 : 0);
    }
}

// LBP3_GFX_WAIT_TRACE (lbp3_gfx_wait_trace.h): the rows for blocked waits. Command-processor
// thread only; guest_written_waits is changed only on this thread, so it is read here unlocked.
Lbp3Trace::Wait Liverpool::TraceGfxWaitBegin(u32 kind, u64 addr, u32 ref, u32 mask, u32 func,
                                             int buf, int off_end, size_t size, bool top) {
    using namespace Lbp3Trace;
    Wait w{};
    w.Begin(0, kind, addr, ref, mask, func);
    const auto& c = compute;
    const u64 c_value = c.parked ? ReadValue(c.kind, c.addr) : 0;
    const bool c_sat = c.parked && c.addr && Satisfied(c.kind, c.addr, c.ref, c.mask, c.func);
    size_t c_pending = 0;
    for (u32 q = 1; q < num_mapped_queues; ++q) {
        std::scoped_lock lock{mapped_queues[q].m_access};
        c_pending += mapped_queues[q].submits.size();
    }
    const bool known = addr && std::find(guest_written_waits.begin(), guest_written_waits.end(),
                                         addr) != guest_written_waits.end();
    Row("GB,%lld,%u,%s,%#llx,%#llx,%#x,%#x,%u,%d,%d,%zu,%d,%d,%#llx,%#llx,%#x,%u,%d,%zu,%d,%llu,%.1f\n",
        Us(w.begin_ns), flips.load(), kind ? "sem" : "wrm", static_cast<unsigned long long>(addr),
        static_cast<unsigned long long>(ReadValue(kind, addr)), ref, mask, func, buf, off_end, size,
        top ? 1 : 0, c.parked ? 1 : 0, static_cast<unsigned long long>(c.addr),
        static_cast<unsigned long long>(c_value), c.ref, c.func, c_sat ? 1 : 0, c_pending,
        known ? 1 : 0, static_cast<unsigned long long>(fault_n.load()),
        DUs(static_cast<s64>(fault_ns.load())));
    return w;
}

void Liverpool::TraceGfxWaitEnd(const Lbp3Trace::Wait& w, u32 kind) {
    using namespace Lbp3Trace;
    const s64 end = NowNs();
    WatchResult r{};
    if (w.addr) r = Disarm(0, w.gen);
    const Lbp3Trace::TraceWrite* lw = w.addr ? LatestWrite(w.addr, w.write_since) : nullptr;
    Row("GE,%lld,%u,%#llx,%#llx,%.1f,%u,%.1f,%.1f,%.1f,%llu,%.1f,%llu,%llu,%.1f,%lld,%lld,%s,%lld,"
        "%d,%#llx,%llu,%.1f,%llu,%llu,%.1f,%llu,%.1f,%.1f,%.1f,%.1f,%.1f\n",
        Us(end), flips.load(), static_cast<unsigned long long>(w.addr),
        static_cast<unsigned long long>(ReadValue(kind, w.addr)), DUs(end - w.begin_ns), w.polls,
        DUs(w.max_gap_ns), DUs(acc_gres_ns - w.gres0), DUs(acc_cres_ns - w.cres0),
        static_cast<unsigned long long>(acc_cres_n - w.cres_n0), DUs(acc_cadv_ns - w.cadv0),
        static_cast<unsigned long long>(compute_packets - w.cpk0),
        static_cast<unsigned long long>(compute_dispatches - w.cdisp0), DUs(acc_pc_ns - w.pc0),
        r.sat_ns >= 0 ? Us(r.sat_ns) : -1LL, r.first_ns >= 0 ? Us(r.first_ns) : -1LL,
        lw ? lw->source : "cpu", lw ? Us(lw->ns) : -1LL, compute.parked ? 1 : 0,
        static_cast<unsigned long long>(compute.addr),
        static_cast<unsigned long long>(fault_n.load() - w.fault_n0),
        DUs(static_cast<s64>(fault_ns.load() - w.fault_ns0)),
        static_cast<unsigned long long>(fault_cp_n.load() - w.fault_cp_n0),
        static_cast<unsigned long long>(protect_cp_n.load() - w.prot_cp_n0),
        DUs(static_cast<s64>(protect_cp_ns.load() - w.prot_cp_ns0)),
        static_cast<unsigned long long>(protect_n.load() - w.prot_n0),
        DUs(static_cast<s64>(protect_ns.load() - w.prot_ns0)),
        DUs(static_cast<s64>(vq_ns.load() - w.vq0)), DUs(static_cast<s64>(vp_ns.load() - w.vp0)),
        DUs(static_cast<s64>(vq_cp_ns.load() - w.vq_cp0)),
        DUs(static_cast<s64>(vp_cp_ns.load() - w.vp_cp0)));
}

Lbp3Trace::Wait Liverpool::TraceComputeWaitBegin(u32 kind, u64 addr, u32 ref, u32 mask,
                                                 u32 func) {
    using namespace Lbp3Trace;
    Wait w{};
    w.Begin(1, kind, addr, ref, mask, func);
    compute = {true, kind, addr, ref, mask, func};
    const bool known = addr && std::find(guest_written_waits.begin(), guest_written_waits.end(),
                                         addr) != guest_written_waits.end();
    Row("CB,%lld,%u,%s,%#llx,%#llx,%#x,%#x,%u,%d,%u\n", Us(w.begin_ns), flips.load(),
        kind ? "sem" : "wrm", static_cast<unsigned long long>(addr),
        static_cast<unsigned long long>(ReadValue(kind, addr)), ref, mask, func, known ? 1 : 0,
        gfx_blocked_on_wait.load());
    return w;
}

void Liverpool::TraceComputeWaitEnd(const Lbp3Trace::Wait& w, u32 kind) {
    using namespace Lbp3Trace;
    const s64 end = NowNs();
    WatchResult r{};
    if (w.addr) r = Disarm(1, w.gen);
    compute.parked = false;
    const Lbp3Trace::TraceWrite* lw = w.addr ? LatestWrite(w.addr, w.write_since) : nullptr;
    Row("CE,%lld,%u,%#llx,%#llx,%.1f,%u,%.1f,%.1f,%.1f,%lld,%lld,%s,%lld\n", Us(end), flips.load(),
        static_cast<unsigned long long>(w.addr),
        static_cast<unsigned long long>(ReadValue(kind, w.addr)), DUs(end - w.begin_ns), w.polls,
        DUs(w.max_gap_ns), DUs(acc_gres_ns - w.gres0), DUs(acc_pc_ns - w.pc0),
        r.sat_ns >= 0 ? Us(r.sat_ns) : -1LL, r.first_ns >= 0 ? Us(r.first_ns) : -1LL,
        lw ? lw->source : "cpu", lw ? Us(lw->ns) : -1LL);
}

// DMA_DATA with an immediate value fills the GPU copy of memory; also write guest memory,
// since WAIT_REG_MEM and the PM4 parser read guest memory on the CPU (LBP3 sets
// cross-queue labels this way: a compute wait on a DMA-written label never ended).
static void MirrorDmaFill(const char* source, VAddr dst, u32 num_bytes, u32 value) {
    NoteLabelWrite(source, dst, value);
    auto* memory = Core::Memory::Instance();
    std::array<u32, 64> chunk;
    chunk.fill(value);
    for (u32 done = 0; done < num_bytes;) {
        const u32 n = std::min<u32>(num_bytes - done, sizeof(chunk));
        memory->TryWriteBacking(reinterpret_cast<void*>(dst + done), chunk.data(), n);
        done += n;
    }
}

static void ReportStuckWait(const char* queue, const PM4CmdWaitRegMem* wait, u32 value) {
    const bool memory = wait->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
    const u64 address = memory ? reinterpret_cast<uintptr_t>(wait->Address()) : wait->Reg();
    LOG_WARNING(Render_Vulkan,
                "WaitRegMem stuck over 1 s on {}: {} {:#x} value={:#x} func={} ref={:#x} mask={:#x}",
                queue, memory ? "memory" : "register", address, value,
                static_cast<u32>(wait->function.Value()), wait->ref, wait->mask);
    const u32 total = recent_label_write_count;
    for (u32 i = total > recent_label_writes.size() ? total - recent_label_writes.size() : 0;
         i < total; ++i) {
        const auto& w = recent_label_writes[i % recent_label_writes.size()];
        if (w.address + 0x1000 > address && w.address < address + 0x1000) {
            LOG_WARNING(Render_Vulkan, "  nearby label write #{}: {} {:#x} = {:#x}", i, w.source,
                        w.address, w.data);
        }
    }
    LOG_WARNING(Render_Vulkan, "  {} label writes so far", total);
}

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}, at {:#x} (header {:08x})",
            offset, span.size(), reinterpret_cast<uintptr_t>(span.data()),
            span.empty() ? 0u : span[0]);
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
    Lbp3Trace::cp_tid = Lbp3Trace::ThreadId();
#ifdef __linux__
    gpu_tid = gettid();
#endif

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_submits || num_commands) {
            if (Lbp3Trace::On()) {
                const s64 t0 = Lbp3Trace::NowNs();
                ProcessCommands();
                Lbp3Trace::acc_pc_ns += Lbp3Trace::NowNs() - t0;
            } else {
                ProcessCommands();
            }

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            if (Lbp3Trace::On()) {
                const u64 packets0 = Lbp3Trace::compute_packets;
                const s64 t0 = Lbp3Trace::NowNs();
                task.resume();
                const s64 dt = Lbp3Trace::NowNs() - t0;
                if (curr_qid == GfxQueueId) {
                    Lbp3Trace::acc_gres_ns += dt;
                } else {
                    Lbp3Trace::acc_cres_ns += dt;
                    ++Lbp3Trace::acc_cres_n;
                    if (Lbp3Trace::compute_packets != packets0) Lbp3Trace::acc_cadv_ns += dt;
                }
            } else {
                task.resume();
            }

            if (task.done()) {
                task.destroy();

                const bool gfx_done = curr_qid == GfxQueueId;
                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();

                    --num_submits;
                    if (gfx_done) --num_gfx_submits;
                    std::scoped_lock lock2{submit_mutex};
                    submit_cv.notify_all();
                }
                // LBP3 (milestone 5): hand the work recorded so far to the GPU as each graphics
                // command buffer finishes, so the GPU runs a frame's first buffers while this
                // thread records the rest, instead of starting only at the frame's readback.
                // A flush submits without waiting; the game is still released only when the
                // graphics queue is idle (ticket 10). LBP3_GPU_FLUSH=off restores the old
                // behaviour; LBP3_GPU_FLUSH=<n> flushes after every n-th graphics buffer.
                static const int flush_every = [] {
                    const char* v = std::getenv("LBP3_GPU_FLUSH");
                    if (!v) return 1;
                    if (std::string_view{v} == "off") return 0;
                    const int n = std::atoi(v);
                    return n > 0 ? n : 1;
                }();
                static int gfx_buffers = 0;
                if (gfx_done && flush_every && rasterizer && ++gfx_buffers % flush_every == 0) {
                    rasterizer->Flush();
                }
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const u32 packet_dwords = header->type3.NumWords() + 1;  // before the packet runs
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            task.handle.destroy();
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, packet_dwords);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                           bool top_level) {
    FIBER_ENTER(dcb_task_name);

    // LBP3 Recompiled (perf/early_flip.md): where this buffer starts and ends, so the PatchedFlip
    // case can tell the runtime's marker (always the NOP 59 dwords before a top-level buffer's
    // end) from a stale copy of it left in reused command memory.
    const u32* const lbp3_dcb_begin = dcb.data();
    const u32* const lbp3_dcb_end = dcb.data() + dcb.size();
    const u32 lbp3_buffer_index =
        top_level ? lbp3_gfx_buffers_since_flip++ : lbp3_gfx_buffers_since_flip - 1;

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    const PM4Header* previous_header = nullptr;  // for diagnosing misparses
    const PM4Header* current_header = nullptr;
    std::array<const PM4Header*, 8> recent_headers{};
    u32 recent_count = 0;
    bool dumped_context = false;
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        previous_header = current_header;
        current_header = header;
        recent_headers[recent_count++ % recent_headers.size()] = header;
        if (type == 0 && header->type0.base.Value() != 0 && !dumped_context) {
            // The first real misparse in this buffer: show how the parser got here.
            dumped_context = true;
            const auto at = reinterpret_cast<uintptr_t>(header);
            std::string headers;
            for (u32 i = recent_count > recent_headers.size() ? recent_count - recent_headers.size()
                                                              : 0;
                 i < recent_count; ++i) {
                const auto* h = recent_headers[i % recent_headers.size()];
                headers += fmt::format(" [{}: {:08x}]",
                                       (reinterpret_cast<uintptr_t>(h) - base_addr) / 4, h->raw);
            }
            const size_t offset = (at - base_addr) / 4;
            const auto* words = reinterpret_cast<const u32*>(at);
            std::string before;
            for (size_t i = std::min<size_t>(offset, 32); i > 0; --i) {
                before += fmt::format(" {:08x}", words[-static_cast<ptrdiff_t>(i)]);
            }
            LOG_WARNING(Render_Vulkan,
                        "PM4 misparse context at dword {} of a buffer at {:#x}: last headers "
                        "(dword: header):{}; 32 words before:{}",
                        offset, base_addr, headers, before);
        }

        switch (type) {
        case 0: {
            // Type-0: write NumWords consecutive registers from base. Guests use zero dwords
            // (register 0, one word) as filler, which does nothing on hardware.
            const u32 base = header->type0.base.Value();
            static std::atomic<int> reported{};
            if (base != 0 || reported++ < 6) {
                const auto at = reinterpret_cast<uintptr_t>(dcb.data());
                const size_t offset = (at - base_addr) / 4;
                std::string previous;
                if (previous_header) {
                    const auto* p = reinterpret_cast<const u32*>(previous_header);
                    const size_t n = std::min<size_t>((at - reinterpret_cast<uintptr_t>(p)) / 4, 12);
                    for (size_t i = 0; i < n; ++i) {
                        previous += fmt::format(" {:08x}", p[i]);
                    }
                }
                std::string next;
                for (size_t i = 0; i < std::min<size_t>(dcb.size(), 6); ++i) {
                    next += fmt::format(" {:08x}", dcb[i]);
                }
                LOG_WARNING(Render_Vulkan,
                            "PM4 type 0 write of {} registers at {:#x}, dword {} of a buffer at "
                            "{:#x}; previous packet:{}; here:{}",
                            header->type0.NumWords(), base, offset, base_addr, previous, next);
            }
            dcb = NextPacket(dcb, header->type0.NumWords() + 1);
            continue;
        }
        default: {
            // Garbage where a packet should be: the guest changed the buffer under us or
            // submitted it before it was complete. Drop the rest instead of aborting.
            const auto at = reinterpret_cast<uintptr_t>(dcb.data());
            std::string words;
            for (size_t i = 0; i < std::min<size_t>(dcb.size(), 8); ++i) {
                words += fmt::format(" {:08x}", dcb[i]);
            }
            LOG_ERROR(Render_Vulkan,
                      "Bad PM4 packet (type {}) at {:#x}, dword {} of a buffer at {:#x}, {} dwords "
                      "left:{}; skipping the rest",
                      type, at, (at - base_addr) / 4, base_addr, dcb.size(), words);
            dcb = {};
            continue;
        }
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    //
                    // LBP3 Recompiled (perf/early_flip.md): the runtime patches the flip
                    // request into the NOP 59 dwords before the end of the frame's last
                    // top-level buffer (runtime videoout.cpp PatchFlipRequest; upstream's
                    // gnmdriver.cpp writes it at the same place) and never removes it. The
                    // game later reuses that memory for other command buffers, and a NOP that
                    // lands on the old header with its first body dword left unwritten carries
                    // the old marker: at the Pod's first frame that flipped after buffer 1 of 8.
                    // Only the marker at that exact position of a top-level buffer is a flip;
                    // keep this rule and PatchFlipRequest's layout in step.
                    const u32* const nop_at = reinterpret_cast<const u32*>(header);
                    if (top_level && nop_at + 59 == lbp3_dcb_end) {
                        lbp3_gfx_buffers_since_flip = 0;
                        if (Lbp3Trace::On()) {
                            const u32 flip = ++Lbp3Trace::flips;
                            Lbp3Trace::Row("F,%lld,%u\n", Lbp3Trace::Us(Lbp3Trace::NowNs()), flip);
                            Lbp3Trace::Flush();
                        }
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    } else {
                        // Once per position (dword offset, buffer size, NOP size), at most 32
                        // lines: the parsed buffer is the runtime's snapshot copy, so the same
                        // stale marker shows up at a different address each time.
                        static std::mutex logged_mutex;
                        static std::set<std::tuple<size_t, size_t, u32, bool>> logged;
                        const size_t offset = static_cast<size_t>(nop_at - lbp3_dcb_begin);
                        const size_t size = static_cast<size_t>(lbp3_dcb_end - lbp3_dcb_begin);
                        std::scoped_lock lk{logged_mutex};
                        if (logged.size() < 32 &&
                            logged.emplace(offset, size, count, top_level).second) {
                            std::fprintf(stderr,
                                         "[gnm] ignored a stale flip marker at 0x%llx: dword %zu of "
                                         "%s buffer %u (size %zu dwords), NOP of %u body dwords\n",
                                         static_cast<unsigned long long>(
                                             reinterpret_cast<uintptr_t>(nop_at)),
                                         offset, top_level ? "top-level" : "an indirect",
                                         lbp3_buffer_index, size, count);
                        }
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(true); });
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexOffset2", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->Draw(true, draw_index_off->index_offset); });
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(false); });
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                regs.max_index_size = index_buffer_num_indices;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_index_indirect->base_vtx_loc,
                                                 draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                regs.max_index_size = index_buffer_num_indices;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count, 0, draw_index_indirect->base_vtx_loc,
                            draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                regs.max_index_size = index_buffer_num_indices;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count,
                            draw_index_indirect->count_indirect_enable.Value()
                                ? draw_index_indirect->count_addr
                                : 0,
                            draw_index_indirect->base_vtx_loc, draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->DispatchDirect(); });
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->DispatchIndirect(indirect_args_addr, offset, size); });
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                index_buffer_num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                LBP3_TRACE_GFX_CTX();
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    NoteLabelWrite("gfx EOS", reinterpret_cast<u64>(address), data);
                    auto* memory = Core::Memory::Instance();
                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                LBP3_TRACE_GFX_CTX();
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                event_eop->SignalFence(
                    [](void* address, u64 data, u32 num_bytes) {
                        NoteLabelWrite("gfx EOP", reinterpret_cast<u64>(address), data);
                        auto* memory = Core::Memory::Instance();
                        ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                    },
                    [] {
                        if (Lbp3Trace::On()) {
                            Lbp3Trace::Row("I,%lld,%u,%d,%d,%llu,%.1f\n",
                                           Lbp3Trace::Us(Lbp3Trace::NowNs()),
                                           Lbp3Trace::flips.load(), Lbp3Trace::ctx_buf,
                                           Lbp3Trace::ctx_off_end,
                                           static_cast<unsigned long long>(
                                               Lbp3Trace::fault_n.load()),
                                           Lbp3Trace::DUs(static_cast<s64>(
                                               Lbp3Trace::fault_ns.load())));
                        }
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                    });
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                    MirrorDmaFill("gfx DMA_DATA fill", dma_data->DstAddress<VAddr>(),
                                  dma_data->NumBytes(), dma_data->data);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    NoteLabelWrite("gfx DMA_DATA from GDS", dma_data->DstAddress<VAddr>(),
                                   dma_data->src_addr_lo);
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    NoteLabelWrite("gfx DMA_DATA copy (value is the source address)",
                                   dma_data->DstAddress<VAddr>(), dma_data->SrcAddress<VAddr>());
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                    u32(dma_data->dst_sel));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                LBP3_TRACE_GFX_CTX();
                if (!write_data->wr_one_addr.Value()) {
                    if (rasterizer) {
                        rasterizer->OnFence();
                    }
                    NoteLabelWrite("gfx WRITE_DATA", reinterpret_cast<u64>(address),
                                   write_data->data[0]);
                    std::memcpy(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    NoteLabelWrite("gfx MEM_SEMAPHORE signal", mem_semaphore->Address<u64>(), 1);
                    mem_semaphore->Signal();
                } else {
                    if (!mem_semaphore->Signaled()) {
                        const auto blocked = BeginBlockedWait(mem_semaphore->Address<u64>(), true);
                        Lbp3Trace::Wait lbp3_tw{};
                        if (Lbp3Trace::On()) {
                            LBP3_TRACE_GFX_CTX();
                            lbp3_tw = TraceGfxWaitBegin(1, mem_semaphore->Address<u64>(), 0, 0, 0,
                                                        Lbp3Trace::ctx_buf, Lbp3Trace::ctx_off_end,
                                                        lbp3_dcb_end - lbp3_dcb_begin, top_level);
                        }
                        const auto wait_start = std::chrono::steady_clock::now();
                        bool reported = false;
                        while (!mem_semaphore->Signaled()) {
                            if (!reported && std::chrono::steady_clock::now() - wait_start >
                                                 std::chrono::seconds(1)) {
                                reported = true;
                                LOG_WARNING(Render_Vulkan,
                                            "MEM_SEMAPHORE wait stuck over 1 s on gfx: {:#x}",
                                            mem_semaphore->Address<u64>());
                            }
                            YIELD_GFX();
                            if (lbp3_tw.on) lbp3_tw.Poll();
                        }
                        if (lbp3_tw.on) TraceGfxWaitEnd(lbp3_tw, 1);
                        EndBlockedWait(blocked);
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                if (!wait_reg_mem->Test(regs.reg_array)) {
                    // The guest may satisfy this wait only after its submit call returns.
                    const auto blocked = BeginBlockedWait(wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory
                                         ? wait_reg_mem->Address<u64>()
                                         : 0, true);
                    Lbp3Trace::Wait lbp3_tw{};
                    if (Lbp3Trace::On()) {
                        LBP3_TRACE_GFX_CTX();
                        const bool memory =
                            wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
                        lbp3_tw = TraceGfxWaitBegin(
                            0, memory ? wait_reg_mem->Address<u64>() : 0, wait_reg_mem->ref,
                            wait_reg_mem->mask, static_cast<u32>(wait_reg_mem->function.Value()),
                            Lbp3Trace::ctx_buf, Lbp3Trace::ctx_off_end,
                            lbp3_dcb_end - lbp3_dcb_begin, top_level);
                    }
                    const auto wait_start = std::chrono::steady_clock::now();
                    bool reported = false;
                    while (!wait_reg_mem->Test(regs.reg_array)) {
                        if (!reported && std::chrono::steady_clock::now() - wait_start >
                                             std::chrono::seconds(1)) {
                            reported = true;
                            const bool memory =
                                wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
                            ReportStuckWait("gfx", wait_reg_mem,
                                            memory ? *wait_reg_mem->Address()
                                                   : regs.reg_array[wait_reg_mem->Reg()]);
                        }
                        YIELD_GFX();
                        if (lbp3_tw.on) lbp3_tw.Poll();
                    }
                    if (lbp3_tw.on) TraceGfxWaitEnd(lbp3_tw, 0);
                    EndBlockedWait(blocked);
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {},
                    false);
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                task.handle.destroy();
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb, count + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                LOG_ERROR(Render_Vulkan,
                          "Unknown PM4 type 3 opcode {:#x} with count {} at {:#x}, dword {} of a "
                          "buffer at {:#x}; skipping the rest",
                          static_cast<u32>(opcode), count, reinterpret_cast<uintptr_t>(header),
                          (reinterpret_cast<uintptr_t>(header) - base_addr) / 4, base_addr);
                dcb = {};
                continue;
            }
            // The size read before the packet ran: executing it can let the guest reuse this
            // memory (an EOP interrupt wakes its callback, a WAIT_REG_MEM lets it run on).
            dcb = NextPacket(dcb, count + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type == 0) {
            // Type-0 register writes; zero dwords (register 0) are filler. The count field
            // sits where type-3's does, so next_dw_off already covers the packet.
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            task.handle.destroy();
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
                MirrorDmaFill("compute DMA_DATA fill", dma_data->DstAddress<VAddr>(),
                              dma_data->NumBytes(), dma_data->data);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                NoteLabelWrite("compute DMA_DATA from GDS", dma_data->DstAddress<VAddr>(),
                               dma_data->src_addr_lo);
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                NoteLabelWrite("compute DMA_DATA copy (value is the source address)", dst_addr,
                               src_addr);
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            ++Lbp3Trace::compute_dispatches;
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            ++Lbp3Trace::compute_dispatches;
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                NoteLabelWrite("compute WRITE_DATA", reinterpret_cast<u64>(write_data->Address<void*>()),
                               write_data->data[0]);
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                NoteLabelWrite("compute MEM_SEMAPHORE signal", mem_semaphore->Address<u64>(), 1);
                mem_semaphore->Signal();
            } else {
                if (!mem_semaphore->Signaled()) {
                    const auto blocked = BeginBlockedWait(mem_semaphore->Address<u64>());
                    Lbp3Trace::Wait lbp3_tw{};
                    if (Lbp3Trace::On()) {
                        lbp3_tw = TraceComputeWaitBegin(1, mem_semaphore->Address<u64>(), 0, 0, 0);
                    }
                    const auto wait_start = std::chrono::steady_clock::now();
                    bool reported = false;
                    while (!mem_semaphore->Signaled()) {
                        if (!reported &&
                            std::chrono::steady_clock::now() - wait_start > std::chrono::seconds(1)) {
                            reported = true;
                            LOG_WARNING(Render_Vulkan,
                                        "MEM_SEMAPHORE wait stuck over 1 s on compute: {:#x}",
                                        mem_semaphore->Address<u64>());
                        }
                        YIELD_ASC(vqid);
                        if (lbp3_tw.on) lbp3_tw.Poll();
                    }
                    if (lbp3_tw.on) TraceComputeWaitEnd(lbp3_tw, 1);
                    EndBlockedWait(blocked);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (!wait_reg_mem->Test(regs.reg_array)) {
                const auto blocked = BeginBlockedWait(wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory
                                     ? wait_reg_mem->Address<u64>()
                                     : 0);
                Lbp3Trace::Wait lbp3_tw{};
                if (Lbp3Trace::On()) {
                    const bool memory =
                        wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
                    lbp3_tw = TraceComputeWaitBegin(
                        0, memory ? wait_reg_mem->Address<u64>() : 0, wait_reg_mem->ref,
                        wait_reg_mem->mask, static_cast<u32>(wait_reg_mem->function.Value()));
                }
                const auto wait_start = std::chrono::steady_clock::now();
                bool reported = false;
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    if (!reported &&
                        std::chrono::steady_clock::now() - wait_start > std::chrono::seconds(1)) {
                        reported = true;
                        const bool memory =
                            wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
                        ReportStuckWait("compute", wait_reg_mem,
                                        memory ? *wait_reg_mem->Address()
                                               : regs.reg_array[wait_reg_mem->Reg()]);
                    }
                    YIELD_ASC(vqid);
                    if (lbp3_tw.on) lbp3_tw.Poll();
                }
                if (lbp3_tw.on) TraceComputeWaitEnd(lbp3_tw, 0);
                EndBlockedWait(blocked);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                rasterizer->OnFence();
            }
            NoteLabelWrite("compute RELEASE_MEM", release_mem->Address<u64>(),
                           release_mem->DataQWord());
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    if (Lbp3Trace::On()) {
                        Lbp3Trace::Row("CI,%lld,%u,%u\n", Lbp3Trace::Us(Lbp3Trace::NowNs()),
                                       Lbp3Trace::flips.load(), static_cast<u32>(pipe_id));
                    }
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);
        ++Lbp3Trace::compute_packets;

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    if (Lbp3Trace::On()) {
        Lbp3Trace::Row("GS,%lld,%u,%u,%zu\n", Lbp3Trace::Us(Lbp3Trace::NowNs()),
                       Lbp3Trace::flips.load(), Lbp3Trace::ThreadId(), dcb.size());
    }

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_gfx_submits;
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];
    if (Lbp3Trace::On()) {
        Lbp3Trace::Row("CS,%lld,%u,%u,%zu\n", Lbp3Trace::Us(Lbp3Trace::NowNs()),
                       Lbp3Trace::flips.load(), Lbp3Trace::ThreadId(), acb.size());
    }

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu
