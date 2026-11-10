// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/renderer_vulkan/lbp3_checkpoints.h"

#include "common/assert.h"

namespace Vulkan {

Semaphore::Semaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

Semaphore::~Semaphore() = default;

void Semaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void Semaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    // LBP3 Recompiled: wait in 1 s steps so a wait that never ends gets reported. A lost
    // device never signals: report it instead of spinning forever.
    constexpr u64 kStepNs = 1'000'000'000;
    int waited_s = 0;
    bool logged_result = false;
    vk::Result result;
    while ((result = instance.GetDevice().waitSemaphores(&wait_info, kStepNs)) != vk::Result::eSuccess) {
        if (result == vk::Result::eErrorDeviceLost) {
            Lbp3Checkpoints::OnDeviceLost(instance.GetGraphicsQueue(), "Semaphore::Wait", tick);
        }
        if (result != vk::Result::eTimeout && !logged_result) {
            logged_result = true;
            std::fprintf(stderr, "[gpu] Semaphore::Wait: waitSemaphores returned %s\n",
                         vk::to_string(result).c_str());
        }
        if (++waited_s == 5) {
            Lbp3Checkpoints::OnLongWait(instance.GetGraphicsQueue(), "Semaphore::Wait", tick, waited_s,
                                        static_cast<int>(result));
        }
    }
    Refresh();
}

} // namespace Vulkan
