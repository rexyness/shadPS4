// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// LBP3 Recompiled: the file the driver's VkPipelineCache is saved to (user\shader_cache\<serial>\
// driver.vkcache). The idea of saving the driver cache beside the shader cache and checking it
// against the GPU before use is ported from LBPPC (emurphy04), GPL-2.0-or-later, commit aad6a91
// (video_core/renderer_vulkan/vk_pipeline_cache.cpp: CreateDriverCache/SaveDriverCache). This
// adds a header of our own in front of the driver's data, so the driver version is checked too
// (NVIDIA keeps its pipelineCacheUUID across some driver updates), and a size and magic so a cut
// or foreign file is rejected before the driver sees it. Pure functions, no Vulkan calls: the
// offline test (runtime/tests/vkcachetest.cpp) checks every rejection.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace Vulkan::DriverCacheFile {

constexpr std::uint32_t kMagic = 0x43504B56u; // "VKPC"
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kVkHeaderVersionOne = 1; // VK_PIPELINE_CACHE_HEADER_VERSION_ONE
constexpr std::size_t kUuidSize = 16;            // VK_UUID_SIZE

// The GPU and driver a saved cache belongs to.
struct Identity {
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t driver_version{};
    std::array<std::uint8_t, kUuidSize> uuid{}; // VkPhysicalDeviceProperties::pipelineCacheUUID
};

struct FileHeader {
    std::uint32_t magic;
    std::uint32_t format_version;
    std::uint32_t vendor_id;
    std::uint32_t device_id;
    std::uint32_t driver_version;
    std::uint8_t uuid[kUuidSize];
    std::uint64_t data_size;
};
static_assert(sizeof(FileHeader) == 48);

// The driver's own header at the start of vkGetPipelineCacheData's output (Vulkan spec,
// VkPipelineCacheHeaderVersionOne).
struct VkHeaderOne {
    std::uint32_t header_size;
    std::uint32_t header_version;
    std::uint32_t vendor_id;
    std::uint32_t device_id;
    std::uint8_t uuid[kUuidSize];
};
static_assert(sizeof(VkHeaderOne) == 32);

// The file's bytes: our header, then the driver's data unchanged.
inline std::vector<std::uint8_t> Wrap(const Identity& id, const std::vector<std::uint8_t>& data) {
    FileHeader h{};
    h.magic = kMagic;
    h.format_version = kFormatVersion;
    h.vendor_id = id.vendor_id;
    h.device_id = id.device_id;
    h.driver_version = id.driver_version;
    std::memcpy(h.uuid, id.uuid.data(), kUuidSize);
    h.data_size = data.size();
    std::vector<std::uint8_t> out(sizeof(h) + data.size());
    std::memcpy(out.data(), &h, sizeof(h));
    if (!data.empty()) {
        std::memcpy(out.data() + sizeof(h), data.data(), data.size());
    }
    return out;
}

// The driver's data from a file's bytes, or empty with `why` set when the file is cut, foreign,
// from another GPU or driver, or its driver header doesn't match. An empty input (no file) gives
// empty data and an empty `why`.
inline std::vector<std::uint8_t> Unwrap(const Identity& id, const std::vector<std::uint8_t>& file,
                                        std::string* why = nullptr) {
    const auto fail = [&](const char* reason) {
        if (why) {
            *why = reason;
        }
        return std::vector<std::uint8_t>{};
    };
    if (why) {
        why->clear();
    }
    if (file.empty()) {
        return {};
    }
    FileHeader h{};
    if (file.size() < sizeof(h)) {
        return fail("shorter than its header");
    }
    std::memcpy(&h, file.data(), sizeof(h));
    if (h.magic != kMagic) {
        return fail("not a driver cache file");
    }
    if (h.format_version != kFormatVersion) {
        return fail("another file format version");
    }
    if (h.vendor_id != id.vendor_id || h.device_id != id.device_id) {
        return fail("made on another GPU");
    }
    if (h.driver_version != id.driver_version) {
        return fail("made by another driver version");
    }
    if (std::memcmp(h.uuid, id.uuid.data(), kUuidSize) != 0) {
        return fail("made with another pipeline cache UUID");
    }
    if (h.data_size != file.size() - sizeof(h)) {
        return fail("cut short or padded");
    }
    VkHeaderOne vh{};
    if (h.data_size < sizeof(vh)) {
        return fail("driver data shorter than the driver's header");
    }
    std::memcpy(&vh, file.data() + sizeof(h), sizeof(vh));
    if (vh.header_size < sizeof(vh) || vh.header_size > h.data_size ||
        vh.header_version != kVkHeaderVersionOne || vh.vendor_id != id.vendor_id ||
        vh.device_id != id.device_id || std::memcmp(vh.uuid, id.uuid.data(), kUuidSize) != 0) {
        return fail("the driver's own header doesn't match this GPU");
    }
    return std::vector<std::uint8_t>(file.begin() + sizeof(h), file.end());
}

} // namespace Vulkan::DriverCacheFile
