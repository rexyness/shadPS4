// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <thread>
#include <vector>

namespace Storage {

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

    // LBP3 Recompiled: how many blobs of a type ForEachBlob will visit, from the file names
    // only (no file is read), for the boot loading screen's progress (common/lbp3_boot.h).
    [[nodiscard]] u32 CountBlobs(BlobType type);

    // LBP3 Recompiled: reads every file of a folder cache once, on `threads` background threads,
    // and throws the bytes away, so the boot preload's one-at-a-time reads find them in memory.
    // On the first start after a reboot each small file costs milliseconds to open (17,000
    // freshly written files: 94.6 s one at a time, 8.3 s on 16 threads). The threads stop and
    // join when the returned pool is destroyed. Nothing for an archived cache.
    [[nodiscard]] std::vector<std::jthread> StartPrefetch(u32 threads);

    // LBP3 Recompiled: where the driver's VkPipelineCache is saved, beside the shader cache
    // (<serial>\driver.vkcache, or <serial>.vkcache next to an archived cache). Ported from
    // LBPPC (emurphy04), GPL-2.0-or-later, commit aad6a91 (DataBase::DriverCachePath). Only
    // meaningful once Open() has run.
    [[nodiscard]] std::filesystem::path DriverCachePath() const;

private:
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
