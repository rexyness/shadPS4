// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    // LBP3 Recompiled: writes the driver's VkPipelineCache to its file now (the runtime calls it
    // through SaveDriverPipelineCacheNow when the game exits; it ends with ExitProcess, so no
    // destructor runs). Ported from LBPPC (emurphy04), GPL-2.0-or-later, commit aad6a91.
    void SaveDriverCache();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

    // LBP3 Recompiled (gap 15): what a geometry shader's copy shader says about its output,
    // parsed once per copy shader (the shader cache doesn't keep Info::gs_copy_data, so a GS
    // loaded from the cache has no attribute count of its own).
    struct Lbp3CopyShaderInfo {
        bool valid{};          // the VS program is a recognisable copy shader
        u32 output_vertices{}; // vertices the GS really writes per primitive (0 = unknown)
        u32 num_attrs{};       // vec4 parameter outputs
        u32 extra_comps{};     // conservative count for misc/clip/cull outputs
        u32 num_comps{};       // dwords per vertex the copy shader reads
    };
    const Lbp3CopyShaderInfo& Lbp3CopyShader(std::span<const u32> code, u64 hash);
    // The GS output vertex count to declare: the game's VGT_GS_MAX_VERT_OUT is a ring budget
    // (64 in LBP3), the copy shader's count is what the GS writes (4).
    u32 Lbp3GsOutputVertices(u32 max_vert_out, const Lbp3CopyShaderInfo& copy) const;
    // False (and lbp3_drop_reason set) when a GS with this output would exceed the device's
    // geometry limits; the draw is then dropped instead of faulting the GPU.
    bool Lbp3GsWithinLimits(u32 output_vertices, const Lbp3CopyShaderInfo& copy,
                            u32 esgs_itemsize, const char* when);
    bool Lbp3GsGuardBeforeCompile();
    bool Lbp3GsGuardSelectedPermutation();

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    // LBP3 Recompiled: the driver's VkPipelineCache, loaded from and saved to disk
    // (LBP3_VK_PIPELINE_CACHE=off: the old in-memory cache made after WarmUp). Ported from LBPPC
    // (emurphy04), GPL-2.0-or-later, commit aad6a91 (CreateDriverCache).
    bool CreateDriverCache(); ///< true when a saved cache was loaded

    // LBP3 Recompiled: the boot preload reads the cached stages one after the other, then builds
    // the pipelines on several threads (LBP3_PIPELINE_THREADS, default cores - 1, at most 16;
    // 1 = the old serial preload). Ported from LBPPC (emurphy04), GPL-2.0-or-later, commit
    // aad6a91 (PreloadJob, BuildPreloaded).
    struct PreloadJob;
    std::vector<PreloadJob>* preload_jobs{};
    void BuildPreloaded(std::vector<PreloadJob>& jobs, u32 threads);

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    // LBP3 Recompiled (gap 15)
    tsl::robin_map<u64, Lbp3CopyShaderInfo> lbp3_copy_shaders;
    std::string lbp3_drop_reason; // why RefreshGraphicsKey failed, when the GS guard says so

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    // LBP3 Recompiled: the driver cache's file and its saver (ported from LBPPC, see above).
    // Declared last: the saver thread is stopped and joined before pipeline_cache is destroyed.
    std::filesystem::path driver_cache_file;
    std::atomic<u32> driver_cache_changes{}; ///< pipelines made since the last save
    std::mutex driver_cache_mutex;
    std::jthread driver_cache_saver;

    // LBP3 Recompiled: background pipeline builds (LBP3_ASYNC_PIPELINES=1), ported from LBPPC
    // (emurphy04), GPL-2.0-or-later, commit 951fb8d. A new graphics pipeline is set up on the GPU
    // thread and built by the driver on worker threads; its draws are skipped until it's ready (no
    // freeze the first time something is drawn; it can appear a moment late instead). GPU thread
    // only, except the counter.
    struct AsyncBuild {
        GraphicsPipelineKey key;
        u64 hash;
        GraphicsPipeline::SerializationSupport sdata;
        GraphicsPipeline* pipeline;
    };
    std::vector<AsyncBuild> async_builds;
    std::atomic<u32> async_done{};
    u32 async_seen{};
    void FinishAsyncBuilds();
    struct BuildWorkers;
    std::unique_ptr<BuildWorkers> build_workers; ///< last: stopped before anything else goes
};

// LBP3 Recompiled: saves the live PipelineCache's driver cache, if there is one with unsaved
// pipelines. Safe to call from any thread, at any time (the game's exit paths).
void SaveDriverPipelineCacheNow();

} // namespace Vulkan
