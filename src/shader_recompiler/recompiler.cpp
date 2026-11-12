// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/lbp3_res.h"
#include "common/logging/classes.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Shader {

namespace {

// LBP3 (exp/res1440): with LBP3_RES=1440 and LBP3_RES_SHADERS=1, the shaders that carry the
// game's 1080p render size as literals (SMAA's RT_METRICS, the composite's 1/480 bloom step,
// the 1920x2160 stacked target's texel) get them rescaled. Both axes scale alike (2560/1920 =
// 1440/1080 = 4/3), so a literal equal to a 1080p size v (1920, 1080 and their halves,
// quarters and doubles) becomes v * 4/3 and one equal to 1/v becomes 3/4 of it. The list
// (Lbp3Res::kShaders, common/lbp3_res.h, the one list for the rescale and the cache salt) comes
// from a scan of the translated cache (perf/res1440.md) plus a scan of every shader in the game
// data (perf/res1440_shader_scan.md): the 20 variants of proc_base_outline_special and
// proc_base_xray_special.gmat (pixel positions to screen UVs with 1/1920 and 1/1080), six more
// final-composite variants (finalcomp_lowres and five unnamed ones, the 1/480 bloom step) and
// tgpshaderblur (1/960, 1/540). Any other shader with such a literal is logged once, unchanged,
// so new ones show up in unseen levels. With the shared cache the listed shaders arrive here
// under their salted hash (GetParams), so they are matched in either form.
bool Lbp3ResShadersOn() {
    return Lbp3Res::ShadersOn();
}

// 0 = not a 1080p size; otherwise the factor to apply (4/3 for a size, 3/4 for a reciprocal).
float Lbp3ResFactor(float f) {
    static constexpr float kSizes[] = {1920.f, 1080.f, 960.f, 540.f, 480.f, 270.f, 3840.f, 2160.f, 7680.f, 4320.f};
    if (!std::isfinite(f) || f <= 0.f) return 0.f;
    for (const float v : kSizes) {
        if (std::fabs(f - v) <= v * 2e-6f) return 4.f / 3.f;
        if (std::fabs(f - 1.f / v) <= (1.f / v) * 2e-6f) return 3.f / 4.f;
    }
    return 0.f;
}

void Lbp3ScaleResolutionLiterals(std::vector<Gcn::GcnInst>& insts, u64 hash) {
    if (!Lbp3ResShadersOn()) return;
    u64 original = hash;
    const bool listed = Lbp3Res::IsListed(hash, &original);
    int changed = 0, seen = 0;
    for (auto& inst : insts) {
        for (u32 i = 0; i < inst.src_count && i < inst.src.size(); ++i) {
            auto& src = inst.src[i];
            if (src.field != Gcn::OperandField::LiteralConst) continue;
            const float f = std::bit_cast<float>(src.code);
            const float k = Lbp3ResFactor(f);
            if (k == 0.f) continue;
            ++seen;
            if (!listed) continue;
            src.code = std::bit_cast<u32>(f * k);
            ++changed;
        }
    }
    if (listed)
        std::fprintf(stderr, "[res] shader %#018llx: %d 1080p literals scaled for 1440p (cache key %#018llx)\n",
                     static_cast<unsigned long long>(original), changed,
                     static_cast<unsigned long long>(hash));
    else if (seen)
        std::fprintf(stderr, "[res] shader %#018llx: %d 1080p-like literals, NOT on the list (unchanged)\n",
                     static_cast<unsigned long long>(hash), seen);
}

} // namespace

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

void EmitControlFlowGraph(IR::Program& program, Pools& pools, Gcn::CFG& cfg,
                          RuntimeInfo& runtime_info, const Profile& profile) {
    Gcn::Translator translator{program.info, runtime_info, profile};
    for (auto& block : cfg) {
        const u32 start = block.begin_index;
        const u32 size = block.end_index - start + 1;
        auto* ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(start, size));
        program.blocks.push_back(ir_block);
    }
    translator.EmitPrologue(program.blocks.front());
    ASSERT_MSG(!program.info.translation_failed, "Shader translation has failed");
    for (auto& block : cfg) {
        auto* ir_block = block.ir_block;
        if (block.branch_true) {
            auto* true_block = block.branch_true->ir_block;
            ir_block->AddBranch(true_block);
        }
        if (block.branch_false) {
            auto* false_block = block.branch_false->ir_block;
            ir_block->AddBranch(false_block);
        }
    }
    program.post_order_blocks = Shader::IR::PostOrder(program.blocks.front());
}

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }
    Lbp3ScaleResolutionLiterals(program.ins_list, info.pgm_hash);

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    EmitControlFlowGraph(program, pools, cfg, runtime_info, profile);

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    Shader::InjectClipDistanceAttributes(program, runtime_info);

    // Run optimization passes on unstructured graph
    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::ReadLaneEliminationPass(program);
    if (info.sw_stage == SwStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.sw_stage == SwStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    Shader::IR::DumpProgram(program, info, "pre-res-discover.");
    auto resources = Shader::Optimization::ResourceDiscoverPass(program, profile);
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    Shader::IR::DumpProgram(program, info, "pre-res-patch.");
    Shader::Optimization::ResourcePatchingPass(program.info, resources, profile);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    Shader::Optimization::LowerUserClipPlanes(program, runtime_info);
    Shader::Optimization::PhiSimplificationPass(program);
    Shader::Optimization::InverseBallotEliminationPass(program);
    Shader::IR::DumpProgram(program, info, "pre-lower-phi.");

    // Prepare for structurization by clearing flow graph and lowering phis
    for (auto* ir_block : program.blocks) {
        ir_block->imm_predecessors.clear();
        ir_block->imm_successors.clear();
        ir_block->ssa_state.Reset();
    }
    Shader::Optimization::LowerPhisToRegsPass(program);

    // Structurize control flow graph and create program.
    program.syntax_list = Shader::Gcn::BuildASL(pools, cfg, info);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front().data.block);

    // Run optimization passes on structured graph
    Shader::Optimization::SsaRepairPass(program);
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::LowerWave64BallotPass(program, runtime_info, profile);
    Shader::Optimization::LowerHardwareIntrinsics(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
