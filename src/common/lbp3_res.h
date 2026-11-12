// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// LBP3 1440p (LBP3_RES=1440 + LBP3_RES_SHADERS=1): the shaders that carry the game's 1080p render
// size as literals get them rescaled when translated (shader_recompiler/recompiler.cpp). Their
// translations differ from the 1080p ones, so they must not share cache entries with them.
//
// Shared cache (default): every other shader and pipeline uses the normal cache folder, and only
// the listed shaders get a different identity: their hash is salted at the source (GetParams), so
// their SPIR-V, their permutation metas and every pipeline key that includes them are stored
// under names a 1080p session never looks up, and a 1080p session's entries are never picked up
// at 1440p. LBP3_RES_CACHE=separate restores the old scheme (unsalted hashes, the whole cache in
// its own <serial>_res1440 folder).

#include <array>
#include <cstdlib>
#include <cstring>

#include "common/types.h"

namespace Lbp3Res {

// The shaders with 1080p render-size literals (perf/res1440.md, a scan of the translated cache).
inline constexpr std::array<u64, 17> kShaders = {
    0x03101c0edb07dfcaull, 0x0b2a33469cd0355aull, 0x2949baf9d0b695fdull, 0x33c89263dcb0a389ull,
    0x3a0615bd0d8d3a5eull, 0x3d19e6252d1eb891ull, 0x4d3d6de48b31b76aull, 0x5ed16706fc9e0eb1ull,
    0x8079a39da0f7bae3ull, 0x8e1db4f9563309a5ull, 0x8f2b909cf6507cebull, 0xb0bb8b371aa7cb79ull,
    0xc9311bbdd52ae955ull, 0xd7506ff3bef3fd6cull, 0xebb656d6240b4b34ull, 0xec6a8d0b0f8c9aedull,
    0xfdf2b5d50f5db65dull,
};

// A fixed, invertible-looking scramble of a listed hash (splitmix64 finaliser of hash ^ "RES1440").
// Pure, so the cache names are stable across runs and tools can compute them.
[[nodiscard]] constexpr u64 Salt(u64 hash) {
    u64 z = hash ^ 0x5245533134343000ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

[[nodiscard]] constexpr bool IsListedOriginal(u64 hash) {
    for (const u64 h : kShaders) {
        if (h == hash) {
            return true;
        }
    }
    return false;
}

// The hash a shader is known by: salted only for listed shaders, only with the rescale on and the
// shared cache in use.
[[nodiscard]] constexpr u64 StoredHash(u64 original, bool shaders_on, bool separate_cache) {
    return shaders_on && !separate_cache && IsListedOriginal(original) ? Salt(original) : original;
}

// Whether a hash as seen by the recompiler (salted or not) belongs to a listed shader; returns the
// original hash through `original` when it does.
[[nodiscard]] constexpr bool IsListed(u64 hash, u64* original = nullptr) {
    for (const u64 h : kShaders) {
        if (h == hash || Salt(h) == hash) {
            if (original) {
                *original = h;
            }
            return true;
        }
    }
    return false;
}

inline bool EnvIs(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

// LBP3_RES=1440 and LBP3_RES_SHADERS=1.
inline bool ShadersOn() {
    static const bool on = EnvIs("LBP3_RES", "1440") && EnvIs("LBP3_RES_SHADERS", "1");
    return on;
}

// LBP3_RES_CACHE=separate: the old scheme (the whole cache in <serial>_res1440, no salting).
inline bool SeparateCache() {
    static const bool separate = EnvIs("LBP3_RES_CACHE", "separate");
    return separate;
}

inline u64 StoredHash(u64 original) {
    return StoredHash(original, ShadersOn(), SeparateCache());
}

} // namespace Lbp3Res
