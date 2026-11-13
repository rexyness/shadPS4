// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// LBP3 render sizes (LBP3_RES): the game renders at a size other than the PS4's 1920x1080. The
// runtime patches the game's own sizes (runtime/src/core/res_patches.cpp); here is the one mapping
// from a 1080p size to the chosen one that every consumer uses: those patches, the shader
// literals below (shader_recompiler/recompiler.cpp), the cache salt, the runtime's capture
// buffers and the launcher.
//
// LBP3_RES: unset or "1080" = 1920x1080 (the game untouched); "1440" = 2560x1440 (the original
// switch); "WxH" = any size from 1280x720 up to 4480 wide (LBPPC found 5120 wide stops the game),
// no narrower than 16:9. Anything else is ignored (1080p) by every consumer alike.
//
// LBP3_RES_SHADERS=1: the shaders that carry the game's 1080p render size as literals get them
// rescaled when translated. Their translations differ from the 1080p ones, so they must not share
// cache entries with them.
//
// Shared cache (default): every other shader and pipeline uses the normal cache folder, and only
// the listed shaders get a different identity: their hash is salted at the source (GetParams), so
// their SPIR-V, their permutation metas and every pipeline key that includes them are stored
// under names a session at another size never looks up. 2560x1440 keeps the salt it always had,
// so existing 1440p caches still match. LBP3_RES_CACHE=separate restores the old scheme
// (unsalted hashes, the whole cache in its own <serial>_res1440 / <serial>_res<W>x<H> folder).

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "common/types.h"

namespace Lbp3Res {

// ---- The render size and the mapping ----

struct Size {
    u32 w = 1920;
    u32 h = 1080;
    constexpr bool operator==(const Size&) const = default;
};

inline constexpr Size kNative{1920, 1080};
inline constexpr Size k1440{2560, 1440};
inline constexpr u32 kMaxWidth = 4480; // LBPPC (emurphy04, aad6a91): 5120 wide stops the game

// Parses an LBP3_RES value. False (and *out untouched) when it isn't a usable size.
constexpr bool Parse(const char* v, Size* out) {
    if (!v || !*v) {
        return false;
    }
    auto num = [](const char*& p, u32& n) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        n = 0;
        while (*p >= '0' && *p <= '9') {
            n = n * 10 + u32(*p - '0');
            if (n > 100000) {
                return false;
            }
            ++p;
        }
        return true;
    };
    const char* p = v;
    u32 a = 0, b = 0;
    if (!num(p, a)) {
        return false;
    }
    Size s{};
    if (*p == '\0') { // a height on its own: the original switches
        if (a == 1080) {
            s = kNative;
        } else if (a == 1440) {
            s = k1440;
        } else {
            return false;
        }
    } else {
        if (*p != 'x' && *p != 'X') {
            return false;
        }
        ++p;
        if (!num(p, b) || *p != '\0') {
            return false;
        }
        s = {a, b};
    }
    // 1280x720 .. 4480 wide, no taller than 4320, no narrower than 16:9 (the canvas only widens).
    if (s.w < 1280 || s.h < 720 || s.w > kMaxWidth || s.h > 4320) {
        return false;
    }
    if (u64(s.w) * 9 < u64(s.h) * 16) {
        return false;
    }
    *out = s;
    return true;
}

// A 1080p width or height scaled to the size, rounded to the nearest pixel.
constexpr u32 ScaleW(u32 v, Size s) {
    return u32((u64(v) * s.w + 960) / 1920);
}
constexpr u32 ScaleH(u32 v, Size s) {
    return u32((u64(v) * s.h + 540) / 1080);
}
// The targets smaller than the screen at 1080p (the quarter-size 480x270 ones and the bloom chain's
// 1280x720, 640x360, 320x180) grow with the size but never shrink: the game cuts a centred 256x256
// square from one of them (0x4014f0), which read past a shrunk target at 720p. The rule and the
// address are LBPPC's (emurphy04, aad6a91, src/probe.c render_res_install).
constexpr u32 SmallW(u32 v, Size s) {
    const u32 r = ScaleW(v, s);
    return r < v ? v : r;
}
constexpr u32 SmallH(u32 v, Size s) {
    const u32 r = ScaleH(v, s);
    return r < v ? v : r;
}

// What a 1080p render-size literal in a shader stands for at the size (0 = not one): widths
// 1920, 960, 3840, 7680 and the quarter-size target's 480; heights 1080, 540, 2160, 4320 and the
// quarter-size target's 270. The quarter-size target follows its own rule (SmallW/SmallH), so a
// literal for it stays as it is where the target kept its 1080p size.
constexpr u32 LiteralSize(u32 v, Size s) {
    switch (v) {
    case 1920:
    case 960:
    case 3840:
    case 7680:
        return ScaleW(v, s);
    case 1080:
    case 540:
    case 2160:
    case 4320:
        return ScaleH(v, s);
    case 480:
        return SmallW(v, s);
    case 270:
        return SmallH(v, s);
    default:
        return 0;
    }
}

inline bool EnvIs(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

// The size LBP3_RES asks for (1920x1080 when unset or unusable). Read once.
inline Size Current() {
    static const Size s = [] {
        Size r = kNative;
        Parse(std::getenv("LBP3_RES"), &r);
        return r;
    }();
    return s;
}

// ---- The shaders with render-size literals and their cache identity ----

// The shaders with 1080p render-size literals: the one list for both the rescale
// (shader_recompiler/recompiler.cpp) and the cache salt below, from a scan of the translated cache
// (perf/res1440.md): SMAA, the final composite, the stacked target's texel. The full game-data scan
// (perf/res1440_shader_scan.md) found 27 more (outline and x-ray material variants, more composite
// variants, tgpshaderblur): those are on branch fix/res1440-shaders-44, held until they are seen
// in game; until then they're only logged ("NOT on the list") if they ever compile at another size.
inline constexpr std::array<u64, 17> kShaders = {
    0x03101c0edb07dfcaull, 0x0b2a33469cd0355aull, 0x2949baf9d0b695fdull, 0x33c89263dcb0a389ull,
    0x3a0615bd0d8d3a5eull, 0x3d19e6252d1eb891ull, 0x4d3d6de48b31b76aull, 0x5ed16706fc9e0eb1ull,
    0x8079a39da0f7bae3ull, 0x8e1db4f9563309a5ull, 0x8f2b909cf6507cebull, 0xb0bb8b371aa7cb79ull,
    0xc9311bbdd52ae955ull, 0xd7506ff3bef3fd6cull, 0xebb656d6240b4b34ull, 0xec6a8d0b0f8c9aedull,
    0xfdf2b5d50f5db65dull,
};

// The salt's key: "RES1440" for 2560x1440 (unchanged since the first 1440p cache), else "RES" with
// the size in the low bits (width << 20 | height: never the 1440 key, never another size's).
constexpr u64 SaltKey(Size s) {
    return s == k1440 ? 0x5245533134343000ull : (0x5245530000000000ull | (u64(s.w) << 20) | s.h);
}

// A fixed scramble of a listed hash (splitmix64 finaliser of hash ^ the size's key). Pure, so the
// cache names are stable across runs and tools can compute them.
[[nodiscard]] constexpr u64 Salt(u64 hash, Size s) {
    u64 z = hash ^ SaltKey(s);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
// The 1440p salt (the original one).
[[nodiscard]] constexpr u64 Salt(u64 hash) {
    return Salt(hash, k1440);
}

[[nodiscard]] constexpr bool IsListedOriginal(u64 hash) {
    for (const u64 h : kShaders) {
        if (h == hash) {
            return true;
        }
    }
    return false;
}

// The hash a shader is known by: salted (for the size) only for listed shaders, only with the
// rescale on and the shared cache in use.
[[nodiscard]] constexpr u64 StoredHash(u64 original, bool shaders_on, bool separate_cache,
                                       Size s = k1440) {
    return shaders_on && !separate_cache && IsListedOriginal(original) ? Salt(original, s)
                                                                        : original;
}

// Whether a hash as seen by the recompiler (salted for the size or not) belongs to a listed
// shader; returns the original hash through `original` when it does.
[[nodiscard]] constexpr bool IsListed(u64 hash, u64* original = nullptr, Size s = k1440) {
    for (const u64 h : kShaders) {
        if (h == hash || Salt(h, s) == hash) {
            if (original) {
                *original = h;
            }
            return true;
        }
    }
    return false;
}

// LBP3_RES at a size other than 1920x1080 and LBP3_RES_SHADERS=1.
inline bool ShadersOn() {
    static const bool on = Current() != kNative && EnvIs("LBP3_RES_SHADERS", "1");
    return on;
}

// LBP3_RES_CACHE=separate: the old scheme (the whole cache in its own folder, no salting).
inline bool SeparateCache() {
    static const bool separate = EnvIs("LBP3_RES_CACHE", "separate");
    return separate;
}

// The separate scheme's folder suffix: "_res1440" for 2560x1440 (as before), else "_res<W>x<H>".
inline std::string SeparateSuffix() {
    const Size s = Current();
    if (s == k1440) {
        return "_res1440";
    }
    return "_res" + std::to_string(s.w) + "x" + std::to_string(s.h);
}

inline u64 StoredHash(u64 original) {
    return StoredHash(original, ShadersOn(), SeparateCache(), Current());
}

} // namespace Lbp3Res
