#pragma once

#include "skimagexx/tv_denoise.hpp"

namespace skimagexx::detail {
// Source-local checked entry, intentionally absent from the installed SDK.
// Exactly one complete prewrite eligibility check precedes the unchanged loop.
// A returned reason has the same immutable text as tv_denoise_ineligibility;
// neither numerical storage nor result is modified on rejection. Accepted work
// assigns result after the loop. Exceptions after acceptance propagate normally.
// The internal caller owns a separate result object (as both current callers do).
const char* tv_denoise_try_finite_into(const TvConstView& input,
    const TvDenoiseWorkspace& workspace, const TvDenoiseParameters& parameters,
    TvDenoiseResult& result, TvDenoiseTrace* trace = nullptr,
    std::size_t trace_capacity = 0);
} // namespace skimagexx::detail
