#pragma once
#include "skimagexx/regular_seeds.hpp"

namespace skimagexx::simd {
void regular_seeds_zero(void* output, std::size_t bytes);
void regular_seeds_u32_run(std::uint32_t* output, std::size_t count, std::uint64_t first);
void regular_seeds_u64_run(std::uint64_t* output, std::size_t count, std::uint64_t first);
}
