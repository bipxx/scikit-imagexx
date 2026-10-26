#pragma once
#include "skimagexx/convert.hpp"
#include <vector>

namespace skimagexx::detail {
struct BlockReduceInput {
    const void* data = nullptr;
    PixelType type = PixelType::float64;
    bool half = false; // uint16 storage for native binary16.
    std::vector<std::size_t> shape;
};
// Already padded, native-endian, aligned C input. Factors divide every axis.
// Output is separate dense C/F storage with the supplied BYTE strides; the
// caller owns both allocations and keeps their descriptors stable for the call.
// bool/signed -> int64 bits, unsigned -> uint64; f16/f32/f64 retain output type,
// with f32 inner-segment accumulation and half stores between segments for f16.
// Half segment geometry follows the prepared native-C view. Rank zero has
// one input/output value. Empty arrays need no data pointer.
// Returns divide=1, overflow=2, underflow=4, invalid=8, restoring worker FEnv.
unsigned block_reduce_sum(const BlockReduceInput&, const std::vector<std::size_t>& factors,
                          void* output, const std::vector<std::size_t>& output_strides);
}

namespace skimagexx::simd {
// Independent adjacent output blocks in one 2D row; returns the processed count.
std::size_t block_reduce_row(const float*, float*, std::size_t columns,
                            std::size_t factor_rows, std::size_t factor_columns, std::size_t count);
std::size_t block_reduce_row(const double*, double*, std::size_t columns,
                            std::size_t factor_rows, std::size_t factor_columns, std::size_t count);
}
