#pragma once
#include "convert.hpp"
#include <vector>

namespace skimagexx {
struct LocalMeanView {
    const void* data=nullptr;
    PixelType type=PixelType::float64;
    bool half=false; // Binary16 uses uint16 storage.
    bool swap=false;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides; // Bytes; negative/zero strides allowed.
};
// Raw-value block means with after-only virtual constant padding. Factors are
// positive and have one entry per axis. Constant points to one input-storage
// value (including its byte order), needed only for nonempty padded output.
// Output is separate dense storage, with positive byte strides in logical axis
// order. Integer/bool inputs produce double, float32/64 retain their precision,
// binary16 produces binary16 with float32 accumulation. No image copy/padding.
// Returns false for unsupported floating ranges; output is then unspecified.
bool downscale_local_mean_into(const LocalMeanView&,const std::vector<std::size_t>& factors,
    const void* constant,void* output,const std::vector<std::size_t>& output_strides);
}
