#pragma once
#include "convert.hpp"
#include <memory>

namespace skimagexx {
enum class LiKind { ready, empty, constant, infinities, boolean_error, unsupported };
struct LiResult { double value=0; bool updated=false; unsigned status=0; };
struct LiMath {
    double (*log)(double,bool,void*)=nullptr;
    void (*poll)(void*)=nullptr;
    void* context=nullptr;
    void (*iteration)(double,bool,void*)=nullptr;
    // Zero selects a single full reduction tree; NumPy before2.3 uses its
    // current ufunc buffer size and then accumulates the block sums in order.
    std::size_t reduction_block=0;
};
// Copies logical contiguous input, removes nonfinite samples in upstream order,
// and shifts finite samples by their minimum. uint8 uses a bounded histogram.
// Current image types: boolean, uint8, float32, float64; uint16 is admitted
// only for values0..255 and at most2^31-1 samples, reusing the byte histogram.
// Other uint16 inputs return LiKind::unsupported. The object is immutable;
// independent run/spacing calls are safe concurrently. Global workers are
// captured for each operation; sums preserve NumPy's pairwise reduction tree.
class LiImage {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    LiImage(const void*,std::size_t,PixelType);
    ~LiImage();
    LiImage(const LiImage&)=delete;
    LiImage& operator=(const LiImage&)=delete;
    LiKind kind() const;
    double value() const;
    double minimum() const;
    double maximum() const; // shifted range
    // NaN denotes a single unique finite value after infinities were removed.
    double spacing() const;
    // Initial guess is already shifted and cast to the input precision.
    // Status1: empty foreground; status32: unsafe floating arithmetic.
    LiResult run(double tolerance,bool has_guess=false,double guess=0,const LiMath* math=nullptr) const;
};
}
