#pragma once
#include <skimagexx/crop.hpp>
#include <skimagexx/convert.hpp>
#include <cstdint>
#include <vector>
namespace skimagexx {
// Numeric boundary: ordinary live POD allocation, byte strides, no forged spans.
// byte_swapped describes source words only; half uses uint16 POD storage.
// Scalar precision plans and NumPy policy delivery are explicit boundaries.
struct MultiblockLbpInput { CropView image; PixelType type=PixelType::uint8; bool byte_swapped=false; bool half=false; };
struct MultiblockLbpColor { std::vector<double> values; std::vector<std::ptrdiff_t> shape; };
struct MultiblockLbpParameters {
    std::ptrdiff_t r=0,c=0,width=1,height=1;
    std::int64_t code=0;
    double alpha=0.5;
    MultiblockLbpColor greater{{1,1,1},{3}},less{{0,0.69,0.96},{3}};
    // Explicit source scalar-promotion plan, not an inferred NumPy version.
    bool first_product_float32=true;
    bool first_product_half=false; // explicit binary16 first-product plan
    bool explicit_complement=false;
    double complement=0.5;
    bool simd=true;
    bool prepare_only=false;
};
struct MultiblockLbpImage {
    std::vector<std::ptrdiff_t> shape;
    std::vector<float> float32;
    std::vector<double> float64;
    std::vector<std::uint16_t> float16;
    bool is_half=false;
    bool is_float32=false;
    int captured_workers=1; // one global snapshot; ordered FP phases serial
    bool sse2_enabled=false;
    std::size_t elements()const{return is_half?float16.size():is_float32?float32.size():float64.size();}
};
// COMPLETE owning numeric operation: copy/gray replication, normalized integer
// conversion and eight live ordered patch updates. Canonical C-order output;
// Binary16 output words are exposed separately. NumPy K-order/base/protocol
// representation and exact Python errors/policy are public-boundary duties.
// Each patch finishes image-product, color-product, broadcast/add/cast before
// the next patch reads its updated image. No alpha/empty shortcut or FMA.
MultiblockLbpImage draw_multiblock_lbp(const MultiblockLbpInput&,const MultiblockLbpParameters&);
// Source-ordered public phases use the same owning SDK numeric machinery.
// Floating buffers are live byte-strided IEEE binary16/32/64; byte order is
// explicit. Output shape is the checked broadcast shape, supplied by caller.
struct MultiblockLbpFloatView { CropView image; bool swapped=false; };
unsigned multiblock_lbp_binary(const MultiblockLbpFloatView&,const MultiblockLbpFloatView&,
    const MultiblockLbpFloatView& output,bool add);
void multiblock_lbp_assign(const MultiblockLbpFloatView& destination,const MultiblockLbpFloatView& source);
}
