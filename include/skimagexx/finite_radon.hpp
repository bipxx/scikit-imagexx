#pragma once
#include <skimagexx/crop.hpp>
#include <skimagexx/convert.hpp>
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace skimagexx {
// Borrowed ordinary POD view. Caller proves allocation/lifetime/strides.
// Integer word type, itemsize and byte order are explicit; no scaling occurs.
struct FiniteRadonInput { CropView image; PixelType type=PixelType::uint32; bool swapped=false; bool half=false; bool complex=false; };
// Source assignment phase notices:1 complex discard before cast,2 invalid cast
// after actual native conversion. Callback is synchronous on caller thread;
// source warning/error policy may throw before the next projection is computed.
using FiniteRadonNotice=void(*)(void*,unsigned);
class FiniteRadonCastingError:public std::invalid_argument { public:FiniteRadonCastingError():std::invalid_argument("ifrt2 in-place uint32 casting requires metadata error delivery"){} };
struct FiniteRadonPlan {
    unsigned accumulator_bits=64; // actual source default sum accumulator
    bool weak_scalar=false;       // NumPy1.x nonnegative scalar value promotion
    bool signed_sum=false;        // boolean sum uses signed platform integer
};
struct FiniteRadonTransform {
    std::size_t rows=0,columns=0;
    std::vector<std::uint32_t> values;
    int captured_workers=1;
    std::vector<unsigned> cast_status;
};
struct InverseFiniteRadonTransform {
    std::size_t rows=0,columns=0;
    std::vector<double> values;
    int captured_workers=1;
};
// Complete owning operations, default0 captures existing global budget once
// (fresh process default1). Results own C-order storage. No prime restriction.
// frt2 supports bool/signed/unsigned integer, binary16/32/64 and complex32/64
// storage. Integer sums narrow mod2^32; floating assignment preserves scalar
// precision, source projection order and synchronous cast-policy notices.
FiniteRadonTransform frt2(const FiniteRadonInput&,int workers=0,FiniteRadonNotice=nullptr,void* context=nullptr);
// ifrt2 supports bool/unsigned source storage, source uint32 intermediate wrap,
// explicit sum/promotion plan, then ordered float64 conversion/division.
// Signed/floating original in-place casting errors belong to public metadata.
// Python representation, arbitrary object arithmetic and dtype-error classes
// remain explicit metadata/provider boundaries. Wider exceptional vector-cast,
// alternative FP-mode/platform domains require their source profile assessment.
InverseFiniteRadonTransform ifrt2(const FiniteRadonInput&,const FiniteRadonPlan& = {},int workers=0,FiniteRadonNotice=nullptr,void* context=nullptr);
// Native source-ordered projections before inverse in-place cast resolution.
// Complete ifrt2 composes this stage before its floating cast error; public
// Python boundary supplies NumPy's precise error class/args after these notices.
FiniteRadonTransform finite_radon_inverse_prefix(const FiniteRadonInput&,int workers=0,FiniteRadonNotice=nullptr,void* context=nullptr);
}
