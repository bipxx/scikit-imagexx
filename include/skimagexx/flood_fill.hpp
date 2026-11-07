#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace skimagexx {
// Numeric POD operation, ordered serial queue. Input byte strides may be signed.
// Caller supplies valid storage for the declared shape/strides. Seed is normalized.
struct FloodGeometry {
    std::vector<std::ptrdiff_t> shape, strides, seed, footprint_shape;
    std::vector<std::uint8_t> footprint;
    bool fortran=false;
};
struct FloodMask {
    std::vector<std::uint8_t> storage;
    std::vector<std::ptrdiff_t> shape, strides, padding;
    std::ptrdiff_t origin=0;
};
template<class T> FloodMask flood(const T*, const FloodGeometry&, bool tolerance=false,
                                  T low=T{}, T high=T{});
// Owning C-order fill and explicit borrowed in-place variant. Casting is caller's
// responsibility; Python uses NumPy's version-specific scalar conversion policy.
template<class T> std::vector<T> flood_fill(const T*, const FloodGeometry&, T value,
                                           bool tolerance=false,T low=T{},T high=T{});
template<class T> void flood_fill_into(T*, const FloodGeometry&, T value,
                                       bool tolerance=false,T low=T{},T high=T{});
}
