#pragma once
#include "components.hpp"
#include "find_boundaries.hpp"
#include <vector>
namespace skimagexx {
// Source compatibility choice is explicit; SciPy1.18 copies when every zoom
// factor is one, while1.15 still prefilters and interpolates every axis.
enum class MarkCubicProfile { scipy115, scipy118 };
struct MarkColor {std::vector<double> values{1,1,0};std::vector<std::size_t> shape{3};};
struct MarkBoundariesParameters {
    detail::FindBoundaryParameters boundary{detail::FindBoundaryMode::outer,1,0,true};
    MarkCubicProfile cubic_profile=MarkCubicProfile::scipy118;
    MarkColor color;
    bool outline=false;
    MarkColor outline_color{{0,0,0},{3}};
};
// Owning native result, never aliases either source. Strides are in ELEMENTS;
// ordinary copies preserve positive dense K-axis order (gray adds inner RGB),
// subpixel always returns C-order. Keep this object alive while borrowing data.
struct MarkImage {
    PixelType type=PixelType::float64;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::vector<float> f32;
    std::vector<double> f64;
    const void* data()const{return type==PixelType::float32?static_cast<const void*>(f32.data()):static_cast<const void*>(f64.data());}
    void* data(){return type==PixelType::float32?static_cast<void*>(f32.data()):static_cast<void*>(f64.data());}
};
// Caller proves source allocation capacity/lifetime and stable descriptors.
// Ordinary numeric image rank2..4, half/f32->f32, other real types->f64;
// native/swapped storage and signed strides supported. Labels use the existing
// exact find_boundaries domain; ordinary roundedwide/floatlabels remain open.
// Colors are prepared float64 values with explicit broadcast shape. No provider
// is called, no FP masks/FMA/reassociation, no arbitrary NumPy type engine.
// Workers0 captures the existing default1 global budget ONCE. Positive workers
// are an already captured budget. Axis recurrence is ordered serial; independent
// boundary/copy/store spans may use the captured budget.
MarkImage mark_boundaries(const ComponentsView& image,const ComponentsView& labels,
    const MarkBoundariesParameters& parameters={},int workers=0);
// Same owned stages expose source callback/error phases to the Python adapter;
// they are not a substitute for the complete operation above.
MarkImage mark_boundaries_prepare(const ComponentsView&,bool subpixel,MarkCubicProfile,int workers);
struct MarkMask {std::vector<std::size_t> shape;std::vector<std::uint8_t> values;};
MarkMask mark_boundaries_mask(const ComponentsView&,const detail::FindBoundaryParameters&,int workers);
MarkMask mark_boundaries_outline(const MarkMask&);
void mark_boundaries_paint_into(void*,PixelType,const std::vector<std::size_t>&,const std::vector<std::ptrdiff_t>&,std::size_t capacity,const MarkMask&,const MarkColor&);
void mark_boundaries_paint(MarkImage&,const MarkMask&,const MarkColor&);
void mark_boundaries_finish(MarkImage&,const ComponentsView&,const MarkBoundariesParameters&,int workers);
}
