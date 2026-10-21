#pragma once
#include "skimagexx/stain.hpp"
#include "color_detail.hpp"

namespace skimagexx::detail {
struct StainSpec {
    StainTransform transform;
    const double* weights;
    std::size_t channels;
    bool intermediate32;
    double adjust;
};
const std::array<double,256>& stain_log_table();
template<class Input> ColorOutput<Input> stain_value(const Input* p,std::ptrdiff_t offset) {
    if constexpr (std::is_same_v<Input,bool>) return reinterpret_cast<const unsigned char*>(p)[offset]!=0;
    else return color_value(p[offset],0,false);
}
template<class Input> ColorOutput<Input> stain_log(ColorOutput<Input> value) {
    using T=ColorOutput<Input>;
    if (value<=T(1e-6)) value=T(1e-6);
    return T(std::log(value));
}
template<class Input,class Output> void stain_pixel(const Input* p,Output* output,const StainSpec& spec,std::ptrdiff_t channels) {
    Output values[3];
    for (int i=0;i<3;++i) {
        auto value=stain_value(p,std::ptrdiff_t(i)*channels);
        if (spec.transform==StainTransform::separate) value=stain_log<Input>(value);
        if (spec.intermediate32) {
            const float v=float(value),adjust=float(spec.adjust);
            values[i]=Output(spec.transform==StainTransform::separate ? float(v/-adjust) : -float(v*adjust));
        } else {
            const double v=double(value);
            values[i]=Output(spec.transform==StainTransform::separate ? v/-spec.adjust : -(v*spec.adjust));
        }
    }
    for (std::size_t j=0;j<spec.channels;++j) {
        const auto* w=spec.weights+3*j;
        Output value=Output(Output(Output(values[0]*Output(w[0]))+Output(values[1]*Output(w[1])))+Output(values[2]*Output(w[2])));
        if (spec.transform==StainTransform::combine) {
            value=Output(std::exp(value));
            output[j]=value<=0 ? Output(0) : value>=1 ? Output(1) : value;
        } else output[j]=value<=0 ? Output(0) : value;
    }
}
}
