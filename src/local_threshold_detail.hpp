#pragma once
#include "skimagexx/local_threshold.hpp"
#include <algorithm>
#include <cmath>
namespace skimagexx::detail {
template<class F> F local_divide(double numerator,std::size_t area,bool wide) {
    const F rounded=F(numerator);
    return wide ? F(double(rounded)/double(area)) : F(rounded/F(area));
}
template<class F> double local_formula(F mean,F sd,LocalThreshold method,const LocalThresholdParameters& p) {
    if(method==LocalThreshold::niblack) {
        if(p.k_double)return double(mean)-p.k*double(sd);
        return double(F(mean-F(F(p.k)*sd)));
    }
    if(p.r_double) {
        const double inner=double(sd)/p.r-1.0;
        return double(mean)*(1.0+p.k*inner);
    }
    const F inner=F(F(sd/F(p.r))-F(1));
    if(p.k_double)return double(mean)*(1.0+p.k*double(inner));
    return double(F(mean*F(F(1)+F(F(p.k)*inner))));
}
}
