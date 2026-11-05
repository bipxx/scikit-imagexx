#include "skimagexx/draw_ellipsoid.hpp"
#include "execution.hpp"
#include <cmath>
#include <limits>
#include <utility>

namespace skimagexx::detail {
bool draw_ellipsoid_prepare(const std::array<double,3>& axes,
    const std::array<double,3>& spacing,DrawEllipsoidPlan& output) {
    DrawEllipsoidPlan plan;
    plan.axes=axes; plan.spacing=spacing; plan.count=1;
    for (std::size_t d=0;d<3;++d) {
        const double a=axes[d],s=spacing[d];
        if (!std::isfinite(a)||!std::isfinite(s)||a<0x1p-128||s<0x1p-128||
            a>0x1p128||s>0x1p128) return false;
        double low=std::ceil(-a-s);
        const double high=std::floor((a+s)+1.0);
        if (std::fmod(high-low,2.0)==0.0) low-=1.0;
        const double probe_length=std::ceil((high-low)/s);
        if (!(probe_length>0.0)||probe_length>ellipsoid_max_elements) return false;
        const auto n=static_cast<std::size_t>(probe_length);
        const double second=low+s,delta=second-low;
        bool zero=false,negative=false;
        double largest_negative=-std::numeric_limits<double>::infinity();
        for (std::size_t i=0;i<n;++i) {
            const double value=i==0?low:(i==1?second:low+static_cast<double>(i)*delta);
            zero=zero||value==0.0;
            if (value<0.0) { negative=true; if(value>largest_negative) largest_negative=value; }
        }
        if (!zero) {
            if (!negative) return false;
            low-=largest_negative;
        }
        const double length=std::ceil((high-low)/s);
        if (!(length>0.0)||length>ellipsoid_max_elements) return false;
        const auto size=static_cast<std::size_t>(length);
        if (size>ellipsoid_max_elements/plan.count) return false;
        plan.count*=size; plan.shape[d]=size; plan.low[d]=low; plan.high[d]=high;
    }
    for(std::size_t d=0;d<3;++d) {
        auto& squared=plan.squared[d]; squared.resize(plan.shape[d]);
        const double s=spacing[d],low=plan.low[d],a=axes[d];
        for(std::size_t i=0;i<squared.size();++i) {
            const double coordinate=static_cast<double>(i)*s+low;
            const double ratio=coordinate/a;
            squared[i]=ratio*ratio;
        }
    }
    output=std::move(plan); return true;
}

void draw_ellipsoid_bool_into(const DrawEllipsoidPlan& p,std::uint8_t* out,int workers) {
    const auto ny=p.shape[1],nz=p.shape[2];
    const double* const z=p.squared[2].data();
    partition(p.shape[0]*ny,p.count>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto row=begin;row<end;++row) {
            const double xy=p.squared[0][row/ny]+p.squared[1][row%ny];
            auto* dst=out+row*nz;
            #if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd
            #endif
            for(std::size_t k=0;k<nz;++k) dst[k]=static_cast<std::uint8_t>(xy+z[k]<=1.0);
        }
    });
}
void draw_ellipsoid_levelset_into(const DrawEllipsoidPlan& p,double* out,int workers) {
    const auto ny=p.shape[1],nz=p.shape[2];
    const double* const z=p.squared[2].data();
    partition(p.shape[0]*ny,p.count>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto row=begin;row<end;++row) {
            const double xy=p.squared[0][row/ny]+p.squared[1][row%ny];
            auto* dst=out+row*nz;
            #if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd
            #endif
            for(std::size_t k=0;k<nz;++k) dst[k]=(xy+z[k])-1.0;
        }
    });
}
}
