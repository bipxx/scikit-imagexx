// B-spline pole recurrences follow SciPy v1.18.1 ni_splines.c (BSD).
// See third_party/scipy-ndimage-LICENSE.txt. SIMD lanes are independent lines;
// each lane preserves scalar recurrence order, including legacy endpoint aliasing.
#include "spline_detail.hpp"
#include <cmath>
#include <immintrin.h>

namespace skimagexx::simd {
namespace {
struct Four {
    __m256d value;
    Four(double x):value(_mm256_set1_pd(x)) {}
    Four(__m256d x):value(x) {}
};
Four operator+(Four a,Four b) {return _mm256_add_pd(a.value,b.value);}
Four operator-(Four a,Four b) {return _mm256_sub_pd(a.value,b.value);}
Four operator*(Four a,Four b) {return _mm256_mul_pd(a.value,b.value);}
Four operator/(Four a,Four b) {return _mm256_div_pd(a.value,b.value);}
}
void spline_filter_four(double* data,std::size_t length,std::size_t stride,
                        const double* poles,int npoles,GaussianBorder border,bool legacy) {
    const auto get=[&](std::size_t i) {return Four(_mm256_loadu_pd(data+i*stride));};
    const auto put=[&](std::size_t i,Four v) {_mm256_storeu_pd(data+i*stride,v.value);};
    double gain=1.;
    for(int k=0;k<npoles;++k)gain*=(1.-poles[k])*(1.-1./poles[k]);
    for(std::size_t i=0;i<length;++i)put(i,get(i)*gain);
    for(int pole=0;pole<npoles;++pole) {
        const double z=poles[pole];
        double zi=z;
        if(border==GaussianBorder::mirror || border==GaussianBorder::constant) {
            const double zn=std::pow(z,static_cast<std::ptrdiff_t>(length)-1);
            auto sum=get(0)+zn*get(length-1);
            for(std::size_t i=1;i<length-1;++i) {
                sum=sum+zi*(get(i)+zn*get(length-1-i));zi*=z;
            }
            put(0,sum/(1-zn*zn));
        } else if(border==GaussianBorder::wrap) {
            auto sum=get(0);
            for(std::size_t i=1;i<length;++i) {sum=sum+zi*get(length-i);zi*=z;}
            put(0,sum/(1-zi));
        } else {
            const double zn=std::pow(z,static_cast<std::ptrdiff_t>(length));
            const auto original=get(0);
            auto sum=original+zn*get(length-1);
            if(legacy)put(0,sum);
            for(std::size_t i=1;i<length;++i) {
                sum=sum+zi*(get(i)+zn*get(length-1-i));zi*=z;
                if(legacy)put(0,sum);
            }
            put(0,legacy ? sum*(z/(1-zn*zn))+original : original+sum*z/(1-zn*zn));
        }
        for(std::size_t i=1;i<length;++i)put(i,get(i)+z*get(i-1));
        if(border==GaussianBorder::mirror || border==GaussianBorder::constant) {
            put(length-1,(z*get(length-2)+get(length-1))*z/(z*z-1));
        } else if(border==GaussianBorder::wrap) {
            auto sum=get(length-1);zi=z;
            for(std::size_t i=0;i<length-1;++i) {sum=sum+zi*get(i);zi*=z;}
            put(length-1,sum*(z/(zi-1)));
        } else {
            put(length-1,get(length-1)*(z/(z-1)));
        }
        for(std::size_t i=length-1;i-->0;)put(i,z*(get(i+1)-get(i)));
    }
}
} // namespace skimagexx::simd
