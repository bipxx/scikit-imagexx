#include "skimagexx/projective.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <cstring>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace skimagexx {
namespace {
double load(const double* data,std::ptrdiff_t row,std::ptrdiff_t column,std::size_t i,std::size_t j) {
    double value;std::memcpy(&value,reinterpret_cast<const unsigned char*>(data)+std::ptrdiff_t(i)*row+std::ptrdiff_t(j)*column,8);return value;
}
}
void projective_apply_into(const double* points,std::ptrdiff_t point_row,std::ptrdiff_t point_column,
    const double* matrix,std::ptrdiff_t matrix_row,std::ptrdiff_t matrix_column,double* output,
    std::size_t count,std::size_t dimensions) {
    (void)projective_apply_checked_into(points,point_row,point_column,matrix,matrix_row,matrix_column,output,count,dimensions);
}
bool projective_apply_checked_into(const double* points,std::ptrdiff_t point_row,std::ptrdiff_t point_column,
    const double* matrix,std::ptrdiff_t matrix_row,std::ptrdiff_t matrix_column,double* output,
    std::size_t count,std::size_t dimensions) {
    const auto max=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/8;
    if(!dimensions || dimensions>=max || count>max/dimensions)throw std::overflow_error("projective shape overflow");
    if(!count)return false;
    if(!points || !matrix || !output || points==output || matrix==output)throw std::invalid_argument("nonnull disjoint projective storage required");
    std::atomic<bool> provider{false};
    detail::partition(count,detail::active_workers(count*dimensions),[&](auto begin,auto end,auto) {
        std::fenv_t previous;std::feholdexcept(&previous);
        std::size_t first=begin;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())first+=simd::projective_apply(reinterpret_cast<const double*>(reinterpret_cast<const unsigned char*>(points)+std::ptrdiff_t(begin)*point_row),point_row,point_column,matrix,matrix_row,matrix_column,output+begin*dimensions,end-begin,dimensions);
#endif
        for(auto i=first;i<end;++i) {
            // The common 2/3/4-D models reuse each coordinate across every
            // homogeneous dot product, avoiding repeated strided loads.
            double cached[4];
            if(dimensions<=4)for(std::size_t j=0;j<dimensions;++j)cached[j]=load(points,point_row,point_column,i,j);
            const auto coordinate=[&](std::size_t j){return dimensions<=4 ? cached[j]:load(points,point_row,point_column,i,j);};
            double divisor=0.;
            for(std::size_t j=0;j<dimensions;++j)divisor+=coordinate(j)*load(matrix,matrix_row,matrix_column,dimensions,j);
            divisor+=load(matrix,matrix_row,matrix_column,dimensions,dimensions);
            if(divisor==0.)divisor=std::numeric_limits<double>::epsilon();
            for(std::size_t d=0;d<dimensions;++d) {
                double value=0.;
                for(std::size_t j=0;j<dimensions;++j)value+=coordinate(j)*load(matrix,matrix_row,matrix_column,d,j);
                value+=load(matrix,matrix_row,matrix_column,d,dimensions);
                output[i*dimensions+d]=value/divisor;
            }
        }
        bool exceptional=std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW)!=0;
        for(auto i=begin*dimensions;i<end*dimensions && !exceptional;++i)exceptional=!std::isfinite(output[i]);
        std::fesetenv(&previous);
        if(exceptional)provider.store(true,std::memory_order_relaxed);
    });
    return provider.load(std::memory_order_relaxed);
}
}
