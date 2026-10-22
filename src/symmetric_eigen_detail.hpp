#pragma once
#include "skimagexx/symmetric_eigen.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace skimagexx::detail {
constexpr double eigen_tolerance=4*std::numeric_limits<double>::epsilon();
template<class T> double eigen_limit(std::size_t dimension,bool pair) {
    if(dimension==1)return double(std::numeric_limits<T>::max());
    if(pair && dimension==2)return std::sqrt(double(std::numeric_limits<T>::max()))/8;
    return double(std::numeric_limits<T>::max())/(16*dimension);
}
// The matrix is normalized before iteration. Work storage is reused for every
// pixel; no per-matrix allocation or eigenvector construction is needed.
inline bool eigen_jacobi(double* matrix,double* values,std::size_t n) {
    bool converged=n==1;
    for(int sweep=0;sweep<48 && !converged;++sweep){
        for(std::size_t p=0;p<n;++p)for(std::size_t q=p+1;q<n;++q){
            const auto off=matrix[p*n+q];if(std::abs(off)<=eigen_tolerance)continue;
            const auto theta=(matrix[q*n+q]-matrix[p*n+p])/(2*off);
            const auto t=std::copysign(1.,theta)/(std::abs(theta)+std::sqrt(1+theta*theta));
            const auto c=1/std::sqrt(1+t*t),s=t*c;
            matrix[p*n+p]-=t*off;matrix[q*n+q]+=t*off;
            matrix[p*n+q]=matrix[q*n+p]=0.;
            for(std::size_t k=0;k<n;++k)if(k!=p && k!=q){
                const auto a=matrix[k*n+p],b=matrix[k*n+q];
                matrix[k*n+p]=matrix[p*n+k]=c*a-s*b;
                matrix[k*n+q]=matrix[q*n+k]=s*a+c*b;
            }
        }
        double maximum=0.;for(std::size_t p=0;p<n;++p)for(std::size_t q=p+1;q<n;++q)maximum=std::max(maximum,std::abs(matrix[p*n+q]));
        converged=maximum<=eigen_tolerance;
    }
    if(!converged)return false;
    for(std::size_t d=0;d<n;++d)values[d]=matrix[d*n+d];
    // Stable ascending order, then reversed like NumPy's LAPACK result.
    // Strict comparisons preserve the signs of equal zero eigenvalues.
    for(std::size_t d=1;d<n;++d){const auto value=values[d];auto j=d;
        for(;j && values[j-1]>value;--j)values[j]=values[j-1];values[j]=value;}
    std::reverse(values,values+n);return true;
}
}
namespace skimagexx::simd {
std::size_t symmetric_eigenvalues(const void* const*,void*,std::size_t,std::size_t,
    std::size_t,std::size_t,bool,bool,bool,bool&);
}
