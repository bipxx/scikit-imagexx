#pragma once
#include "skimagexx/symmetric_eigen.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace skimagexx::detail {
// Ascending DLASRT followed by NumPy's reversal. Adapted from Reference-LAPACK
// v3.12.1 SRC/dlasrt.f; see third_party/lapack-LICENSE.txt for its BSD license.
// The partition order matters for equal values with different zero signs.
inline void eigen_sort_descending(double* values,std::size_t n){
    using Index=std::ptrdiff_t;
    if(n<2)return;
    std::pair<Index,Index> stack[32];std::size_t depth=1;
    stack[0]={0,Index(n)-1};
    while(depth){
        const auto [start,end]=stack[--depth];
        if(end-start<=20){
            for(auto i=start+1;i<=end;++i){
                for(auto j=i;j>start && values[j]<values[j-1];--j)std::swap(values[j],values[j-1]);
            }
        }else{
            const auto first=values[start],last=values[end],middle=values[(start+end)/2];
            const auto pivot=first<last ? (middle<first ? first : middle<last ? middle : last)
                                        : (middle<last ? last : middle<first ? middle : first);
            auto left=start-1,right=end+1;
            while(true){
                do{--right;}while(values[right]>pivot);
                do{++left;}while(values[left]<pivot);
                if(left>=right)break;
                std::swap(values[left],values[right]);
            }
            if(right-start>end-right-1){stack[depth++]={start,right};stack[depth++]={right+1,end};}
            else{stack[depth++]={right+1,end};stack[depth++]={start,right};}
        }
    }
    std::reverse(values,values+n);
}
constexpr double eigen_tolerance=4*std::numeric_limits<double>::epsilon();
template<class T> double eigen_limit(std::size_t dimension,bool pair) {
    if(dimension==1)return double(std::numeric_limits<T>::max());
    if(pair && dimension==2)return std::sqrt(double(std::numeric_limits<T>::max()))/8;
    return double(std::numeric_limits<T>::max())/(16*dimension);
}
// The matrix is normalized before iteration. Work storage is reused for every
// pixel; no per-matrix allocation or eigenvector construction is needed.
inline bool eigen_jacobi(double* matrix,double* values,std::size_t n) {
    if(n>32){
        bool diagonal=true;
        for(std::size_t r=0;r<n;++r)for(std::size_t c=r+1;c<n;++c)diagonal=diagonal && matrix[r*n+c]==0.;
        // Reference-LAPACK's lower-triangle reduction updates the trailing
        // block after column 32, even for an already diagonal matrix. Its
        // zero BLAS update clears negative diagonal zeros in that block.
        if(diagonal)for(std::size_t d=32;d<n;++d)if(matrix[d*n+d]==0.)matrix[d*n+d]=0.;
    }
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
    eigen_sort_descending(values,n);return true;
}
}
namespace skimagexx::simd {
std::size_t symmetric_eigenvalues(const void* const*,void*,std::size_t,std::size_t,
    std::size_t,std::size_t,bool,bool,bool,bool&);
}
