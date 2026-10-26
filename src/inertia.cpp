#include "inertia_internal.hpp"
namespace skimagexx::detail {
namespace {
template<class T> void tensor(const T* diagonal,const T* mixed,std::size_t n,
                              T total,T mass,T* output) {
    for(std::size_t i=0;i<n;++i) {
        // Materialize each typed source operation before the caller tests FP.
        volatile T numerator=total-diagonal[i];
        volatile T quotient=numerator/mass;
        output[i*n+i]=quotient;
    }
    std::size_t pair=0;
    for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j,++pair) {
        volatile T numerator=-mixed[pair];
        volatile T first=numerator/mass;
        output[i*n+j]=first;
        volatile T second=numerator/mass;
        output[j*n+i]=second;
    }
}
}
void inertia_tensor(const float* d,const float* m,std::size_t n,float s,float mass,float* out) {
    tensor(d,m,n,s,mass,out);
}
void inertia_tensor(const double* d,const double* m,std::size_t n,double s,double mass,double* out) {
    tensor(d,m,n,s,mass,out);
}
}
