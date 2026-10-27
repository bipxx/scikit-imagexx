#include "mesh_surface_area.hpp"
#include "execution.hpp"
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace skimagexx::detail {
namespace {
template<class T>T checked(T value,bool& accepted){
    using Word=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
    Word word;std::memcpy(&word,&value,sizeof(word));
    constexpr Word sign=Word(1)<<(sizeof(T)*8-1);
    constexpr Word exponent=sizeof(T)==4?Word(0x7f800000u):Word(UINT64_C(0x7ff0000000000000));
    const auto magnitude=word&~sign;
    accepted&=magnitude==0||((magnitude&exponent)!=0&&(magnitude&exponent)!=exponent);
    return value;
}
template<class T>T face_norm(const T* v,bool& accepted){
    T a[3],b[3],cross[3],squares[3];
    for(std::size_t j=0;j<3;++j){
        const auto first=checked(v[j],accepted);
        a[j]=checked(T(first-checked(v[3+j],accepted)),accepted);
        b[j]=checked(T(first-checked(v[6+j],accepted)),accepted);
    }
    for(std::size_t j=0;j<3;++j){
        const auto k=(j+1)%3,l=(j+2)%3;
        const auto first=checked(T(a[k]*b[l]),accepted),second=checked(T(a[l]*b[k]),accepted);
        cross[j]=checked(T(first-second),accepted);
        squares[j]=checked(T(cross[j]*cross[j]),accepted);
    }
    const auto sum=checked(T(checked(T(squares[0]+squares[1]),accepted)+squares[2]),accepted);
    return checked(T(std::sqrt(sum)),accepted);
}
template<class T>ColocScalar area(const MeshAreaInput& input,ColocReductionPlan plan,int workers){
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(input.vertices_per_face>limit/(3*sizeof(T))||
       (input.faces&&input.faces>limit/(input.vertices_per_face*3*sizeof(T))))
        throw std::overflow_error("mesh gathered extent overflow");
    // The frozen engineering envelope requires (F+16)*u < 1/4.
    if(input.faces>=(std::uint64_t(1)<<(std::numeric_limits<T>::digits-2))-16)
        throw ColocProviderRequired("mesh reduction length is a provider frontier");
    if(!coloc_fp_supported())throw ColocProviderRequired("mesh floating environment is a provider frontier");
    const auto stride=input.vertices_per_face*3;
    const auto team=input.faces>=parallel_min?workers:1;
    std::vector<T> norms(input.faces);
    std::vector<unsigned char> refused(static_cast<std::size_t>(team),0);
    const auto* data=static_cast<const T*>(input.data);const bool vectorized=has_avx2();
    partition(input.faces,team,[&](auto begin,auto end,auto worker){
        std::fenv_t saved;
        if(!coloc_fp_supported()||std::feholdexcept(&saved)){refused[worker]=1;return;}
        bool accepted=true;std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
        if(vectorized&&begin<end)done=simd::mesh_face_norms(data+begin*stride,input.type,end-begin,stride,norms.data()+begin,accepted);
#else
        (void)vectorized;
#endif
        for(auto i=begin+done;i<end;++i)norms[i]=face_norm(data+i*stride,accepted);
        const auto flags=std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
        refused[worker]=!accepted||flags;
        std::fesetenv(&saved);
    });
    for(auto value:refused)if(value)throw ColocProviderRequired("mesh exceptional face arithmetic is a provider frontier");
    const std::size_t shape=input.faces;const std::ptrdiff_t step=sizeof(T);
    const ColocView view{norms.data(),norms.data(),norms.size()*sizeof(T),{&shape,1},{&step,1},input.type,false};
    auto result=coloc_sum(view,plan);T sum;std::memcpy(&sum,&result.bits,sizeof(T));
    bool accepted=true;checked(sum,accepted);
    if(result.fp_status||!accepted)throw ColocProviderRequired("mesh exceptional reduction is a provider frontier");
    return result;
}
}
ColocScalar mesh_surface_area(const MeshAreaInput& input,ColocReductionPlan plan,int workers){
    if(workers<1||workers>1024)throw std::invalid_argument("mesh worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
    if(input.vertices_per_face<3||(input.faces&&!input.data))throw std::invalid_argument("invalid prepared mesh shape or buffer");
    if(input.type==ColocType::float32)return area<float>(input,plan,workers);
    if(input.type==ColocType::float64)return area<double>(input,plan,workers);
    throw std::invalid_argument("prepared mesh requires float32 or float64");
}
}
