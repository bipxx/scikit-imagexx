#include "skimagexx/corner_response.hpp"
#include "skimagexx/image.hpp"
#include "corner_response_detail.hpp"
#include "execution.hpp"
#include <atomic>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx { namespace {
template<class T,class O> bool execute(const void* const* fields,O* output,O* second,std::size_t count,CornerResponse method,double parameter){
    std::atomic<bool> valid{true};
    const auto work=count>std::numeric_limits<std::size_t>::max()/12 ? count : count*12;
    detail::partition(count,detail::active_workers(work),[&](auto begin,auto end,auto){
        bool supported=true;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin=simd::corner_response(fields,output,second,begin,end,std::is_same_v<T,float>,!std::is_same_v<T,O>,method,parameter,supported);
#endif
        if(!supported){valid.store(false,std::memory_order_relaxed);return;}
        for(;begin<end;++begin){O other{};
            if(!detail::corner_pixel(static_cast<const T*>(fields[0])[begin],static_cast<const T*>(fields[1])[begin],static_cast<const T*>(fields[2])[begin],output[begin],other,method,parameter)){
                valid.store(false,std::memory_order_relaxed);return;
            }
            if(second)second[begin]=other;
        }
    });
    return valid.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
bool corner_response_into(const void* const* fields,void* output,void* second,std::size_t count,bool single,CornerResponse method,double parameter,bool wide){
    if(unsigned(method)>unsigned(CornerResponse::foerstner) || (wide && method!=CornerResponse::harris && method!=CornerResponse::noble))throw std::invalid_argument("invalid corner response plan");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double))throw std::overflow_error("corner response size overflow");
    if(!count)return true;
    if(!fields || !output || !fields[0] || !fields[1] || !fields[2] || (method==CornerResponse::foerstner && !second))throw std::invalid_argument("nonnull corner response buffers required");
    if(method!=CornerResponse::foerstner)second=nullptr;
    if((method==CornerResponse::harris || method==CornerResponse::noble) && (!std::isfinite(parameter) || (single && !wide && std::abs(parameter)>std::numeric_limits<float>::max())))return false;
    if(single && !wide)return execute<float,float>(fields,static_cast<float*>(output),static_cast<float*>(second),count,method,parameter);
    if(single)return execute<float,double>(fields,static_cast<double*>(output),static_cast<double*>(second),count,method,parameter);
    return execute<double,double>(fields,static_cast<double*>(output),static_cast<double*>(second),count,method,parameter);
}
}
