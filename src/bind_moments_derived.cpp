#include "bind_array_storage.hpp"
#include "half.hpp"
#include "moments_hu.hpp"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cfenv>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace py=pybind11;
namespace {
class Capture {
    std::fenv_t saved_;
public:
    Capture(){if(std::feholdexcept(&saved_))throw std::runtime_error("cannot capture normalized moments floating environment");}
    ~Capture(){std::fesetenv(&saved_);}
    unsigned flags()const {
        const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
        return ((f&FE_DIVBYZERO)?1u:0u)|((f&FE_OVERFLOW)?2u:0u)|
               ((f&FE_UNDERFLOW)?4u:0u)|((f&FE_INVALID)?8u:0u);
    }
};
float half_round(float value) {
    std::uint32_t word;std::memcpy(&word,&value,4);
    const auto encoded=skimagexx::detail::encode_half(value);const auto exponent=word&0x7f800000u;
    if(exponent!=0x7f800000u){
        if((encoded&0x7fffu)==0x7c00u)std::feraiseexcept(FE_OVERFLOW);
        else if(exponent<=0x38000000u && skimagexx::detail::decode_half(encoded)!=value)std::feraiseexcept(FE_UNDERFLOW);
    }
    return skimagexx::detail::decode_half(encoded);
}
template<class T> py::tuple divide(const py::object&a,const py::object&b,bool half=false) {
    // Conversions are under the GIL, BEFORE capture. Only scalar copies enter
    // native arithmetic; no borrowed array allocation is read or released.
    const T first=a.cast<T>(),denominator=b.cast<T>();T value;unsigned flags;
    {
        Capture capture;value=first/denominator;
        if constexpr(std::is_same_v<T,float>){if(half)value=half_round(value);}
        flags=capture.flags();
    }
    return py::make_tuple(value,flags);
}
template<class T> void check_hu(const py::array&input) {
    skimagexx::bindings::check_array_storage(input);
    if(!py::array_t<T,0>::check_(input))throw py::type_error("Hu moments require native float32 or float64");
    if(input.ndim()!=2)throw py::type_error("No matching signature found");
    if(!input.writeable())throw py::value_error("buffer source array is read-only");
    if(input.shape(0)<4 || input.shape(1)<4){
        PyErr_SetString(PyExc_NotImplementedError,"Hu moments require an assessed extent of at least 4x4");throw py::error_already_set();
    }
}
template<class T> py::array hu(const py::array&input) {
    py::array_t<T> result(7);
    T* output=result.mutable_data();
    // Validate after the last allocation, immediately before native reads.
    check_hu<T>(input);
    const auto row=input.strides(0),col=input.strides(1);
    alignas(T) T packed[16];const T* data;std::ptrdiff_t rs,cs;
    if(reinterpret_cast<std::uintptr_t>(input.data())%alignof(T) || row%sizeof(T) || col%sizeof(T)) {
        const auto bytes=static_cast<const char*>(input.data());
        for(std::ptrdiff_t r=0;r<4;++r)for(std::ptrdiff_t c=0;c<4;++c)
            std::memcpy(packed+r*4+c,bytes+r*row+c*col,sizeof(T));
        data=packed;rs=4;cs=1;
    } else {data=static_cast<const T*>(input.data());rs=row/std::ptrdiff_t(sizeof(T));cs=col/std::ptrdiff_t(sizeof(T));}
    skimagexx::detail::moments_hu(data,rs,cs,output);
    return result;
}
}

void bind_moments_derived(py::module_&module) {
    module.def("moments_normalize",[](const py::object&a,const py::object&b,char dtype){
        if(dtype=='e')return divide<float>(a,b,true);
        if(dtype=='f')return divide<float>(a,b);
        if(dtype=='d')return divide<double>(a,b);
        if(dtype=='F')return divide<std::complex<float>>(a,b);
        if(dtype=='D')return divide<std::complex<double>>(a,b);
        throw py::type_error("normalized moments require a standard floating scalar type");
    },py::arg("numerator"),py::arg("denominator"),py::arg("dtype"));
    module.def("moments_hu",[](const py::array&input){
        return py::array_t<float,0>::check_(input)?hu<float>(input):hu<double>(input);
    },py::arg("nu").noconvert());
}
