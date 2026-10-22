#include "skimagexx/gabor_filter.hpp"
#include "skimagexx/gabor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace py=pybind11;
namespace {
skimagexx::PixelType type_of(py::dtype dtype) {
    using skimagexx::PixelType;
    const auto kind=dtype.kind();const auto size=dtype.itemsize();
    if(kind=='b')return PixelType::boolean;
    if(kind=='f' && size==4)return PixelType::float32;
    if(kind=='f' && size==8)return PixelType::float64;
#define INTEGER(n,ut,st) if(size==n){if(kind=='u')return PixelType::ut;if(kind=='i')return PixelType::st;}
    INTEGER(1,uint8,int8) INTEGER(2,uint16,int16) INTEGER(4,uint32,int32) INTEGER(8,uint64,int64)
#undef INTEGER
    throw py::type_error("Gabor requires a real native NumPy dtype");
}
}
void bind_gabor(py::module_& m) {
    m.def("gabor_kernel_accurate",[](std::size_t height,std::size_t width,double ct,double st,
            double sx2,double sy2,double frequency,double offset,double normalization,bool single){
        if(height>std::size_t(PY_SSIZE_T_MAX) || width>std::size_t(PY_SSIZE_T_MAX))throw py::value_error("Gabor dimensions overflow");
        py::array output(single ? py::dtype::of<std::complex<float>>() : py::dtype::of<std::complex<double>>(),
            std::vector<py::ssize_t>{py::ssize_t(height),py::ssize_t(width)});
        {py::gil_scoped_release release;skimagexx::gabor_kernel_accurate_into(output.mutable_data(),single,height,width,
            {ct,st,sx2,sy2,frequency,offset,normalization});}
        return output;
    });
    m.def("gabor_filter",[](py::array image,py::array kernel,int border,double cval,bool collision){
        if(image.ndim()!=2 || kernel.ndim()!=2)throw py::value_error("Gabor image and kernel must be two-dimensional");
        auto input=py::array::ensure(image,collision ? 0 : py::array::c_style);
        auto weights=py::array::ensure(kernel,py::array::c_style);
        if(!input || !weights)throw py::type_error("Gabor buffer preparation failed");
        if(!input.dtype().attr("isnative").cast<bool>() || !weights.dtype().attr("isnative").cast<bool>())throw py::type_error("Gabor core requires native byte order");
        const auto type=type_of(input.dtype());
        if(reinterpret_cast<std::uintptr_t>(input.data())%std::size_t(input.itemsize()))input=py::array::ensure(input.attr("copy")(),py::array::c_style);
        const bool single=weights.dtype().kind()=='c' && weights.itemsize()==8;
        if(!single && !(weights.dtype().kind()=='c' && weights.itemsize()==16))throw py::type_error("Gabor kernel must be complex64 or complex128");
        if(reinterpret_cast<std::uintptr_t>(weights.data())%std::size_t(weights.itemsize()/2))weights=py::array::ensure(weights.attr("copy")(),py::array::c_style);
        const auto dtype=py::dtype(input.dtype().attr("name").cast<std::string>());
        const std::vector<py::ssize_t> shape{input.shape(0),input.shape(1)};
        py::array real(dtype,shape),imaginary(dtype,shape);
        {py::gil_scoped_release release;skimagexx::gabor_filter_into(input.data(),type,real.mutable_data(),imaginary.mutable_data(),
            std::size_t(input.shape(0)),std::size_t(input.shape(1)),weights.data(),single,
            std::size_t(weights.shape(0)),std::size_t(weights.shape(1)),skimagexx::GaussianBorder(border),cval,collision);}
        return py::make_tuple(real,imaginary);
    },py::arg("image"),py::arg("kernel"),py::arg("border"),py::arg("cval"),py::arg("zero_stride_collision")=false);
}
