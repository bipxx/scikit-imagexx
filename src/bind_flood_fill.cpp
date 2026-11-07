#include "skimagexx/flood_fill.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
template<class T>skimagexx::FloodMask run(const py::array& a,const skimagexx::FloodGeometry& g,bool tol,py::object low,py::object high){
    return skimagexx::flood(static_cast<const T*>(a.data()),g,tol,tol?low.cast<T>():T{},tol?high.cast<T>():T{});
}
}
void bind_flood_fill(py::module_& m){
 m.def("flood_mask",[](py::array a,std::vector<std::ptrdiff_t> seed,py::array_t<std::uint8_t,py::array::c_style> footprint,bool fortran,bool tol,py::object low,py::object high){
    skimagexx::FloodGeometry g{{a.shape(),a.shape()+a.ndim()},{a.strides(),a.strides()+a.ndim()},std::move(seed),{footprint.shape(),footprint.shape()+footprint.ndim()},{footprint.data(),footprint.data()+footprint.size()},fortran};
    skimagexx::FloodMask result;const char kind=a.dtype().kind();const auto bytes=a.itemsize();
    if(!py::bool_(a.dtype().attr("isnative")))throw py::value_error("Big-endian buffer not supported on little-endian compiler");
#define CASE(K,N,T) if(kind==K&&bytes==N)result=run<T>(a,g,tol,low,high);else
    CASE('b',1,std::uint8_t) CASE('u',1,std::uint8_t) CASE('u',2,std::uint16_t) CASE('u',4,std::uint32_t)
    CASE('u',8,std::uint64_t) CASE('i',1,std::int8_t) CASE('i',2,std::int16_t)
    CASE('i',4,std::int32_t) CASE('i',8,std::int64_t) CASE('f',4,float) CASE('f',8,double)
    {if(kind=='f'&&bytes==2)throw py::type_error("dtype of `image` is float16 which is not supported, try upcasting to float32");throw py::type_error("No matching signature found");}
#undef CASE
    auto shape=result.shape;for(std::size_t i=0;i<shape.size();++i)shape[i]+=2*result.padding[i];
    py::array output(py::dtype::of<std::uint8_t>(),shape,result.strides);
    std::memcpy(output.mutable_data(),result.storage.data(),result.storage.size());return output;
 },py::arg("image").noconvert(),py::arg("seed"),py::arg("footprint").noconvert(),py::arg("fortran"),py::arg("tolerance"),py::arg("low"),py::arg("high"));
 m.def("flood_assign",[](py::array output,py::array mask,py::array values){
    if(!output.writeable())throw py::value_error("assignment destination is read-only");
    if(output.ndim()!=mask.ndim()||output.itemsize()!=values.itemsize()||!values.owndata()||!(values.flags()&py::array::c_style))throw py::value_error("flood assignment descriptor mismatch");
    for(py::ssize_t a=0;a<output.ndim();++a)if(output.shape(a)!=mask.shape(a))throw py::value_error("flood mask shape mismatch");
    const auto kind=output.dtype().kind();
    if(output.dtype().has_fields()||!(kind=='b'||kind=='i'||kind=='u'||kind=='f'||kind=='c')||output.itemsize()>16)throw py::type_error("flood assignment requires numeric POD dtype");
    if(mask.itemsize()!=1||mask.dtype().kind()!='b'||!output.dtype().is(values.dtype()))throw py::type_error("flood assignment requires matching numeric dtype");
    py::ssize_t selected=0;
    for(py::ssize_t k=0;k<output.size();++k){auto i=k;py::ssize_t target=0,flag=0;
      for(py::ssize_t a=output.ndim();a-->0;){auto c=i%output.shape(a);i/=output.shape(a);target+=c*output.strides(a);flag+=c*mask.strides(a);}
      if(*(static_cast<const char*>(mask.data())+flag)){if(selected>=values.size())throw py::value_error("flood assignment value count mismatch");
        std::memcpy(static_cast<char*>(output.mutable_data())+target,static_cast<const char*>(values.data())+selected*values.itemsize(),output.itemsize());++selected;}}
    if(selected!=values.size())throw py::value_error("flood assignment value count mismatch");
 },py::arg("output").noconvert(),py::arg("mask").noconvert(),py::arg("values").noconvert());
}
