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
template<int Bytes>void assign_values(py::array& output,const py::array& mask,const py::array& values){
    py::ssize_t selected=0;
    const auto count=output.size();
    if(count){
      const auto rank=output.ndim();
      const auto width=rank?output.shape(rank-1):1;
      const auto lines=count/width;
      const auto target_step=rank?output.strides(rank-1):0;
      const auto flag_step=rank?mask.strides(rank-1):0;
      const py::ssize_t bytes=Bytes?Bytes:output.itemsize();
      auto* destination=static_cast<char*>(output.mutable_data());
      const auto* flags=static_cast<const char*>(mask.data());
      const auto* replacement=static_cast<const char*>(values.data());
      std::vector<py::ssize_t> index(rank?rank-1:0,0);
      py::ssize_t target_base=0,flag_base=0;
      // Keep logical C order, including F/reversed/zero strides and live
      // mask aliases. No buffered mask or parallel/SIMD store reordering:
      // each selected value is read immediately before its ordered copy.
      for(py::ssize_t line=0;line<lines;++line){
        auto target=target_base,flag=flag_base;
        for(py::ssize_t x=0;x<width;++x){
          if(flags[flag]){if(selected>=values.size())throw py::value_error("flood assignment value count mismatch");
            std::memcpy(destination+target,replacement+selected*bytes,bytes);++selected;}
          if(x+1<width){target+=target_step;flag+=flag_step;}
        }
        if(line+1<lines)for(py::ssize_t a=rank-1;a-->0;){
          if(++index[a]<output.shape(a)){target_base+=output.strides(a);flag_base+=mask.strides(a);break;}
          index[a]=0;
          target_base-=(output.shape(a)-1)*output.strides(a);
          flag_base-=(mask.shape(a)-1)*mask.strides(a);
        }
      }
    }
    if(selected!=values.size())throw py::value_error("flood assignment value count mismatch");
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
    // Dispatch once so ordinary POD copies are fixed-size inline moves.
    // The generic path preserves any other descriptor admitted above.
    switch(output.itemsize()){
      case 1:return assign_values<1>(output,mask,values);
      case 2:return assign_values<2>(output,mask,values);
      case 4:return assign_values<4>(output,mask,values);
      case 8:return assign_values<8>(output,mask,values);
      case 16:return assign_values<16>(output,mask,values);
      default:return assign_values<0>(output,mask,values);
    }
 },py::arg("output").noconvert(),py::arg("mask").noconvert(),py::arg("values").noconvert());
}
