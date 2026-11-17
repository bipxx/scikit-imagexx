#include "skimagexx/flood_fill.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <limits>
namespace py=pybind11;
namespace {
template<class T>skimagexx::FloodMask run(const py::array& a,const skimagexx::FloodGeometry& g,bool tol,py::object low,py::object high){
    return skimagexx::flood(static_cast<const T*>(a.data()),g,tol,tol?low.cast<T>():T{},tol?high.cast<T>():T{});
}
// Conservative positive byte intervals exclude live input/output aliases.
// Unsupported descriptors retain the ordered element-by-element fallback.
bool positive_interval(const py::array& a,std::uintptr_t& begin,std::uintptr_t& end){
    const auto limit=std::uintptr_t(std::numeric_limits<py::ssize_t>::max());
    auto span=std::uintptr_t(a.itemsize());
    for(py::ssize_t axis=0;axis<a.ndim();++axis){
        if(a.shape(axis)<=0||a.strides(axis)<0)return false;
        const auto n=std::uintptr_t(a.shape(axis)-1),step=std::uintptr_t(a.strides(axis));
        if(n&&step>(limit-span)/n)return false;
        span+=n*step;
    }
    begin=reinterpret_cast<std::uintptr_t>(a.data());
    if(span>std::numeric_limits<std::uintptr_t>::max()-begin)return false;
    end=begin+span;return true;
}
void next_mask_line(const py::array& mask,py::ssize_t inner,
                    std::vector<py::ssize_t>& index,py::ssize_t& base){
    for(py::ssize_t axis=mask.ndim();axis-->0;){
        if(axis==inner)continue;
        if(++index[axis]<mask.shape(axis)){base+=mask.strides(axis);return;}
        index[axis]=0;base-=(mask.shape(axis)-1)*mask.strides(axis);
    }
}
py::ssize_t selected_count(const py::array& mask){
    const auto* flags=static_cast<const char*>(mask.data());
    if(!mask.ndim())return flags[0]!=0;
    if(mask.flags()&(py::array::c_style|py::array::f_style))
        return std::count_if(flags,flags+mask.size(),[](char value){return value!=0;});
    auto inner=mask.ndim()-1;
    if(mask.strides(inner)!=1){inner=0;if(mask.strides(inner)!=1)return -1;}
    const auto width=mask.shape(inner),lines=mask.size()/width;
    std::vector<py::ssize_t> index(mask.ndim(),0);
    py::ssize_t base=0,selected=0;
    for(py::ssize_t line=0;line<lines;++line){
        selected+=std::count_if(flags+base,flags+base+width,[](char value){return value!=0;});
        if(line+1<lines)next_mask_line(mask,inner,index,base);
    }
    return selected;
}
bool assign_disjoint_runs(py::array& output,const py::array& mask,const py::array& values){
    if(!output.size()||!(output.flags()&py::array::c_style))return false;
    std::uintptr_t ob,oe,mb,me,vb,ve;
    if(!positive_interval(output,ob,oe)||!positive_interval(mask,mb,me)||
       !positive_interval(values,vb,ve)||!(oe<=mb||me<=ob)||!(oe<=vb||ve<=ob))return false;
    // No store can change mask/RHS and this synchronous path has no callback.
    // Admit the exact count first. A mismatch keeps the original partial writes
    // before its error, rather than failing early or buffering aliased inputs.
    const auto selected=selected_count(mask);
    if(selected<0||selected!=values.size())return false;
    auto* destination=static_cast<char*>(output.mutable_data());
    const auto* replacement=static_cast<const char*>(values.data());
    if(selected==output.size()){
        std::memcpy(destination,replacement,std::size_t(oe-ob));return true;
    }
    const auto inner=output.ndim()-1,width=output.shape(inner);
    const auto lines=output.size()/width,step=mask.strides(inner);
    const auto bytes=std::size_t(output.itemsize()),row_bytes=std::size_t(width)*bytes;
    const auto* flags=static_cast<const char*>(mask.data());
    std::vector<py::ssize_t> index(mask.ndim(),0);py::ssize_t base=0;
    for(py::ssize_t line=0;line<lines;++line){
        for(py::ssize_t x=0;x<width;){
            while(x<width&&!flags[base+x*step])++x;
            if(x==width)break;
            const auto first=x;
            if(step==1){
                const auto* row=flags+base;
                const auto* stop=static_cast<const char*>(std::memchr(row+x,0,std::size_t(width-x)));
                x=stop?stop-row:width;
            }else while(x<width&&flags[base+x*step])++x;
            if(x>first){const auto length=std::size_t(x-first)*bytes;
                std::memcpy(destination+std::size_t(line)*row_bytes+std::size_t(first)*bytes,
                            replacement,length);replacement+=length;}
        }
        if(line+1<lines)next_mask_line(mask,inner,index,base);
    }
    return true;
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
    if(assign_disjoint_runs(output,mask,values))return;
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
