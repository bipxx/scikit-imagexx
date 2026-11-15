#include "skimagexx/mark_boundaries.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <algorithm>
#include <limits>
namespace py=pybind11;
namespace {
using namespace skimagexx;
void storage(const py::array& a){
    if(!a.owndata()){bindings::check_array_storage(a);return;}
    if(!a.size())return;const auto limit=std::numeric_limits<std::uintptr_t>::max();std::uintptr_t low=0,high=0;
    for(py::ssize_t d=0;d<a.ndim();++d){const auto stride=a.strides(d);const auto magnitude=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);auto& extent=stride<0?low:high;const auto repeat=std::uintptr_t(a.shape(d)-1);if(repeat&&magnitude>(limit-extent)/repeat)throw py::value_error("mark extent overflow");extent+=repeat*magnitude;}
    const auto address=reinterpret_cast<std::uintptr_t>(a.data());if(low||high>limit-address||std::uintptr_t(a.itemsize())>limit-address-high||high+std::uintptr_t(a.itemsize())>std::uintptr_t(a.nbytes()))throw py::value_error("mark array escapes owning allocation");
}
ComponentsView view(const py::array& a,bool image){
    if(Py_TYPE(a.ptr())!=py::detail::npy_api::get().PyArray_Type_)throw py::type_error("mark exact ndarray descriptor required");
    ComponentsView v;const auto k=a.dtype().kind();const auto n=a.itemsize();using P=PixelType;
    if(k=='b'&&n==1)v.type=P::boolean;
    else if(k=='f'&&n==2&&image){v.type=P::float32;v.half=true;}
    else if(k=='f'&&n==4)v.type=P::float32;
    else if(k=='f'&&n==8)v.type=P::float64;
    else if(k=='u')switch(n){case 1:v.type=P::uint8;break;case 2:v.type=P::uint16;break;case 4:v.type=P::uint32;break;case 8:v.type=P::uint64;break;default:throw py::type_error("unsupported mark dtype");}
    else if(k=='i')switch(n){case 1:v.type=P::int8;break;case 2:v.type=P::int16;break;case 4:v.type=P::int32;break;case 8:v.type=P::int64;break;default:throw py::type_error("unsupported mark dtype");}
    else throw py::type_error("unsupported mark dtype");
    v.swap=!a.dtype().attr("isnative").cast<bool>();v.data=a.data();for(py::ssize_t d=0;d<a.ndim();++d){v.shape.push_back(std::size_t(a.shape(d)));v.strides.push_back(a.strides(d));}storage(a);return v;
}
py::array array(const MarkImage& image){std::vector<py::ssize_t> shape(image.shape.begin(),image.shape.end()),strides;const auto bytes=pixel_size(image.type);for(auto s:image.strides)strides.push_back(py::ssize_t(s*bytes));auto dtype=image.type==PixelType::float32?py::dtype::of<float>():py::dtype::of<double>();py::array out(dtype,shape,strides);if(out.nbytes())std::memcpy(out.mutable_data(),image.data(),std::size_t(out.nbytes()));return out;}
MarkMask mask(const py::array& a){if(a.dtype().kind()!='b'||!a.attr("flags").attr("c_contiguous").cast<bool>())throw py::type_error("mark canonical mask required");bindings::check_array_storage(a);MarkMask out;for(py::ssize_t d=0;d<a.ndim();++d)out.shape.push_back(std::size_t(a.shape(d)));out.values.resize(a.size());std::memcpy(out.values.data(),a.data(),out.values.size());for(auto& value:out.values)value=value!=0;return out;}
py::array array(const MarkMask& value){std::vector<py::ssize_t> shape(value.shape.begin(),value.shape.end());py::array out(py::dtype::of<bool>(),shape);if(!value.values.empty())std::memcpy(out.mutable_data(),value.values.data(),value.values.size());return out;}
}
void bind_mark_boundaries(py::module_& m){
 // Source NumPy mapping.c uses this exact FORCECAST RHS boundary for a
 // non-array color, rather than Python np.asarray's different flag path.
 m.def("mark_boundaries_color",[](py::object value,py::dtype dtype){
    if(py::isinstance<py::array>(value))return py::reinterpret_borrow<py::array>(value);
    Py_INCREF(dtype.ptr());auto* result=py::detail::npy_api::get().PyArray_FromAny_(value.ptr(),dtype.ptr(),0,0,py::array::forcecast,nullptr);
    if(!result)throw py::error_already_set();return py::reinterpret_steal<py::array>(result);
 });
 m.def("mark_boundaries_prepare",[](const py::array& image,bool subpixel,bool modern,int workers){auto v=view(image,true);MarkImage out;{py::gil_scoped_release release;out=mark_boundaries_prepare(v,subpixel,modern?MarkCubicProfile::scipy118:MarkCubicProfile::scipy115,workers);}return array(out);});
 m.def("mark_boundaries_mask",[](const py::array& labels,int mode,std::uint64_t background,bool representable,int workers){if(mode<0||mode>3)throw py::value_error("mark mode outside assessed range");auto v=view(labels,false);detail::FindBoundaryParameters p;p.mode=detail::FindBoundaryMode(mode);p.background_bits=background;p.background_representable=representable;MarkMask out;{py::gil_scoped_release release;out=mark_boundaries_mask(v,p,workers);}return array(out);});
 m.def("mark_boundaries_count",[](const py::array& input){auto value=mask(input);return std::count(value.values.begin(),value.values.end(),std::uint8_t(1));});
 m.def("mark_boundaries_outline",[](const py::array& input){auto v=mask(input);MarkMask out;{py::gil_scoped_release release;out=mark_boundaries_outline(v);}return array(out);});
 m.def("mark_boundaries_paint",[](py::array image,const py::array& selection,const py::array& colors){auto v=view(image,true);if(!image.writeable()||!image.owndata()||!colors.attr("flags").attr("c_contiguous").cast<bool>()||(colors.dtype().kind()!='f'||(colors.itemsize()!=4&&colors.itemsize()!=8)))throw py::value_error("mark prepared output/color storage required");bindings::check_array_storage(colors);auto selected=mask(selection);MarkColor c;c.shape.clear();c.values.resize(colors.size());for(py::ssize_t d=0;d<colors.ndim();++d)c.shape.push_back(std::size_t(colors.shape(d)));if(colors.itemsize()==4){const auto* p=static_cast<const float*>(colors.data());for(std::size_t i=0;i<c.values.size();++i)c.values[i]=p[i];}else{std::memcpy(c.values.data(),colors.data(),c.values.size()*sizeof(double));}for(auto& stride:v.strides)stride/=image.itemsize();{py::gil_scoped_release release;mark_boundaries_paint_into(image.mutable_data(),v.type,v.shape,v.strides,std::size_t(image.size()),selected,c);}return true;});
}
