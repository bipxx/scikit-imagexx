#include "skimagexx/array_views.hpp"
#include "skimagexx/slice_along_axes.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include <cstdint>
namespace py=pybind11;
namespace {
skimagexx::CropView view(const py::array& a){return {a.data(),{a.shape(),a.shape()+a.ndim()},{a.strides(),a.strides()+a.ndim()},std::size_t(a.itemsize())};}
py::tuple metadata(const skimagexx::CropView& a){return py::make_tuple(a.shape,a.strides);}
std::pair<std::uintptr_t,std::uintptr_t> bounds(const py::array& a) {
    if(!a.size())return {0,0};
    auto low=reinterpret_cast<std::uintptr_t>(a.data()),high=low;
    const auto maximum=std::numeric_limits<std::uintptr_t>::max();
    for(py::ssize_t i=0;i<a.ndim();++i) {
        auto stride=a.strides(i);auto count=std::uintptr_t(a.shape(i)-1);
        auto magnitude=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
        if(magnitude&&count>maximum/magnitude)throw py::value_error("slice storage span overflow");
        auto distance=count*magnitude;
        if(stride<0){if(distance>low)throw py::value_error("slice storage span overflow");low-=distance;}
        else {if(distance>maximum-high)throw py::value_error("slice storage span overflow");high+=distance;}
    }
    if(std::size_t(a.itemsize())>maximum-high)throw py::value_error("slice storage span overflow");
    return {low,high+std::size_t(a.itemsize())};
}
}
// Geometry is native; NumPy's as_strided is an explicit metadata/base-chain
// provider in the Python boundary. No payload read/copy or dtype conversion.
void bind_util_views(py::module_& m) {
    m.def("util_view_as_blocks",[](py::array a,const std::vector<std::ptrdiff_t>& block){return metadata(skimagexx::view_as_blocks(view(a),block));},py::arg("array").noconvert(),py::arg("block"));
    m.def("util_view_as_windows",[](py::array a,const std::vector<std::ptrdiff_t>& window,const std::vector<std::ptrdiff_t>& step){return metadata(skimagexx::view_as_windows(view(a),window,step));},py::arg("array").noconvert(),py::arg("window"),py::arg("step"));
    m.def("util_slice_view",[](py::array a,py::tuple supplied)->py::object {
        if(supplied.size()!=std::size_t(a.ndim()))throw py::value_error("slice rank mismatch");
        std::vector<skimagexx::SliceAlongAxesPair> pairs;std::vector<std::ptrdiff_t> axes;
        for(py::ssize_t i=0;i<a.ndim();++i) {
            Py_ssize_t start,stop,step,length;
            if(!PySlice_Check(supplied[i].ptr()))throw py::type_error("slice requires basic slices");
            if(PySlice_GetIndicesEx(supplied[i].ptr(),a.shape(i),&start,&stop,&step,&length)<0)throw py::error_already_set();
            if(step!=1)throw py::value_error("slice requires step one");
            pairs.push_back({start,stop});axes.push_back(i);
        }
        auto result=skimagexx::slice_along_axes(view(a),pairs,&axes,false);py::tuple index(pairs.size());
        for(std::size_t i=0;i<pairs.size();++i)index[i]=py::slice(result.plan.axes[i].start,result.plan.axes[i].stop,1);
        auto* out=PyObject_GetItem(a.ptr(),index.ptr());if(!out)throw py::error_already_set();return py::reinterpret_steal<py::object>(out);
    },py::arg("array").noconvert(),py::arg("slices").noconvert());
    m.def("util_slice_copy_into",[](py::array input,py::array output) {
        if(!output.writeable()||!output.owndata()||!(output.flags()&py::array::c_style))throw py::value_error("slice output must be owning writable C storage");
        if(!input.itemsize()||py::cast<bool>(input.dtype().attr("hasobject"))||PyObject_RichCompareBool(input.dtype().ptr(),output.dtype().ptr(),Py_EQ)!=1)throw py::type_error("slice copy requires matching POD dtype");
        if(input.ndim()!=output.ndim())throw py::value_error("slice output shape mismatch");
        std::vector<skimagexx::CropSlice> all;
        for(py::ssize_t i=0;i<input.ndim();++i){if(input.shape(i)!=output.shape(i))throw py::value_error("slice output shape mismatch");all.push_back({0,input.shape(i)});}
        const auto source=bounds(input),target=bounds(output);
        if(source.first<target.second&&target.first<source.second)throw py::value_error("slice output overlaps input");
        skimagexx::crop_copy_into(view(input),all,output.mutable_data(),{output.strides(),output.strides()+output.ndim()});
    },py::arg("input").noconvert(),py::arg("output").noconvert());
}
