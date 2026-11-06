#include "skimagexx/crop.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <limits>
#include <cstdint>
namespace py=pybind11;
namespace {
skimagexx::CropView view(const py::array& a) {
    return {a.data(),{a.shape(),a.shape()+a.ndim()},{a.strides(),a.strides()+a.ndim()},std::size_t(a.itemsize())};
}
// Conservative byte intervals suffice: shipping outputs are owning and dense.
std::pair<std::uintptr_t,std::uintptr_t> bounds(const py::array& a) {
    if(!a.size())return {0,0};
    auto low=reinterpret_cast<std::uintptr_t>(a.data()),high=low;
    const auto maximum=std::numeric_limits<std::uintptr_t>::max();
    for(py::ssize_t i=0;i<a.ndim();++i) {
        const auto stride=a.strides(i);const auto count=std::uintptr_t(a.shape(i)-1);
        const auto step=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
        if(step && count>maximum/step)throw py::value_error("crop storage span overflow");
        const auto distance=count*step;
        if(stride<0){if(distance>low)throw py::value_error("crop storage span overflow");low-=distance;}
        else {if(distance>maximum-high)throw py::value_error("crop storage span overflow");high+=distance;}
    }
    if(std::size_t(a.itemsize())>maximum-high)throw py::value_error("crop storage span overflow");
    return {low,high+std::size_t(a.itemsize())};
}
std::vector<skimagexx::CropSlice> slices(const py::array& a,const py::tuple& supplied) {
    if(supplied.size()!=std::size_t(a.ndim()))throw py::value_error("crop slice rank mismatch");
    std::vector<skimagexx::CropSlice> result;result.reserve(supplied.size());
    for(auto value:supplied) {
        Py_ssize_t start,stop,step;
        if(!PySlice_Check(value.ptr()))throw py::type_error("crop requires basic slices");
        if(PySlice_Unpack(value.ptr(),&start,&stop,&step)<0)throw py::error_already_set();
        if(step!=1)throw py::value_error("crop requires step-one slices");
        result.push_back({start,stop});
    }
    return result;
}
}
void bind_crop(py::module_& m) {
    m.def("crop_view",[](py::array a,py::tuple supplied)->py::object {
        const auto prepared=slices(a,supplied);
        auto plan=skimagexx::crop_view(view(a),prepared);py::tuple index(plan.axes.size());
        for(std::size_t i=0;i<plan.axes.size();++i)index[i]=py::slice(plan.axes[i].start,plan.axes[i].stop,1);
        auto* result=PyObject_GetItem(a.ptr(),index.ptr());if(!result)throw py::error_already_set();return py::reinterpret_steal<py::object>(result);
    },py::arg("array").noconvert(),py::arg("slices").noconvert());
    m.def("crop_copy_into",[](py::array input,py::array output) {
        if(!output.writeable())throw py::value_error("crop output is read-only");
        if(input.itemsize()==0)throw py::type_error("crop requires nonzero itemsize");
        if(input.dtype().has_fields()||input.dtype().kind()=='O'||PyObject_RichCompareBool(input.dtype().ptr(),output.dtype().ptr(),Py_EQ)!=1)throw py::type_error("crop requires matching POD dtype");
        if(input.ndim()!=output.ndim())throw py::value_error("crop output shape mismatch");
        std::vector<skimagexx::CropSlice> all;
        for(py::ssize_t i=0;i<input.ndim();++i){if(input.shape(i)!=output.shape(i))throw py::value_error("crop output shape mismatch");all.push_back({0,input.shape(i)});}
        if(!output.owndata())throw py::value_error("crop output must own its storage");
        if(output.size()) {
            std::vector<std::pair<py::ssize_t,py::ssize_t>> axes;
            for(py::ssize_t i=0;i<output.ndim();++i)if(output.shape(i)>1)axes.push_back({output.strides(i),output.shape(i)});
            std::sort(axes.begin(),axes.end());auto expected=output.itemsize();
            for(const auto& axis:axes) {
                if(axis.first!=expected)throw py::value_error("crop output must have dense positive strides");
                if(axis.second>std::numeric_limits<py::ssize_t>::max()/expected)throw py::value_error("crop output span overflow");
                expected*=axis.second;
            }
        }
        const auto source=bounds(input),target=bounds(output);
        if(source.first<target.second && target.first<source.second)throw py::value_error("crop output overlaps input");
        // Shipping output comes from empty_like: dtype/layout allocation only.
        skimagexx::crop_copy_into(view(input),all,output.mutable_data(),{output.strides(),output.strides()+output.ndim()});
    },py::arg("input").noconvert(),py::arg("output").noconvert());
}
