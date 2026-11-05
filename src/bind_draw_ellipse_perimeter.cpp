#include <skimagexx/draw_ellipse_perimeter.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
namespace py=pybind11;
namespace {
using Int=std::intptr_t;
constexpr double shape_limit=0x1p52;
bool scalar_bound(py::handle value,double& output) {
    if(PyLong_CheckExact(value.ptr())||PyBool_Check(value.ptr())) {
        int overflow=0;const auto integer=PyLong_AsLongLongAndOverflow(value.ptr(),&overflow);
        if(PyErr_Occurred())throw py::error_already_set();
        if(overflow||integer<-(1LL<<52)||integer>(1LL<<52))return false;
        output=double(integer);return true;
    }
    if(!PyFloat_CheckExact(value.ptr()))return false;
    output=PyFloat_AS_DOUBLE(value.ptr());return std::isfinite(output)&&std::abs(output)<=shape_limit;
}
template<class T> bool array_bounds(const py::array& a,double& h,double& w) {
    double values[2];const auto* data=static_cast<const char*>(a.data());
    for(int i=0;i<2;++i) {
        T value;std::memcpy(&value,data+std::ptrdiff_t(i)*a.strides(0),sizeof(T));
        if constexpr(std::is_integral_v<T> && sizeof(T)==8) {
            if constexpr(std::is_unsigned_v<T>) {if(value>(1ULL<<52))return false;}
            else {if(value<-(1LL<<52)||value>(1LL<<52))return false;}
        }
        const double bound=double(value);if(!std::isfinite(bound)||std::abs(bound)>shape_limit)return false;
        values[i]=bound;
    }
    h=values[0];w=values[1];return true;
}
bool shape_bounds(const py::object& shape,double& h,double& w) {
    if(PyTuple_CheckExact(shape.ptr())||PyList_CheckExact(shape.ptr())) {
        const auto n=PySequence_Size(shape.ptr());if(n<2)return false;
        return scalar_bound(shape[py::int_(0)],h)&&scalar_bound(shape[py::int_(1)],w);
    }
    if(!py::isinstance<py::array>(shape))return false;
    const auto a=py::cast<py::array>(shape);
    skimagexx::bindings::check_array_storage(a);
    if(a.ndim()!=1||a.size()<2||!a.dtype().attr("isnative").cast<bool>())return false;
    const auto kind=a.dtype().kind();const auto bytes=a.itemsize();
    if(kind=='i') {
        if(bytes==1)return array_bounds<std::int8_t>(a,h,w);
        if(bytes==2)return array_bounds<std::int16_t>(a,h,w);
        if(bytes==4)return array_bounds<std::int32_t>(a,h,w);
        if(bytes==8)return array_bounds<std::int64_t>(a,h,w);
    } else if(kind=='u') {
        if(bytes==1)return array_bounds<std::uint8_t>(a,h,w);
        if(bytes==2)return array_bounds<std::uint16_t>(a,h,w);
        if(bytes==4)return array_bounds<std::uint32_t>(a,h,w);
        if(bytes==8)return array_bounds<std::uint64_t>(a,h,w);
    } else if(kind=='f') {
        if(bytes==4)return array_bounds<float>(a,h,w);
        if(bytes==8)return array_bounds<double>(a,h,w);
    }
    return false;
}
py::array coordinates(std::size_t count,bool scalar) {
    std::vector<py::ssize_t> shape,strides;
    if(!scalar){shape.push_back(py::ssize_t(count));strides.push_back(count?py::ssize_t(sizeof(Int)):0);}
    return py::array(py::dtype::of<Int>(),shape,strides);
}
}
void bind_draw_ellipse_perimeter(py::module_& module) {
    module.def("ellipse_perimeter_orientation",[](py::handle value) {
        const double result=PyFloat_AsDouble(value.ptr());
        if(PyErr_Occurred())throw py::error_already_set();
        return result;
    },py::arg("value"));
    module.def("draw_ellipse_perimeter",[](Int row,Int column,Int rr,Int cr,double angle,const py::object& shape,int workers)->py::object {
        if(workers<1||workers>1024)throw py::value_error("ellipse worker snapshot outside [1,1024]");
        skimagexx::detail::EllipsePerimeterRequest request{row,column,rr,cr,angle};
        request.clipped=!shape.is_none() && !(rr==0&&cr==0);
        if(request.clipped&&!shape_bounds(shape,request.height,request.width))return py::none();
        skimagexx::detail::EllipsePerimeter prepared;
        if(!skimagexx::detail::prepare_ellipse_perimeter(request,prepared))return py::none();
        const auto count=skimagexx::detail::ellipse_perimeter_count(prepared);
        auto rows=coordinates(count,prepared.scalar_result),columns=coordinates(count,prepared.scalar_result);
        const skimagexx::detail::DrawCoordinatePair output{static_cast<Int*>(rows.mutable_data()),static_cast<Int*>(columns.mutable_data()),1,1};
        {py::gil_scoped_release release;skimagexx::detail::ellipse_perimeter_fill(prepared,output,workers);}
        return py::make_tuple(rows,columns);
    },py::arg("row"),py::arg("column"),py::arg("row_radius"),py::arg("column_radius"),py::arg("orientation"),py::arg("shape"),py::arg("workers"));
}
