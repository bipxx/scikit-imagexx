#include <skimagexx/circle_perimeter.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace py=pybind11;
namespace {
using Int=std::intptr_t;
template<class T> bool shape_values(const py::array& value,Int& height,Int& width) {
    const auto* data=static_cast<const unsigned char*>(value.data());
    Int bounds[2];
    for(std::size_t i=0;i<2;++i) {
        T item;std::memcpy(&item,data+std::ptrdiff_t(i)*value.strides(0),sizeof(T));
        if constexpr(std::is_unsigned_v<T>) {
            if constexpr(sizeof(T)>=sizeof(Int))if(item>T(std::numeric_limits<Int>::max()))return false;
        } else if constexpr(sizeof(T)>sizeof(Int)) {
            if(item<T(std::numeric_limits<Int>::min())||item>T(std::numeric_limits<Int>::max()))return false;
        }
        bounds[i]=Int(item);
    }
    height=bounds[0];width=bounds[1];return true;
}
bool integer_shape(const py::object& object,Int& height,Int& width) {
    if(PyTuple_CheckExact(object.ptr())) {
        if(PyTuple_GET_SIZE(object.ptr())!=2)return false;
        auto* a=PyTuple_GET_ITEM(object.ptr(),0);auto* b=PyTuple_GET_ITEM(object.ptr(),1);
        if(!PyLong_CheckExact(a)||!PyLong_CheckExact(b))return false;
        height=PyLong_AsSsize_t(a);if(PyErr_Occurred()){PyErr_Clear();return false;}
        width=PyLong_AsSsize_t(b);if(PyErr_Occurred()){PyErr_Clear();return false;}
        return true;
    }
    if(!py::isinstance<py::array>(object))return false;
    const auto value=py::reinterpret_borrow<py::array>(object);
    if(value.ndim()!=1||value.size()<2||!value.dtype().attr("isnative").cast<bool>())return false;
    const auto kind=value.dtype().kind();const auto bytes=value.itemsize();
    if((kind!='i'&&kind!='u')||(bytes!=1&&bytes!=2&&bytes!=4&&bytes!=8))return false;
    skimagexx::bindings::check_array_storage(value);
    if(kind=='i') {
        if(bytes==1)return shape_values<std::int8_t>(value,height,width);
        if(bytes==2)return shape_values<std::int16_t>(value,height,width);
        if(bytes==4)return shape_values<std::int32_t>(value,height,width);
        return shape_values<std::int64_t>(value,height,width);
    }
    if(bytes==1)return shape_values<std::uint8_t>(value,height,width);
    if(bytes==2)return shape_values<std::uint16_t>(value,height,width);
    if(bytes==4)return shape_values<std::uint32_t>(value,height,width);
    return shape_values<std::uint64_t>(value,height,width);
}
py::array coordinates(std::size_t count) {
    return py::array(py::dtype("intp"),std::array<py::ssize_t,1>{py::ssize_t(count)},
                     std::array<py::ssize_t,1>{count?py::ssize_t(sizeof(Int)):0});
}
}

void bind_circle_perimeter(py::module_& module) {
    module.def("draw_circle_perimeter",[](Int row,Int column,Int radius,int method,
                                         const py::object& shape,int workers)->py::object {
        if(workers<1||workers>1024)throw py::value_error("circle worker snapshot outside [1,1024]");
        if(method!=0&&method!=1)return py::none();
        skimagexx::detail::CirclePerimeterRequest request{row,column,radius,
            method?skimagexx::detail::CirclePerimeterMethod::andres:skimagexx::detail::CirclePerimeterMethod::bresenham};
        request.clipped=!shape.is_none();
        if(request.clipped&&!integer_shape(shape,request.height,request.width))return py::none();
        skimagexx::detail::CirclePerimeter prepared;
        if(!skimagexx::detail::prepare_circle_perimeter(request,prepared))return py::none();
        std::size_t count;
        {py::gil_scoped_release release;count=skimagexx::detail::circle_perimeter_count(prepared,workers);}
        auto rows=coordinates(count),columns=coordinates(count);
        const skimagexx::detail::DrawCoordinatePair output{static_cast<Int*>(rows.mutable_data()),static_cast<Int*>(columns.mutable_data()),1,1};
        {py::gil_scoped_release release;skimagexx::detail::circle_perimeter_fill(prepared,output,workers);}
        return py::make_tuple(rows,columns);
    },py::arg("row"),py::arg("column"),py::arg("radius"),py::arg("method"),py::arg("shape"),py::arg("workers"));
}
