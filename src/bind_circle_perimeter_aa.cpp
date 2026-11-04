#include <skimagexx/circle_perimeter_aa.hpp>
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
py::array output_array(std::size_t count,const py::dtype& dtype) {
    return py::array(dtype,std::array<py::ssize_t,1>{py::ssize_t(count)},
                     std::array<py::ssize_t,1>{count?py::ssize_t(dtype.itemsize()):0});
}
}

void bind_circle_perimeter_aa(py::module_& module) {
    module.def("draw_circle_perimeter_aa",[](Int row,Int column,Int radius,
                                             const py::object& shape,int workers)->py::object {
        if(workers<1||workers>1024)throw py::value_error("circle AA worker snapshot outside [1,1024]");
        skimagexx::detail::CirclePerimeterAARequest request{row,column,radius};
        request.clipped=!shape.is_none();
        if(request.clipped&&!integer_shape(shape,request.height,request.width))return py::none();
        skimagexx::detail::CirclePerimeterAA prepared;
        if(!skimagexx::detail::prepare_circle_perimeter_aa(request,prepared))return py::none();
        std::size_t count;
        {py::gil_scoped_release release;count=skimagexx::detail::circle_perimeter_aa_count(prepared,workers);}
        auto rows=output_array(count,py::dtype("intp")),columns=output_array(count,py::dtype("intp"));
        auto weights=output_array(count,py::dtype("float64"));
        const skimagexx::detail::DrawCoordinatePair output{
            static_cast<Int*>(rows.mutable_data()),static_cast<Int*>(columns.mutable_data()),1,1};
        auto* values=static_cast<double*>(weights.mutable_data());
        {py::gil_scoped_release release;skimagexx::detail::circle_perimeter_aa_fill(prepared,output,values,1,workers);}
        return py::make_tuple(rows,columns,weights);
    },py::arg("row"),py::arg("column"),py::arg("radius"),py::arg("shape"),py::arg("workers"));
}
