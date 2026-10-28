#include <skimagexx/subdivide_polygon.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>

namespace py=pybind11;
void bind_subdivide_polygon(py::module_& module) {
    module.def("subdivide_polygon_apply",[](const py::array& coordinates,unsigned degree,
        bool circular,bool preserve_ends,int workers)->py::object {
        if(workers<1||workers>1024)throw py::value_error("polygon worker snapshot outside [1,1024]");
        using Coordinates=py::array_t<double,py::array::c_style>;
        if(!Coordinates::check_(coordinates)||!coordinates.dtype().attr("isnative").cast<bool>()
            ||coordinates.ndim()!=2||coordinates.shape(1)!=2
            ||reinterpret_cast<std::uintptr_t>(coordinates.data())%alignof(double))return py::none();
        skimagexx::bindings::check_array_storage(coordinates);
        const auto count=std::size_t(coordinates.shape(0));
        // Structural planning for fresh output, not a second numerical scan.
        if(degree<1||degree>7||!count)return py::none();
        const auto length=std::size_t((degree+3)/2);
        if(!circular&&count<length)return py::none();
        const auto points=circular?count:count-length+1;
        const auto extra=std::size_t(circular?1:preserve_ends?2:0);
        const auto limit=std::size_t(std::numeric_limits<py::ssize_t>::max())/(2*sizeof(double));
        if(points>(limit-extra)/2)throw py::value_error("polygon output shape is not addressable");
        const auto rows=points*2+extra;
        // A fresh owning C array establishes disjointness and output capacity.
        Coordinates output({py::ssize_t(rows),py::ssize_t(2)});
        const skimagexx::detail::SubdividePolygonInput input{
            static_cast<const double*>(coordinates.data()),count,
            coordinates.strides(0)/py::ssize_t(sizeof(double)),
            coordinates.strides(1)/py::ssize_t(sizeof(double))};
        auto* destination=output.mutable_data();bool accepted;
        {
            py::gil_scoped_release release;
            accepted=skimagexx::detail::subdivide_polygon_into(
                input,degree,circular,preserve_ends,destination,workers);
        }
        if(!accepted)return py::none();
        return output;
    },py::arg("coordinates").noconvert(),py::arg("degree"),py::arg("circular"),
       py::arg("preserve_ends"),py::arg("workers"));
}
