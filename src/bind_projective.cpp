#include "skimagexx/projective.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py=pybind11;
namespace {
py::tuple apply(py::array_t<double,0> points,py::array_t<double,0> matrix) {
        if(points.ndim()!=2 || matrix.ndim()!=2 || matrix.shape(0)!=matrix.shape(1) || matrix.shape(0)!=points.shape(1)+1 || points.shape(1)<1)
            throw py::value_error("projective coordinate/matrix shape mismatch");
        if(!points.dtype().attr("isnative").cast<bool>() || !matrix.dtype().attr("isnative").cast<bool>())throw py::type_error("projective native float64 views required");
        py::array_t<double> output({points.shape(0),points.shape(1)});
        bool provider;
        {py::gil_scoped_release release;
        provider=skimagexx::projective_apply_checked_into(points.data(),points.strides(0),points.strides(1),matrix.data(),matrix.strides(0),matrix.strides(1),output.mutable_data(),std::size_t(points.shape(0)),std::size_t(points.shape(1)));}
        return py::make_tuple(output,provider);
}
}
void bind_projective(py::module_& module) {
    module.def("projective_apply",[](py::array_t<double,0> points,py::array_t<double,0> matrix){py::object output=apply(points,matrix)[0];return output;});
    module.def("projective_apply_checked",&apply);
}
