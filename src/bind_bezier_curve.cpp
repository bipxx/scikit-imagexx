#include <skimagexx/bezier_curve.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py=pybind11;

void bind_bezier_curve(py::module_& module) {
    module.def("bezier_curve_weight",[](py::handle value) {
        const double weight=PyFloat_AsDouble(value.ptr());
        if(PyErr_Occurred())throw py::error_already_set();
        return weight;
    },py::arg("value"));
    module.def("draw_bezier_curve",[](std::intptr_t r0,std::intptr_t c0,
        std::intptr_t r1,std::intptr_t c1,std::intptr_t r2,std::intptr_t c2,
        double weight,bool clipped,double height,double width,int workers)->py::object {
        if(workers<1||workers>1024)throw py::value_error("Bezier worker snapshot outside [1,1024]");
        skimagexx::detail::BezierCurve p;
        const skimagexx::detail::BezierCurveRequest q{r0,c0,r1,c1,r2,c2,weight,clipped,height,width};
        bool accepted;
        {
            py::gil_scoped_release release;
            accepted=skimagexx::detail::prepare_bezier_curve(q,p);
        }
        if(!accepted)return py::none();
        const auto n=static_cast<py::ssize_t>(skimagexx::detail::bezier_curve_count(p));
        const std::vector<py::ssize_t> shape{n},strides{n?py::ssize_t(sizeof(std::intptr_t)):0};
        py::array rows(py::dtype::of<std::intptr_t>(),shape,strides);
        py::array columns(py::dtype::of<std::intptr_t>(),shape,strides);
        {
            py::gil_scoped_release release;
            skimagexx::detail::bezier_curve_fill(p,{static_cast<std::intptr_t*>(rows.mutable_data()),static_cast<std::intptr_t*>(columns.mutable_data()),1,1},workers);
        }
        return py::make_tuple(rows,columns);
    },py::arg("r0"),py::arg("c0"),py::arg("r1"),py::arg("c1"),py::arg("r2"),py::arg("c2"),py::arg("weight"),py::arg("clipped"),py::arg("height"),py::arg("width"),py::arg("workers"));
}
