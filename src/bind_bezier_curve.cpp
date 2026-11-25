#include <skimagexx/bezier_curve.hpp>
#include "bind_helpers.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
namespace py=pybind11;

void bind_bezier_curve(py::module_& module) {
    using namespace skimagexx::bindings;
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
        const auto [min_row,max_row]=std::minmax({r0,r1,r2});
        const auto [min_col,max_col]=std::minmax({c0,c1,c2});
        const auto rows_span=std::uintptr_t(max_row)-std::uintptr_t(min_row);
        const auto cols_span=std::uintptr_t(max_col)-std::uintptr_t(min_col);
        // Preparation walks the curve before clipping, so estimate from the
        // control-point bounds rather than the eventual output count.
        const bool large=rows_span>=4095||cols_span>=4095||rows_span+cols_span>=4095;
        bool accepted;
        {
            optional_gil_release release(large);
            accepted=skimagexx::detail::prepare_bezier_curve(q,p);
        }
        if(!accepted)return py::none();
        const auto n=static_cast<py::ssize_t>(skimagexx::detail::bezier_curve_count(p));
        const py::ssize_t shape[]{n};
        auto rows=empty_c_array<std::intptr_t>(shape,1,true);
        auto columns=empty_c_array<std::intptr_t>(shape,1,true);
        const skimagexx::detail::DrawCoordinatePair output{rows.mutable_data(),columns.mutable_data(),1,1};
        {
            optional_gil_release release(n>=4096);
            skimagexx::detail::bezier_curve_fill(p,output,workers);
        }
        return py::make_tuple(rows,columns);
    },py::arg("r0"),py::arg("c0"),py::arg("r1"),py::arg("c1"),py::arg("r2"),py::arg("c2"),py::arg("weight"),py::arg("clipped"),py::arg("height"),py::arg("width"),py::arg("workers"));
}
