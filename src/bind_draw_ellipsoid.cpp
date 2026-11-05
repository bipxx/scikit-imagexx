#include <skimagexx/draw_ellipsoid.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cfenv>
namespace py=pybind11;
void bind_draw_ellipsoid(py::module_& module) {
    module.def("draw_ellipsoid", [](const std::array<double,3>& axes,
        const std::array<double,3>& spacing,bool levelset,int workers)->py::object {
        if(workers<1 || workers>1024)throw py::value_error("drawing worker snapshot outside [1,1024]");
        if(std::fegetround()!=FE_TONEAREST)return py::none();
        skimagexx::detail::DrawEllipsoidPlan plan;
        if(!skimagexx::detail::draw_ellipsoid_prepare(axes,spacing,plan))return py::none();
        const std::vector<py::ssize_t> shape{py::ssize_t(plan.shape[0]),py::ssize_t(plan.shape[1]),py::ssize_t(plan.shape[2])};
        if(levelset) {
            py::array_t<double> output(shape);auto* data=output.mutable_data();
            {py::gil_scoped_release release;skimagexx::detail::draw_ellipsoid_levelset_into(plan,data,workers);}
            return output;
        }
        py::array_t<bool> output(shape);auto* data=reinterpret_cast<std::uint8_t*>(output.mutable_data());
        {py::gil_scoped_release release;skimagexx::detail::draw_ellipsoid_bool_into(plan,data,workers);}
        return output;
    },py::arg("axes"),py::arg("spacing"),py::arg("levelset"),py::arg("workers"));
}
