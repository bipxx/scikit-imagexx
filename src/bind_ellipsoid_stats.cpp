#include <skimagexx/ellipsoid_stats.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace {

bool exact_builtin_radius(py::handle value, double& radius) {
    if (PyFloat_CheckExact(value.ptr())) {
        radius = PyFloat_AS_DOUBLE(value.ptr());
        return true;
    }
    if (!PyLong_CheckExact(value.ptr()) && !PyBool_Check(value.ptr())) return false;
    int overflow = 0;
    const auto integer = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred()) throw py::error_already_set();
    constexpr long long exact_limit = 1LL << 53;
    if (overflow || integer < -exact_limit || integer > exact_limit) return false;
    radius = static_cast<double>(integer);
    return true;
}

}

void bind_ellipsoid_stats(py::module_& module) {
    module.def("_ellipsoid_stats_builtin", [](py::handle a, py::handle b, py::handle c) -> py::object {
        double x, y, z;
        if (!exact_builtin_radius(a, x) || !exact_builtin_radius(b, y) || !exact_builtin_radius(c, z))
            return py::none();
        if (x <= 0 || y <= 0 || z <= 0)
            throw py::value_error("Parameters a, b, and c must all be > 0");
        const auto result = skimagexx::ellipsoid_stats(x, y, z);
        if (!result) return py::none();
        return py::make_tuple(result->volume, result->surface_area);
    });
    module.def("_ellipsoid_stats_rg", [](double x, double y, double z) -> py::object {
        const auto result = skimagexx::detail::ellipsoid_carlson_rg(x, y, z);
        if (!result) return py::none();
        return py::float_(*result);
    });
}
