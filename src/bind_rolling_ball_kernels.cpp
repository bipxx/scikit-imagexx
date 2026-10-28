#include <skimagexx/rolling_ball_kernels.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cmath>
#include <cstdint>

namespace py = pybind11;
namespace {
using Array = py::array_t<double, py::array::c_style>;
struct Prepared {
    std::array<skimagexx::detail::RollingAxis, 32> axes;
    std::size_t ndim, count;
};

bool prepare(const py::tuple& axes, py::array& output, const py::array* semi,
             int workers, Prepared& prepared) {
    if (workers < 1 || workers > 1024)
        throw py::value_error("rolling-kernel worker snapshot outside [1,1024]");
    if (axes.empty() || axes.size() > 32 || output.ndim() != py::ssize_t(axes.size())
            || !Array::check_(output) || !output.writeable() || !output.owndata()
            || reinterpret_cast<std::uintptr_t>(output.data()) % alignof(double)) return false;
    skimagexx::bindings::check_array_storage(output);
    const auto destination = reinterpret_cast<std::uintptr_t>(output.data());
    const auto end = destination + std::size_t(output.nbytes());
    const auto overlaps = [&](const py::array& input) {
        const auto start = reinterpret_cast<std::uintptr_t>(input.data());
        return output.size() && input.size() && destination < start + input.nbytes() && start < end;
    };
    if (semi) {
        if (!Array::check_(*semi) || semi->ndim() != 1 || semi->size() != py::ssize_t(axes.size())
                || reinterpret_cast<std::uintptr_t>(semi->data()) % alignof(double)) return false;
        skimagexx::bindings::check_array_storage(*semi);
        if (overlaps(*semi)) return false;
    }
    prepared.ndim = axes.size();
    prepared.count = std::size_t(output.size());
    const bool empty = output.size() == 0;
    bool empty_axis = false;
    std::size_t product = 1;
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        if (!py::array::check_(axes[axis])) return false;
        const auto values = py::reinterpret_borrow<py::array>(axes[axis]);
        if (!Array::check_(values) || values.ndim() != 1 || values.size() != output.shape(axis)
                || reinterpret_cast<std::uintptr_t>(values.data()) % alignof(double)) return false;
        skimagexx::bindings::check_array_storage(values);
        if (overlaps(values)) return false;
        const auto length = std::size_t(values.size());
        empty_axis = empty_axis || !length;
        if (!empty) {
            if (!length || product > std::size_t(PY_SSIZE_T_MAX) / sizeof(double) / length)
                return false;
            product *= length;
        }
        const auto scale = semi ? static_cast<const double*>(semi->data())[axis] : 1.;
        if (!std::isfinite(scale) || scale <= 0) return false;
        prepared.axes[axis] = {static_cast<const double*>(values.data()),
                               std::size_t(values.size()), scale};
    }
    return empty ? empty_axis : product == prepared.count;
}
}

void bind_rolling_ball_kernels(py::module_& module) {
    module.def("ball_kernel_apply", [](const py::tuple& axes, double radius,
               double radius_squared, py::array output, int workers) {
        Prepared prepared;
        if (!prepare(axes, output, nullptr, workers, prepared)) return false;
        const skimagexx::detail::RollingGrid grid{prepared.axes.data(), prepared.ndim, prepared.count};
        auto* destination = static_cast<double*>(output.mutable_data());
        py::gil_scoped_release release;
        return skimagexx::detail::ball_kernel_into(grid, radius, radius_squared, destination, workers);
    }, py::arg("axes").noconvert(), py::arg("radius"), py::arg("radius_squared"),
       py::arg("output").noconvert(), py::arg("workers"));
    module.def("ellipsoid_kernel_apply", [](const py::tuple& axes, const py::array& semi,
               double intensity, py::array output, int workers) {
        Prepared prepared;
        if (!prepare(axes, output, &semi, workers, prepared)) return false;
        const skimagexx::detail::RollingGrid grid{prepared.axes.data(), prepared.ndim, prepared.count};
        auto* destination = static_cast<double*>(output.mutable_data());
        py::gil_scoped_release release;
        return skimagexx::detail::ellipsoid_kernel_into(grid, intensity, destination, workers);
    }, py::arg("axes").noconvert(), py::arg("semi_axes").noconvert(), py::arg("intensity"),
       py::arg("output").noconvert(), py::arg("workers"));
}
