#include <skimagexx/rectangle_perimeter.hpp>
#include "bind_helpers.hpp"

#include <cmath>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace skimagexx::bindings {
bool read_perimeter_vector(py::handle value, std::vector<double>& output, bool rectangle);
bool read_perimeter_shape(py::handle value, detail::PerimeterShape& output);
py::object perimeter_result(detail::PerimeterStatus status, const detail::PerimeterCoordinates& output);
}

void bind_rectangle_perimeter(py::module_& module) {
    using namespace skimagexx;
    using Int = std::intptr_t;
    module.def("rectangle_perimeter", [](py::handle start, py::handle second, bool use_extent,
                                         py::handle dimensions, bool clip) -> py::object {
        detail::PerimeterShape shape;
        std::vector<double> a, b;
        if (!bindings::read_perimeter_shape(dimensions, shape) ||
            !bindings::read_perimeter_vector(start, a, true) || !bindings::read_perimeter_vector(second, b, true))
            return py::none();
        if (clip && !shape) throw py::value_error("Must specify clipping shape");
        // Estimate the unclipped edge traversal from the admitted integer bounds.
        constexpr double limit = 1 << 26;
        const bool bounded = std::abs(a[0]) <= limit && std::abs(a[1]) <= limit &&
                             std::abs(b[0]) <= limit && std::abs(b[1]) <= limit;
        const bool large = bounded &&
            2 * (std::abs(use_extent ? b[0] : b[0] - a[0]) +
                 std::abs(use_extent ? b[1] : b[1] - a[1])) + (use_extent ? 8 : 12) >= 4096;
        detail::PerimeterCoordinates output;
        detail::PerimeterStatus status;
        {
            bindings::optional_gil_release release(large);
            status = detail::rectangle_perimeter({static_cast<Int>(a[0]), static_cast<Int>(a[1])},
                                                  {static_cast<Int>(b[0]), static_cast<Int>(b[1])},
                                                  use_extent, shape, clip, output);
        }
        return bindings::perimeter_result(status, output);
    }, py::arg("start"), py::arg("end_or_extent"), py::arg("use_extent").noconvert(),
       py::arg("shape"), py::arg("clip").noconvert());
}
