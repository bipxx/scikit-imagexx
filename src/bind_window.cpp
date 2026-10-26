#include "window_internal.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace py = pybind11;
namespace {
using Window = py::array_t<double, py::array::c_style>;
using Coordinates = py::array_t<float, py::array::c_style>;
}

void bind_window(py::module_& module) {
    module.def("window_sample", [](const Window& window, const Coordinates& coordinates,
                                    int order, int workers) {
        if (window.ndim() != 1 || window.size() == 0)
            throw py::value_error("window samples must be a nonempty one-dimensional array");
        const auto length = window.size();
        const std::vector<py::ssize_t> shape(coordinates.shape(), coordinates.shape() + coordinates.ndim());
        py::array_t<double> output(shape);
        // Allocation can run callbacks: recheck the prepared typed buffers at
        // that boundary. Public window construction supplies bounded arrays.
        if (window.ndim() != 1 || window.size() != length || !Window::check_(window) ||
            !Coordinates::check_(coordinates) || coordinates.ndim() != py::ssize_t(shape.size()) ||
            !std::equal(shape.begin(), shape.end(), coordinates.shape()) ||
            reinterpret_cast<std::uintptr_t>(window.data()) % alignof(double) ||
            reinterpret_cast<std::uintptr_t>(coordinates.data()) % alignof(float))
            throw py::value_error("window inputs changed or are unaligned at allocation");
        {
            py::gil_scoped_release release;
            skimagexx::detail::window_sample(window.data(), std::size_t(length), coordinates.data(),
                std::size_t(coordinates.size()), output.mutable_data(), order, workers);
        }
        return output;
    }, py::arg("window").noconvert(), py::arg("coordinates").noconvert(),
       py::arg("order"), py::arg("workers"));
}
