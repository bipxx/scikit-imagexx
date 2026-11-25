#include "window_internal.hpp"
#include "bind_helpers.hpp"
#include <algorithm>
#include <cstdint>

namespace py = pybind11;
namespace {
using Window = py::array_t<double, py::array::c_style>;
using Coordinates = py::array_t<float, py::array::c_style>;
}

void bind_window(py::module_& module) {
    using namespace skimagexx::bindings;
    module.def("window_sample", [](const Window& window, const Coordinates& coordinates,
                                    int order, int workers) {
        if (window.ndim() != 1 || window.size() == 0)
            throw py::value_error("window samples must be a nonempty one-dimensional array");
        const auto length = window.size();
        const auto rank = static_cast<int>(coordinates.ndim());
        auto output = empty_c_array<double>(coordinates.shape(), rank);
        // Allocation can run callbacks: recheck the prepared typed buffers at
        // that boundary. Public window construction supplies bounded arrays.
        if (window.ndim() != 1 || window.size() != length || !Window::check_(window) ||
            !Coordinates::check_(coordinates) || coordinates.ndim() != rank ||
            !std::equal(output.shape(), output.shape() + rank, coordinates.shape()) ||
            reinterpret_cast<std::uintptr_t>(window.data()) % alignof(double) ||
            reinterpret_cast<std::uintptr_t>(coordinates.data()) % alignof(float))
            throw py::value_error("window inputs changed or are unaligned at allocation");
        auto* dst = output.mutable_data();
        {
            optional_gil_release release(output.size() >= 4096);
            skimagexx::detail::window_sample(window.data(), std::size_t(length), coordinates.data(),
                std::size_t(coordinates.size()), dst, order, workers);
        }
        return output;
    }, py::arg("window").noconvert(), py::arg("coordinates").noconvert(),
       py::arg("order"), py::arg("workers"));
}
