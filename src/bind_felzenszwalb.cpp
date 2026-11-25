#include "bind_array_storage.hpp"
#include "skimagexx/felzenszwalb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace py = pybind11;

void bind_felzenszwalb(py::module_& module) {
    module.def("felzenszwalb_prepared", [](const py::array& image, double scale,
                                         std::ptrdiff_t min_size) -> py::object {
        const auto dtype = image.dtype();
        if (Py_TYPE(image.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
            dtype.char_() != 'd' || !dtype.equal(py::dtype::of<double>()) ||
            !dtype.attr("metadata").is_none() || !(image.flags() & py::array::c_style) ||
            !image.writeable() || reinterpret_cast<std::uintptr_t>(image.data()) % alignof(double)) {
            throw py::type_error("felzenszwalb requires writable contiguous float64 storage");
        }
        if (image.ndim() != 3 || image.size() == 0 || image.shape(2) > 3 ||
            image.size() > std::numeric_limits<std::ptrdiff_t>::max() / 4 || !std::isfinite(scale)) {
            throw py::value_error("invalid felzenszwalb image or scale");
        }
        skimagexx::bindings::check_array_storage(image);
        const auto height = image.shape(0);
        const auto width = image.shape(1);
        const auto channels = image.shape(2);
        const auto count = skimagexx::felzenszwalb_edge_count(height, width);
        py::array_t<double> costs(count);
        const auto* values = static_cast<const double*>(image.data());
        auto* weights = costs.mutable_data();
        bool finite;
        {
            py::gil_scoped_release release;
            skimagexx::felzenszwalb_costs_into(values, weights, height, width, channels);
            finite = std::all_of(weights, weights + count, [](double v) { return std::isfinite(v); });
        }
        if (!finite) return py::none();

        // Equal-cost edges retain the runtime's NumPy quicksort tie ordering.
        const auto order = costs.attr("argsort")().cast<py::array>();
        const auto* indices = static_cast<const std::ptrdiff_t*>(order.data());
        py::array_t<std::ptrdiff_t> output({height, width});
        auto* labels = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::felzenszwalb_labels_into(weights, indices, labels, height, width, scale, min_size);
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("scale"), py::arg("min_size"));
}
