#include "skimagexx/hough_ellipse.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <type_traits>

namespace py = pybind11;

void bind_hough_ellipse(py::module_& module) {
    using Record = skimagexx::HoughEllipseDetection;
    static_assert(std::is_trivially_copyable_v<Record>);
    const auto fields = py::make_tuple("accumulator", "yc", "xc", "a", "b", "orientation");
    const auto f64 = py::dtype::of<double>();
    const auto dtype = py::dtype(py::list(fields),
        py::list(py::make_tuple(py::dtype::of<std::intptr_t>(), f64, f64, f64, f64, f64)),
        py::list(py::make_tuple(offsetof(Record, accumulator), offsetof(Record, yc), offsetof(Record, xc),
                                offsetof(Record, a), offsetof(Record, b), offsetof(Record, orientation))), sizeof(Record));
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    module.def("hough_ellipse_bounded", [ndarray, dtype](const py::array& image, int threshold, double accuracy,
               int min_size, std::optional<int> max_size) -> py::object {
        if (!py::type::handle_of(image).is(ndarray) || image.ndim() != 2 || !(image.flags() & py::array::c_style) ||
            (!image.dtype().equal(py::dtype::of<bool>()) && !image.dtype().equal(py::dtype::of<std::uint8_t>())))
            throw py::value_error("hough_ellipse requires an ordinary C-order Boolean or uint8 image");
        const auto height = image.shape(0), width = image.shape(1);
        if (height < 0 || height > 64 || width < 0 || width > 64)
            return py::none();
        skimagexx::bindings::check_array_storage(image);
        const auto* input = static_cast<const std::uint8_t*>(image.data());
        const auto count = std::size_t(image.size());
        std::optional<skimagexx::HoughEllipseResult> results;
        {
            py::gil_scoped_release release;
            results = skimagexx::hough_ellipse_bounded(input, count, int(height), int(width),
                threshold, accuracy, min_size, max_size);
        }
        if (!results)
            return py::none();
        if (!results->has_foreground)
            return py::array_t<double>(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(6)});
        const auto size = results->detections.size();
        py::array array(dtype, std::vector<py::ssize_t>{py::ssize_t(size)});
        if (size)
            std::memcpy(array.mutable_data(), results->detections.data(), size * sizeof(Record));
        return array;
    }, py::arg("image").noconvert(), py::arg("threshold"), py::arg("accuracy"),
       py::arg("min_size"), py::arg("max_size"));
}
