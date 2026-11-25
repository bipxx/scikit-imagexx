#include "skimagexx/hough_circle.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cfenv>

namespace py = pybind11;

void bind_hough_circle(py::module_& module) {
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    module.def("hough_circle_accumulate", [ndarray](const py::array& image, const std::vector<int>& radii,
               bool normalize, bool full_output) -> py::object {
        if (!py::type::handle_of(image).is(ndarray) || image.ndim() != 2 || !(image.flags() & py::array::c_style) ||
            (!image.dtype().equal(py::dtype::of<bool>()) && !image.dtype().equal(py::dtype::of<std::uint8_t>())))
            throw py::value_error("hough_circle requires an ordinary C-order Boolean or uint8 image");
        const auto height = image.shape(0), width = image.shape(1);
        if (height < 1 || height > 512 || width < 1 || width > 512 || std::fegetround() != FE_TONEAREST)
            return py::none();
        const auto plane = skimagexx::hough_circle_plane(int(height), int(width), radii, full_output);
        if (!plane)
            return py::none();
        py::array_t<double> result({py::ssize_t(radii.size()), py::ssize_t(plane->height), py::ssize_t(plane->width)});
        skimagexx::bindings::check_array_storage(image);
        if (image.ndim() != 2 || image.shape(0) != height || image.shape(1) != width ||
            !(image.flags() & py::array::c_style) ||
            (!image.dtype().equal(py::dtype::of<bool>()) && !image.dtype().equal(py::dtype::of<std::uint8_t>())))
            throw py::value_error("hough_circle image changed during output preparation");
        const auto* input = static_cast<const std::uint8_t*>(image.data());
        auto* accumulator = result.mutable_data();
        const auto output_count = std::size_t(result.size());
        {
            py::gil_scoped_release release;
            skimagexx::hough_circle_into(input, std::size_t(height) * width, int(height), int(width),
                radii, normalize, full_output, accumulator, output_count);
        }
        return result;
    }, py::arg("image").noconvert(), py::arg("radii"), py::arg("normalize"), py::arg("full_output"));
}
