#include "skimagexx/hough_line.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>

#include <cfenv>
#include <cmath>
#include <cstring>

namespace py = pybind11;

namespace {

void check_theta(const py::array& theta, const py::object& ndarray) {
    if (!py::type::handle_of(theta).is(ndarray) || theta.ndim() != 1 ||
        !theta.dtype().equal(py::dtype::of<double>()))
        throw py::value_error("hough_line requires ordinary float64 theta storage");
    skimagexx::bindings::check_array_storage(theta);
    for (py::ssize_t index = 0; index < theta.size(); ++index) {
        double angle;
        std::memcpy(&angle, theta.data(index), sizeof(angle));
        if (!std::isfinite(angle))
            throw py::value_error("hough_line requires finite theta");
    }
}

bool aligned_c(const py::array& value, std::size_t alignment) {
    return (value.flags() & py::array::c_style) &&
           reinterpret_cast<std::uintptr_t>(value.data()) % alignment == 0;
}

const double* checked_trig(const py::array& value, py::ssize_t count) {
    if (value.ndim() != 1 || value.size() != count ||
        !value.dtype().equal(py::dtype::of<double>()) || !aligned_c(value, alignof(double)))
        throw py::value_error("hough_line requires aligned native trigonometric buffers");
    skimagexx::bindings::check_array_storage(value);
    return static_cast<const double*>(value.data());
}

}  // namespace

void bind_hough_line(py::module_& module) {
    const auto numpy = py::module_::import("numpy");
    const py::object ndarray = numpy.attr("ndarray");
    const py::object cosine = numpy.attr("cos");
    const py::object sine = numpy.attr("sin");
    module.def("hough_line_check_theta", [ndarray](const py::array& theta) {
        check_theta(theta, ndarray);
    }, py::arg("theta").noconvert());

    module.def("hough_line_accumulate", [ndarray, cosine, sine](const py::array& image,
               const py::array& theta) {
        if (std::fegetround() != FE_TONEAREST)
            throw py::value_error("hough_line requires round-to-nearest");
        if (!py::type::handle_of(image).is(ndarray) || image.ndim() != 2 || !aligned_c(image, 1) ||
            (!image.dtype().equal(py::dtype::of<bool>()) &&
             !image.dtype().equal(py::dtype::of<std::uint8_t>())))
            throw py::value_error("hough_line requires an ordinary C-order Boolean or uint8 image");
        const auto height = std::size_t(image.shape(0));
        const auto width = std::size_t(image.shape(1));
        const auto count = theta.size();
        const auto offset = skimagexx::hough_line_offset(height, width, std::size_t(count));
        check_theta(theta, ndarray);
        if (!aligned_c(theta, alignof(double)))
            throw py::value_error("hough_line requires aligned C-order theta");
        const auto cosines = cosine(theta).cast<py::array>();
        const auto sines = sine(theta).cast<py::array>();
        py::array_t<std::uint64_t> accumulator({py::ssize_t(2 * offset + 1), count});
        // Provider calls and output allocation precede final input buffer access.
        if (image.ndim() != 2 || image.shape(0) != py::ssize_t(height) ||
            image.shape(1) != py::ssize_t(width) || !aligned_c(image, 1) ||
            (!image.dtype().equal(py::dtype::of<bool>()) &&
             !image.dtype().equal(py::dtype::of<std::uint8_t>())))
            throw py::value_error("hough_line image changed during preparation");
        skimagexx::bindings::check_array_storage(image);
        const auto* cos_data = checked_trig(cosines, count);
        const auto* sin_data = checked_trig(sines, count);
        const auto* pixels = static_cast<const std::uint8_t*>(image.data());
        auto* votes = accumulator.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::hough_line_into(pixels, height * width, height, width, cos_data, sin_data,
                                      std::size_t(count), votes, std::size_t(2 * offset + 1) * count);
        }
        return accumulator;
    }, py::arg("image").noconvert(), py::arg("theta").noconvert());
}
