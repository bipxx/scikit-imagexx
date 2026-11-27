#include "skimagexx/probabilistic_hough_line.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>

#include <cfenv>

namespace py = pybind11;

void bind_probabilistic_hough_line(py::module_& module) {
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    const auto ordinary = [ndarray](const py::array& value, int ndim) {
        return py::type::handle_of(value).is(ndarray) && value.ndim() == ndim && (value.flags() & py::array::c_style) &&
               value.attr("flags").attr("aligned").cast<bool>() && value.dtype().attr("metadata").is_none();
    };
    module.def("probabilistic_hough_foreground", [ordinary](const py::array& image) -> py::object {
        if (!ordinary(image, 2) ||
            (!image.dtype().equal(py::dtype::of<bool>()) && !image.dtype().equal(py::dtype::of<std::uint8_t>())) ||
            image.shape(0) < 1 || image.shape(0) > 256 || image.shape(1) < 1 || image.shape(1) > 256)
            throw py::value_error("probabilistic_hough_line requires a bounded ordinary byte mask");
        skimagexx::bindings::check_array_storage(image);
        if (std::fegetround() != FE_TONEAREST)
            return py::none();
        const auto* bytes = static_cast<const std::uint8_t*>(image.data());
        const auto size = image.size();
        std::size_t count = 0;
        for (py::ssize_t i = 0; i < size; ++i)
            count += bytes[i] != 0;
        if (count > 4096)
            return py::none();
        return py::int_(count);
    }, py::arg("image").noconvert());
    module.def("probabilistic_hough_check_theta", [](const py::array& theta) {
        skimagexx::bindings::check_array_storage(theta);
    }, py::arg("theta").noconvert());
    module.def("probabilistic_hough_line_ordered", [ordinary](const py::array& image, const py::array& cosines,
               const py::array& sines, const py::array& permutation, int threshold, int line_length, int line_gap) {
        if (!ordinary(image, 2) || image.shape(0) < 1 || image.shape(0) > 256 ||
            image.shape(1) < 1 || image.shape(1) > 256 ||
            (!image.dtype().equal(py::dtype::of<bool>()) && !image.dtype().equal(py::dtype::of<std::uint8_t>())) ||
            !ordinary(cosines, 1) || !ordinary(sines, 1) || cosines.size() != sines.size() ||
            !cosines.dtype().equal(py::dtype::of<double>()) || !sines.dtype().equal(py::dtype::of<double>()) ||
            !ordinary(permutation, 1) || !permutation.dtype().equal(py::dtype::of<std::intptr_t>()))
            throw py::value_error("probabilistic_hough_line requires ordinary mask, trig and permutation storage");
        for (const auto* value : {&image, &cosines, &sines, &permutation})
            skimagexx::bindings::check_array_storage(*value);

        const auto* mask = static_cast<const std::uint8_t*>(image.data());
        const auto* c = static_cast<const double*>(cosines.data());
        const auto* s = static_cast<const double*>(sines.data());
        const auto* order = static_cast<const std::intptr_t*>(permutation.data());
        const auto size = std::size_t(image.size());
        const auto height = int(image.shape(0)), width = int(image.shape(1));
        const auto angles = std::size_t(cosines.size()), count = std::size_t(permutation.size());
        std::vector<skimagexx::ProbabilisticHoughSegment> lines;
        {
            py::gil_scoped_release release;
            lines = skimagexx::probabilistic_hough_line_ordered(mask, size, height, width, c, s, angles,
                order, count, threshold, line_length, line_gap);
        }
        py::list result;
        for (const auto& line : lines)
            result.append(py::make_tuple(py::make_tuple(line.first.x, line.first.y),
                                         py::make_tuple(line.second.x, line.second.y)));
        return result;
    }, py::arg("image").noconvert(), py::arg("cosines").noconvert(), py::arg("sines").noconvert(),
       py::arg("permutation").noconvert(), py::arg("threshold"), py::arg("line_length"), py::arg("line_gap"));
}
