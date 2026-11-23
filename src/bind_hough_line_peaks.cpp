#include "skimagexx/hough_line_peaks.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

namespace {

template<class T>
void check_array(const py::array& value, const py::object& ndarray, int rank) {
    if (!py::type::of(value).is(ndarray) || value.ndim() != rank ||
        !value.dtype().equal(py::dtype::of<T>()) || !(value.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T))
        throw py::value_error("hough_line_peaks requires aligned native C-order arrays");
    skimagexx::bindings::check_array_storage(value);
}

bool finite(const py::array& value) {
    const auto* data = static_cast<const double*>(value.data());
    return std::all_of(data, data + value.size(), [](double item) { return std::isfinite(item); });
}

}  // namespace

void bind_hough_line_peaks(py::module_& module) {
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    module.def("hough_line_peak_indices", [ndarray](const py::array& accumulator, const py::array& angles,
               const py::array& distances, int distance_radius, int angle_radius,
               std::optional<double> threshold) -> py::object {
        check_array<std::uint64_t>(accumulator, ndarray, 2);
        check_array<double>(angles, ndarray, 1);
        check_array<double>(distances, ndarray, 1);
        const auto height = accumulator.shape(0);
        const auto width = accumulator.shape(1);
        if (height < 1 || height > 1024 || width < 1 || width > 512 ||
            distances.size() != height || angles.size() != width)
            throw py::value_error("hough_line_peaks accumulator shape and coordinate lengths differ");
        if (!finite(angles) || !finite(distances))
            return py::none();
        const auto* input = static_cast<const std::uint64_t*>(accumulator.data());
        const auto count = std::size_t(accumulator.size());
        std::vector<skimagexx::HoughLinePeak> peaks;
        try {
            py::gil_scoped_release release;
            peaks = skimagexx::hough_line_peak_indices(input, count, int(height), int(width),
                distance_radius, angle_radius, threshold);
        } catch (const std::overflow_error&) {
            return py::none();
        }
        const auto size = py::ssize_t(peaks.size());
        py::array_t<std::uint64_t> votes(size);
        py::array_t<std::int32_t> rows(size), columns(size);
        auto* counts = votes.mutable_data();
        auto* row = rows.mutable_data();
        auto* column = columns.mutable_data();
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            counts[i] = peaks[i].votes;
            row[i] = peaks[i].distance;
            column[i] = peaks[i].angle;
        }
        return py::make_tuple(votes, rows, columns);
    }, py::arg("hspace").noconvert(), py::arg("angles").noconvert(), py::arg("dists").noconvert(),
       py::arg("min_distance"), py::arg("min_angle"), py::arg("threshold"));
}
