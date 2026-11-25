#include "skimagexx/hough_circle_peaks.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>

namespace py = pybind11;

void bind_hough_circle_peaks(py::module_& module) {
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    module.def("hough_circle_peak_check_radii", [](const py::array& radii) {
        skimagexx::bindings::check_array_storage(radii);
    }, py::arg("radii").noconvert());
    module.def("hough_circle_peak_indices", [ndarray](const py::array& votes, int min_xdistance,
               int min_ydistance, std::optional<double> threshold) -> py::object {
        if (!py::type::handle_of(votes).is(ndarray) || votes.ndim() != 3 || !(votes.flags() & py::array::c_style) ||
            !votes.attr("flags").attr("aligned").cast<bool>() || !votes.dtype().attr("metadata").is_none() ||
            !votes.dtype().equal(py::dtype::of<double>()) || votes.shape(0) < 1 || votes.shape(0) > 16 ||
            votes.shape(1) < 1 || votes.shape(1) > 256 || votes.shape(2) < 1 || votes.shape(2) > 256)
            throw py::value_error("hough_circle_peaks requires a bounded ordinary float64 volume");
        skimagexx::bindings::check_array_storage(votes);

        const int count = int(votes.shape(0));
        const int height = int(votes.shape(1)), width = int(votes.shape(2));
        const auto* data = static_cast<const double*>(votes.data());
        const auto plane_size = std::size_t(height * width);
        std::vector<std::vector<skimagexx::HoughCirclePeak>> planes;
        planes.reserve(count);
        {
            py::gil_scoped_release release;
            for (int i = 0; i < count; ++i) {
                auto peaks = skimagexx::hough_circle_peak_indices(data + i * plane_size, plane_size,
                    height, width, min_xdistance, min_ydistance, threshold);
                if (!peaks)
                    break;
                planes.push_back(std::move(*peaks));
            }
        }
        if (planes.size() != std::size_t(count))
            return py::none();
        py::list result;
        for (const auto& peaks : planes) {
            const auto size = py::ssize_t(peaks.size());
            py::array_t<double> intensity(size);
            py::array_t<std::intptr_t> x(size), y(size);
            auto* intensity_data = intensity.mutable_data();
            auto* x_data = x.mutable_data();
            auto* y_data = y.mutable_data();
            for (py::ssize_t i = 0; i < size; ++i) {
                intensity_data[i] = peaks[i].vote;
                x_data[i] = peaks[i].column;
                y_data[i] = peaks[i].row;
            }
            result.append(py::make_tuple(std::move(intensity), std::move(x), std::move(y)));
        }
        return result;
    }, py::arg("votes").noconvert(), py::arg("min_xdistance"), py::arg("min_ydistance"), py::arg("threshold"));
}
