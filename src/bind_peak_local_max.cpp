#include "skimagexx/peak_local_max.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

namespace {

bool exact_array(const py::handle array) {
    return Py_TYPE(array.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

void check_array(const py::array& array) {
    if (!exact_array(array) || array.ndim() != 2 ||
        !(array.flags() & py::array::c_style) || !array.writeable() || !array.attr("flags").attr("aligned").cast<bool>() ||
        !array.dtype().attr("isnative").cast<bool>() || !array.dtype().attr("metadata").is_none())
        throw py::value_error("peak routines require aligned native C-order arrays");
    skimagexx::bindings::check_array_storage(array);
}

void check_image(const py::array& image) {
    check_array(image);
    const char code = image.dtype().char_();
    if (!((code == 'B' && image.itemsize() == 1) || (code == 'f' && image.itemsize() == py::ssize_t(sizeof(float))) ||
          (code == 'd' && image.itemsize() == py::ssize_t(sizeof(double)))) || image.size() < 1 || image.size() > 16777216 ||
        image.shape(0) > 65536 || image.shape(1) > 65536)
        throw py::value_error("unsupported peak image dtype or extent");
}

template<class T>
std::vector<skimagexx::PeakCoordinate> coordinates(const py::array& corners) {
    const auto* data = static_cast<const T*>(corners.data());
    std::vector<skimagexx::PeakCoordinate> result;
    result.reserve(std::size_t(corners.shape(0)));
    for (py::ssize_t i = 0; i < corners.shape(0); ++i) {
        const auto row = data[2 * i], column = data[2 * i + 1];
        if (row < 1 || row > 65536 || column < 1 || column > 65536)
            throw std::overflow_error("corner coordinates outside native range");
        result.push_back({std::ptrdiff_t(row), std::ptrdiff_t(column)});
    }
    return result;
}

}  // namespace

void bind_peak_local_max(py::module_& module) {
    module.def("peak_local_max_2d", [](const py::array& image, const py::object& label_arg,
        int distance, std::optional<double> absolute, std::optional<double> relative,
        std::array<int, 2> border, std::size_t max_peaks, std::size_t max_per_label,
        int norm, bool corner, bool indices, bool wide_relative_product, double p_norm) -> py::object {
        check_image(image);
        if (norm < 0 || norm > 3)
            throw py::value_error("unsupported peak norm");
        const int height = int(image.shape(0)), width = int(image.shape(1));
        skimagexx::PeakLabels labels;
        if (!label_arg.is_none()) {
            if (!exact_array(label_arg))
                throw py::value_error("unsupported peak label array");
            const auto array = py::reinterpret_borrow<py::array>(label_arg);
            check_array(array);
            if (array.shape(0) != height || array.shape(1) != width || array.dtype().kind() != 'i')
                throw py::value_error("incompatible peak label array");
            if (array.itemsize() == 4)
                labels = static_cast<const std::int32_t*>(array.data());
            else if (array.itemsize() == 8)
                labels = static_cast<const std::int64_t*>(array.data());
            else
                throw py::value_error("unsupported peak label dtype");
        }
        const char code = image.dtype().char_();
        const auto* data = image.data();
        const skimagexx::LocalPeakOptions options{distance, absolute, relative, border, max_peaks, max_per_label,
                                                 static_cast<skimagexx::PeakNorm>(norm), corner, wide_relative_product, p_norm};
        std::vector<skimagexx::PeakCoordinate> points;
        try {
            py::gil_scoped_release release;
            if (code == 'f')
                points = skimagexx::peak_local_max(static_cast<const float*>(data), height, width, options, labels);
            else if (code == 'd')
                points = skimagexx::peak_local_max(static_cast<const double*>(data), height, width, options, labels);
            else
                points = skimagexx::peak_local_max(static_cast<const std::uint8_t*>(data), height, width, options, labels);
        } catch (const std::overflow_error&) {
            return py::none();
        }
        if (!indices) {
            py::array_t<bool> output(std::array<py::ssize_t, 2>{height, width});
            auto* mask = output.mutable_data();
            std::fill_n(mask, std::size_t(height) * std::size_t(width), false);
            for (const auto& point : points)
                mask[point[0] * width + point[1]] = true;
            return output;
        }
        py::array_t<py::ssize_t> output(std::array<py::ssize_t, 2>{py::ssize_t(points.size()), 2});
        auto* coordinates = output.mutable_data();
        for (const auto& point : points) {
            *coordinates++ = point[0];
            *coordinates++ = point[1];
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("labels"), py::arg("min_distance"), py::arg("threshold_abs"),
       py::arg("threshold_rel"), py::arg("border"), py::arg("max_peaks"), py::arg("max_peaks_per_label"),
       py::arg("norm"), py::arg("corner"), py::arg("indices"), py::arg("wide_relative_product") = false,
       py::arg("p_norm") = 2);

    module.def("corner_subpix_2d", [](const py::array& image, const py::array& corners,
        int window, double critical_dot, double critical_edge) -> py::object {
        check_image(image);
        check_array(corners);
        const auto count = corners.shape(0);
        if (corners.shape(1) != 2 || count > 262144 || corners.dtype().kind() != 'i' ||
            !(corners.itemsize() == 4 || corners.itemsize() == 8))
            throw py::value_error("unsupported corner coordinates");
        const bool wide_coordinates = corners.itemsize() == 8;
        const char code = image.dtype().char_();
        const int height = int(image.shape(0)), width = int(image.shape(1));
        const auto* data = image.data();
        const std::array<py::ssize_t, 2> shape{count, 2};
        py::array output(code == 'f' ? py::dtype::of<float>() : py::dtype::of<double>(), shape);
        auto* dest = output.mutable_data();
        try {
            const auto points = wide_coordinates ? coordinates<std::int64_t>(corners) : coordinates<std::int32_t>(corners);
            py::gil_scoped_release release;
            if (code == 'f')
                skimagexx::corner_subpix_into(static_cast<const float*>(data), height, width, points.data(),
                    points.size(), window, critical_dot, critical_edge, static_cast<float*>(dest));
            else if (code == 'd')
                skimagexx::corner_subpix_into(static_cast<const double*>(data), height, width, points.data(),
                    points.size(), window, critical_dot, critical_edge, static_cast<double*>(dest));
            else
                skimagexx::corner_subpix_into(static_cast<const std::uint8_t*>(data), height, width, points.data(),
                    points.size(), window, critical_dot, critical_edge, static_cast<double*>(dest));
        } catch (const std::overflow_error&) {
            return py::none();
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("corners").noconvert(), py::arg("window_size"),
       py::arg("critical_dot"), py::arg("critical_edge"));
}
