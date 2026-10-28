#include "resample_detail.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style>;

template<class T>
py::object sample(const py::array& image, const Array<double>& coordinates,
                  skimagexx::PixelType type, int order,
                  skimagexx::GaussianBorder border, double cval, int workers) {
    const auto height = image.shape(0), width = image.shape(1);
    const std::vector<py::ssize_t> shape(coordinates.shape() + 1,
                                       coordinates.shape() + coordinates.ndim());
    py::array_t<T> output(shape);
    // Allocation can run callbacks. Validate the actual prepared buffers once
    // here, then snapshot every shape/pointer before releasing the GIL.
    skimagexx::bindings::check_array_storage(image);
    skimagexx::bindings::check_array_storage(coordinates);
    if (!Array<T>::check_(image) || image.ndim() != 2 ||
        image.shape(0) != height || image.shape(1) != width ||
        !Array<double>::check_(coordinates) || coordinates.ndim() != 3 ||
        coordinates.shape(0) != 2 ||
        !std::equal(shape.begin(), shape.end(), coordinates.shape() + 1) ||
        reinterpret_cast<std::uintptr_t>(image.data()) % alignof(T) ||
        reinterpret_cast<std::uintptr_t>(coordinates.data()) % alignof(double))
        throw py::value_error("profile sampling inputs changed or are unaligned at allocation");
    const skimagexx::detail::ProfileInput input{image.data(), type,
        std::size_t(height), std::size_t(width)};
    const auto* positions = coordinates.data();
    const auto count = std::size_t(output.size());
    auto* destination = output.mutable_data();
    bool accepted;
    {
        py::gil_scoped_release release;
        accepted = skimagexx::detail::profile_sample(input, positions, count,
                    destination, order, border, cval, workers);
    }
    if (!accepted) return py::none();
    return output;
}
}

void bind_profile_line(py::module_& module) {
    module.def("profile_coordinates", [](const Array<double>& rows,
               const Array<double>& cols, py::ssize_t width,
               double row_width, double col_width, int workers) -> py::object {
        if (rows.ndim() != 1 || cols.ndim() != 1 || rows.size() != cols.size() ||
            rows.size() == 0 || width <= 0)
            throw py::value_error("profile grid requires matching nonempty center lines and positive width");
        const auto length = rows.size();
        py::array_t<double> output({py::ssize_t(2), length, width});
        // Validate actual buffers after allocation, then keep fixed snapshots.
        skimagexx::bindings::check_array_storage(rows);
        skimagexx::bindings::check_array_storage(cols);
        if (!Array<double>::check_(rows) || !Array<double>::check_(cols) ||
            rows.ndim() != 1 || cols.ndim() != 1 || rows.size() != length || cols.size() != length ||
            reinterpret_cast<std::uintptr_t>(rows.data()) % alignof(double) ||
            reinterpret_cast<std::uintptr_t>(cols.data()) % alignof(double))
            throw py::value_error("profile center lines changed or are unaligned at allocation");
        const auto* row_data = rows.data(); const auto* col_data = cols.data();
        auto* destination = output.mutable_data(); bool accepted;
        {
            py::gil_scoped_release release;
            accepted = skimagexx::detail::profile_coordinates_into(row_data, col_data,
                std::size_t(length), std::size_t(width), row_width, col_width, destination, workers);
        }
        if (!accepted) return py::none();
        return output;
    }, py::arg("rows").noconvert(), py::arg("cols").noconvert(),
       py::arg("width"), py::arg("row_width"), py::arg("col_width"), py::arg("workers"));
    module.def("profile_sample", [](const py::array& image,
               const Array<double>& coordinates, int order, const std::string& mode,
               double cval, int workers) -> py::object {
        if (image.ndim() != 2 || image.shape(0) == 0 || image.shape(1) == 0)
            throw py::value_error("profile sampling requires a nonempty two-dimensional image");
        if (coordinates.ndim() != 3 || coordinates.shape(0) != 2)
            throw py::value_error("profile coordinates require shape (2, length, width)");
        if (order != 0 && order != 1)
            throw py::value_error("profile sampling requires order zero or one");
        using skimagexx::GaussianBorder;
        GaussianBorder border;
        if (mode == "reflect") border = GaussianBorder::reflect;
        else if (mode == "mirror") border = GaussianBorder::mirror;
        else if (mode == "nearest") border = GaussianBorder::nearest;
        else if (mode == "grid-constant") border = GaussianBorder::constant;
        else if (mode == "grid-wrap") border = GaussianBorder::wrap;
        else throw py::value_error("unsupported profile sampling mode");
        using skimagexx::PixelType;
        if (Array<bool>::check_(image))
            return sample<bool>(image, coordinates, PixelType::boolean, order, border, cval, workers);
        if (Array<std::uint8_t>::check_(image))
            return sample<std::uint8_t>(image, coordinates, PixelType::uint8, order, border, cval, workers);
        if (Array<float>::check_(image))
            return sample<float>(image, coordinates, PixelType::float32, order, border, cval, workers);
        if (Array<double>::check_(image))
            return sample<double>(image, coordinates, PixelType::float64, order, border, cval, workers);
        throw py::value_error("profile sampling requires prepared native contiguous bool, uint8, float32 or float64");
    }, py::arg("image").noconvert(), py::arg("coordinates").noconvert(),
       py::arg("order"), py::arg("mode"), py::arg("cval"), py::arg("workers"));
}
