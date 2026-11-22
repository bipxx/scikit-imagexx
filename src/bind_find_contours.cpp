#include "skimagexx/find_contours.hpp"

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <optional>
#include <type_traits>

namespace py = pybind11;

namespace {

std::optional<bool> high_option(const py::handle& option) {
    if (!PyUnicode_CheckExact(option.ptr())) {
        return std::nullopt;
    }
    if (PyUnicode_CompareWithASCIIString(option.ptr(), "low") == 0) {
        return false;
    }
    if (PyUnicode_CompareWithASCIIString(option.ptr(), "high") == 0) {
        return true;
    }
    return std::nullopt;
}

std::optional<double> scalar_level(const py::handle& level) {
    if (level.is_none()) {
        return 0.0;
    }
    if (PyFloat_CheckExact(level.ptr())) {
        return PyFloat_AS_DOUBLE(level.ptr());
    }
    if (PyLong_CheckExact(level.ptr())) {
        int overflow = 0;
        const auto value = PyLong_AsLongLongAndOverflow(level.ptr(), &overflow);
        if (!overflow) {
            return static_cast<double>(value);
        }
    }
    return std::nullopt;
}

bool plain_array(const py::handle& input) {
    return Py_TYPE(input.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

bool plain_dtype(const py::dtype& dtype, const py::dtype& expected) {
    return dtype.is(expected)
        || (dtype.num() == expected.num()
            && py::detail::npy_api::get().PyArray_EquivTypes_(dtype.ptr(), expected.ptr())
            && dtype.attr("isnative").cast<bool>() && dtype.attr("metadata").is_none());
}

bool has_storage(const py::array& array) {
    try {
        skimagexx::bindings::check_array_storage(array);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

template <class Pixel>
bool run(const py::array& array, const std::uint8_t* mask, skimagexx::ContourOptions options,
         skimagexx::Contours& result) {
    return reinterpret_cast<std::uintptr_t>(array.data()) % alignof(Pixel) == 0
        && skimagexx::find_contours(static_cast<const Pixel*>(array.data()), mask,
              std::size_t(array.shape(0)), std::size_t(array.shape(1)), options, result);
}

py::object find_contours(const py::handle& input, const py::handle& level, const py::handle& connectivity,
                         const py::handle& orientation, const py::handle& mask) {
    const auto connected = high_option(connectivity);
    const auto value = scalar_level(level);
    if (!connected || !high_option(orientation) || !value || !plain_array(input)) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(input);
    if (image.ndim() != 2 || !(image.flags() & py::array::c_style)
        || !skimagexx::find_contours_shape_supported(std::size_t(image.shape(0)), std::size_t(image.shape(1)))
        || !has_storage(image)) {
        return py::none();
    }

    const std::uint8_t* mask_data = nullptr;
    if (!mask.is_none()) {
        if (!plain_array(mask)) {
            return py::none();
        }
        const auto array = py::reinterpret_borrow<py::array>(mask);
        if (array.ndim() != 2 || array.shape(0) != image.shape(0) || array.shape(1) != image.shape(1)
            || !(array.flags() & py::array::c_style) || !plain_dtype(array.dtype(), py::dtype::of<bool>())
            || !has_storage(array)) {
            return py::none();
        }
        mask_data = static_cast<const std::uint8_t*>(array.data());
    }
    const auto dtype = image.dtype();
    const skimagexx::ContourOptions options{*value, level.is_none(), *connected};
    skimagexx::Contours result;
    // Both owners remain held by the call. Keep the GIL while reading them;
    // all output allocations happen after those reads have finished.
    bool supported = false;
    if (plain_dtype(dtype, py::dtype::of<double>())) {
        supported = run<double>(image, mask_data, options, result);
    } else if (plain_dtype(dtype, py::dtype::of<float>())) {
        supported = run<float>(image, mask_data, options, result);
    } else if (plain_dtype(dtype, py::dtype::of<std::uint8_t>())) {
        supported = run<std::uint8_t>(image, mask_data, options, result);
    }
    if (!supported) {
        return py::none();
    }

    static_assert(std::is_trivially_copyable_v<skimagexx::ContourPoint>);
    static_assert(sizeof(skimagexx::ContourPoint) == 2 * sizeof(double));
    static_assert(offsetof(skimagexx::ContourPoint, column) == sizeof(double));
    py::list output(result.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto& contour = result[i];
        py::array_t<double> array(std::vector<py::ssize_t>{static_cast<py::ssize_t>(contour.size()), 2});
        std::memcpy(array.mutable_data(), contour.data(), contour.size() * sizeof(skimagexx::ContourPoint));
        output[i] = std::move(array);
    }
    return output;
}

}  // namespace

void bind_find_contours(py::module_& module) {
    module.def("find_contours_finite", &find_contours, py::arg("image"), py::arg("level"),
               py::arg("fully_connected"), py::arg("positive_orientation"), py::arg("mask"));
}
