#include <skimagexx/corner_fast.hpp>

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <vector>

namespace py = pybind11;

namespace {

enum class FastKind { none, u8, f32, f64 };

FastKind fast_kind(const py::array& image) {
    if (Py_TYPE(image.ptr()) != py::detail::npy_api::get().PyArray_Type_
        || image.ndim() != 2
        || !(image.flags() & py::array::c_style)
        || !image.attr("flags").attr("aligned").cast<bool>()) {
        return FastKind::none;
    }
    const auto dtype = image.dtype();
    if (!dtype.attr("metadata").is_none()) {
        return FastKind::none;
    }
    if (dtype.equal(py::dtype::of<std::uint8_t>())) {
        return FastKind::u8;
    }
    if (dtype.equal(py::dtype::of<float>())) {
        return image.writeable() ? FastKind::f32 : FastKind::none;
    }
    if (dtype.equal(py::dtype::of<double>())) {
        return image.writeable() ? FastKind::f64 : FastKind::none;
    }
    return FastKind::none;
}

template <typename OutputT, typename Fn>
py::object run_fast(const py::array& image, FastKind expected_kind,
    py::ssize_t rows, py::ssize_t columns, Fn&& fn) {
    py::array_t<OutputT> owner(std::vector<py::ssize_t>{rows, columns});
    if (fast_kind(image) != expected_kind || image.shape(0) != rows
        || image.shape(1) != columns) {
        return py::none();
    }
    if (!image.owndata()) {
        try {
            skimagexx::bindings::check_array_storage(image);
        } catch (const py::value_error&) {
            return py::none();
        }
    }
    if (!fn(owner.mutable_data())) {
        return py::none();
    }
    return owner;
}

py::object make_fast_response(
    const py::handle& input, const py::handle& n_handle, const py::handle& threshold_handle) {
    if (!py::isinstance<py::array>(input)
        || !PyLong_CheckExact(n_handle.ptr())
        || !PyFloat_CheckExact(threshold_handle.ptr())) {
        return py::none();
    }
    int overflow = 0;
    const long supplied_n = PyLong_AsLongAndOverflow(n_handle.ptr(), &overflow);
    if (overflow || supplied_n < skimagexx::corner_fast_min_n
        || supplied_n > skimagexx::corner_fast_max_n) {
        return py::none();
    }
    const double threshold = PyFloat_AS_DOUBLE(threshold_handle.ptr());
    if (!std::isfinite(threshold) || threshold < 0.0
        || threshold > skimagexx::corner_fast_max_threshold) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(input);
    const FastKind kind = fast_kind(image);
    if (kind == FastKind::none) {
        return py::none();
    }
    const auto rows = image.shape(0);
    const auto columns = image.shape(1);
    if (rows < 0 || columns < 0 || !skimagexx::corner_fast_shape_supported(
            static_cast<std::size_t>(rows), static_cast<std::size_t>(columns))) {
        return py::none();
    }
    const int n = static_cast<int>(supplied_n);
    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(columns);
    if (kind == FastKind::u8) {
        return run_fast<double>(image, kind, rows, columns, [&](double* out) {
            return skimagexx::corner_fast_u8_into(
                static_cast<const std::uint8_t*>(image.data()), r, c, n, threshold, out);
        });
    }
    if (kind == FastKind::f32) {
        return run_fast<float>(image, kind, rows, columns, [&](float* out) {
            return skimagexx::corner_fast_f32_into(
                static_cast<const float*>(image.data()), r, c, n,
                static_cast<float>(threshold), out);
        });
    }
    return run_fast<double>(image, kind, rows, columns, [&](double* out) {
        return skimagexx::corner_fast_f64_into(
            static_cast<const double*>(image.data()), r, c, n, threshold, out);
    });
}

}  // namespace

void bind_corner_fast(py::module_& module) {
    module.def("corner_fast_apply", &make_fast_response,
        py::arg("image"), py::arg("n"), py::arg("threshold"));
}
