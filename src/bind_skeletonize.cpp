#include "skeletonize_internal.hpp"

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

const char* noncanonical = "skeletonize requires canonical Boolean storage";

void check_fallback_boolean(const py::array& image) {
    // Layouts outside the native leaf still pass to the compatibility route.
    // Validate raw Boolean storage before offering it to the upstream Zhang LUT.
    for (py::ssize_t row = 0; row < image.shape(0); ++row) {
        for (py::ssize_t col = 0; col < image.shape(1); ++col) {
            if (*static_cast<const std::uint8_t*>(image.data(row, col)) > 1) {
                throw py::value_error(noncanonical);
            }
        }
    }
}

py::object skeletonize(const py::handle& input, const py::handle& method) {
    const auto* type = py::detail::npy_api::get().PyArray_Type_;
    if (Py_TYPE(input.ptr()) != type || !(method.is_none() || (PyUnicode_CheckExact(method.ptr())
            && PyUnicode_CompareWithASCIIString(method.ptr(), "zhang") == 0))) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(input);
    if (image.ndim() != 2 || image.shape(0) == 0 || image.shape(1) == 0) {
        return py::none();
    }
    const auto dtype = image.dtype();
    const bool boolean = dtype.kind() == 'b' && dtype.itemsize() == 1;
    if (!boolean && !(dtype.kind() == 'u' && dtype.itemsize() == 1)) {
        return py::none();
    }
    try {
        skimagexx::bindings::check_array_storage(image);
    } catch (const py::value_error&) {
        throw py::value_error("skeletonize requires bounded input storage");
    }

    const auto rows = std::size_t(image.shape(0));
    const auto cols = std::size_t(image.shape(1));
    if (!(image.flags() & py::array::c_style) || rows > 512 || cols > 512) {
        if (boolean) {
            check_fallback_boolean(image);
        }
        return py::none();
    }
    if (boolean && !image.writeable()) {
        throw py::value_error("buffer source array is read-only");
    }
    // Read the checked input while its Python owner and the GIL are retained.
    const auto kind = boolean ? skimagexx::SkeletonizeInput::boolean : skimagexx::SkeletonizeInput::uint8;
    std::vector<std::uint8_t> padded;
    if (!skimagexx::detail::skeletonize_zhang_padded(
            static_cast<const std::uint8_t*>(image.data()), rows, cols, kind, padded)) {
        throw py::value_error(noncanonical);
    }

    py::array_t<bool> output(std::vector<py::ssize_t>{image.shape(0), image.shape(1)});
    static_assert(sizeof(bool) == 1);
    skimagexx::detail::copy_skeletonize_zhang_rows(
        padded, rows, cols, reinterpret_cast<std::uint8_t*>(output.mutable_data()));
    return output;
}

}  // namespace

void bind_skeletonize(py::module_& module) {
    module.def("skeletonize_zhang", &skeletonize, py::arg("image"), py::arg("method"));
}
