#include <skimagexx/corner_moravec.hpp>

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace {

bool byte_image(const py::array& image) {
    const auto dtype = image.dtype();
    return image.ndim() == 2 && (image.flags() & py::array::c_style)
        && image.itemsize() == 1 && dtype.kind() == 'u'
        && dtype.attr("metadata").is_none();
}

py::object make_response(const py::handle& input, const py::handle& radius) {
    if (Py_TYPE(input.ptr()) != py::detail::npy_api::get().PyArray_Type_
        || !PyLong_CheckExact(radius.ptr())) {
        return py::none();
    }
    int overflow = 0;
    const long supplied = PyLong_AsLongAndOverflow(radius.ptr(), &overflow);
    if (overflow || supplied < 1 || supplied > skimagexx::corner_moravec_max_window) {
        return py::none();
    }
    const int window = static_cast<int>(supplied);
    const auto image = py::reinterpret_borrow<py::array>(input);
    if (!byte_image(image)) {
        return py::none();
    }
    const auto rows = image.shape(0);
    const auto columns = image.shape(1);
    if (!skimagexx::corner_moravec_shape_supported(std::size_t(rows), std::size_t(columns), window)) {
        return py::none();
    }

    py::array_t<double> owner(std::vector<py::ssize_t>{rows, columns});
    // NumPy allocated the output. Check the input's current descriptor and
    // contiguous backing before borrowing its bytes for the native scan.
    if (!byte_image(image) || image.shape(0) != rows || image.shape(1) != columns) {
        return py::none();
    }
    if (!image.owndata()) {
        try {
            skimagexx::bindings::check_array_storage(image);
        } catch (const py::value_error&) {
            return py::none();
        }
    }
    if (!skimagexx::corner_moravec_u8_into(static_cast<const std::uint8_t*>(image.data()),
            std::size_t(rows), std::size_t(columns), window, owner.mutable_data())) {
        return py::none();
    }
    return owner;
}

}  // namespace

void bind_corner_moravec(py::module_& module) {
    module.def("corner_moravec_u8", &make_response, py::arg("image"), py::arg("window_size"));
}
