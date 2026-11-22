#include "skimagexx/thin.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_thin(py::module_& module) {
    module.def("thin_binary", [](py::handle value, std::size_t max_iterations) -> py::object {
        if (Py_TYPE(value.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
            max_iterations > skimagexx::thin_max_iterations)
            return py::none();
        const auto image = py::reinterpret_borrow<py::array>(value);
        if (image.ndim() != 2 || !(image.flags() & py::array::c_style) ||
            image.itemsize() != 1 || image.shape(0) <= 0 || image.shape(1) <= 0 ||
            image.shape(0) > static_cast<py::ssize_t>(skimagexx::thin_max_extent) ||
            image.shape(1) > static_cast<py::ssize_t>(skimagexx::thin_max_extent))
            return py::none();
        const auto kind = image.dtype().kind();
        if (kind != 'b' && kind != 'u') return py::none();
        try {
            if (!image.owndata()) skimagexx::bindings::check_array_storage(image);
        } catch (const py::value_error&) {
            return py::none();
        }
        const auto rows = image.shape(0);
        const auto cols = image.shape(1);
        py::array_t<bool> output(py::array::ShapeContainer{rows, cols});
        if (!skimagexx::thin_into(
                static_cast<const std::uint8_t*>(image.data()), static_cast<std::size_t>(rows),
                static_cast<std::size_t>(cols), kind == 'b', max_iterations,
                static_cast<std::uint8_t*>(static_cast<void*>(output.mutable_data()))))
            return py::none();
        return output;
    }, py::arg("image"), py::arg("max_iterations").noconvert());
}
