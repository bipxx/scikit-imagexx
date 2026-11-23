#include "skimagexx/convex_hull_image.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_convex_hull_image(py::module_& module) {
    module.def("convex_hull_image_binary", [](py::handle value, bool offset, bool borders) -> py::object {
        if (Py_TYPE(value.ptr()) != py::detail::npy_api::get().PyArray_Type_) return py::none();
        const auto image = py::reinterpret_borrow<py::array>(value);
        if (image.ndim() != 2 || !(image.flags() & py::array::c_style) || image.itemsize() != 1)
            return py::none();
        const auto kind = image.dtype().kind();
        if (kind != 'b' && kind != 'u') return py::none();
        const auto rows = image.shape(0), cols = image.shape(1);
        const auto limit = static_cast<py::ssize_t>(skimagexx::convex_hull_image_max_extent);
        if (rows <= 0 || cols <= 0 || rows > limit || cols > limit) return py::none();
        try {
            if (!image.owndata()) skimagexx::bindings::check_array_storage(image);
        } catch (const py::value_error&) {
            return py::none();
        }
        py::array_t<bool> output(py::array::ShapeContainer{rows, cols});
        if (!skimagexx::convex_hull_image_into(
                static_cast<const std::uint8_t*>(image.data()), static_cast<std::size_t>(rows),
                static_cast<std::size_t>(cols), offset, borders,
                reinterpret_cast<std::uint8_t*>(output.mutable_data())))
            return py::none();
        return output;
    }, py::arg("image"), py::arg("offset").noconvert(), py::arg("borders").noconvert());
}
