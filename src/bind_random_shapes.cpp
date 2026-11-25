#include "skimagexx/random_shapes.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>

#include <array>

namespace py = pybind11;

namespace {

skimagexx::RandomShapeIndexView index_view(const py::array& values) {
    const auto size = values.size();
    const auto stride = size < 2 ? py::ssize_t(sizeof(std::intptr_t)) : values.strides(0);
    if (stride != sizeof(std::intptr_t) && stride != 2 * sizeof(std::intptr_t))
        throw py::value_error("random_shapes requires packed or interleaved coordinates");
    return {static_cast<const std::intptr_t*>(values.data()), std::size_t(stride) / sizeof(std::intptr_t)};
}

}  // namespace

void bind_random_shapes(py::module_& module) {
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    const auto ordinary = [ndarray](const py::array& array, int ndim, const py::dtype& dtype) {
        if (!py::type::handle_of(array).is(ndarray) || array.ndim() != ndim)
            return false;
        const auto descriptor = array.dtype();
        return descriptor.equal(dtype) && descriptor.attr("metadata").is_none() &&
            array.attr("flags").attr("aligned").cast<bool>();
    };
    module.def("random_shapes_paint", [ordinary](py::array image, py::array occupied, const py::array& rows,
               const py::array& columns, const py::array& color, bool allow_overlap) {
        const auto bytes = py::dtype::of<std::uint8_t>();
        const auto indices = py::dtype::of<std::intptr_t>();
        if (!ordinary(image, 3, bytes) || !ordinary(occupied, 2, bytes) || !ordinary(color, 1, bytes) ||
            !(image.flags() & py::array::c_style) || !(occupied.flags() & py::array::c_style) ||
            !(color.flags() & py::array::c_style) || !image.writeable() || !occupied.writeable() ||
            image.shape(0) < 1 || image.shape(0) > 256 || image.shape(1) < 1 || image.shape(1) > 256 ||
            image.shape(2) < 1 || image.shape(2) > 8 || occupied.shape(0) != image.shape(0) ||
            occupied.shape(1) != image.shape(1) || color.size() != image.shape(2) ||
            !ordinary(rows, 1, indices) || !ordinary(columns, 1, indices) || rows.size() != columns.size())
            throw py::value_error("random_shapes requires ordinary packed paint buffers");
        const std::array<const py::array*, 5> arrays{&image, &occupied, &rows, &columns, &color};
        for (const auto* value : arrays)
            skimagexx::bindings::check_array_storage(*value);
        const auto y = index_view(rows), x = index_view(columns);
        const auto count = std::size_t(rows.size());
        auto* pixels = static_cast<std::uint8_t*>(image.mutable_data());
        auto* mask = static_cast<std::uint8_t*>(occupied.mutable_data());
        const auto* channels = static_cast<const std::uint8_t*>(color.data());
        const auto pixel_count = std::size_t(image.size()), mask_count = std::size_t(occupied.size());
        const auto height = int(image.shape(0)), width = int(image.shape(1)), channel_count = int(image.shape(2));
        py::gil_scoped_release release;
        return skimagexx::random_shape_paint_into(pixels, pixel_count, mask, mask_count, height, width, channel_count,
            y, x, count, channels, std::size_t(channel_count), allow_overlap);
    }, py::arg("image").noconvert(), py::arg("occupied").noconvert(), py::arg("rows").noconvert(),
       py::arg("columns").noconvert(), py::arg("color").noconvert(), py::arg("allow_overlap"));
}
