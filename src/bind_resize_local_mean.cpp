#include "skimagexx/resize_local_mean.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>

namespace py = pybind11;

void bind_resize_local_mean(py::module_& module) {
    using skimagexx::PixelType;
    const py::object ndarray = py::module_::import("numpy").attr("ndarray");
    module.def("resize_local_mean_bounded", [ndarray](const py::array& image, int rows, int columns,
               bool preserve_range) -> py::object {
        if (!py::type::of(image).is(ndarray) || image.ndim() != 2 || !(image.flags() & py::array::c_style) ||
            !image.attr("flags").attr("aligned").cast<bool>() || !image.dtype().attr("metadata").is_none() ||
            image.shape(0) < 16 || image.shape(0) > 2048 || image.shape(1) < 16 || image.shape(1) > 2048 ||
            rows < 1 || rows > image.shape(0) || columns < 1 || columns > image.shape(1))
            throw py::value_error("resize_local_mean requires a bounded ordinary C-order plane");
        const auto dtype = image.dtype();
        PixelType type;
        if (dtype.equal(py::dtype::of<std::uint8_t>()))
            type = PixelType::uint8;
        else if (dtype.equal(py::dtype::of<std::uint16_t>()))
            type = PixelType::uint16;
        else
            throw py::value_error("resize_local_mean requires native uint8 or uint16");

        skimagexx::bindings::check_array_storage(image);
        const auto* source = image.data();
        const auto source_count = std::size_t(image.size());
        const auto height = int(image.shape(0)), width = int(image.shape(1));
        py::array_t<double> result(py::array::ShapeContainer{py::ssize_t(rows), py::ssize_t(columns)});
        auto* output = result.mutable_data();
        const auto output_count = std::size_t(rows * columns);
        bool done;
        {
            py::gil_scoped_release release;
            done = skimagexx::resize_local_mean_into(source, type, source_count, height, width,
                output, output_count, rows, columns, preserve_range);
        }
        if (!done)
            return py::none();
        return result.attr("view")();
    }, py::arg("image").noconvert(), py::arg("rows"), py::arg("columns"), py::arg("preserve_range"));
}
