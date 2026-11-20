#include "skimagexx/local_extrema.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace py = pybind11;
namespace {

using skimagexx::LocalExtremaKind;
using skimagexx::LocalExtremaNeighbors;

template<class T>
py::array mask(const py::array& image, LocalExtremaKind kind,
               LocalExtremaNeighbors neighbors, bool allow_borders) {
    const auto rows = image.shape(0);
    const auto columns = image.shape(1);
    const py::ssize_t pad = allow_borders ? 1 : 0;
    const auto max = std::numeric_limits<py::ssize_t>::max();
    if (rows > max - 2 * pad || columns > max - 2 * pad ||
        rows + 2 * pad > max / (columns + 2 * pad)) {
        throw py::value_error("Local extrema mask size overflows");
    }
    const auto stride = columns + 2 * pad;
    py::array_t<std::uint8_t> flags({rows + 2 * pad, stride});
    auto* output = flags.mutable_data();
    if (allow_borders) {
        std::fill_n(output, stride, std::uint8_t{3});
        std::fill_n(output + (rows + 1) * stride, stride, std::uint8_t{3});
        for (py::ssize_t r = 1; r <= rows; ++r) {
            output[r * stride] = 3;
            output[r * stride + stride - 1] = 3;
        }
    }
    const auto offset = pad * (stride + 1);
    skimagexx::local_extrema_2d_into(
        static_cast<const T*>(image.data()), std::size_t(image.size()),
        std::size_t(rows), std::size_t(columns), kind, neighbors, allow_borders,
        output + offset, std::size_t(flags.size() - offset), std::size_t(stride));
    return flags;
}

}  // namespace

void bind_local_extrema(py::module_& module) {
    module.def("local_extrema_integer_mask",
        [](const py::array& image, bool minima, int connectivity, bool allow_borders) -> py::array {
            if (image.ndim() != 2 || image.shape(0) < 3 || image.shape(1) < 3 ||
                !(image.flags() & py::array::c_style) ||
                !image.attr("flags").attr("aligned").cast<bool>() ||
                !image.dtype().attr("isnative").cast<bool>() ||
                (connectivity != 1 && connectivity != 2)) {
                throw py::value_error("Local extrema require an aligned C-order 2D image and connectivity 1 or 2");
            }
            const auto kind = minima ? LocalExtremaKind::minima : LocalExtremaKind::maxima;
            const auto neighbors = connectivity == 1 ? LocalExtremaNeighbors::four : LocalExtremaNeighbors::eight;
            const auto dtype = image.dtype();
            if (dtype.equal(py::dtype::of<std::uint8_t>())) return mask<std::uint8_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::int8_t>())) return mask<std::int8_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::uint16_t>())) return mask<std::uint16_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::int16_t>())) return mask<std::int16_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::uint32_t>())) return mask<std::uint32_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::int32_t>())) return mask<std::int32_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::uint64_t>())) return mask<std::uint64_t>(image, kind, neighbors, allow_borders);
            if (dtype.equal(py::dtype::of<std::int64_t>())) return mask<std::int64_t>(image, kind, neighbors, allow_borders);
            throw py::type_error("Local extrema require an integer dtype");
        }, py::arg("image").noconvert(), py::arg("minima"), py::arg("connectivity"),
        py::arg("allow_borders"));
}
