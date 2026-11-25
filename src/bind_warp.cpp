#include "skimagexx/warp.hpp"
#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using namespace skimagexx;

namespace {
bool is_dtype(const py::dtype& dtype, const py::dtype& expected) {
    return dtype.is(expected) ||
           (dtype.char_() == expected.char_() && dtype.equal(expected) &&
            bindings::is_native_byteorder(dtype));
}

py::object sample(const py::array& image, std::array<py::ssize_t, 2> dimensions,
                  int mapping, const std::array<double, 9>& params, int order,
                  int border, double cval, bool clip, bool preserve_range,
                  bool legacy_reflect, const py::object& mapped, bool round_clip_cval) {
    if (Py_TYPE(image.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
        (image.ndim() != 2 && image.ndim() != 3) || image.size() == 0 ||
        !(image.flags() & py::array::c_style) || !image.writeable() ||
        dimensions[0] <= 0 || dimensions[1] <= 0 || mapping < 0 || mapping > 3 ||
        border < 0 || border > 4 || order < 0 || order > 3 || !std::isfinite(cval))
        return py::none();
    const auto dtype = image.dtype();
    if (!dtype.attr("metadata").is_none()) return py::none();
    PixelType type;
    if (is_dtype(dtype, py::dtype::of<double>())) type = PixelType::float64;
    else if (is_dtype(dtype, py::dtype::of<float>())) type = PixelType::float32;
    else if (is_dtype(dtype, py::dtype::of<std::uint8_t>())) type = PixelType::uint8;
    else if (is_dtype(dtype, py::dtype::of<bool>()) && order == 0) type = PixelType::boolean;
    else return py::none();
    if (reinterpret_cast<std::uintptr_t>(image.data()) % std::size_t(dtype.itemsize()))
        return py::none();
    if (!image.owndata()) {
        try {
            bindings::check_array_storage(image);
        } catch (const py::value_error&) {
            return py::none();
        }
    }
    const auto channels = image.ndim() == 2 ? 1 : image.shape(2);
    // A SciPy spline prefilter for color images also traverses the band axis.
    if (image.ndim() == 3 && order > 1 && !(mapping == 0 && order == 3))
        return py::none();
    const WarpShape source{std::size_t(image.shape(0)), std::size_t(image.shape(1)),
                           std::size_t(channels)};
    const WarpShape target{std::size_t(dimensions[0]), std::size_t(dimensions[1]), source.channels};
    const double* homogeneous = nullptr;
    if (!mapped.is_none()) {
        if (mapping != 0 || order != 0 ||
            Py_TYPE(mapped.ptr()) != py::detail::npy_api::get().PyArray_Type_)
            return py::none();
        const auto h = py::reinterpret_borrow<py::array>(mapped);
        const auto descriptor = h.dtype();
        if (h.ndim() != 2 || h.shape(1) != 3 || h.shape(0) / dimensions[0] != dimensions[1] ||
            h.shape(0) % dimensions[0] || !(h.flags() & py::array::c_style) ||
            !descriptor.attr("metadata").is_none() || !is_dtype(descriptor, py::dtype::of<double>()) ||
            reinterpret_cast<std::uintptr_t>(h.data()) % alignof(double))
            return py::none();
        if (!h.owndata()) {
            try {
                bindings::check_array_storage(h);
            } catch (const py::value_error&) {
                return py::none();
            }
        }
        homogeneous = static_cast<const double*>(h.data());
    }
    const py::ssize_t shape[]{dimensions[0], dimensions[1], channels};
    const int rank = image.ndim() == 3 ? 3 : 2;
    const auto out_dtype = type == PixelType::uint8 && order != 0 ? py::dtype::of<double>() : dtype;
    if (dimensions[0] > PY_SSIZE_T_MAX / out_dtype.itemsize() / dimensions[1] / channels)
        return py::none();
    auto output = bindings::empty_c_array(out_dtype, shape, rank);
    const auto* input_data = image.data();
    auto* out_data = output.mutable_data();
    bool done;
    {
        py::gil_scoped_release release;
        done = warp_into(input_data, out_data, type, source, target,
                         {static_cast<WarpMapping>(mapping), params, homogeneous},
                         {order, static_cast<GaussianBorder>(border), cval, clip,
                          preserve_range, legacy_reflect, round_clip_cval});
    }
    return done ? py::object(output) : py::none();
}

py::object coords(const std::vector<py::ssize_t>& dimensions,
                  const std::array<double, 9>& matrix, bool single) {
    if ((dimensions.size() != 2 && dimensions.size() != 3) ||
        std::any_of(dimensions.begin(), dimensions.end(), [](auto n) { return n <= 0; }))
        return py::none();
    const bool color = dimensions.size() == 3;
    const WarpShape shape{std::size_t(dimensions[0]), std::size_t(dimensions[1]),
                          color ? std::size_t(dimensions[2]) : 1};
    const py::ssize_t output_shape[]{py::ssize_t(dimensions.size()), dimensions[0],
                                     dimensions[1], color ? dimensions[2] : 1};
    auto output = bindings::empty_c_array(
        single ? py::dtype::of<float>() : py::dtype::of<double>(), output_shape,
        static_cast<int>(dimensions.size()) + 1);
    void* data = output.mutable_data();
    bool done;
    {
        py::gil_scoped_release release;
        const WarpMap map{WarpMapping::matrix, matrix};
        if (single) done = warp_coords_into(static_cast<float*>(data), shape, color, map);
        else done = warp_coords_into(static_cast<double*>(data), shape, color, map);
    }
    return done ? py::object(output) : py::none();
}
}

void bind_warp(py::module_& module) {
    module.def("warp_sample", &sample, py::arg("image").noconvert(),
        py::arg("shape"), py::arg("mapping"), py::arg("params"), py::arg("order"),
        py::arg("border"), py::arg("cval"), py::arg("clip"),
        py::arg("preserve_range"), py::arg("legacy_reflect"), py::arg("mapped") = py::none(),
        py::arg("round_clip_cval") = true);
    module.def("warp_coordinates", &coords);
    module.def("warp_matrix_storage", [](const py::array& matrix) {
        if (Py_TYPE(matrix.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
            !matrix.dtype().attr("metadata").is_none()) return false;
        try {
            bindings::check_array_storage(matrix);
        } catch (const py::value_error&) {
            return false;
        }
        return true;
    }, py::arg("matrix").noconvert());
}
