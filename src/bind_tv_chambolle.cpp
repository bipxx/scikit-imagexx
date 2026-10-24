#include "skimagexx/tv_chambolle.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
skimagexx::TvScalarType scalar_type(const py::array& value) {
    if (value.dtype().kind() != 'f' ||
        !value.dtype().attr("isnative").cast<bool>())
        throw py::type_error("TV stages require native float32 or float64 storage");
    if (value.itemsize() == 4) return skimagexx::TvScalarType::float32;
    if (value.itemsize() == 8) return skimagexx::TvScalarType::float64;
    throw py::type_error("TV stages require native float32 or float64 storage");
}

std::vector<std::size_t> spatial_shape(const py::array& value,
                                     py::ssize_t first_axis = 0) {
    std::vector<std::size_t> shape;
    for (py::ssize_t axis = first_axis; axis < value.ndim(); ++axis)
        shape.push_back(static_cast<std::size_t>(value.shape(axis)));
    return shape;
}

std::vector<std::ptrdiff_t> spatial_strides(const py::array& value) {
    std::vector<std::ptrdiff_t> strides;
    for (py::ssize_t axis = 0; axis < value.ndim(); ++axis)
        strides.push_back(static_cast<std::ptrdiff_t>(value.strides(axis)));
    return strides;
}

void require_writable(const py::array& value) {
    if (!value.writeable()) throw py::value_error("TV stage output is read-only");
}

void require_planes(const py::array& value) {
    if (value.ndim() < 2 || value.ndim() > 5 ||
        value.shape(0) != value.ndim() - 1)
        throw py::value_error("TV stage planes must have shape (D,)+spatial_shape");
    if (!(value.flags() & py::array::c_style))
        throw py::value_error("TV stage planes must be C contiguous");
}

skimagexx::TvConstView const_view(const py::array& value) {
    return {value.data(), scalar_type(value), spatial_shape(value),
            spatial_strides(value)};
}

skimagexx::TvView writable_view(py::array& value) {
    require_writable(value);
    return {value.mutable_data(), scalar_type(value), spatial_shape(value),
            spatial_strides(value)};
}

skimagexx::TvConstPlanes const_planes(const py::array& value) {
    require_planes(value);
    return {value.data(), scalar_type(value), spatial_shape(value, 1)};
}

skimagexx::TvPlanes writable_planes(py::array& value) {
    require_planes(value);
    require_writable(value);
    return {value.mutable_data(), scalar_type(value), spatial_shape(value, 1)};
}
}

// This bridge exposes only the private materialized stages. It performs no
// Python allocation, conversion, reduction, convergence or channel processing.
// Validate Python buffer metadata with the GIL held. The native entry validates
// shape/type/interval/alias relationships before writing; Python owners remain
// alive across the released-GIL call through the lambda's array arguments.
void bind_tv_chambolle(py::module_& module) {
    module.attr("tv_chambolle_interface") = skimagexx::tv_chambolle_interface;
    module.attr("tv_chambolle_capabilities") =
        py::int_(skimagexx::tv_chambolle_capabilities());
    module.def("tv_chambolle_divergence_into", [](py::array p, py::array d) {
        const auto input = const_planes(p);
        const auto output = writable_view(d);
        py::gil_scoped_release release;
        skimagexx::tv_chambolle_divergence_into(input, output);
    }, py::arg("p").noconvert(), py::arg("d").noconvert());
    module.def("tv_chambolle_gradient_into", [](py::array out, py::array g) {
        const auto input = const_view(out);
        const auto output = writable_planes(g);
        py::gil_scoped_release release;
        skimagexx::tv_chambolle_gradient_into(input, output);
    }, py::arg("out").noconvert(), py::arg("g").noconvert());
    module.def("tv_chambolle_norm_into", [](py::array g, py::array norm) {
        const auto input = const_planes(g);
        const auto output = writable_view(norm);
        py::gil_scoped_release release;
        skimagexx::tv_chambolle_norm_into(input, output);
    }, py::arg("g").noconvert(), py::arg("norm").noconvert());
    module.def("tv_chambolle_dual_into",
        [](py::array p, py::array g, py::array denominator, double tau_work) {
            const auto output = writable_planes(p);
            const auto gradient = const_planes(g);
            const auto divisor = const_view(denominator);
            py::gil_scoped_release release;
            skimagexx::tv_chambolle_dual_into(output, gradient, divisor, tau_work);
        }, py::arg("p").noconvert(), py::arg("g").noconvert(),
           py::arg("denominator").noconvert(), py::arg("tau_work"));
}
