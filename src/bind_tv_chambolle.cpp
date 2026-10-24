#include "skimagexx/tv_chambolle.hpp"
#include "skimagexx/tv_denoise.hpp"
#include "tv_denoise_internal.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <stdexcept>
#include <string>
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

    // A separate private finite-loop interface. Stage63 is never evidence of
    // loop support. Scalar/shape metadata and owner lifetimes are marshalled
    // with the GIL held; no dtype/itemsize or Python object access after release.
    module.attr("tv_chambolle_finite_interface") = skimagexx::tv_denoise_interface;
    module.attr("tv_chambolle_finite_capabilities") = py::int_(skimagexx::tv_denoise_capabilities);
    module.attr("tv_chambolle_finite_profiles") = py::make_tuple("1.24.4", "2.5.4");
    module.def("tv_chambolle_finite_run",
        [](py::array image, py::array p, py::array g, py::array d,
           double weight, double eps, std::size_t max_iterations,
           const std::string& profile, std::size_t buffer_elements, bool trace) {
            skimagexx::TvReductionProfile reduction_profile;
            if (profile == "1.24.4") reduction_profile = skimagexx::TvReductionProfile::numpy_1244_buffered;
            else if (profile == "2.5.4") reduction_profile = skimagexx::TvReductionProfile::numpy_254_aligned_contiguous;
            else throw py::value_error("TV finite loop requires an explicitly audited NumPy profile");
            if (image.ndim() < 1 || image.ndim() > 4 || image.size() == 0 || max_iterations < 1 || max_iterations > 20000) {
                PyErr_SetString(PyExc_NotImplementedError, "TV finite loop requires nonempty rank1..4 and iterations1..20000");
                throw py::error_already_set();
            }
            const auto input = const_view(image);
            std::vector<py::ssize_t> shape;
            for (py::ssize_t axis = 0; axis < image.ndim(); ++axis) shape.push_back(image.shape(axis));
            const py::dtype dtype = image.dtype();
            py::array out(dtype, shape);
            py::array squared(dtype, shape);
            py::array norm(dtype, shape);
            const skimagexx::TvDenoiseWorkspace workspace{
                writable_planes(p), writable_planes(g), writable_view(d),
                writable_view(out), writable_view(squared), writable_view(norm),
                const_planes(p), const_planes(g), const_view(out), const_view(norm)};
            const skimagexx::TvDenoiseParameters parameters{
                weight, eps, max_iterations, {reduction_profile, buffer_elements}};
            std::vector<skimagexx::TvDenoiseTrace> rows(trace ? max_iterations : 0);
            skimagexx::TvDenoiseResult result{};
            const char* reason = nullptr;
            {
                py::gil_scoped_release release;
                // One complete native guard, with no Python callback between
                // validation and execution. Unexpected loop exceptions still
                // propagate; only an explicit rejection becomes a provider retry.
                reason = skimagexx::detail::tv_denoise_try_finite_into(input, workspace,
                    parameters, result, trace ? rows.data() : nullptr, rows.size());
            }
            if (reason != nullptr) {
                PyErr_SetString(PyExc_NotImplementedError, reason);
                throw py::error_already_set();
            }
            py::list trace_rows;
            if (trace) {
                for (std::size_t i = 0; i < result.iterations; ++i) {
                    const auto& row = rows[i];
                    trace_rows.append(py::make_tuple(row.square_sum_bits, row.norm_sum_bits,
                        row.energy_bits, row.difference_bits, row.threshold_bits,
                        row.energy_is_float32, row.stopped));
                }
            }
            // Do not expose uninitialized internal out storage for the one-
            // iteration alias case. All numerical updates still ran in C++.
            py::object returned = result.input_alias ? py::object(image) : py::object(out);
            return py::make_tuple(returned, result.iterations, result.input_alias, trace_rows);
        }, py::arg("image").noconvert(), py::arg("p").noconvert(),
           py::arg("g").noconvert(), py::arg("d").noconvert(),
           py::arg("weight"), py::arg("eps"), py::arg("max_iterations"),
           py::arg("profile"), py::arg("buffer_elements"), py::arg("trace") = false);
}
