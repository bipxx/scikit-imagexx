#include "skimagexx/polynomial.hpp"
#include "bind_array_storage.hpp"

#include <cstdint>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace {

bool doubles(const py::array& input, py::ssize_t dimensions) {
    if (input.ndim() != dimensions || !(input.flags() & py::array::c_style) ||
        !input.dtype().is(py::dtype::of<double>()) ||
        reinterpret_cast<std::uintptr_t>(input.data()) % alignof(double) != 0) return false;
    try {
        skimagexx::bindings::check_array_storage(input);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

const double* data(const py::array& input) {
    return static_cast<const double*>(input.data());
}

py::object apply(const py::array& coords, const py::array& params) {
    if (!doubles(coords, 2) || !doubles(params, 2) ||
        coords.shape(1) != 2 || params.shape(0) != 2) return py::none();
    py::array_t<double> output({coords.shape(0), py::ssize_t{2}});
    const auto* points = data(coords);
    const auto* coefficients = data(params);
    const auto count = static_cast<std::size_t>(coords.shape(0));
    const auto terms = static_cast<std::size_t>(params.shape(1));
    auto* dst = output.mutable_data();
    bool finite;
    if (count < 1024) {
        finite = skimagexx::polynomial_apply(points, count, coefficients, terms, dst);
    } else {
        py::gil_scoped_release release;
        finite = skimagexx::polynomial_apply(points, count, coefficients, terms, dst);
    }
    if (!finite) return py::none();
    return output;
}

py::object constraints(const py::array& src, const py::array& dst, unsigned order,
                       const py::object& weights) {
    if (order > skimagexx::polynomial_max_order || !doubles(src, 2) || !doubles(dst, 2) ||
        src.shape(1) != 2 || dst.shape(1) != 2 || dst.shape(0) != src.shape(0)) return py::none();
    const auto count = src.shape(0);
    const py::ssize_t columns = (order + 1) * (order + 2) + 1;
    // An underdetermined nullspace is intentionally handled by the provider.
    if (count < columns / 2 || count > PY_SSIZE_T_MAX / (2 * columns)) return py::none();
    const double* confidences = nullptr;
    if (!weights.is_none()) {
        if (!py::isinstance<py::array>(weights)) return py::none();
        const auto weights_array = py::reinterpret_borrow<py::array>(weights);
        if (!doubles(weights_array, 1) || weights_array.shape(0) != count) return py::none();
        confidences = data(weights_array);
    }
    py::array_t<double> equations({2 * count, columns});
    const auto* src_points = data(src);
    const auto* dst_points = data(dst);
    auto* rows = equations.mutable_data();
    bool valid;
    {
        py::gil_scoped_release release;
        valid = skimagexx::polynomial_constraints(src_points, dst_points, confidences,
            static_cast<std::size_t>(count), order, rows);
    }
    if (!valid) return py::none();
    return equations;
}

}  // namespace

void bind_polynomial(py::module_& module) {
    using py::arg;
    module.def("polynomial_apply", &apply, arg("coords").noconvert(), arg("params").noconvert());
    module.def("polynomial_constraints", &constraints, arg("src").noconvert(),
               arg("dst").noconvert(), arg("order"), arg("weights"));
}

#ifdef SKIMAGEXX_POLYNOMIAL_PRIVATE_MODULE
PYBIND11_MODULE(_polynomial_native, module) {
    bind_polynomial(module);
}
#endif
