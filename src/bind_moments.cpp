#include "moments_internal.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

namespace py = pybind11;
namespace {
using Shape = std::vector<py::ssize_t>;
Shape dimensions(const py::array& value) {
    return Shape(value.shape(), value.shape() + value.ndim());
}
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
template<class T> void check(const py::array& value, const Shape& shape, bool aligned = true) {
    skimagexx::bindings::check_array_storage(value);
    if (dimensions(value) != shape || !Array<T>::check_(value) ||
        (aligned && reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T)))
        throw py::value_error("moments input changed during allocation");
}
template<class T> Array<T> prepare(const py::array& value) {
    skimagexx::bindings::check_array_storage(value);
    Array<T> result(value);
    if (reinterpret_cast<std::uintptr_t>(result.data()) % alignof(T)) {
        const auto shape = dimensions(result);
        py::array_t<T> aligned(shape);
        check<T>(result, shape, false);
        std::memcpy(aligned.mutable_data(), result.data(), std::size_t(result.nbytes()));
        return Array<T>(aligned);
    }
    return result;
}
template<class T> py::tuple project(const py::array& calc, const py::array& powers,
                                    std::size_t axis, int workers) {
    auto values = prepare<T>(calc), weights = prepare<T>(powers);
    const auto shape = dimensions(values), power_shape = dimensions(weights);
    if (axis >= shape.size() || power_shape.size() != 2 || power_shape[0] != shape[axis])
        throw py::value_error("moments projection shape/axis mismatch");
    auto output_shape = shape;
    output_shape[axis] = power_shape[1];
    py::array_t<T> output(output_shape);
    // Preparation and output allocation may run callbacks. Recheck the two
    // established typed buffers here before releasing the GIL for native reads.
    check<T>(values, shape);
    check<T>(weights, power_shape);
    const std::vector<std::size_t> native_shape(shape.begin(), shape.end());
    unsigned status;
    {
        py::gil_scoped_release release;
        status = skimagexx::detail::moments_project(values.data(), native_shape, weights.data(),
            std::size_t(power_shape[1]), axis, output.mutable_data(), workers);
    }
    return py::make_tuple(output, status);
}
}

void bind_moments(py::module_& module) {
    module.def("moments_project", [](const py::array& calc, const py::array& powers,
                                     std::size_t axis, int workers) {
        // The source frontend chooses the common dtype once from actual
        // array descriptors, after its literal coordinate/power promotion.
        const auto kind = calc.dtype().kind();
        const auto bytes = calc.itemsize();
        if (kind != powers.dtype().kind() || bytes != powers.itemsize())
            throw py::type_error("moments projection requires matching component types");
        if (kind == 'f' && bytes == 4) return project<float>(calc, powers, axis, workers);
        if (kind == 'f' && bytes == 8) return project<double>(calc, powers, axis, workers);
        if (kind == 'c' && bytes == 8) return project<std::complex<float>>(calc, powers, axis, workers);
        if (kind == 'c' && bytes == 16) return project<std::complex<double>>(calc, powers, axis, workers);
        PyErr_SetString(PyExc_NotImplementedError, "moments projection supports standard float and complex types");
        throw py::error_already_set();
    }, py::arg("calc").noconvert(), py::arg("powers").noconvert(),
       py::arg("axis"), py::arg("workers"));
}
