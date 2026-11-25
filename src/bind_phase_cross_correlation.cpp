#include "skimagexx/phase_cross_correlation.hpp"

#include "bind_array_storage.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/numpy.h>
#include <type_traits>

namespace py = pybind11;

namespace {

bool admitted_array(const py::array& array) {
    return array.ndim() == 2 && array.shape(0) > 0 && array.shape(1) > 0 &&
           (array.flags() & py::array::c_style) && array.writeable();
}

template<class T>
bool typed_array(const py::array& array) {
    const auto dtype = array.dtype();
    // NumPy gives longdouble a distinct type number even when its byte
    // representation is identical to double, as it is on Windows.
    return dtype.num() == py::dtype::of<T>().num() &&
           array.itemsize() == static_cast<py::ssize_t>(sizeof(T)) &&
           dtype.attr("isnative").cast<bool>() && dtype.attr("metadata").is_none() &&
           reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) == 0;
}

template<class T, class Input>
py::object execute(const py::array& reference, const py::array& moving,
                   skimagexx::PhaseCorrelationOptions options, const py::object& scalar,
                   const py::object& error_scalar) {
    if (!typed_array<Input>(reference) || !typed_array<Input>(moving)) return py::none();
    if (!reference.owndata()) skimagexx::bindings::check_array_storage(reference);
    if (!moving.owndata()) skimagexx::bindings::check_array_storage(moving);
    const auto height = static_cast<std::size_t>(reference.shape(0));
    const auto width = static_cast<std::size_t>(reference.shape(1));
    const auto* a = static_cast<const Input*>(reference.data());
    const auto* b = static_cast<const Input*>(moving.data());
    skimagexx::PhaseCorrelationResult<T> result{};
    bool success;
    {
        py::gil_scoped_release release;
        if constexpr (std::is_same_v<T, Input>)
            success = skimagexx::phase_cross_correlation(a, b, height, width, options, result);
        else
            success = skimagexx::phase_cross_correlation_fourier(a, b, height, width, options, result);
    }
    if (!success) return py::none();
    py::array_t<T> shift(2);
    std::copy(result.shift.begin(), result.shift.end(), shift.mutable_data());
    return py::make_tuple(shift, error_scalar(result.error), scalar(result.phasediff));
}

} // namespace

void bind_phase_cross_correlation(py::module_& module) {
    auto numpy = py::module_::import("numpy");
    auto single = numpy.attr("float32");
    auto wide = numpy.attr("float64");
    auto legacy = numpy.attr("result_type")(numpy.attr("complex64")(0), 1.0)
                     .attr("itemsize").cast<int>() > static_cast<int>(sizeof(std::complex<float>));
    py::object single_error = legacy ? wide : single;
    module.def("phase_cross_correlation",
        [single = py::object(single), wide = py::object(wide), single_error]
        (py::handle left, py::handle right, int factor, bool fourier, bool normalize) -> py::object {
            const auto* array_type = py::detail::npy_api::get().PyArray_Type_;
            if (Py_TYPE(left.ptr()) != array_type || Py_TYPE(right.ptr()) != array_type ||
                factor < 1 || factor > skimagexx::phase_correlation_max_upsample)
                return py::none();
            const auto reference = py::reinterpret_borrow<py::array>(left);
            const auto moving = py::reinterpret_borrow<py::array>(right);
            if (!admitted_array(reference) || !admitted_array(moving) ||
                reference.shape(0) != moving.shape(0) || reference.shape(1) != moving.shape(1))
                return py::none();
            const skimagexx::PhaseCorrelationOptions options{factor, normalize};
            const auto itemsize = reference.itemsize();
            if (fourier) {
                if (itemsize == sizeof(std::complex<float>))
                    return execute<float, std::complex<float>>(reference, moving, options, single, single_error);
                return execute<double, std::complex<double>>(reference, moving, options, wide, wide);
            }
            if (itemsize == sizeof(float))
                return execute<float, float>(reference, moving, options, single, single_error);
            return execute<double, double>(reference, moving, options, wide, wide);
        }, py::arg("reference"), py::arg("moving"), py::arg("factor").noconvert(),
        py::arg("fourier").noconvert(), py::arg("normalize").noconvert());
}
