#include "bind_array_storage.hpp"
#include "skimagexx/registration_flows.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace {
char native_precision(const py::array& input) {
    const auto dtype = input.dtype();
    if (!dtype.attr("metadata").is_none() || !dtype.attr("isnative").cast<bool>())
        return 0;
    if (dtype.char_() == 'f' && dtype.equal(py::dtype::of<float>()))
        return 'f';
    if (dtype.char_() == 'd' && dtype.equal(py::dtype::of<double>()))
        return 'd';
    return 0;
}

template<class T, class Source>
const T* prepare_values(const void* data, std::size_t size, std::vector<T>& converted) {
    const auto* values = static_cast<const Source*>(data);
    // Leave room for products of gradients and their tensor determinants.
    constexpr double limit = std::is_same_v<T, float> ? 0x1p24 : 0x1p200;
    if constexpr (std::is_same_v<T, Source>) {
        for (std::size_t i = 0; i < size; ++i)
            if (!(std::abs(values[i]) <= limit))
                return nullptr;
        return values;
    } else {
        converted.resize(size);
        for (std::size_t i = 0; i < size; ++i) {
            if (!(std::abs(values[i]) <= limit))
                return nullptr;
            converted[i] = static_cast<T>(values[i]);
        }
        return converted.data();
    }
}

template<class T, class Solve>
py::object run_flow(const py::handle& reference, const py::handle& moving, Solve solve) {
    auto* ndarray = py::detail::npy_api::get().PyArray_Type_;
    if (Py_TYPE(reference.ptr()) != ndarray || Py_TYPE(moving.ptr()) != ndarray)
        return py::none();
    const auto first = py::reinterpret_borrow<py::array>(reference);
    const auto second = py::reinterpret_borrow<py::array>(moving);
    if (first.ndim() != 2 || second.ndim() != 2 ||
        first.shape(0) != second.shape(0) || first.shape(1) != second.shape(1) ||
        first.shape(0) < 2 || first.shape(1) < 2)
        return py::none();

    py::array_t<T> result({py::ssize_t{2}, first.shape(0), first.shape(1)});
    auto* output = result.mutable_data();
    const auto height = static_cast<std::size_t>(result.shape(1));
    const auto width = static_cast<std::size_t>(result.shape(2));
    char precision[2];
    const void* data[2];
    const py::array* inputs[] = {&first, &second};
    for (std::size_t i = 0; i < 2; ++i) {
        const auto& input = *inputs[i];
        if (input.ndim() != 2 || input.shape(0) != result.shape(1) ||
            input.shape(1) != result.shape(2) ||
            !(input.flags() & py::array::c_style) || !input.writeable())
            return py::none();
        precision[i] = native_precision(input);
        if (!precision[i])
            return py::none();
        data[i] = input.data();
        const auto alignment = precision[i] == 'f' ? alignof(float) : alignof(double);
        if (reinterpret_cast<std::uintptr_t>(data[i]) % alignment)
            return py::none();
        if (!input.owndata())
            skimagexx::bindings::check_array_storage(input);
    }
    bool supported;
    {
        py::gil_scoped_release release;
        std::vector<T> converted[2];
        const T* values[2];
        for (std::size_t i = 0; i < 2; ++i) {
            values[i] = precision[i] == 'f'
                ? prepare_values<T, float>(data[i], height * width, converted[i])
                : prepare_values<T, double>(data[i], height * width, converted[i]);
        }
        supported = values[0] && values[1];
        if (supported)
            supported = solve(values[0], values[1], output, height, width);
    }
    if (!supported)
        return py::none();
    return result;
}
} // namespace

void bind_registration_flows(py::module_& module) {
    module.def("optical_flow_ilk", [](py::handle first, py::handle second, bool single,
                                      int radius, int warps, bool gaussian, bool prefilter) {
        if (radius < 0 || radius > 64 || warps <= 0)
            return py::object(py::none());
        const auto solve = [=](auto* a, auto* b, auto* out, auto h, auto w) {
            return skimagexx::optical_flow_ilk_into(a, b, out, h, w, radius, warps, gaussian, prefilter);
        };
        return single ? run_flow<float>(first, second, solve) : run_flow<double>(first, second, solve);
    });
    module.def("optical_flow_tvl1", [](py::handle first, py::handle second, bool single,
                                       double attachment, double tightness, int warps,
                                       int iterations, double tol, bool prefilter) {
        if (!std::isfinite(attachment) || attachment <= 0 ||
            !std::isfinite(tightness) || tightness <= 0 ||
            !std::isfinite(tol) || tol < 0 || warps <= 0 || iterations <= 0)
            return py::object(py::none());
        const auto solve = [=](auto* a, auto* b, auto* out, auto h, auto w) {
            using T = std::remove_pointer_t<decltype(out)>;
            const auto lambda = attachment * tightness;
            const auto diffusion = 0.25 / tightness;
            if (lambda > std::numeric_limits<T>::max() ||
                diffusion > std::numeric_limits<T>::max() ||
                static_cast<T>(lambda) == 0 || static_cast<T>(diffusion) == 0)
                return false;
            skimagexx::optical_flow_tvl1_into(a, b, out, h, w, attachment, tightness,
                                              warps, iterations, tol, prefilter);
            return std::all_of(out, out + 2 * h * w, [](T value) { return std::isfinite(value); });
        };
        return single ? run_flow<float>(first, second, solve) : run_flow<double>(first, second, solve);
    });
}
