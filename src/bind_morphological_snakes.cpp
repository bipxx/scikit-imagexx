#include "skimagexx/morphological_snakes.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <type_traits>

namespace py = pybind11;

namespace {

void check_array(const py::array& array) {
    if (Py_TYPE(array.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
        !(array.flags() & py::array::c_style) || !array.writeable() ||
        !array.attr("flags").attr("aligned").cast<bool>() ||
        !array.dtype().attr("isnative").cast<bool>() ||
        !array.dtype().attr("metadata").is_none())
        throw py::value_error("morphological snakes require ordinary writable native C-order arrays");
    skimagexx::bindings::check_array_storage(array);
}

std::vector<std::size_t> shape_of(const py::array& array, py::ssize_t minimum) {
    if ((array.ndim() != 2 && array.ndim() != 3) || array.size() > 8 * 1024 * 1024 ||
        std::any_of(array.shape(), array.shape() + array.ndim(),
                    [&](auto n) { return n < minimum; }))
        throw py::value_error("unsupported morphological snakes shape");
    return {array.shape(), array.shape() + array.ndim()};
}

template<class T>
bool bounded_values(const T* data, std::size_t count) {
    using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    const auto as_bits = [](T value) {
        Bits bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };
    const auto minimum = as_bits(T(1e-12)), maximum = as_bits(T(1e12));
    for (std::size_t i = 0; i < count; ++i) {
        // Strip only the sign bit. Refusing signaling NaNs must not trigger
        // an arithmetic exception before the upstream policy handles them.
        Bits bits;
        std::memcpy(&bits, data + i, sizeof(bits));
        const auto magnitude = bits & (Bits(-1) >> 1);
        if (magnitude != 0 && (magnitude < minimum || magnitude > maximum))
            return false;
    }
    return true;
}

bool single_precision(const py::array& image) {
    const auto code = image.dtype().char_();
    if (code != 'f' && code != 'd')
        throw py::type_error("morphological snakes require float32 or float64 images");
    return code == 'f';
}

template<class F>
py::object dispatch(const py::array& image, F call) {
    const auto single = single_precision(image);
    if (single)
        return call(static_cast<const float*>(nullptr));
    return call(static_cast<const double*>(nullptr));
}

template<class F>
py::object contour(const py::array& image, const py::array& initial, F call) {
    check_array(image);
    check_array(initial);
    const auto shape = shape_of(image, 2);
    if (shape != shape_of(initial, 2) || !initial.dtype().equal(py::dtype::of<std::int8_t>()))
        throw py::value_error("initial level set must be int8 and match image shape");
    py::array_t<std::int8_t> output(shape);
    auto* result = output.mutable_data();
    check_array(image);
    check_array(initial);
    if (shape != shape_of(image, 2) || shape != shape_of(initial, 2) ||
        initial.dtype().char_() != 'b')
        throw py::value_error("initial level set or image changed during allocation");
    return dispatch(image, [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* data = static_cast<T*>(image.data());
        const auto* start = static_cast<const std::int8_t*>(initial.data());
        call(data, start, result, shape);
        return output;
    });
}

} // namespace

void bind_morphological_snakes(py::module_& module) {
    module.def("morphological_snakes_supported", [](const py::array& image) {
        check_array(image);
        const auto single = single_precision(image);
        if (std::fegetround() != FE_TONEAREST)
            return false;
        const auto count = static_cast<std::size_t>(image.size());
        const auto* data = image.data();
        py::gil_scoped_release release;
        return single ? bounded_values(static_cast<const float*>(data), count)
                      : bounded_values(static_cast<const double*>(data), count);
    }, py::arg("image").noconvert());

    module.def("inverse_gaussian_gradient", [](const py::array& image, double alpha, double sigma) {
        check_array(image);
        const auto shape = shape_of(image, 1);
        return dispatch(image, [&](auto* tag) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(tag)>>;
            py::array_t<T> output(shape);
            check_array(image);
            if (shape != shape_of(image, 1) || image.dtype().char_() != py::dtype::of<T>().char_())
                throw py::value_error("gradient image changed during allocation");
            const auto* data = static_cast<const T*>(image.data());
            auto* result = output.mutable_data();
            {
                py::gil_scoped_release release;
                skimagexx::inverse_gaussian_gradient_into(data, result, shape, alpha, sigma);
            }
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("alpha"), py::arg("sigma"));

    module.def("morphological_chan_vese",
        [](const py::array& image, const py::array& initial, std::size_t iterations,
           std::size_t smoothing, double lambda1, double lambda2, bool reverse) {
            return contour(image, initial, [&](auto* data, auto* start, auto* output, const auto& shape) {
                py::gil_scoped_release release;
                skimagexx::morphological_chan_vese_into(data, start, output, shape, iterations,
                    smoothing, lambda1, lambda2, reverse);
            });
        }, py::arg("image").noconvert(), py::arg("initial").noconvert(), py::arg("iterations"),
           py::arg("smoothing"), py::arg("lambda1"), py::arg("lambda2"), py::arg("phase"));

    module.def("morphological_geodesic_active_contour",
        [](const py::array& image, const py::array& initial, std::size_t iterations,
           std::size_t smoothing, int balloon, double threshold, bool reverse) {
            return contour(image, initial, [&](auto* data, auto* start, auto* output, const auto& shape) {
                py::gil_scoped_release release;
                skimagexx::morphological_geodesic_active_contour_into(data, start, output, shape,
                    iterations, smoothing, threshold, balloon, reverse);
            });
        }, py::arg("image").noconvert(), py::arg("initial").noconvert(), py::arg("iterations"),
           py::arg("smoothing"), py::arg("balloon"), py::arg("threshold"), py::arg("phase"));
}
