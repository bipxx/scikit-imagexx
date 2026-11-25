#include <skimagexx/image.hpp>
#include <skimagexx/restoration_extra.hpp>
#include "bind_array_storage.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <type_traits>

namespace py = pybind11;
namespace {
using skimagexx::RestorationShape;

template<class T> const T* dense(const py::array& input) {
    if (input.dtype().num() != py::dtype::of<T>().num() ||
        !input.dtype().attr("isnative").cast<bool>() ||
        !(input.flags() & py::array::c_style) || input.size() == 0 ||
        reinterpret_cast<std::uintptr_t>(input.data()) % alignof(T))
        throw py::value_error("restoration requires a nonempty native aligned dense array");
    skimagexx::bindings::check_array_storage(input);
    return static_cast<const T*>(input.data());
}

RestorationShape shape_of(const py::array& input) {
    if (input.ndim() == 0) throw py::value_error("restoration requires an array with axes");
    return {input.shape(), input.shape() + input.ndim()};
}

RestorationShape spectrum_of(RestorationShape shape) {
    if (shape.empty()) throw py::value_error("restoration requires an array with axes");
    shape.back() = shape.back() / 2 + 1;
    return shape;
}

template<class Fn> auto real_dispatch(const py::array& input, Fn fn) {
    if (input.dtype().num() == py::dtype::of<float>().num())
        return fn(dense<float>(input));
    return fn(dense<double>(input));
}
} // namespace

void bind_restoration_extra(py::module_& module) {
    module.def("restoration_extra_transfer", [](const py::array& input, const RestorationShape& shape) {
        return real_dispatch(input, [&](auto kernel) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(kernel)>>;
            const auto kernel_shape = shape_of(input);
            py::array_t<std::complex<T>> output(spectrum_of(shape));
            auto* out = output.mutable_data();
            const int workers = skimagexx::get_num_threads();
            py::gil_scoped_release release;
            skimagexx::restoration_transfer(kernel, kernel_shape, shape, out, workers);
            return output;
        });
    }, py::arg("input").noconvert(), py::arg("shape"));

    module.def("restoration_extra_laplacian", [](const RestorationShape& shape, bool single) -> py::array {
        if (single) {
            py::array_t<float> output(spectrum_of(shape));
            auto* out = output.mutable_data();
            py::gil_scoped_release release;
            skimagexx::restoration_laplacian(shape, out);
            return output;
        }
        py::array_t<double> output(spectrum_of(shape));
        auto* out = output.mutable_data();
        py::gil_scoped_release release;
        skimagexx::restoration_laplacian(shape, out);
        return output;
    });

    module.def("restoration_extra_wiener", [](const py::array& image, const py::array& transfer,
                                              const py::array& reg, double balance, bool clip) {
        return real_dispatch(image, [&](auto source) -> py::object {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(image);
            const auto spectrum = spectrum_of(shape);
            if (shape_of(transfer) != spectrum || shape_of(reg) != spectrum)
                throw py::value_error("restoration spectrum shape mismatch");
            const auto* tf = dense<std::complex<T>>(transfer);
            const auto* regularizer = dense<T>(reg);
            py::array_t<T> output(shape);
            auto* out = output.mutable_data();
            const int workers = skimagexx::get_num_threads();
            bool success;
            {
                py::gil_scoped_release release;
                success = skimagexx::restoration_wiener(source, shape, tf, regularizer,
                                                       T(balance), clip, out, workers);
            }
            if (!success) return py::none();
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("transfer").noconvert(),
       py::arg("reg").noconvert(), py::arg("balance"), py::arg("clip"));

    module.def("restoration_extra_gibbs_sample", [](const py::array& data, const py::array& weighted,
                                                    const py::array& precision, const py::array& left,
                                                    const py::array& right) {
        const auto shape = shape_of(data);
        if (shape_of(weighted) != shape || shape_of(precision) != shape ||
            shape_of(left) != shape || shape_of(right) != shape)
            throw py::value_error("restoration Gibbs spectrum shape mismatch");
        const auto* a = dense<double>(left);
        const auto* b = dense<double>(right);
        return real_dispatch(precision, [&](auto p) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
            const auto* y = dense<std::complex<T>>(data);
            const auto* h = dense<std::complex<T>>(weighted);
            py::array_t<std::complex<T>> output(shape);
            auto* out = output.mutable_data();
            const auto count = std::size_t(data.size());
            py::gil_scoped_release release;
            skimagexx::restoration_gibbs_sample(y, h, p, a, b, count, out);
            return output;
        });
    }, py::arg("data").noconvert(), py::arg("weighted").noconvert(), py::arg("precision").noconvert(),
       py::arg("left").noconvert(), py::arg("right").noconvert());

    module.def("restoration_extra_interpolate", [](const py::array& image, std::size_t spatial_rank) {
        return real_dispatch(image, [&](auto source) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(image);
            py::array_t<T> output(shape);
            auto* out = output.mutable_data();
            py::gil_scoped_release release;
            skimagexx::restoration_interpolate(source, shape, spatial_rank, out);
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("spatial_rank"));

    module.def("restoration_extra_roll", [](const py::array& input, const RestorationShape& shift,
                                            const py::object& accumulator) {
        return real_dispatch(input, [&](auto source) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(input);
            const bool add = !accumulator.is_none();
            auto output = add ? accumulator.cast<py::array>() : py::array(py::dtype::of<T>(), shape);
            if (add) {
                dense<T>(output);
                if (!output.owndata() || !output.writeable() || shape_of(output) != shape)
                    throw py::value_error("restoration roll requires an owning writable accumulator");
                const auto a = reinterpret_cast<std::uintptr_t>(source);
                const auto b = reinterpret_cast<std::uintptr_t>(output.data());
                const auto bytes = std::size_t(output.nbytes());
                if ((a < b ? b - a : a - b) < bytes)
                    throw py::value_error("restoration roll buffers overlap");
            }
            auto* out = static_cast<T*>(output.mutable_data());
            py::gil_scoped_release release;
            skimagexx::restoration_roll(source, shape, shift, out, add);
            return output;
        });
    }, py::arg("input").noconvert(), py::arg("shift"), py::arg("accumulator") = py::none());
}
