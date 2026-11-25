#include <skimagexx/image.hpp>
#include <skimagexx/restoration_extra.hpp>
#include "bind_array_storage.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
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
    if (!input.owndata()) skimagexx::bindings::check_array_storage(input);
    return static_cast<const T*>(input.data());
}

RestorationShape shape_of(const py::array& input) {
    if (input.ndim() == 0) throw py::value_error("restoration requires an array with axes");
    return {input.shape(), input.shape() + input.ndim()};
}

bool same_shape(const py::array& input, const py::array& reference) {
    return input.ndim() == reference.ndim() &&
           std::equal(input.shape(), input.shape() + input.ndim(), reference.shape());
}

bool overlaps(const py::array& a, const py::array& b) {
    const auto left = reinterpret_cast<std::uintptr_t>(a.data());
    const auto right = reinterpret_cast<std::uintptr_t>(b.data());
    return left <= right ? right - left < std::size_t(a.nbytes())
                         : left - right < std::size_t(b.nbytes());
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
                                              const py::object& reg, double balance, bool clip) {
        return real_dispatch(image, [&](auto source) -> py::object {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(image);
            const auto spectrum = spectrum_of(shape);
            if (shape_of(transfer) != spectrum)
                throw py::value_error("restoration spectrum shape mismatch");
            const auto* tf = dense<std::complex<T>>(transfer);
            const T* regularizer = nullptr;
            if (!reg.is_none()) {
                const auto array = reg.cast<py::array>();
                if (shape_of(array) != spectrum)
                    throw py::value_error("restoration regularizer shape mismatch");
                regularizer = dense<T>(array);
            }
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
       py::arg("reg"), py::arg("balance"), py::arg("clip"));

    module.def("restoration_extra_gibbs_forward", [](const py::array& image) {
        return real_dispatch(image, [&](auto source) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(image);
            py::array_t<std::complex<T>> output(spectrum_of(shape));
            auto* out = output.mutable_data();
            const int workers = skimagexx::get_num_threads();
            py::gil_scoped_release release;
            skimagexx::restoration_gibbs_forward(source, shape, out, workers);
            return output;
        });
    }, py::arg("image").noconvert());

    module.def("restoration_extra_gibbs_inverse", [](const py::array& data, const RestorationShape& shape) {
        if (shape_of(data) != spectrum_of(shape))
            throw py::value_error("restoration Gibbs spectrum shape mismatch");
        const auto inverse = [&](auto tag) -> py::array {
            using T = decltype(tag);
            const auto* source = dense<std::complex<T>>(data);
            py::array_t<T> output(shape);
            auto* out = output.mutable_data();
            const int workers = skimagexx::get_num_threads();
            py::gil_scoped_release release;
            skimagexx::restoration_gibbs_inverse(source, shape, out, workers);
            return output;
        };
        if (data.dtype().num() == py::dtype::of<std::complex<float>>().num()) return inverse(float{});
        return inverse(double{});
    }, py::arg("data").noconvert(), py::arg("shape"));

    module.def("restoration_extra_gibbs_step", [](const py::array& data, const py::array& transfer,
                                                     const py::array& reg, const py::array& power,
                                                     const py::array& reg_power, double noise,
                                                     double prior, const py::array& normals,
                                                     py::array& sample, py::array& residuals) {
        if (data.ndim() != 2 || !same_shape(transfer, data) || !same_shape(reg, data) ||
            !same_shape(power, data) || !same_shape(reg_power, data) || !same_shape(sample, data))
            throw py::value_error("restoration Gibbs spectrum shape mismatch");
        if (normals.ndim() != 3 || normals.shape(0) != 2 || normals.shape(1) != data.shape(0) ||
            normals.shape(2) != data.shape(1) || !same_shape(normals, residuals))
            throw py::value_error("restoration Gibbs workspace shape mismatch");
        for (const auto* input : {&data, &transfer, &reg, &power, &reg_power, &normals}) {
            if (overlaps(*input, sample) || overlaps(*input, residuals))
                throw py::value_error("restoration Gibbs input and output overlap");
        }
        if (overlaps(sample, residuals)) throw py::value_error("restoration Gibbs outputs overlap");
        const auto* draws = dense<double>(normals);
        real_dispatch(reg, [&](auto regularizer) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(regularizer)>>;
            using C = std::complex<T>;
            const auto* y = dense<C>(data);
            const auto* h = dense<C>(transfer);
            const auto* p = dense<T>(power);
            const auto* rp = dense<T>(reg_power);
            dense<C>(sample);
            dense<C>(residuals);
            auto* out = static_cast<C*>(sample.mutable_data());
            auto* errors = static_cast<C*>(residuals.mutable_data());
            const auto count = std::size_t(data.size());
            py::gil_scoped_release release;
            skimagexx::restoration_gibbs_step(y, h, regularizer, p, rp, T(noise), T(prior),
                                             draws, count, out, errors);
        });
    }, py::arg("data").noconvert(), py::arg("transfer").noconvert(), py::arg("reg").noconvert(),
       py::arg("power").noconvert(), py::arg("reg_power").noconvert(), py::arg("noise"),
       py::arg("prior"), py::arg("normals").noconvert(), py::arg("sample").noconvert(),
       py::arg("residuals").noconvert());

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
                if (overlaps(input, output))
                    throw py::value_error("restoration roll buffers overlap");
            }
            auto* out = static_cast<T*>(output.mutable_data());
            py::gil_scoped_release release;
            skimagexx::restoration_roll(source, shape, shift, out, add);
            return output;
        });
    }, py::arg("input").noconvert(), py::arg("shift"), py::arg("accumulator") = py::none());
}
