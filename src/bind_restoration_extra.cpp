#include <skimagexx/image.hpp>
#include <skimagexx/restoration_extra.hpp>
#include <skimagexx/tv_denoise.hpp>
#include "bind_array_storage.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

namespace skimagexx {
template<class T>
std::array<double, 3> restoration_gibbs_accumulate(const std::complex<T>*,
    const std::complex<T>*, const T*, const T*, const T*, T, T, const double*,
    std::size_t, std::size_t, std::size_t, std::complex<T>*);

namespace detail {
const char* tv_denoise_try_finite_reuse_into(const TvConstView&, const TvDenoiseWorkspace&,
    const TvDenoiseParameters&, TvDenoiseResult&);
}
}

namespace py = pybind11;
namespace {
using skimagexx::RestorationShape;

template<class T> const T* dense(const py::array& input) {
    if (input.dtype().num() != py::dtype::of<T>().num() ||
        !PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject*>(input.ptr())) ||
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

template<class T>
bool tv_batch(const T* input, const T* interp, const RestorationShape& shape,
              const std::vector<std::array<std::size_t, 2>>& jobs, std::size_t stride,
              const skimagexx::TvDenoiseParameters& parameters, T* output) {
    using namespace skimagexx;
    const auto height = shape[0], width = shape[1], size = height * width;
    const auto type = std::is_same_v<T, float> ? TvScalarType::float32 : TvScalarType::float64;
    const std::vector<std::ptrdiff_t> strides{std::ptrdiff_t(width * sizeof(T)), sizeof(T)};
    std::vector<T> shifted(size), p(2 * size), g(2 * size), d(size), out(size), squared(size), norm(size);
    const TvDenoiseWorkspace workspace{
        {p.data(), type, shape}, {g.data(), type, shape}, {d.data(), type, shape, strides},
        {out.data(), type, shape, strides}, {squared.data(), type, shape, strides},
        {norm.data(), type, shape, strides}, {p.data(), type, shape}, {g.data(), type, shape},
        {out.data(), type, shape, strides}, {norm.data(), type, shape, strides}};
    TvConstView source{shifted.data(), type, shape, strides};
    RestorationShape shift(2);
    bool add = false;
    if (interp) std::fill_n(output, size, T(0));
    for (const auto& job : jobs) {
        const auto [y, x] = job;
        const auto each_masked = [&](const auto& action) {
            for (auto row = y; row < height; row += stride)
                for (auto col = x; col < width; col += stride)
                    action(row * width + col);
        };
        if (interp) {
            if (!add) std::copy_n(input, size, shifted.data());
            each_masked([&](auto i) { shifted[i] = interp[i]; });
        } else if (y || x) {
            shift[0] = y;
            shift[1] = x;
            restoration_roll(input, shape, shift, shifted.data(), false);
        }
        source.data = interp || y || x ? shifted.data() : input;
        TvDenoiseResult result{};
        if (detail::tv_denoise_try_finite_reuse_into(source, workspace, parameters, result))
            return false;
        const auto* denoised = result.input_alias ? static_cast<const T*>(source.data) : out.data();
        if (interp) {
            each_masked([&](auto i) {
                output[i] = denoised[i];
                shifted[i] = input[i];
            });
        } else if (y || x) {
            shift[0] = y ? height - y : 0;
            shift[1] = x ? width - x : 0;
            restoration_roll(denoised, shape, shift, output, add);
        } else if (add) {
            std::transform(denoised, denoised + size, output, output, std::plus<T>());
        } else {
            std::copy_n(denoised, size, output);
        }
        add = true;
    }
    return true;
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

    module.def("restoration_extra_gibbs_accumulate", [](
        const py::array& data, const py::array& transfer, const py::array& reg,
        const py::array& power, const py::array& reg_power, double noise, double prior,
        const py::array& normals, py::array& total, std::size_t count) {
        if (data.ndim() != 2 || !same_shape(data, transfer) || !same_shape(data, reg) ||
            !same_shape(data, power) || !same_shape(data, reg_power) || !same_shape(data, total) ||
            normals.ndim() != 3 || normals.shape(0) != 2 || normals.shape(1) != data.shape(0) ||
            normals.shape(2) != data.shape(1))
            throw py::value_error("restoration Gibbs workspace shape mismatch");
        for (const auto* input : {&data, &transfer, &reg, &power, &reg_power, &normals})
            if (overlaps(*input, total))
                throw py::value_error("restoration Gibbs input and output overlap");
        const auto* draws = dense<double>(normals);
        const auto height = std::size_t(data.shape(0)), width = std::size_t(data.shape(1));
        return real_dispatch(reg, [&](auto regularizer) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(regularizer)>>;
            using C = std::complex<T>;
            const auto* y = dense<C>(data);
            const auto* h = dense<C>(transfer);
            const auto* p = dense<T>(power);
            const auto* rp = dense<T>(reg_power);
            dense<C>(total);
            auto* out = static_cast<C*>(total.mutable_data());
            py::gil_scoped_release release;
            return skimagexx::restoration_gibbs_accumulate(y, h, regularizer, p, rp,
                T(noise), T(prior), draws, height, width, count, out);
        });
    }, py::arg("data").noconvert(), py::arg("transfer").noconvert(), py::arg("reg").noconvert(),
       py::arg("power").noconvert(), py::arg("reg_power").noconvert(), py::arg("noise"),
       py::arg("prior"), py::arg("normals").noconvert(), py::arg("total").noconvert(), py::arg("count"));

    module.def("restoration_extra_tv_batch", [](const py::array& image, const py::object& interpolation,
        const std::vector<std::array<std::size_t, 2>>& jobs, std::size_t stride, double weight,
        double eps, std::size_t iterations, bool modern, std::size_t buffer_elements) -> py::object {
        using namespace skimagexx;
        const auto shape = shape_of(image);
        if (shape.size() != 2 || jobs.empty() || !stride ||
            std::max(shape[0], shape[1]) > std::numeric_limits<std::size_t>::max() - stride)
            throw py::value_error("restoration TV batch requires a nonempty 2D job list");
        const auto interp = interpolation.is_none() ? py::array() : interpolation.cast<py::array>();
        if (!interpolation.is_none() && !same_shape(image, interp))
            throw py::value_error("restoration TV interpolation shape mismatch");
        if (interpolation.is_none())
            for (const auto& job : jobs)
                if (job[0] >= shape[0] || job[1] >= shape[1])
                    throw py::value_error("restoration TV shift is not reduced");
        const auto profile = modern ? TvReductionProfile::numpy_254_aligned_contiguous
                                    : TvReductionProfile::numpy_1244_buffered;
        const TvDenoiseParameters parameters{weight, eps, iterations, {profile, buffer_elements}};
        return real_dispatch(image, [&](auto input) -> py::object {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            const auto* values = interpolation.is_none() ? nullptr : dense<T>(interp);
            py::array_t<T> output(shape);
            auto* out = output.mutable_data();
            bool success;
            {
                py::gil_scoped_release release;
                success = tv_batch(input, values, shape, jobs, stride, parameters, out);
            }
            if (!success) return py::none();
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("interpolation"), py::arg("jobs"), py::arg("stride"),
       py::arg("weight"), py::arg("eps"), py::arg("iterations"), py::arg("modern"),
       py::arg("buffer_elements"));

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
