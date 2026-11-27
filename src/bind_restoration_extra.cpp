#include <skimagexx/image.hpp>
#include <skimagexx/restoration_extra.hpp>
#include <skimagexx/tv_denoise.hpp>
#include <skimagexx/tv_bregman.hpp>
#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace skimagexx {
template<class T>
std::array<double, 3> restoration_gibbs_accumulate(const std::complex<T>*,
    const std::complex<T>*, const T*, const T*, const T*, T, T, const double*,
    std::size_t, std::size_t, std::size_t, std::complex<T>*, bool);

void restoration_interpolate_phase(const float*, std::size_t, std::size_t,
                                   std::array<std::size_t, 2>, std::size_t, float*);
void restoration_interpolate_phase(const double*, std::size_t, std::size_t,
                                   std::array<std::size_t, 2>, std::size_t, double*);

template<class T>
void restoration_replace_phase(const T*, std::size_t, std::size_t,
                               std::array<std::size_t, 2>, std::size_t, T*);

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
        !skimagexx::bindings::is_native_byteorder(input) ||
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
void report_roll_add_errors(int errors) {
    if (!errors) return;
    // Let NumPy apply its active errstate once, as for one in-place add.
    // The actual accumulator is private to cycle spinning.
    py::array_t<T> lhs(2), rhs(2);
    auto* a = lhs.mutable_data();
    auto* b = rhs.mutable_data();
    a[0] = b[0] = errors & FE_OVERFLOW ? std::numeric_limits<T>::max() : T(0);
    a[1] = errors & FE_INVALID ? std::numeric_limits<T>::infinity() : T(0);
    b[1] = -a[1];
    lhs.attr("__iadd__")(rhs);
}

using Phase = std::array<std::size_t, 2>;

std::size_t phase_extent(std::size_t extent, std::size_t offset, std::size_t stride) {
    return offset < extent ? 1 + (extent - 1 - offset) / stride : 0;
}

void check_phase(const RestorationShape& shape, Phase phase, std::size_t stride) {
    if (shape.size() != 2 || !stride || std::max(phase[0], phase[1]) >= stride ||
        std::max(shape[0], shape[1]) > std::numeric_limits<std::size_t>::max() - stride)
        throw py::value_error("invalid restoration mask phase");
}

RestorationShape phase_shape(const RestorationShape& shape, Phase phase, std::size_t stride) {
    return {phase_extent(shape[0], phase[0], stride), phase_extent(shape[1], phase[1], stride)};
}

template<class Action>
void visit_phase(const RestorationShape& shape, Phase phase, std::size_t stride, Action action) {
    std::size_t compact = 0;
    for (auto row = phase[0]; row < shape[0]; row += stride) {
        const auto start = row * shape[1];
        for (auto col = phase[1]; col < shape[1]; col += stride)
            action(start + col, compact++);
    }
}

template<class T, class Denoise>
bool denoise_batch(const T* input, const T* interp, const RestorationShape& shape,
                   const std::vector<Phase>& jobs, std::size_t stride, T* shifted, T* output,
                   bool masked, bool compact, bool compact_interp, Denoise denoise) {
    using namespace skimagexx;
    const auto height = shape[0], width = shape[1], size = height * width;
    if (masked) {
        std::copy_n(input, size, shifted);
        if (!compact && jobs.size() == 1) std::fill_n(output, size, T(0));
    }
    RestorationShape shift(2);
    bool add = false;
    for (const auto& job : jobs) {
        const auto [y, x] = job;
        if (interp) {
            visit_phase(shape, job, stride, [&](auto i, auto j) {
                shifted[i] = interp[compact_interp ? j : i];
            });
        } else if (masked) {
            restoration_replace_phase(input, height, width, job, stride, shifted);
        } else if (y || x) {
            shift[0] = y;
            shift[1] = x;
            restoration_roll(input, shape, shift, shifted, false);
        }
        const auto* denoised = denoise(masked || y || x ? shifted : input);
        if (!denoised) return false;
        if (masked) {
            visit_phase(shape, job, stride, [&](auto i, auto j) {
                output[compact ? j : i] = denoised[i];
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

template<class T>
bool tv_batch(const T* input, const T* interp, const RestorationShape& shape,
              const std::vector<Phase>& jobs, std::size_t stride,
              const skimagexx::TvDenoiseParameters& parameters, T* output,
              bool masked, bool compact, bool compact_interp) {
    using namespace skimagexx;
    const auto size = shape[0] * shape[1];
    const auto type = std::is_same_v<T, float> ? TvScalarType::float32 : TvScalarType::float64;
    const std::vector<std::ptrdiff_t> strides{std::ptrdiff_t(shape[1] * sizeof(T)), sizeof(T)};
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(T) / 9)
        throw std::bad_array_new_length();
    const std::unique_ptr<T[]> scratch(new T[9 * size]);
    auto* shifted = scratch.get();
    auto* p = shifted + size;
    auto* g = p + 2 * size;
    auto* d = g + 2 * size;
    auto* out = d + size;
    auto* squared = out + size;
    auto* norm = squared + size;
    const TvDenoiseWorkspace workspace{
        {p, type, shape}, {g, type, shape}, {d, type, shape, strides},
        {out, type, shape, strides}, {squared, type, shape, strides},
        {norm, type, shape, strides}, {p, type, shape}, {g, type, shape},
        {out, type, shape, strides}, {norm, type, shape, strides}};
    TvConstView source{input, type, shape, strides};
    return denoise_batch(input, interp, shape, jobs, stride, shifted, output,
                         masked, compact, compact_interp,
                         [&](const T* values) -> const T* {
        source.data = values;
        TvDenoiseResult result{};
        if (detail::tv_denoise_try_finite_reuse_into(source, workspace, parameters, result))
            return nullptr;
        return result.input_alias ? values : out;
    });
}

template<class T>
bool bregman_batch(const T* input, const T* interp, const RestorationShape& shape,
                   const std::vector<Phase>& jobs, std::size_t stride,
                   skimagexx::TvBregmanParameters parameters, T* output,
                   bool masked, bool compact, bool compact_interp) {
    using namespace skimagexx;
    const auto height = shape[0], width = shape[1], size = height * width;
    if (height < 2 || width < 2) return false;
    const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(T) / 5;
    if (height + 2 > limit / (width + 2)) throw std::bad_array_new_length();
    const auto padded = (height + 2) * (width + 2);
    const std::unique_ptr<T[]> scratch(new T[5 * padded]);
    const std::unique_ptr<T[]> images(new T[2 * size]);
    auto* shifted = images.get();
    auto* denoised = shifted + size;
    auto* out = scratch.get();
    const TvBregmanWorkspace workspace{
        {out, padded}, {out + padded, padded}, {out + 2 * padded, padded},
        {out + 3 * padded, padded}, {out + 4 * padded, padded}};
    const auto type = std::is_same_v<T, float> ? TvBregmanType::float32 : TvBregmanType::float64;
    parameters.weight = T(parameters.weight);
    return denoise_batch(input, interp, shape, jobs, stride, shifted, output,
                         masked, compact, compact_interp,
                         [&](const T* values) -> const T* {
        std::fill_n(out, padded, T(0));
        // The native solver rejects an unsupported input before writing.
        try {
            tv_bregman_into({values, type, height, width, 1, size}, workspace, parameters);
        } catch (const std::invalid_argument&) {
            return nullptr;
        }
        for (std::size_t row = 0; row < height; ++row) {
            const auto* interior = out + (row + 1) * (width + 2) + 1;
            for (std::size_t col = 0; col < width; ++col) {
                const auto value = interior[col];
                // Keep exceptional cycle-spin additions on NumPy's errstate path.
                if (!(std::abs(value) <= T(1048576))) return nullptr;
                denoised[row * width + col] = value;
            }
        }
        return denoised;
    });
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
            {
                py::gil_scoped_release release;
                skimagexx::restoration_transfer(kernel, kernel_shape, shape, out, workers);
            }
            return output;
        });
    }, py::arg("input").noconvert(), py::arg("shape"));

    module.def("restoration_extra_laplacian", [](const RestorationShape& shape, bool single) -> py::array {
        if (single) {
            py::array_t<float> output(spectrum_of(shape));
            auto* out = output.mutable_data();
            {
                py::gil_scoped_release release;
                skimagexx::restoration_laplacian(shape, out);
            }
            return output;
        }
        py::array_t<double> output(spectrum_of(shape));
        auto* out = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::restoration_laplacian(shape, out);
        }
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
            {
                py::gil_scoped_release release;
                skimagexx::restoration_gibbs_forward(source, shape, out, workers);
            }
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
            {
                py::gil_scoped_release release;
                skimagexx::restoration_gibbs_inverse(source, shape, out, workers);
            }
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
        const py::array& normals, py::array& total, std::size_t count, bool measure_delta) {
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
                T(noise), T(prior), draws, height, width, count, out, measure_delta);
        });
    }, py::arg("data").noconvert(), py::arg("transfer").noconvert(), py::arg("reg").noconvert(),
       py::arg("power").noconvert(), py::arg("reg_power").noconvert(), py::arg("noise"),
       py::arg("prior"), py::arg("normals").noconvert(), py::arg("total").noconvert(), py::arg("count"),
       py::arg("measure_delta") = true);

    module.def("restoration_extra_denoise_batch", [](const py::array& image,
        const py::object& interpolation, std::vector<Phase> jobs, std::size_t stride, bool masked, bool compact,
        bool bregman, double weight, double eps, int iterations,
        bool profile_or_isotropic, std::size_t buffer_elements) -> py::object {
        using namespace skimagexx;
        const auto shape = shape_of(image);
        if (shape.size() != 2 || !stride || stride > 64 || iterations < 1 ||
            std::max(shape[0], shape[1]) > std::numeric_limits<std::size_t>::max() - stride)
            throw py::value_error("invalid restoration denoise batch");
        auto output_shape = shape;
        if (masked && jobs.empty() && !compact) {
            jobs.reserve(stride * stride);
            for (std::size_t row = 0; row < stride; ++row)
                for (std::size_t col = 0; col < stride; ++col)
                    jobs.push_back({row, col});
        } else if (masked && jobs.size() == 1) {
            check_phase(shape, jobs.front(), stride);
            if (compact) output_shape = phase_shape(shape, jobs.front(), stride);
        } else if (masked || compact || jobs.empty()) {
            throw py::value_error("invalid restoration denoise jobs");
        } else {
            for (const auto& job : jobs)
                if (job[0] >= shape[0] || job[1] >= shape[1])
                    throw py::value_error("restoration denoise shift is not reduced");
        }
        const bool has_interp = !interpolation.is_none();
        const auto interp = has_interp ? interpolation.cast<py::array>() : py::array();
        const bool compact_interp = has_interp && !same_shape(image, interp);
        if (has_interp && (!masked || (compact_interp && (!compact || shape_of(interp) != output_shape))))
            throw py::value_error("restoration denoise interpolation shape mismatch");
        const auto profile = profile_or_isotropic ? TvReductionProfile::numpy_254_aligned_contiguous
                                                  : TvReductionProfile::numpy_1244_buffered;
        const TvDenoiseParameters tv{weight, eps, std::size_t(iterations),
                                     {profile, buffer_elements}};
        const TvBregmanParameters bg{weight, iterations, eps, profile_or_isotropic};
        return real_dispatch(image, [&](auto input) -> py::object {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            const auto* values = has_interp ? dense<T>(interp) : nullptr;
            py::array_t<T> output(output_shape);
            auto* out = output.mutable_data();
            bool success;
            {
                py::gil_scoped_release release;
                success = bregman
                    ? bregman_batch(input, values, shape, jobs, stride, bg, out, masked, compact, compact_interp)
                    : tv_batch(input, values, shape, jobs, stride, tv, out, masked, compact, compact_interp);
            }
            if (!success) return py::none();
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("interpolation"), py::arg("jobs"), py::arg("stride"), py::arg("masked"),
       py::arg("compact"), py::arg("bregman"), py::arg("weight"), py::arg("eps"),
       py::arg("iterations"), py::arg("profile_or_isotropic"), py::arg("buffer_elements"));

    module.def("restoration_extra_phase_input", [](const py::array& image, const py::array& interp,
                                                    Phase phase, std::size_t stride) {
        const auto shape = shape_of(image);
        check_phase(shape, phase, stride);
        const bool compact_interp = !same_shape(image, interp);
        if (compact_interp && shape_of(interp) != phase_shape(shape, phase, stride))
            throw py::value_error("restoration interpolation shape mismatch");
        const auto size = std::size_t(image.size());
        return real_dispatch(image, [&](auto input) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            const auto* interpolated = dense<T>(interp);
            py::array_t<T> output(shape);
            auto* out = output.mutable_data();
            {
                py::gil_scoped_release release;
                std::copy_n(input, size, out);
                visit_phase(shape, phase, stride, [&](auto i, auto j) {
                    out[i] = interpolated[compact_interp ? j : i];
                });
            }
            return output;
        });
    });

    module.def("restoration_extra_phase_store", [](const py::array& denoised, py::array& output,
                                                    Phase phase, std::size_t stride, bool compact) {
        const auto shape = shape_of(denoised);
        check_phase(shape, phase, stride);
        const auto expected = compact
            ? phase_shape(shape, phase, stride)
            : shape;
        if (shape_of(output) != expected || overlaps(denoised, output))
            throw py::value_error("invalid restoration mask destination");
        real_dispatch(denoised, [&](auto input) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            dense<T>(output);
            auto* out = static_cast<T*>(output.mutable_data());
            py::gil_scoped_release release;
            visit_phase(shape, phase, stride, [&](auto i, auto j) { out[compact ? j : i] = input[i]; });
        });
    });

    module.def("restoration_extra_squared_error", [](const py::array& image, const py::array& denoised,
        Phase phase, std::size_t stride, bool modern_mean) -> py::object {
        using namespace skimagexx;
        const auto shape = shape_of(image);
        check_phase(shape, phase, stride);
        const auto expected = phase_shape(shape, phase, stride);
        if (shape_of(denoised) != expected)
            throw py::value_error("restoration loss shape mismatch");
        const auto size = std::size_t(denoised.size());
        return real_dispatch(image, [&](auto input) -> py::object {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            const auto* values = dense<T>(denoised);
            py::array_t<T> squared(size);
            auto* out = squared.mutable_data();
            bool success = false;
            double mean = 0;
            const bool reduce = modern_mean && std::is_same_v<T, double>;
            {
                py::gil_scoped_release release;
                std::fenv_t original;
                if (std::feholdexcept(&original) == 0) {
                    visit_phase(shape, phase, stride, [&](auto i, auto j) {
                        const T diff = input[i] - values[j];
                        out[j] = diff * diff;
                    });
                    success = !std::fetestexcept(FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
                    std::fesetenv(&original);
                }
                if (success && reduce) {
                    // Match the audited contiguous float64 NumPy sum association.
                    try {
                        const auto sum = tv_sum_contiguous({out, size, TvReductionType::float64},
                            {TvReductionProfile::numpy_254_aligned_contiguous, size});
                        double value;
                        std::memcpy(&value, &sum.bits, sizeof(value));
                        const auto count = static_cast<double>(size);
                        success = sum.fp_status == 0 &&
                            (value == 0 || value >= std::numeric_limits<double>::min() * count);
                        if (success) mean = value / count;
                    } catch (const std::invalid_argument&) {
                        // Keep NumPy's reducer for unsupported floating environments.
                        success = false;
                    }
                }
            }
            if (!success) return py::none();
            if (reduce) return py::float_(mean);
            return squared;
        });
    });

    module.def("restoration_extra_interpolate", [](const py::array& image, std::size_t spatial_rank) {
        return real_dispatch(image, [&](auto source) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(image);
            py::array_t<T> output(shape);
            auto* out = output.mutable_data();
            {
                py::gil_scoped_release release;
                skimagexx::restoration_interpolate(source, shape, spatial_rank, out);
            }
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("spatial_rank"));

    module.def("restoration_extra_interpolate_phase", [](const py::array& image, Phase phase,
                                                          std::size_t stride) {
        const auto shape = shape_of(image);
        check_phase(shape, phase, stride);
        const auto height = shape[0], width = shape[1];
        return real_dispatch(image, [&](auto source) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            py::array_t<T> output(phase_shape(shape, phase, stride));
            auto* out = output.mutable_data();
            {
                py::gil_scoped_release release;
                skimagexx::restoration_interpolate_phase(source, height, width, phase, stride, out);
            }
            return output;
        });
    }, py::arg("image").noconvert(), py::arg("phase"), py::arg("stride"));

    module.def("restoration_extra_roll", [](const py::array& input, RestorationShape shift,
                                            const py::object& accumulator, bool inverse) {
        return real_dispatch(input, [&](auto source) -> py::array {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
            const auto shape = shape_of(input);
            if (inverse) {
                if (shift.size() != shape.size())
                    throw py::value_error("restoration roll shift rank mismatch");
                for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                    if (shift[axis] >= shape[axis])
                        throw py::value_error("restoration roll shift is not reduced");
                    if (shift[axis]) shift[axis] = shape[axis] - shift[axis];
                }
            }
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
            int errors = 0;
            bool rolled = false;
            {
                py::gil_scoped_release release;
                if (!add) {
                    skimagexx::restoration_roll(source, shape, shift, out, false);
                    rolled = true;
                } else {
                    std::fenv_t original;
                    if (std::feholdexcept(&original) == 0) {
                        try {
                            skimagexx::restoration_roll(source, shape, shift, out, true);
                        } catch (...) {
                            std::fesetenv(&original);
                            throw;
                        }
                        errors = std::fetestexcept(FE_OVERFLOW | FE_INVALID);
                        std::fesetenv(&original);
                        rolled = true;
                    }
                }
            }
            if (!rolled) {
                py::array_t<T> aligned(shape);
                auto* dst = aligned.mutable_data();
                {
                    py::gil_scoped_release release;
                    skimagexx::restoration_roll(source, shape, shift, dst, false);
                }
                output.attr("__iadd__")(aligned);
            }
            report_roll_add_errors<T>(errors);
            return output;
        });
    }, py::arg("input").noconvert(), py::arg("shift"), py::arg("accumulator") = py::none(),
       py::arg("inverse") = false);
}
