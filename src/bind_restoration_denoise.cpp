#include "skimagexx/restoration_denoise.hpp"

#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace {

bool exact_c_array(const py::array& array) {
    return Py_TYPE(array.ptr()) == py::detail::npy_api::get().PyArray_Type_ &&
           (array.flags() & py::array::c_style) && array.writeable() &&
           array.size() > 0;
}

template<class T>
bool descriptor(const py::array& array, char code) {
    const auto dtype = array.dtype();
    return (dtype.is(py::dtype::of<T>()) ||
            (dtype.char_() == code && dtype.equal(py::dtype::of<T>()))) &&
           dtype.attr("metadata").is_none() &&
           reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) == 0;
}

void check_storage(const py::array& array) {
    if (!array.owndata()) skimagexx::bindings::check_array_storage(array);
}

template<class Function>
py::object dispatch(const py::array& image, Function run) {
    if (!exact_c_array(image)) return py::none();
    if (descriptor<float>(image, 'f')) {
        check_storage(image);
        return run(static_cast<const float*>(image.data()));
    }
    if (descriptor<double>(image, 'd')) {
        check_storage(image);
        return run(static_cast<const double*>(image.data()));
    }
    return py::none();
}

struct Dimensions {
    std::size_t depth, rows, cols, channels;
};

Dimensions dimensions(const py::array& image, int spatial_rank) {
    return {
        spatial_rank == 3 ? static_cast<std::size_t>(image.shape(0)) : 1,
        spatial_rank >= 2 ? static_cast<std::size_t>(image.shape(spatial_rank - 2)) : 1,
        static_cast<std::size_t>(image.shape(spatial_rank - 1)),
        image.ndim() == spatial_rank ? 1 : static_cast<std::size_t>(image.shape(spatial_rank))
    };
}

bool valid_rank(const py::array& image, int rank, int min_rank, int max_rank) {
    return rank >= min_rank && rank <= max_rank &&
           (image.ndim() == rank || image.ndim() == rank + 1);
}

template<class T, class Function>
py::object array_result(const py::array& image, Function compute) {
    auto output = skimagexx::bindings::empty_c_array<T>(image.shape(), static_cast<int>(image.ndim()));
    auto* data = output.mutable_data();
    bool ok;
    {
        py::gil_scoped_release release;
        ok = compute(data);
    }
    if (!ok) return py::none();
    return output;
}

}  // namespace

void bind_restoration_denoise(pybind11::module_& module) {
    module.def("restoration_nl_means", [](const py::array& image, int rank, int patch,
                                         int distance, double h, double sigma) {
        if (!valid_rank(image, rank, 2, 3)) return py::object(py::none());
        const auto d = dimensions(image, rank);
        return dispatch(image, [&](const auto* input) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            return array_result<T>(image, [&](T* out) {
                return skimagexx::denoise_nl_means(input, out, d.depth, d.rows, d.cols,
                                                  d.channels, rank == 3, patch, distance, h, sigma);
            });
        });
    }, py::arg("image").noconvert(), py::arg("rank"), py::arg("patch"), py::arg("distance"),
       py::arg("h"), py::arg("sigma"));

    module.def("restoration_estimate_sigma", [](const py::array& image, int rank, bool channels) {
        if (!valid_rank(image, rank, 1, 2) || image.ndim() != rank + channels)
            return py::object(py::none());
        const auto d = dimensions(image, rank);
        if (d.channels > 16) return py::object(py::none());
        return dispatch(image, [&](const auto* input) -> py::object {
            std::vector<double> values(d.channels);
            {
                py::gil_scoped_release release;
                for (std::size_t channel = 0; channel < d.channels; ++channel) {
                    values[channel] = skimagexx::estimate_sigma(input, d.rows, d.cols,
                                                               d.channels, channel, rank == 2);
                    if (!std::isfinite(values[channel])) break;
                }
            }
            for (const auto value : values)
                if (!std::isfinite(value)) return py::none();
            if (!channels) return py::float_(values.front());
            py::list result;
            for (const auto value : values) result.append(py::float_(value));
            return result;
        });
    }, py::arg("image").noconvert(), py::arg("rank"), py::arg("channels"));

    module.def("restoration_wavelet", [](const py::array& image, int rank, int length, int levels,
                                        double sigma, bool automatic, bool soft, bool bayes) {
        if (!valid_rank(image, rank, 1, 2)) return py::object(py::none());
        const auto d = dimensions(image, rank);
        return dispatch(image, [&](const auto* input) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            return array_result<T>(image, [&](T* out) {
                return skimagexx::denoise_wavelet(input, out, d.rows, d.cols, d.channels,
                                                rank == 2, length, levels, sigma,
                                                automatic, soft, bayes);
            });
        });
    }, py::arg("image").noconvert(), py::arg("rank"), py::arg("length"), py::arg("levels"),
       py::arg("sigma"), py::arg("automatic"), py::arg("soft"), py::arg("bayes"));

    module.def("restoration_inpaint", [](const py::array& image, const py::array& mask) {
        if (!valid_rank(image, 2, 2, 2) || !exact_c_array(mask) || !descriptor<bool>(mask, '?') ||
            mask.ndim() != 2 || mask.shape(0) != image.shape(0) || mask.shape(1) != image.shape(1))
            return py::object(py::none());
        check_storage(mask);
        const auto d = dimensions(image, 2);
        const auto* bits = static_cast<const std::uint8_t*>(mask.data());
        return dispatch(image, [&](const auto* input) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            return array_result<T>(image, [&](T* out) {
                return skimagexx::inpaint_biharmonic(input, bits, out, d.rows, d.cols, d.channels);
            });
        });
    }, py::arg("image").noconvert(), py::arg("mask").noconvert());

    module.def("restoration_unwrap", [](const py::array& image, bool deterministic) {
        if (image.ndim() < 1 || image.ndim() > 3) return py::object(py::none());
        const int rank = static_cast<int>(image.ndim());
        const auto d = dimensions(image, rank);
        return dispatch(image, [&](const auto* input) {
            return array_result<double>(image, [&](double* out) {
                return skimagexx::unwrap_phase(input, out, d.depth, d.rows, d.cols, rank,
                                              deterministic);
            });
        });
    }, py::arg("image").noconvert(), py::arg("deterministic"));

    module.def("restoration_richardson_lucy", [](const py::array& image, const py::array& psf,
                                                 int iterations, bool clip, double epsilon) {
        if (image.ndim() != 2 || psf.ndim() != 2 || !exact_c_array(psf))
            return py::object(py::none());
        const auto d = dimensions(image, 2);
        const auto p = dimensions(psf, 2);
        return dispatch(image, [&](const auto* input) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(input)>>;
            if (!descriptor<T>(psf, std::is_same_v<T, float> ? 'f' : 'd'))
                return py::object(py::none());
            check_storage(psf);
            const auto* kernel = static_cast<const T*>(psf.data());
            return array_result<T>(image, [&](T* out) {
                return skimagexx::richardson_lucy(input, kernel, out, d.rows, d.cols,
                                                p.rows, p.cols, iterations, clip, epsilon);
            });
        });
    }, py::arg("image").noconvert(), py::arg("psf").noconvert(), py::arg("iterations"),
       py::arg("clip"), py::arg("epsilon"));
}
