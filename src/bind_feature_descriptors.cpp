#include "skimagexx/feature_descriptors.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace py = pybind11;
using namespace skimagexx;

namespace {

enum class Pixel { unsupported, u8, f32, f64, u64 };

Pixel prepare(const py::array& array, int min_ndim = 2, int max_ndim = 2) {
    if (Py_TYPE(array.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
        array.ndim() < min_ndim || array.ndim() > max_ndim ||
        !(array.flags() & py::array::c_style) || !array.writeable() ||
        !(array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_))
        return Pixel::unsupported;
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        if (array.shape(axis) <= 0 || array.shape(axis) > (1 << 24)) return Pixel::unsupported;
    }
    if (array.size() > (1 << 26)) return Pixel::unsupported;
    const auto dtype = array.dtype();
    if (!dtype.attr("metadata").is_none()) return Pixel::unsupported;
    Pixel pixel = Pixel::unsupported;
    if (dtype.char_() == 'B' && dtype.equal(py::dtype::of<std::uint8_t>())) pixel = Pixel::u8;
    if (dtype.char_() == 'f' && dtype.equal(py::dtype::of<float>())) pixel = Pixel::f32;
    if (dtype.char_() == 'd' && dtype.equal(py::dtype::of<double>())) pixel = Pixel::f64;
    if (dtype.kind() == 'u' && dtype.equal(py::dtype::of<std::uint64_t>())) pixel = Pixel::u64;
    if (pixel != Pixel::unsupported && !array.owndata()) bindings::check_array_storage(array);
    return pixel;
}

template<class T>
bool finite_pixels(const T* p, py::ssize_t size) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::all_of(p, p + size, [](T v) { return std::isfinite(v); });
    }
    return true;
}

bool addressable(std::initializer_list<py::ssize_t> dims) {
    py::ssize_t elements = 1;
    for (auto d : dims) {
        if (d <= 0 || d > (PY_SSIZE_T_MAX / 8) / elements) return false;
        elements *= d;
    }
    return true;
}

template<class Input, class Output>
py::object hog_result(const py::array& array, const HogOptions& o, bool visualize,
                      bool transform_sqrt, bool flatten) {
    const auto nr = o.rows / o.cell_rows - o.block_rows + 1;
    const auto nc = o.cols / o.cell_cols - o.block_cols + 1;
    if (!addressable({nr, nc, o.block_rows, o.block_cols, o.orientations})) return py::none();
    const std::vector<py::ssize_t> shape = flatten
        ? std::vector<py::ssize_t>{py::ssize_t(nr) * nc * o.block_rows * o.block_cols * o.orientations}
        : std::vector<py::ssize_t>{nr, nc, o.block_rows, o.block_cols, o.orientations};
    py::array_t<Output> output(shape), view;
    if (visualize) view = py::array_t<Output>({o.rows, o.cols});
    auto* hist = output.mutable_data();
    auto* view_data = visualize ? view.mutable_data() : nullptr;
    const auto* source = static_cast<const Input*>(array.data());
    const auto count = array.size();
    bool supported = true;
    {
        py::gil_scoped_release release;
        if (transform_sqrt) {
            std::vector<Output> transformed(static_cast<std::size_t>(count));
            for (py::ssize_t i = 0; i < count; ++i) {
                if (!(source[i] >= 0) || !std::isfinite(source[i])) {
                    supported = false;
                    break;
                }
                transformed[static_cast<std::size_t>(i)] = std::sqrt(Output(source[i]));
            }
            if (supported) hog_into(transformed.data(), o, hist, view_data);
        } else {
            supported = finite_pixels(source, count);
            if (supported) hog_into(source, o, hist, view_data);
        }
    }
    if (!supported) return py::none();
    if (visualize) return py::make_tuple(output, view);
    return output;
}

template<class Input, class Output>
py::object daisy_result(const py::array& array, const DaisyOptions& o,
                        const std::vector<double>& sigmas, const std::vector<double>& radii) {
    const int oh = 1 + (o.rows - 2 * o.radius - 1) / o.step;
    const int ow = 1 + (o.cols - 2 * o.radius - 1) / o.step;
    const auto width = (py::ssize_t{o.rings} * o.histograms + 1) * o.orientations;
    if (!addressable({oh, ow, width})) return py::none();
    py::array_t<Output> output({py::ssize_t(oh), py::ssize_t(ow), width});
    const auto* source = static_cast<const Input*>(array.data());
    auto* dest = output.mutable_data();
    {
        py::gil_scoped_release release;
        daisy_into(source, o, sigmas.data(), radii.data(), dest);
    }
    return output;
}

template<class Input, class T = Input>
py::object template_result(const py::array& image, const py::array& pattern, bool pad, double constant) {
    const TemplateMatchOptions options{static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)),
                                       static_cast<int>(pattern.shape(0)), static_cast<int>(pattern.shape(1)),
                                       pad, constant};
    const int rows = pad ? options.rows : options.rows - options.template_rows + 1;
    const int cols = pad ? options.cols : options.cols - options.template_cols + 1;
    py::array_t<T> output({rows, cols});
    auto* dest = output.mutable_data();
    const auto* input = static_cast<const Input*>(image.data());
    const auto* needle = static_cast<const Input*>(pattern.data());
    const auto image_count = image.size(), pattern_count = pattern.size();
    bool ok;
    {
        py::gil_scoped_release release;
        if constexpr (std::is_same_v<Input, T>) {
            ok = match_template_into(input, needle, options, dest);
        } else {
            const std::vector<T> values(input, input + image_count);
            const std::vector<T> weights(needle, needle + pattern_count);
            ok = match_template_into(values.data(), weights.data(), options, dest);
        }
    }
    if (!ok) return py::none();
    return output;
}

template<class T>
py::object basic_result(const py::array& array, const BasicFeatureOptions& o, const std::vector<double>& sigmas) {
    const auto per_scale = int(o.intensity) + int(o.edges) + 2 * int(o.texture);
    const auto features = py::ssize_t(o.channels) * per_scale * static_cast<py::ssize_t>(sigmas.size());
    if (!addressable({o.rows, o.cols, features})) return py::none();
    py::array_t<float> output({py::ssize_t(o.rows), py::ssize_t(o.cols), features});
    auto* dest = output.mutable_data();
    const auto* source = static_cast<const T*>(array.data());
    const auto count = array.size();
    bool supported;
    {
        py::gil_scoped_release release;
        supported = finite_pixels(source, count);
        if (supported) multiscale_basic_features_into(source, o, sigmas.data(), static_cast<int>(sigmas.size()), dest);
    }
    if (!supported) return py::none();
    return output;
}

} // namespace

void bind_feature_descriptors(py::module_& module) {
    module.def("local_binary_pattern_u8", [](const py::array& image, int points, double radius, int method) -> py::object {
        if (points <= 0 || points > 64 || !std::isfinite(radius) || radius < 0 || radius > 1e6 ||
            method < 0 || method > 4 || (method <= 1 && points > 31) ||
            prepare(image) != Pixel::u8) return py::none();
        const auto rows = static_cast<int>(image.shape(0)), cols = static_cast<int>(image.shape(1));
        py::array_t<double> output({rows, cols});
        const auto* source = static_cast<const std::uint8_t*>(image.data());
        auto* dest = output.mutable_data();
        {
            py::gil_scoped_release release;
            local_binary_pattern_into(source, rows, cols, points, radius, static_cast<LbpMethod>(method), dest);
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("points"), py::arg("radius"), py::arg("method"));

    module.def("multiblock_lbp_native", [](const py::array& image, int r, int c, int w, int h) -> py::object {
        const auto type = prepare(image);
        if (type != Pixel::f32 && type != Pixel::f64 && type != Pixel::u64) return py::none();
        if (r < 0 || c < 0 || w <= 0 || h <= 0 ||
            r >= image.shape(0) || c >= image.shape(1) ||
            h > (image.shape(0) - r) / 3 || w > (image.shape(1) - c) / 3)
            throw py::value_error("the 3x3 LBP grid must fit inside the integral image");
        const int cols = static_cast<int>(image.shape(1));
        const auto* data = image.data();
        const auto count = image.size();
        int result = 0;
        bool compatible = true;
        {
            py::gil_scoped_release release;
            if (type == Pixel::f32) result = multiblock_lbp(static_cast<const float*>(data), cols, r, c, w, h);
            else {
                const auto* doubles = static_cast<const double*>(data);
                if (type == Pixel::f64) {
                    // NumPy converts the whole table and can warn on overflow,
                    // including entries beyond the 3x3 window.
                    compatible = std::none_of(doubles, doubles + count, [](double v) {
                        return std::abs(v) > std::numeric_limits<float>::max() && std::isfinite(v);
                    });
                }
                // Only four grid boundary coordinates per axis can contribute.
                float grid[16]{};
                for (int y = 0; compatible && y < 4; ++y) {
                    const auto row = r + y * h - 1;
                    if (row < 0) continue;
                    for (int x = 0; x < 4; ++x) {
                        const auto col = c + x * w - 1;
                        if (col < 0) continue;
                        const auto i = std::size_t(row) * cols + col;
                        grid[y * 4 + x] = type == Pixel::f64
                            ? static_cast<float>(doubles[i])
                            : static_cast<float>(static_cast<const std::uint64_t*>(data)[i]);
                    }
                }
                if (compatible) result = multiblock_lbp(grid, 4, 1, 1, 1, 1);
            }
        }
        if (!compatible) return py::none();
        return py::int_(result);
    }, py::arg("image").noconvert(), py::arg("r"), py::arg("c"), py::arg("width"), py::arg("height"));

    module.def("hog_native", [](const py::array& image, int bins, int cr, int cc, int br,
                               int bc, int norm, bool visualize, bool sqrt, bool flatten) -> py::object {
        const auto type = prepare(image, 2, 3);
        if (type == Pixel::unsupported || type == Pixel::u64 || bins < 1 || bins > 128 ||
            cr < 1 || cc < 1 || br < 1 || bc < 1 || norm < 0 || norm > 3 ||
            (visualize && std::min(cr, cc) < 2))
            return py::none();
        const HogOptions options{static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)),
                                 image.ndim() == 2 ? 1 : static_cast<int>(image.shape(2)),
                                 cr, cc, br, bc, bins, static_cast<HogNorm>(norm)};
        if (cr > options.rows || cc > options.cols || br > options.rows / cr || bc > options.cols / cc ||
            cr > std::numeric_limits<int>::max() / cc) return py::none();
        if (type == Pixel::u8) return hog_result<std::uint8_t, double>(image, options, visualize, sqrt, flatten);
        if (type == Pixel::f32) return hog_result<float, float>(image, options, visualize, sqrt, flatten);
        return hog_result<double, double>(image, options, visualize, sqrt, flatten);
    }, py::arg("image").noconvert(), py::arg("bins"), py::arg("cell_rows"), py::arg("cell_cols"),
       py::arg("block_rows"), py::arg("block_cols"), py::arg("norm"), py::arg("visualize"),
       py::arg("sqrt"), py::arg("flatten"));

    module.def("daisy_native", [](const py::array& image, int step, int radius, int histograms, int orientations,
                                 int norm, const std::vector<double>& sigmas,
                                 const std::vector<double>& radii) -> py::object {
        const auto type = prepare(image);
        if (type == Pixel::unsupported || type == Pixel::u64 || step <= 0 || radius <= 0 ||
            radius > (std::min(image.shape(0), image.shape(1)) - 1) / 2 ||
            histograms <= 0 || histograms > 128 || orientations <= 0 || orientations > 64 ||
            norm < 0 || norm > 3 || radii.empty() || radii.size() > 64 || sigmas.size() != radii.size() + 1 ||
            !std::all_of(radii.begin(), radii.end(), [=](double r) { return r >= 0 && r <= radius; }) ||
            !std::all_of(sigmas.begin(), sigmas.end(), [](double s) { return s >= 0 && s <= 256; }))
            return py::none();
        const DaisyOptions options{static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)), step, radius,
                                   static_cast<int>(radii.size()), histograms, orientations, static_cast<DaisyNorm>(norm)};
        if (type == Pixel::u8) return daisy_result<std::uint8_t, double>(image, options, sigmas, radii);
        if (type == Pixel::f32) return daisy_result<float, float>(image, options, sigmas, radii);
        return daisy_result<double, double>(image, options, sigmas, radii);
    }, py::arg("image").noconvert(), py::arg("step"), py::arg("radius"), py::arg("histograms"),
       py::arg("orientations"), py::arg("norm"), py::arg("sigmas"), py::arg("radii"));

    module.def("match_template_native", [](const py::array& image, const py::array& pattern,
                                          bool pad, double constant) -> py::object {
        const auto type = prepare(image);
        if ((type != Pixel::f32 && type != Pixel::f64 && type != Pixel::u8) || type != prepare(pattern) ||
            pattern.shape(0) > image.shape(0) || pattern.shape(1) > image.shape(1) ||
            !std::isfinite(constant) || (type == Pixel::f32 && std::abs(constant) > std::numeric_limits<float>::max()))
            return py::none();
        if (type == Pixel::f32) return template_result<float>(image, pattern, pad, constant);
        if (type == Pixel::u8) return template_result<std::uint8_t, double>(image, pattern, pad, constant);
        return template_result<double>(image, pattern, pad, constant);
    }, py::arg("image").noconvert(), py::arg("template").noconvert(), py::arg("pad"), py::arg("constant"));

    module.def("multiscale_basic_features_native", [](const py::array& image, bool intensity, bool edges,
                                                      bool texture, const std::vector<double>& sigmas) -> py::object {
        const auto type = prepare(image, 2, 3);
        if (type == Pixel::unsupported || type == Pixel::u64 || !(intensity || edges || texture) ||
            (texture && std::min(image.shape(0), image.shape(1)) < 2) ||
            sigmas.empty() || sigmas.size() > 64 ||
            !std::all_of(sigmas.begin(), sigmas.end(), [](double s) { return s > 0 && s <= 256; }))
            return py::none();
        const BasicFeatureOptions options{static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)),
                                          image.ndim() == 2 ? 1 : static_cast<int>(image.shape(2)),
                                          intensity, edges, texture};
        if (type == Pixel::u8) return basic_result<std::uint8_t>(image, options, sigmas);
        if (type == Pixel::f32) return basic_result<float>(image, options, sigmas);
        return basic_result<double>(image, options, sigmas);
    }, py::arg("image").noconvert(), py::arg("intensity"), py::arg("edges"), py::arg("texture"), py::arg("sigmas"));
}
