#include "skimagexx/orb.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>

namespace py = pybind11;
using skimagexx::PixelType;

namespace {

PixelType source_type(const py::array& image) {
    const auto dtype = image.dtype();
    if (!dtype.attr("isnative").cast<bool>())
        throw py::type_error("ORB requires native byte order");
    const auto kind = dtype.kind();
    const auto size = dtype.itemsize();
    if (kind == 'b') return PixelType::boolean;
    if (kind == 'u' && size == 1) return PixelType::uint8;
    if (kind == 'i' && size == 1) return PixelType::int8;
    if (kind == 'u' && size == 2) return PixelType::uint16;
    if (kind == 'i' && size == 2) return PixelType::int16;
    if (kind == 'u' && size == 4) return PixelType::uint32;
    if (kind == 'i' && size == 4) return PixelType::int32;
    if (kind == 'u' && size == 8) return PixelType::uint64;
    if (kind == 'i' && size == 8) return PixelType::int64;
    if (kind == 'f' && size == 4) return PixelType::float32;
    if (kind == 'f' && size == 8) return PixelType::float64;
    throw py::type_error("unsupported ORB input dtype");
}

void require_contiguous(const py::array& array, py::ssize_t ndim) {
    if (array.ndim() != ndim || !(array.flags() & py::array::c_style)
            || !array.attr("flags").attr("aligned").cast<bool>())
        throw py::value_error("ORB requires aligned contiguous arrays of the expected rank");
}

bool single_precision(const py::array& image) {
    require_contiguous(image, 2);
    const auto type = source_type(image);
    if (type != PixelType::float32 && type != PixelType::float64)
        throw py::type_error("ORB octave images require floating-point values");
    return type == PixelType::float32;
}

py::object prepare(const py::array& image) {
    require_contiguous(image, 2);
    const auto source = source_type(image);
    const auto dtype = source == PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
    py::array output(dtype, {image.shape(0), image.shape(1)});
    const auto* in = image.data();
    auto* out = output.mutable_data();
    bool valid;
    {
        py::gil_scoped_release release;
        valid = skimagexx::orb_prepare(in, out, source, std::size_t(image.shape(0)), std::size_t(image.shape(1)));
    }
    if (!valid) return py::none();
    return output;
}

py::array reduce(const py::array& image, py::ssize_t rows, py::ssize_t cols,
                 const std::vector<double>& weights, bool legacy_zoom) {
    const auto single = single_precision(image);
    if (rows < 1 || cols < 1 || rows > image.shape(0) || cols > image.shape(1))
        throw py::value_error("invalid ORB target shape");
    py::array output(image.dtype(), {rows, cols});
    const auto* in = image.data();
    auto* out = output.mutable_data();
    {
        py::gil_scoped_release release;
        skimagexx::orb_reduce(in, out, single, std::size_t(image.shape(0)), std::size_t(image.shape(1)),
                             std::size_t(rows), std::size_t(cols), weights, legacy_zoom);
    }
    return output;
}

template<class T>
py::tuple native_points(const skimagexx::OrbOctave& octave) {
    const auto count = py::ssize_t(octave.points.size());
    py::array_t<T> orientations(count), responses(count);
    if (!octave.had_fast_peaks) {
        py::array_t<T> points({count, py::ssize_t(2)});
        return py::make_tuple(points, orientations, responses);
    }
    py::array_t<py::ssize_t> points({count, py::ssize_t(2)});
    auto* coords = points.mutable_data();
    auto* angles = orientations.mutable_data();
    auto* scores = responses.mutable_data();
    for (std::size_t i = 0; i < octave.points.size(); ++i) {
        const auto& point = octave.points[i];
        coords[2 * i] = point.row;
        coords[2 * i + 1] = point.col;
        angles[i] = T(point.orientation);
        scores[i] = T(point.response);
    }
    return py::make_tuple(points, orientations, responses);
}

py::tuple detect(const py::array& image, int fast_n, double threshold, double k,
                 const std::array<double, 9>& weights) {
    const auto single = single_precision(image);
    const auto* in = image.data();
    skimagexx::OrbOctave octave;
    {
        py::gil_scoped_release release;
        octave = skimagexx::orb_detect_octave(in, single, std::size_t(image.shape(0)),
                                             std::size_t(image.shape(1)), fast_n, threshold, k, weights);
    }
    return single ? native_points<float>(octave) : native_points<double>(octave);
}

py::tuple extract(const py::array& image, const py::array& coordinates, const py::array& orientations) {
    const auto single = single_precision(image);
    require_contiguous(coordinates, 2);
    require_contiguous(orientations, 1);
    if (source_type(coordinates) != PixelType::float64 || coordinates.shape(1) != 2
            || orientations.shape(0) != coordinates.shape(0)
            || source_type(orientations) != (single ? PixelType::float32 : PixelType::float64))
        throw py::value_error("invalid ORB keypoint buffers");
    const auto count = std::size_t(coordinates.shape(0));
    const auto* in = image.data();
    const auto* points = static_cast<const double*>(coordinates.data());
    const auto* angles = orientations.data();
    skimagexx::OrbDescriptors result;
    {
        py::gil_scoped_release release;
        result = skimagexx::orb_extract_octave(in, single, std::size_t(image.shape(0)),
                                              std::size_t(image.shape(1)), points, angles, count);
    }
    constexpr auto size = py::ssize_t(skimagexx::orb_descriptor_size);
    py::array_t<std::uint8_t> descriptors({py::ssize_t(result.data.size() / size), size});
    py::array_t<bool> mask(py::ssize_t(result.mask.size()));
    std::copy(result.data.begin(), result.data.end(), descriptors.mutable_data());
    std::copy(result.mask.begin(), result.mask.end(), mask.mutable_data());
    return py::make_tuple(descriptors, mask);
}

} // namespace

void bind_orb(py::module_& module) {
    module.def("orb_image", &prepare, py::arg("image").noconvert());
    module.def("orb_reduce", &reduce, py::arg("image").noconvert(), py::arg("rows"),
               py::arg("cols"), py::arg("weights"), py::arg("legacy_zoom"));
    module.def("orb_detect_octave", &detect, py::arg("image").noconvert(), py::arg("fast_n"),
               py::arg("threshold"), py::arg("k"), py::arg("weights"));
    module.def("orb_extract_octave", &extract, py::arg("image").noconvert(),
               py::arg("coordinates").noconvert(), py::arg("orientations").noconvert());
}
