#include "skimagexx/sift.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <limits>

namespace py = pybind11;

namespace {
using skimagexx::PixelType;

PixelType pixel_type(const py::dtype& type) {
    switch (type.normalized_num()) {
        case py::dtype::num_of<bool>(): return PixelType::boolean;
        case py::dtype::num_of<std::uint8_t>(): return PixelType::uint8;
        case py::dtype::num_of<std::int8_t>(): return PixelType::int8;
        case py::dtype::num_of<std::uint16_t>(): return PixelType::uint16;
        case py::dtype::num_of<std::int16_t>(): return PixelType::int16;
        case py::dtype::num_of<std::uint32_t>(): return PixelType::uint32;
        case py::dtype::num_of<std::int32_t>(): return PixelType::int32;
        case py::dtype::num_of<std::uint64_t>(): return PixelType::uint64;
        case py::dtype::num_of<std::int64_t>(): return PixelType::int64;
        case py::dtype::num_of<float>(): return PixelType::float32;
        case py::dtype::num_of<double>(): return PixelType::float64;
        default: throw py::value_error("Unsupported SIFT input dtype");
    }
}

void check_storage(const py::array& array) {
    if (array.ndim() != 2 || !(array.flags() & py::array::c_style) ||
        !array.attr("flags").attr("aligned").cast<bool>() ||
        !array.dtype().attr("isnative").cast<bool>() ||
        array.shape(0) > std::numeric_limits<int>::max() ||
        array.shape(1) > std::numeric_limits<int>::max())
        throw py::value_error("SIFT requires a contiguous 2-D native-endian array");
    if (!array.owndata()) skimagexx::bindings::check_array_storage(array);
}

py::array descriptors(const std::vector<std::uint8_t>& bytes, py::ssize_t rows, int columns) {
    py::array_t<std::uint8_t> output(py::array::ShapeContainer{rows, py::ssize_t(columns)});
    std::copy(bytes.begin(), bytes.end(), output.mutable_data());
    return output;
}

int index(double val) {
    if (!(val >= 0 && val <= std::numeric_limits<int>::max()) || val != int(val))
        throw py::value_error("Expected a SIFT scale or octave index");
    return int(val);
}
} // namespace

void bind_sift(py::module_& module) {
    module.def("sift", [](const py::array& image, int upsampling, int n_octaves, int n_scales,
                          double sigma_min, double sigma_in, double c_dog, double c_edge,
                          int n_bins, double lambda_ori, double c_max, double lambda_descr,
                          int n_hist, int n_ori, bool describe, py::object stored) -> py::object {
        check_storage(image);
        const auto type = pixel_type(image.dtype());
        const skimagexx::SiftOptions options{upsampling, n_octaves, n_scales, sigma_min, sigma_in,
                                             c_dog, c_edge, n_bins, lambda_ori, c_max, lambda_descr,
                                             n_hist, n_ori};
        const int height = int(image.shape(0)), width = int(image.shape(1));
        if (!stored.is_none()) {
            const auto array = stored.cast<py::array>();
            check_storage(array);
            if (array.shape(1) != 6 || !array.dtype().is(py::dtype::of<double>()))
                throw py::value_error("Invalid SIFT keypoint array");
            const auto* input = static_cast<const double*>(array.data());
            std::vector<skimagexx::SiftKeypoint> keys;
            keys.reserve(array.shape(0));
            for (py::ssize_t i = 0; i < array.shape(0); ++i, input += 6)
                keys.push_back({input[0], input[1], input[2], input[3], index(input[4]), index(input[5])});
            std::vector<std::uint8_t> output;
            bool zero_norm = false;
            {
                py::gil_scoped_release unlock;
                output = skimagexx::sift_extract(image.data(), height, width, type, options, keys, &zero_norm);
            }
            if (zero_norm) throw py::value_error("SIFT requires nonzero finite descriptor norms");
            return descriptors(output, array.shape(0), n_hist * n_hist * n_ori);
        }
        skimagexx::SiftResult result;
        {
            py::gil_scoped_release unlock;
            result = skimagexx::sift_detect(image.data(), height, width, type, options, describe);
        }
        if (result.descriptor_has_zero_norm)
            throw py::value_error("SIFT requires nonzero finite descriptor norms");
        const auto count = static_cast<py::ssize_t>(result.keypoints.size());
        py::array_t<double> positions(py::array::ShapeContainer{count, py::ssize_t(2)});
        py::array_t<double> sigmas(count), orientations(count);
        py::array_t<std::intptr_t> scales(count), octaves(count);
        auto* xy = positions.mutable_data();
        auto* sigma = sigmas.mutable_data();
        auto* angle = orientations.mutable_data();
        auto* scale = scales.mutable_data();
        auto* octave = octaves.mutable_data();
        for (const auto& key : result.keypoints) {
            *xy++ = key.row;
            *xy++ = key.col;
            *sigma++ = key.sigma;
            *angle++ = key.orientation;
            *scale++ = key.scale;
            *octave++ = key.octave;
        }
        py::object values = py::none();
        if (describe) values = descriptors(result.descriptors, count, n_hist * n_hist * n_ori);
        return py::make_tuple(positions, sigmas, orientations, scales, octaves, values,
                              result.promoted_positions, result.multiple_orientations);
    });
}
