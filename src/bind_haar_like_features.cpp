#include "skimagexx/haar_like_features.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace {
using namespace skimagexx;

std::size_t checked_total(std::size_t total, std::size_t next) {
    constexpr auto limit = std::numeric_limits<py::ssize_t>::max() / sizeof(std::uint64_t);
    if (next > limit - total) {
        throw std::overflow_error("Haar output exceeds addressable array storage");
    }
    return total + next;
}

std::pair<std::vector<std::size_t>, std::size_t>
counts(int width, int height, const std::vector<HaarKind>& kinds) {
    std::vector<std::size_t> sizes;
    std::size_t total = 0;
    for (auto kind : kinds) {
        const auto size = haar_feature_count(width, height, kind);
        total = checked_total(total, size);
        sizes.push_back(size);
    }
    return {std::move(sizes), total};
}

py::tuple coordinates(int width, int height, const std::vector<HaarKind>& kinds,
                      const py::tuple& names) {
    if (kinds.size() != names.size() || kinds.empty()) {
        throw py::value_error("not enough values to unpack (expected 2, got 0)");
    }
    const auto [sizes, total] = counts(width, height, kinds);
    const auto count = static_cast<py::ssize_t>(total);
    const auto object_dtype = py::dtype::of<PyObject*>();
    const py::array::ShapeContainer shape{count};
    const py::array::StridesContainer strides{count ? static_cast<py::ssize_t>(sizeof(PyObject*)) : 0};
    // NumPy initializes object slots to null when the native constructor owns the data.
    // In particular, unfilled slots remain safe if building a Python rectangle throws.
    py::array coordinates(object_dtype, shape, strides);
    py::array labels(object_dtype, shape, strides);
    auto** boxes = static_cast<PyObject**>(coordinates.mutable_data());
    auto** types = static_cast<PyObject**>(labels.mutable_data());
    for (std::size_t k = 0; k < kinds.size(); ++k) {
        std::vector<HaarLikeFeature> features(sizes[k]);
        haar_coordinates(width, height, kinds[k], features.data(), features.size());
        const auto blocks = haar_rectangle_count(kinds[k]);
        const py::handle type = names[k];
        for (const auto& feature : features) {
            py::list rectangles(blocks);
            for (std::size_t i = 0; i < blocks; ++i) {
                const auto& box = feature.rectangles[i];
                py::list endpoints(2);
                // All entries in each newly allocated list still hold null pointers.
                PyList_SET_ITEM(endpoints.ptr(), 0, py::make_tuple(box.top, box.left).release().ptr());
                PyList_SET_ITEM(endpoints.ptr(), 1, py::make_tuple(box.bottom, box.right).release().ptr());
                PyList_SET_ITEM(rectangles.ptr(), i, endpoints.release().ptr());
            }
            *boxes++ = rectangles.release().ptr();
            *types++ = type.inc_ref().ptr();
        }
    }
    return py::make_tuple(std::move(coordinates), std::move(labels));
}

template<class T>
PixelType checked_pixel_type(const py::dtype& dtype, PixelType type) {
    const auto canonical = py::dtype::of<T>();
    if (dtype.is(canonical) || (dtype.itemsize() == sizeof(T)
            && dtype.attr("isnative").cast<bool>() && dtype.equal(canonical))) {
        return type;
    }
    throw py::type_error("No matching signature found");
}

PixelType pixel_type(const py::dtype& dtype) {
    switch (dtype.normalized_num()) {
    case py::dtype::num_of<bool>(): return checked_pixel_type<bool>(dtype, PixelType::boolean);
    case py::dtype::num_of<std::uint8_t>(): return checked_pixel_type<std::uint8_t>(dtype, PixelType::uint8);
    case py::dtype::num_of<std::int8_t>(): return checked_pixel_type<std::int8_t>(dtype, PixelType::int8);
    case py::dtype::num_of<std::uint16_t>(): return checked_pixel_type<std::uint16_t>(dtype, PixelType::uint16);
    case py::dtype::num_of<std::int16_t>(): return checked_pixel_type<std::int16_t>(dtype, PixelType::int16);
    case py::dtype::num_of<std::uint32_t>(): return checked_pixel_type<std::uint32_t>(dtype, PixelType::uint32);
    case py::dtype::num_of<std::int32_t>(): return checked_pixel_type<std::int32_t>(dtype, PixelType::int32);
    case py::dtype::num_of<std::uint64_t>(): return checked_pixel_type<std::uint64_t>(dtype, PixelType::uint64);
    case py::dtype::num_of<std::int64_t>(): return checked_pixel_type<std::int64_t>(dtype, PixelType::int64);
    case py::dtype::num_of<float>(): return checked_pixel_type<float>(dtype, PixelType::float32);
    case py::dtype::num_of<double>(): return checked_pixel_type<double>(dtype, PixelType::float64);
    default: throw py::type_error("No matching signature found");
    }
}

struct Prepared {
    const void* data;
    PixelType type;
    int width, height;
};

Prepared prepared(const py::array& image) {
    if (image.ndim() != 2 || !(image.flags() & py::array::c_style) || !image.owndata() ||
        !image.attr("flags").attr("aligned").cast<bool>() ||
        image.shape(0) >= std::numeric_limits<int>::max() ||
        image.shape(1) >= std::numeric_limits<int>::max()) {
        throw py::value_error("Expected an owning, contiguous two-dimensional Haar ROI");
    }
    return {image.data(), pixel_type(image.dtype()), static_cast<int>(image.shape(1)),
            static_cast<int>(image.shape(0))};
}

py::array new_values(PixelType type, std::size_t count) {
    const auto size = static_cast<py::ssize_t>(checked_total(0, count));
    if (type == PixelType::float32) return py::array_t<float>(size);
    if (type == PixelType::float64) return py::array_t<double>(size);
    return py::array_t<std::int64_t>(size);
}

py::array generated(const py::array& image, const std::vector<HaarKind>& kinds) {
    const auto input = prepared(image);
    const auto [sizes, total] = counts(input.width, input.height, kinds);
    if (!total) {
        throw py::value_error("need at least one array to concatenate");
    }
    auto result = new_values(input.type, total);
    auto* output = static_cast<unsigned char*>(result.mutable_data());
    const auto itemsize = result.itemsize();
    py::gil_scoped_release release;
    for (std::size_t k = 0; k < kinds.size(); ++k) {
        haar_generated_values(input.data, input.type, input.width, input.height, kinds[k], output, sizes[k]);
        output += sizes[k] * itemsize;
    }
    return result;
}

py::array selected(const py::array& image, const std::vector<HaarKind>& kinds,
                   const std::vector<std::vector<std::array<int, 4>>>& rows) {
    const auto input = prepared(image);
    if (rows.size() != kinds.size()) {
        throw py::value_error("Inconsistent Haar feature selection");
    }
    std::vector<HaarLikeFeature> features;
    features.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        HaarLikeFeature feature{kinds[i], {}};
        const auto count = haar_rectangle_count(feature.kind);
        if (rows[i].size() != count) {
            throw py::value_error("Haar feature has too few rectangles");
        }
        for (std::size_t j = 0; j < count; ++j) {
            const auto& box = rows[i][j];
            feature.rectangles[j] = {box[0], box[1], box[2], box[3]};
        }
        features.push_back(feature);
    }
    auto result = new_values(input.type, features.size());
    auto* output = result.mutable_data();
    py::gil_scoped_release release;
    haar_selected_values(input.data, input.type, input.width, input.height,
                         features.data(), output, features.size());
    return result;
}
} // namespace

void bind_haar_like_features(py::module_& module) {
    py::enum_<HaarKind>(module, "_HaarKind")
        .value("two_x", HaarKind::two_x).value("two_y", HaarKind::two_y)
        .value("three_x", HaarKind::three_x).value("three_y", HaarKind::three_y)
        .value("four", HaarKind::four);
    module.def("haar_coordinates", &coordinates);
    module.def("haar_generated_values", &generated);
    module.def("haar_selected_values", &selected);
}
