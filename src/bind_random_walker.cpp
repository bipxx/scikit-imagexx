#include "skimagexx/random_walker.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cfenv>
#include <cstdint>

namespace py = pybind11;

namespace {
using skimagexx::PixelType;

bool exact_array(py::handle value) {
    return Py_TYPE(value.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

bool eligible(const py::array& array) {
    return (array.flags() & py::array::c_style) && array.writeable()
           && array.attr("flags").attr("aligned").cast<bool>()
           && array.dtype().attr("metadata").is_none()
           && array.dtype().attr("isnative").cast<bool>();
}

std::optional<PixelType> data_type(const py::dtype& dtype) {
    if (dtype.equal(py::dtype::of<std::uint8_t>())) return PixelType::uint8;
    if (dtype.char_() == 'g') return std::nullopt;
    if (dtype.equal(py::dtype::of<float>())) return PixelType::float32;
    if (dtype.equal(py::dtype::of<double>())) return PixelType::float64;
    return std::nullopt;
}

std::optional<PixelType> marker_type(const py::dtype& dtype) {
    if (dtype.equal(py::dtype::of<std::int8_t>())) return PixelType::int8;
    if (dtype.equal(py::dtype::of<std::uint8_t>())) return PixelType::uint8;
    if (dtype.equal(py::dtype::of<std::int16_t>())) return PixelType::int16;
    if (dtype.equal(py::dtype::of<std::uint16_t>())) return PixelType::uint16;
    if (dtype.equal(py::dtype::of<std::int32_t>())) return PixelType::int32;
    if (dtype.equal(py::dtype::of<std::uint32_t>())) return PixelType::uint32;
    if (dtype.equal(py::dtype::of<std::int64_t>())) return PixelType::int64;
    if (dtype.equal(py::dtype::of<std::uint64_t>())) return PixelType::uint64;
    return std::nullopt;
}

template<class T>
void write_labels(const std::vector<std::int64_t>& labels, void* output) {
    std::transform(labels.begin(), labels.end(), static_cast<T*>(output),
                   [](std::int64_t value) { return static_cast<T>(value); });
}

void write_labels(PixelType type, const std::vector<std::int64_t>& labels, void* output) {
    switch (type) {
        case PixelType::int8: write_labels<std::int8_t>(labels, output); break;
        case PixelType::uint8: write_labels<std::uint8_t>(labels, output); break;
        case PixelType::int16: write_labels<std::int16_t>(labels, output); break;
        case PixelType::uint16: write_labels<std::uint16_t>(labels, output); break;
        case PixelType::int32: write_labels<std::int32_t>(labels, output); break;
        case PixelType::uint32: write_labels<std::uint32_t>(labels, output); break;
        case PixelType::int64: write_labels<std::int64_t>(labels, output); break;
        case PixelType::uint64: write_labels<std::uint64_t>(labels, output); break;
        default: throw py::type_error("random walker requires integer output labels");
    }
}
}  // namespace

void bind_random_walker(py::module_& module) {
    module.def("random_walker", [](py::handle data, py::handle labels, double beta,
                                  double tolerance, double probability_tolerance,
                                  const std::array<double, 3>& spacing,
                                  bool return_probabilities) -> py::object {
        if (!exact_array(data) || !exact_array(labels) || std::fegetround() != FE_TONEAREST)
            return py::none();
        const auto image = py::reinterpret_borrow<py::array>(data);
        const auto markers = py::reinterpret_borrow<py::array>(labels);
        const auto rank = image.ndim();
        if ((rank != 2 && rank != 3) || markers.ndim() != rank ||
            !eligible(image) || !eligible(markers) || image.size() == 0 ||
            !std::equal(image.shape(), image.shape() + rank, markers.shape()))
            return py::none();
        const auto input_type = data_type(image.dtype());
        const auto label_dtype = markers.dtype();
        const auto labels_type = marker_type(label_dtype);
        if (!input_type || !labels_type) return py::none();

        skimagexx::bindings::check_array_storage(image);
        skimagexx::bindings::check_array_storage(markers);
        const std::vector<std::size_t> shape(image.shape(), image.shape() + rank);
        std::vector<py::ssize_t> output_shape(image.shape(), image.shape() + rank);
        const auto* pixels = image.data();
        const auto* seeds = markers.data();
        const skimagexx::RandomWalkerOptions options{beta, tolerance, probability_tolerance,
                                                    spacing, return_probabilities};
        std::optional<skimagexx::RandomWalkerResult> result;
        {
            py::gil_scoped_release release;
            result = skimagexx::random_walker(pixels, *input_type, seeds, *labels_type, shape, options);
        }
        if (!result) return py::none();

        if (return_probabilities)
            output_shape.insert(output_shape.begin(), static_cast<py::ssize_t>(result->classes));
        py::array output(return_probabilities ? py::dtype::of<double>() : label_dtype, output_shape);
        if (return_probabilities)
            std::copy(result->probabilities.begin(), result->probabilities.end(),
                      static_cast<double*>(output.mutable_data()));
        else
            write_labels(*labels_type, result->labels, output.mutable_data());
        return py::make_tuple(output, result->convergence_warning, result->probability_warning);
    }, py::arg("data"), py::arg("labels"), py::arg("beta"), py::arg("tol"),
       py::arg("prob_tol"), py::arg("spacing"), py::arg("return_full_prob"));
}
