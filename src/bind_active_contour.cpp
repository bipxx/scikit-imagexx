#include "skimagexx/active_contour.hpp"

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace py = pybind11;

namespace {

using skimagexx::ActiveContourBoundary;
using skimagexx::ActiveContourOptions;

std::optional<ActiveContourBoundary> parse_boundary(const py::handle& value) {
    if (!PyUnicode_CheckExact(value.ptr())) {
        return std::nullopt;
    }
    const auto equal = [&value](const char* label) {
        return PyUnicode_CompareWithASCIIString(value.ptr(), label) == 0;
    };
    if (equal("periodic")) return ActiveContourBoundary::periodic;
    if (equal("fixed") || equal("fixed-fixed")) return ActiveContourBoundary::fixed;
    if (equal("free") || equal("free-free")) return ActiveContourBoundary::free;
    if (equal("fixed-free")) return ActiveContourBoundary::fixed_free;
    if (equal("free-fixed")) return ActiveContourBoundary::free_fixed;
    return std::nullopt;
}

std::optional<double> scalar(const py::handle& value) {
    if (PyFloat_CheckExact(value.ptr())) {
        return PyFloat_AS_DOUBLE(value.ptr());
    }
    if (PyLong_CheckExact(value.ptr())) {
        int overflow = 0;
        const auto integer = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
        if (!overflow) {
            return static_cast<double>(integer);
        }
    }
    return std::nullopt;
}

bool exact_array(const py::handle& value) {
    return Py_TYPE(value.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

bool available_storage(const py::array& array) {
    if (!(array.flags() & py::array::c_style) || !array.writeable()) {
        return false;
    }
    try {
        skimagexx::bindings::check_array_storage(array);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

template<class T>
bool plain_dtype(const py::array& array) {
    const auto dtype = array.dtype();
    const auto expected = py::dtype::of<T>();
    return (dtype.is(expected)
            || (dtype.num() == expected.num()
                && py::detail::npy_api::get().PyArray_EquivTypes_(dtype.ptr(), expected.ptr())
                && dtype.attr("isnative").cast<bool>() && dtype.attr("metadata").is_none()))
           && reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) == 0;
}

template<class Pixel, class Work>
py::object run(const py::array& image, const py::array& snake, std::size_t channels,
               ActiveContourOptions options) {
    const auto coordinates = [&snake](auto sample_type) {
        using Sample = decltype(sample_type);
        const auto* initial = static_cast<const Sample*>(snake.data());
        std::vector<Work> copy(std::size_t(snake.size()));
        std::transform(initial, initial + snake.size(), copy.begin(),
                       [](Sample coordinate) { return static_cast<Work>(coordinate); });
        return copy;
    };
    std::vector<Work> snake_copy;
    if (plain_dtype<double>(snake)) {
        snake_copy = coordinates(double{});
    } else if (plain_dtype<float>(snake)) {
        snake_copy = coordinates(float{});
    } else {
        return py::none();
    }

    // Copy while holding the GIL. The solver and the Sobel worker pool only
    // access these snapshots while Python threads can run.
    const auto* pixels = static_cast<const Pixel*>(image.data());
    const std::vector<Pixel> image_copy(pixels, pixels + image.size());
    const auto rows = std::size_t(image.shape(0));
    const auto columns = std::size_t(image.shape(1));
    const auto points = std::size_t(snake.shape(0));
    std::vector<Work> fitted;
    bool supported;
    {
        py::gil_scoped_release unlocked;
        supported = skimagexx::active_contour(image_copy.data(), rows, columns, channels,
                                              snake_copy.data(), points, options, fitted);
    }
    if (!supported) {
        return py::none();
    }
    py::array_t<Work> output(std::vector<py::ssize_t>{static_cast<py::ssize_t>(points), 2});
    std::memcpy(output.mutable_data(), fitted.data(), fitted.size() * sizeof(Work));
    return output;
}

py::object active_contour(const py::handle& image_object, const py::handle& snake_object,
                          const py::handle& alpha, const py::handle& beta,
                          const py::handle& w_line, const py::handle& w_edge,
                          const py::handle& gamma, const py::handle& max_px_move,
                          const py::handle& max_num_iter, const py::handle& convergence,
                          const py::handle& boundary_condition) {
    const auto a = scalar(alpha);
    const auto b = scalar(beta);
    const auto line = scalar(w_line);
    const auto edge = scalar(w_edge);
    const auto g = scalar(gamma);
    const auto move = scalar(max_px_move);
    const auto stop = scalar(convergence);
    const auto boundary = parse_boundary(boundary_condition);
    if (!a || !b || !line || !edge || !g || !move || !stop || !boundary
        || !PyLong_CheckExact(max_num_iter.ptr())
        || !exact_array(image_object) || !exact_array(snake_object)) {
        return py::none();
    }
    const auto iterations = scalar(max_num_iter);
    if (!iterations || *iterations < 1 || *iterations > 10000) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(image_object);
    const auto snake = py::reinterpret_borrow<py::array>(snake_object);
    if ((image.ndim() != 2 && (image.ndim() != 3 || image.shape(2) != 3))
        || snake.ndim() != 2 || snake.shape(1) != 2) {
        return py::none();
    }
    const std::size_t channels = image.ndim() == 3 ? 3 : 1;
    if (!skimagexx::active_contour_shape_supported(std::size_t(image.shape(0)), std::size_t(image.shape(1)),
                                                 channels, std::size_t(snake.shape(0)))
        || !available_storage(image) || !available_storage(snake)) {
        return py::none();
    }
    const ActiveContourOptions options{*a, *b, *line, *edge, *g, *move,
                                       static_cast<int>(*iterations), *stop, *boundary};
    if (plain_dtype<double>(image)) {
        return run<double, double>(image, snake, channels, options);
    }
    if (plain_dtype<float>(image)) {
        return run<float, float>(image, snake, channels, options);
    }
    if (plain_dtype<std::uint8_t>(image)) {
        return run<std::uint8_t, double>(image, snake, channels, options);
    }
    return py::none();
}

}  // namespace

void bind_active_contour(py::module_& module) {
    module.def("active_contour_finite", &active_contour, py::arg("image"), py::arg("snake"),
               py::arg("alpha"), py::arg("beta"), py::arg("w_line"), py::arg("w_edge"),
               py::arg("gamma"), py::arg("max_px_move"), py::arg("max_num_iter"),
               py::arg("convergence"), py::arg("boundary_condition"));
}
