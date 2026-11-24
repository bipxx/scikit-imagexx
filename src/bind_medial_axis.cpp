#include "skimagexx/medial_axis.hpp"

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>

namespace py = pybind11;

namespace {

bool plain_array(const py::handle& input) {
    return Py_TYPE(input.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

bool bounded(const py::array& array) {
    try {
        skimagexx::bindings::check_array_storage(array);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

bool boolean_image(const py::array& array) {
    return array.ndim() == 2 && array.dtype().equal(py::dtype::of<bool>())
        && array.dtype().attr("metadata").is_none()
        && (array.flags() & py::array::c_style)
        && array.shape(0) > 0 && array.shape(1) > 0
        && std::size_t(array.shape(0)) <= skimagexx::medial_axis_max_extent
        && std::size_t(array.shape(1)) <= skimagexx::medial_axis_max_extent && bounded(array);
}

bool same_shape(const py::array& a, const py::array& b) {
    return a.ndim() == 2 && b.ndim() == 2 && a.shape(0) == b.shape(0) && a.shape(1) == b.shape(1);
}

py::object prepare(const py::handle& input, const py::handle& mask) {
    if (!plain_array(input) || (!mask.is_none() && !plain_array(mask))) {
        return py::none();
    }
    const auto image = py::reinterpret_borrow<py::array>(input);
    if (!boolean_image(image)) {
        return py::none();
    }
    const std::uint8_t* mask_data = nullptr;
    if (!mask.is_none()) {
        const auto array = py::reinterpret_borrow<py::array>(mask);
        if (!same_shape(image, array) || !boolean_image(array)) {
            return py::none();
        }
        mask_data = static_cast<const std::uint8_t*>(array.data());
    }
    py::array_t<bool> prepared(std::vector<py::ssize_t>{image.shape(0), image.shape(1)});
    static_assert(sizeof(bool) == 1);
    auto* output = reinterpret_cast<std::uint8_t*>(prepared.mutable_data());
    const auto* data = static_cast<const std::uint8_t*>(image.data());
    std::size_t foreground = 0;
    for (std::size_t i = 0; i < std::size_t(image.size()); ++i) {
        const auto selected = mask_data ? mask_data[i] : 1;
        // Normalize only the established Boolean domain. Other descriptors
        // and unchecked buffers stay on the compatibility route.
        if (data[i] > 1 || selected > 1) {
            return py::none();
        }
        output[i] = static_cast<std::uint8_t>(data[i] & selected);
        foreground += output[i];
    }
    return py::make_tuple(std::move(prepared), foreground);
}

template <class T>
bool valid_provider_buffer(const py::array& array) {
    return (array.flags() & py::array::c_style) && array.dtype().equal(py::dtype::of<T>())
        && reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) == 0 && bounded(array);
}

py::array_t<bool> ordered(const py::handle& input, const py::handle& field, const py::handle& ties) {
    if (!plain_array(input) || !plain_array(field) || !plain_array(ties)) {
        throw py::value_error("medial_axis requires ordinary ndarray provider values");
    }
    const auto image = py::reinterpret_borrow<py::array>(input);
    const auto distance = py::reinterpret_borrow<py::array>(field);
    const auto priority = py::reinterpret_borrow<py::array>(ties);
    if (!boolean_image(image) || !same_shape(image, distance) || priority.ndim() != 1
        || !valid_provider_buffer<double>(distance) || !valid_provider_buffer<std::int64_t>(priority)) {
        throw py::value_error("medial_axis requires bounded matching EDT and permutation buffers");
    }
    std::vector<std::uint8_t> result;
    if (!skimagexx::medial_axis_ordered(
            static_cast<const std::uint8_t*>(image.data()),
            static_cast<const double*>(distance.data()), std::size_t(distance.size()),
            static_cast<const std::int64_t*>(priority.data()), std::size_t(priority.size()),
            std::size_t(image.shape(0)), std::size_t(image.shape(1)), result)) {
        throw py::value_error("medial_axis requires finite distances and unique foreground priorities");
    }
    py::array_t<bool> output(std::vector<py::ssize_t>{image.shape(0), image.shape(1)});
    std::memcpy(output.mutable_data(), result.data(), result.size());
    return output;
}

}  // namespace

void bind_medial_axis(py::module_& module) {
    module.def("medial_axis_prepare", &prepare, py::arg("image"), py::arg("mask"));
    module.def("medial_axis_ordered", &ordered, py::arg("prepared"), py::arg("distance"), py::arg("priorities"));
}
