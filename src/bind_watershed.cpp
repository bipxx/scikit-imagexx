#include "bind_array_storage.hpp"
#include <skimagexx/watershed.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL SKIMAGEXX_WATERSHED_ARRAY_API
#include <numpy/arrayobject.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

namespace py = pybind11;
namespace {

bool real_dtype(const py::dtype& dtype) {
    const auto bytes = dtype.itemsize();
    if (dtype.kind() == 'f')
        return dtype.char_() == 'e' || dtype.char_() == 'f' || dtype.char_() == 'd';
    return (dtype.kind() == 'b' || dtype.kind() == 'i' || dtype.kind() == 'u') &&
           (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8);
}

void check_storage(const py::array& value) {
    if (!PyArray_CheckExact(value.ptr()))
        throw py::type_error("watershed requires an exact ndarray");
    auto* array = reinterpret_cast<PyArrayObject*>(value.ptr());
    const auto dtype = value.dtype();
    if (!real_dtype(dtype) || PyDataType_METADATA(PyArray_DESCR(array)) || !PyArray_ISNOTSWAPPED(array))
        throw py::type_error("watershed requires an untagged native fixed-width real dtype");
    if (!PyArray_ISCARRAY(array))
        throw py::value_error("watershed requires writable aligned C-order storage");
    skimagexx::bindings::check_array_storage(value);
}

py::array cast_array(const py::array& value, int type) {
    auto* result = PyArray_Cast(reinterpret_cast<PyArrayObject*>(value.ptr()), type);
    if (!result)
        throw py::error_already_set();
    return py::reinterpret_steal<py::array>(result);
}

bool same_shape(const py::array& first, const py::array& second) {
    return first.ndim() == second.ndim() &&
           std::equal(first.shape(), first.shape() + first.ndim(), second.shape());
}

template<class T>
void apply(const py::array& image, const py::array& markers, const std::uint8_t* mask,
           const std::vector<std::size_t>& shape, double compactness, bool line,
           const skimagexx::WatershedNeighborhood* neighborhood, py::array& output,
           bool boolean_markers) {
    const auto* values = static_cast<const double*>(image.data());
    const auto* labels = static_cast<const T*>(markers.data());
    auto* result = static_cast<T*>(output.mutable_data());
    const auto count = static_cast<std::size_t>(image.size());
    py::gil_scoped_release release;
    skimagexx::watershed_into(values, labels, mask, count, shape.data(), shape.size(),
                              compactness, line, result, neighborhood, boolean_markers);
}

}  // namespace

void bind_watershed(py::module_& module) {
    if (_import_array() < 0)
        throw py::error_already_set();
    module.def("watershed_check_storage", &check_storage, py::arg("array").noconvert());
    module.def("watershed", [](const py::array& image, const py::array& markers,
                               const py::object& mask, double compactness, bool line,
                               const py::object& padding, const py::object& neighbors) {
        skimagexx::WatershedNeighborhood neighborhood;
        if (padding.is_none() != neighbors.is_none())
            throw py::value_error("watershed padding and offsets must be provided together");
        if (!padding.is_none()) {
            neighborhood.padding = padding.cast<std::vector<std::size_t>>();
            neighborhood.offsets = neighbors.cast<std::vector<std::ptrdiff_t>>();
        }

        check_storage(image);
        check_storage(markers);
        if (image.ndim() == 0 || !same_shape(image, markers))
            throw py::value_error("watershed requires matching positive-rank image and marker shapes");
        const auto dtype = markers.dtype();
        const auto kind = dtype.kind();
        if (kind != 'i' && kind != 'u' && kind != 'b')
            throw py::type_error("watershed requires integer or boolean marker labels");

        py::array mask_array;
        if (!mask.is_none()) {
            if (!PyArray_CheckExact(mask.ptr()))
                throw py::type_error("watershed mask requires an exact ndarray");
            mask_array = py::reinterpret_borrow<py::array>(mask);
            check_storage(mask_array);
            if (!same_shape(image, mask_array))
                throw py::value_error("watershed requires matching image and mask shapes");
            if (mask_array.dtype().kind() != 'b')
                mask_array = cast_array(mask_array, NPY_BOOL);
        }

        py::array values = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(image.ptr())) == NPY_DOUBLE
            ? image : cast_array(image, NPY_DOUBLE);

        const std::vector<py::ssize_t> dimensions(image.shape(), image.shape() + image.ndim());
        const std::vector<std::size_t> shape(dimensions.begin(), dimensions.end());
        py::array output(dtype, dimensions, image.size() ? std::vector<py::ssize_t>{} :
                                                         std::vector<py::ssize_t>(shape.size(), 0));
        const auto* active = mask.is_none() ? nullptr : static_cast<const std::uint8_t*>(mask_array.data());
        const auto* custom = padding.is_none() ? nullptr : &neighborhood;
        const auto run = [&](auto label) {
            apply<decltype(label)>(values, markers, active, shape, compactness,
                                   line, custom, output, kind == 'b');
        };
        switch (dtype.itemsize()) {
        case 1:
            if (kind == 'i') run(std::int8_t{});
            else run(std::uint8_t{});
            break;
        case 2:
            if (kind == 'i') run(std::int16_t{});
            else run(std::uint16_t{});
            break;
        case 4:
            if (kind == 'i') run(std::int32_t{});
            else run(std::uint32_t{});
            break;
        case 8:
            if (kind == 'i') run(std::int64_t{});
            else run(std::uint64_t{});
            break;
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("markers").noconvert(), py::arg("mask") = py::none(),
       py::arg("compactness") = 0.0, py::arg("watershed_line") = false,
       py::arg("padding") = py::none(), py::arg("neighbors") = py::none());
}
