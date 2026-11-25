#include <skimagexx/polygon_perimeter.hpp>

#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <type_traits>

namespace py = pybind11;
namespace skimagexx::bindings {
namespace {

using Int = std::intptr_t;
constexpr long long limit = 1LL << 26;

bool builtin_integer(py::handle value, Int& output) {
    if (!PyLong_CheckExact(value.ptr())) return false;
    int overflow = 0;
    const auto integer = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred()) throw py::error_already_set();
    if (overflow || integer < -limit || integer > limit) return false;
    output = static_cast<Int>(integer);
    return true;
}

template<class T>
bool read_array(const py::array& array, std::vector<double>& output) {
    const auto* data = static_cast<const char*>(array.data());
    const auto stride = array.strides(0);
    for (std::size_t i = 0; i < output.size(); ++i) {
        T value;
        std::memcpy(&value, data + static_cast<std::ptrdiff_t>(i) * stride, sizeof(T));
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                if (value < -limit || value > limit) return false;
            } else if (value > static_cast<unsigned long long>(limit)) {
                return false;
            }
        }
        output[i] = static_cast<double>(value);
    }
    return true;
}

bool sequence(py::handle value) {
    return PyTuple_CheckExact(value.ptr()) || PyList_CheckExact(value.ptr());
}

py::handle sequence_item(py::handle value, Py_ssize_t i) {
    return PyTuple_CheckExact(value.ptr()) ? PyTuple_GET_ITEM(value.ptr(), i) : PyList_GET_ITEM(value.ptr(), i);
}

} // namespace

bool read_perimeter_vector(py::handle value, std::vector<double>& output, bool rectangle) {
    if (sequence(value)) {
        const auto n = PySequence_Size(value.ptr());
        if (rectangle ? n != 2 : n > 4096) return false;
        output.resize(static_cast<std::size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            const auto item = sequence_item(value, i);
            if (!rectangle && PyFloat_CheckExact(item.ptr())) {
                output[i] = PyFloat_AS_DOUBLE(item.ptr());
            } else {
                Int integer;
                if (!builtin_integer(item, integer)) return false;
                output[i] = static_cast<double>(integer);
            }
        }
        return true;
    }

    if (Py_TYPE(value.ptr()) != py::detail::npy_api::get().PyArray_Type_) return false;
    const auto array = py::reinterpret_borrow<py::array>(value);
    if (array.ndim() != 1 || !(array.flags() & py::array::c_style) || !array.writeable()
        || (rectangle ? array.size() != 2 : array.size() > 4096)) return false;
    const auto dtype = array.dtype();
    if (!dtype.attr("metadata").is_none() || !dtype.attr("isnative").cast<bool>()) return false;
    check_array_storage(array);
    output.resize(static_cast<std::size_t>(array.size()));
    const auto kind = dtype.kind();
    const auto bytes = dtype.itemsize();
    // Array extents use NumPy integer addition. Restrict to intp so every
    // accepted pair and sum represent the same values as the literal source.
    if (rectangle) return kind == 'i' && bytes == sizeof(Int) && read_array<Int>(array, output);
    if (kind == 'i') {
        if (bytes == 1) return read_array<std::int8_t>(array, output);
        if (bytes == 2) return read_array<std::int16_t>(array, output);
        if (bytes == 4) return read_array<std::int32_t>(array, output);
        if (bytes == 8) return read_array<std::int64_t>(array, output);
    } else if (kind == 'u') {
        if (bytes == 1) return read_array<std::uint8_t>(array, output);
        if (bytes == 2) return read_array<std::uint16_t>(array, output);
        if (bytes == 4) return read_array<std::uint32_t>(array, output);
        if (bytes == 8) return read_array<std::uint64_t>(array, output);
    } else if (kind == 'f') {
        if (bytes == 4 && dtype.char_() == 'f') return read_array<float>(array, output);
        if (bytes == 8 && dtype.char_() == 'd') return read_array<double>(array, output);
    }
    return false;
}

bool read_perimeter_shape(py::handle value, detail::PerimeterShape& output) {
    if (value.is_none()) return true;
    if (!sequence(value) || PySequence_Size(value.ptr()) < 2) return false;
    std::array<Int, 2> shape;
    if (!builtin_integer(sequence_item(value, 0), shape[0]) || !builtin_integer(sequence_item(value, 1), shape[1]))
        return false;
    output = shape;
    return true;
}

py::object perimeter_result(detail::PerimeterStatus status, const detail::PerimeterCoordinates& output) {
    if (status == detail::PerimeterStatus::outside_domain) return py::none();
    if (status == detail::PerimeterStatus::no_polygon) throw py::index_error("list index out of range");
    const auto count = static_cast<py::ssize_t>(output.rows.size());
    const py::ssize_t stride = count ? sizeof(Int) : 0;
    const std::array<py::ssize_t, 1> shape{count}, strides{stride};
    py::array rows(py::dtype::of<Int>(), shape, strides);
    py::array columns(py::dtype::of<Int>(), shape, strides);
    if (count) {
        std::memcpy(rows.mutable_data(), output.rows.data(), count * sizeof(Int));
        std::memcpy(columns.mutable_data(), output.columns.data(), count * sizeof(Int));
    }
    return py::make_tuple(rows, columns);
}

} // namespace skimagexx::bindings

void bind_polygon_perimeter(py::module_& module) {
    using namespace skimagexx;
    module.def("polygon_perimeter", [](py::handle r, py::handle c, py::handle dimensions, bool clip) -> py::object {
        if (clip && dimensions.is_none()) throw py::value_error("Must specify clipping shape");
        detail::PerimeterShape shape;
        std::vector<double> rows, columns;
        if (!bindings::read_perimeter_shape(dimensions, shape) ||
            !bindings::read_perimeter_vector(r, rows, false) || !bindings::read_perimeter_vector(c, columns, false))
            return py::none();
        if (!clip && (rows.empty() || columns.empty()))
            throw py::value_error("zero-size array to reduction operation minimum which has no identity");
        if (rows.size() != columns.size()) return py::none();
        std::vector<detail::PerimeterVertex> vertices;
        vertices.reserve(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) vertices.push_back({rows[i], columns[i]});
        detail::PerimeterCoordinates output;
        detail::PerimeterStatus status;
        {
            py::gil_scoped_release release;
            status = detail::polygon_perimeter(vertices, shape, clip, output);
        }
        return bindings::perimeter_result(status, output);
    }, py::arg("r"), py::arg("c"), py::arg("shape"), py::arg("clip").noconvert());
}
