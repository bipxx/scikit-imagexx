#include "skimagexx/io_suite.hpp"
#include "bind_array_storage.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace io = skimagexx::io;

namespace {

std::string_view view(py::bytes bytes) {
    return {PyBytes_AS_STRING(bytes.ptr()), static_cast<std::size_t>(PyBytes_GET_SIZE(bytes.ptr()))};
}

std::size_t stride_magnitude(py::ssize_t step) {
    return step < 0 ? std::size_t(-(step + 1)) + 1 : std::size_t(step);
}

bool bounded(const py::array& array, py::ssize_t item_size) {
    if (array.owndata() && (array.flags() & py::array::c_style)) return true;
    const auto base = array.base();
    if (base && py::isinstance<py::array>(base)) {
        const auto owner = py::reinterpret_borrow<py::array>(base);
        if (owner.owndata() && (owner.flags() & (py::array::c_style | py::array::f_style))) {
            // Ordinary NumPy slices have one dense owning parent. Keep its
            // byte interval check allocation-free and use the shared policy
            // for other owner chains and bytes-backed arrays.
            const auto start = reinterpret_cast<std::uintptr_t>(owner.data());
            const auto data = reinterpret_cast<std::uintptr_t>(array.data());
            const auto length = static_cast<std::size_t>(owner.nbytes());
            if (data < start || data - start > length) return false;
            auto low = static_cast<std::size_t>(data - start);
            if (array.flags() & py::array::c_style) {
                return static_cast<std::size_t>(array.size()) <= (length - low) / static_cast<std::size_t>(item_size);
            }
            if (static_cast<std::size_t>(item_size) > length - low) return false;
            auto high = length - low - static_cast<std::size_t>(item_size);
            for (auto axis = array.ndim(); axis-- > 0;) {
                const auto count = static_cast<std::size_t>(array.shape(axis) - 1);
                if (!count) continue;
                const auto step = array.strides(axis);
                const auto distance = stride_magnitude(step);
                auto& available = step < 0 ? low : high;
                if (distance > available / count) return false;
                available -= distance * count;
            }
            return true;
        }
    }
    try {
        skimagexx::bindings::check_array_storage(array);
        return true;
    } catch (const py::value_error&) {
        return false;
    }
}

bool row_major_axes(const py::array& array) {
    std::size_t previous = 0;
    for (auto axis = array.ndim(); axis-- > 0;) {
        if (array.shape(axis) <= 1) continue;
        const auto magnitude = stride_magnitude(array.strides(axis));
        if (!magnitude || magnitude < previous) return false;
        previous = magnitude;
    }
    return true;
}

py::object concatenate(const py::sequence& images, const py::type& array_type, bool expanded) {
    auto* sequence = images.ptr();
    const bool list = PyList_CheckExact(sequence);
    if (!list && !PyTuple_CheckExact(sequence)) return py::none();
    const auto count = list ? PyList_GET_SIZE(sequence) : PyTuple_GET_SIZE(sequence);
    if (!count) return py::none();
    std::vector<py::array> arrays;
    arrays.reserve(static_cast<std::size_t>(count));
    std::vector<py::ssize_t> shape{count};
    py::dtype dtype;
    py::ssize_t item_size = 0;
    for (py::ssize_t i = 0; i < count; ++i) {
        auto* item = list ? PyList_GET_ITEM(sequence, i) : PyTuple_GET_ITEM(sequence, i);
        if (reinterpret_cast<PyObject*>(Py_TYPE(item)) != array_type.ptr()) return py::none();
        auto array = py::reinterpret_borrow<py::array>(item);
        if (expanded && (array.ndim() == 0 || array.shape(0) != 1)) return py::none();
        const auto offset = expanded ? 1 : 0;
        const auto* dimensions = expanded ? array.shape() + 1 : array.shape();
        auto current = array.dtype();
        if (arrays.empty()) {
            dtype = current;
            if (!std::strchr("buifc", dtype.kind()) || !dtype.attr("isnative").cast<bool>()
                || !dtype.attr("metadata").is_none() || array.size() == 0) {
                return py::none();
            }
            if (array.ndim() > offset) {
                shape.insert(shape.end(), dimensions, array.shape() + array.ndim());
            }
            item_size = dtype.itemsize();
        }
        if (!current.is(dtype) || array.ndim() - offset + 1 != static_cast<py::ssize_t>(shape.size())
            || !std::equal(shape.begin() + 1, shape.end(), dimensions)
            || (!(array.flags() & py::array::c_style) && !row_major_axes(array))
            || !bounded(array, item_size)) {
            return py::none();
        }
        arrays.push_back(std::move(array));
    }
    const auto& first = arrays.front();
    const auto bytes = first.nbytes();
    if (count > PY_SSIZE_T_MAX / bytes) return py::none();
    py::array output(dtype, shape);
    auto target = static_cast<char*>(output.mutable_data());
    for (const auto& array : arrays) {
        if (array.flags() & py::array::c_style) {
            std::memcpy(target, array.data(), static_cast<std::size_t>(bytes));
        } else {
            io::copy_image(array.data(), target, array.shape(), array.strides(),
                           static_cast<std::size_t>(array.ndim()), static_cast<std::size_t>(item_size));
        }
        target += bytes;
    }
    return output;
}

PyObject* pop(PyObject* state, PyObject* const* args, Py_ssize_t count, PyObject* keywords) {
    if (count != 0 || (keywords && PyTuple_GET_SIZE(keywords))) {
        return PyObject_Vectorcall(PyTuple_GET_ITEM(state, 1), args,
                                   static_cast<std::size_t>(count), keywords);
    }
    return PyObject_CallNoArgs(PyTuple_GET_ITEM(state, 0));
}

PyObject* push(PyObject* state, PyObject* const* args, Py_ssize_t count, PyObject* keywords) {
    if (count != 1 || (keywords && PyTuple_GET_SIZE(keywords))) {
        return PyObject_Vectorcall(PyTuple_GET_ITEM(state, 2), args,
                                   static_cast<std::size_t>(count), keywords);
    }
    auto* array_type = reinterpret_cast<PyTypeObject*>(PyTuple_GET_ITEM(state, 1));
    if (!PyObject_TypeCheck(args[0], array_type)) {
        return PyObject_Vectorcall(PyTuple_GET_ITEM(state, 2), args, 1, nullptr);
    }
    if (PyList_Append(PyTuple_GET_ITEM(state, 0), args[0]) < 0) return nullptr;
    Py_RETURN_NONE;
}

using KeywordMethod = PyObject* (*)(PyObject*, PyObject* const*, Py_ssize_t, PyObject*);

PyCFunction keyword_method(KeywordMethod method) {
    // PyMethodDef stores handlers for several calling conventions as PyCFunction.
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(method));
}

}  // namespace

void bind_io_suite(py::module_& module) {
    module.def("io_stack_api", [](py::list stack, py::type array_type,
                                  py::object original_push, py::object original_pop) {
        static PyMethodDef pop_def = {"pop", keyword_method(pop), METH_FASTCALL | METH_KEYWORDS,
                                     "pop($self, /)\n--\n\nPop an image."};
        static PyMethodDef push_def = {"push", keyword_method(push), METH_FASTCALL | METH_KEYWORDS,
                                      "push($self, /, img)\n--\n\nPush an image."};
        auto push_state = py::make_tuple(stack, array_type, original_push);
        auto pop_state = py::make_tuple(stack.attr("pop"), original_pop);
        auto origin = py::str("skimagexx._io_suite");
        auto pop_func = py::reinterpret_steal<py::object>(PyCFunction_NewEx(&pop_def, pop_state.ptr(), origin.ptr()));
        if (!pop_func) throw py::error_already_set();
        auto push_func = py::reinterpret_steal<py::object>(PyCFunction_NewEx(&push_def, push_state.ptr(), origin.ptr()));
        if (!push_func) throw py::error_already_set();
        return py::make_tuple(std::move(pop_func), std::move(push_func));
    });
    module.def("io_concatenate", &concatenate);
    module.def("io_decode_netpbm", [](py::bytes bytes, std::size_t max_pixels) -> py::object {
        const auto image = io::netpbm_view(view(bytes));
        if (!image || image->height > max_pixels / image->width) return py::none();
        std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(image->height),
                                       static_cast<py::ssize_t>(image->width)};
        if (image->channels == 3) shape.push_back(3);
        py::array_t<unsigned char> output(shape);
        std::memcpy(output.mutable_data(), image->pixels.data(), image->pixels.size());
        return output;
    });
    module.def("io_netpbm_header", [](py::array array, bool rgb) -> py::object {
        if (array.itemsize() != 1 || array.dtype().kind() != 'u'
            || array.ndim() != (rgb ? 3 : 2) || (rgb && array.shape(2) != 3)
            || array.size() == 0 || !(array.flags() & py::array::c_style) || !bounded(array, 1)) {
            return py::none();
        }
        return py::bytes(io::netpbm_header(static_cast<std::size_t>(array.shape(1)),
                                          static_cast<std::size_t>(array.shape(0)), rgb ? 3 : 1));
    });
    module.def("io_read_features", [](py::bytes bytes, bool surf) -> py::object {
        const auto data = io::feature_layout(view(bytes), surf ? io::FeatureFormat::surf : io::FeatureFormat::sift);
        if (!data || data->descriptor_size > std::numeric_limits<int>::max() / sizeof(double)
            || data->count > std::size_t(PY_SSIZE_T_MAX) / sizeof(double) / data->values_per_record) {
            return py::none();
        }
        const auto size = data->count * data->values_per_record;
        py::array_t<double> values(static_cast<py::ssize_t>(size));
        auto* output = values.mutable_data();
        bool parsed;
        {
            py::gil_scoped_release release;
            parsed = io::read_feature_values(data->values, output, size);
        }
        if (!parsed) return py::none();
        return py::make_tuple(std::move(values), data->descriptor_size);
    });
}

#ifdef SKIMAGEXX_IO_SUITE_PRIVATE
PYBIND11_MODULE(_io_suite_native, module) {
    bind_io_suite(module);
}
#endif
