#pragma once

#include <pybind11/numpy.h>

namespace skimagexx::bindings {

// Match py::array's explicit C strides, including its handling of empty axes.
// The fixed-rank callers can pass their shape directly without ShapeContainer.
inline pybind11::array empty_c_array(const pybind11::dtype& dtype,
                                    const pybind11::ssize_t* shape, int ndim) {
    namespace py = pybind11;
    if (ndim > 4) return py::array(dtype, py::array::ShapeContainer(shape, shape + ndim));

    Py_intptr_t dimensions[4];
    Py_intptr_t strides[4];
    for (int axis = 0; axis < ndim; ++axis)
        dimensions[axis] = shape[axis];
    if (ndim > 0) {
        strides[ndim - 1] = dtype.itemsize();
        for (int axis = ndim - 1; axis > 0; --axis)
            strides[axis - 1] = strides[axis] * dimensions[axis];
    }
    auto& api = py::detail::npy_api::get();
    auto* result = api.PyArray_NewFromDescr_(
        api.PyArray_Type_, dtype.inc_ref().ptr(), ndim,
        ndim ? dimensions : nullptr, ndim ? strides : nullptr,
        nullptr, 0, nullptr);
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<py::array>(result);
}

template <typename T>
pybind11::array_t<T> empty_c_array(const pybind11::ssize_t* shape, int ndim) {
    auto result = empty_c_array(pybind11::dtype::of<T>(), shape, ndim);
    return pybind11::reinterpret_steal<pybind11::array_t<T>>(result.release());
}

template <typename T, std::size_t Rank>
pybind11::array_t<T> empty_c_array(const pybind11::ssize_t (&shape)[Rank]) {
    return empty_c_array<T>(shape, static_cast<int>(Rank));
}

// Wrap a strided view of an existing ndarray, keeping its dtype and retaining
// the base with the same flags as py::array's owner-backed constructor.
template <std::size_t Rank>
pybind11::array array_view(const pybind11::array& owner,
                         const pybind11::ssize_t (&shape)[Rank],
                         const pybind11::ssize_t (&strides)[Rank], const void* data) {
    namespace py = pybind11;
    auto& api = py::detail::npy_api::get();
    auto dtype = owner.dtype();
    auto result = py::reinterpret_steal<py::array>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_, dtype.release().ptr(), static_cast<int>(Rank),
        reinterpret_cast<const Py_intptr_t*>(shape),
        reinterpret_cast<const Py_intptr_t*>(strides), const_cast<void*>(data),
        owner.flags() & ~py::detail::npy_api::NPY_ARRAY_OWNDATA_, nullptr));
    if (!result) throw py::error_already_set();
    if (api.PyArray_SetBaseObject_(result.ptr(), owner.inc_ref().ptr()) < 0)
        throw py::error_already_set();
    return result;
}

} // namespace skimagexx::bindings
