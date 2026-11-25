#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <pybind11/numpy.h>

namespace skimagexx::bindings {

inline constexpr int numpy_max_rank = 64;

inline bool is_aligned(const pybind11::array& array) {
    return (array.flags() & pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_) != 0;
}

inline bool is_native_byteorder(const pybind11::dtype& dtype) {
    // Structured and subarray dtypes can contain swapped fields even when
    // the outer dtype has no byte order of its own.
    if (dtype.num() == pybind11::detail::npy_api::NPY_VOID_)
        return dtype.attr("isnative").cast<bool>();
    const char order = dtype.byteorder();
    const std::uint16_t one = 1;
    const bool little_endian = *reinterpret_cast<const unsigned char*>(&one) != 0;
    return order == '=' || order == '|' || order == (little_endian ? '<' : '>');
}

inline bool is_native_byteorder(const pybind11::array& array) {
    return is_native_byteorder(array.dtype());
}

template <typename Array>
Array copy_array(const Array& array) {
    namespace py = pybind11;
    constexpr int c_order = 0;
    auto* result = py::detail::npy_api::get().PyArray_NewCopy_(array.ptr(), c_order);
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<Array>(result);
}

namespace {

class optional_gil_release {
public:
    explicit optional_gil_release(bool release) {
        if (release) guard_.emplace();
    }

    optional_gil_release(const optional_gil_release&) = delete;
    optional_gil_release& operator=(const optional_gil_release&) = delete;

private:
    std::optional<pybind11::gil_scoped_release> guard_;
};

} // namespace

inline pybind11::array empty_array(const pybind11::dtype& dtype,
                                  const pybind11::ssize_t* shape,
                                  const pybind11::ssize_t* strides, int ndim) {
    namespace py = pybind11;
    Py_intptr_t dimensions[numpy_max_rank];
    const auto* numpy_shape = reinterpret_cast<const Py_intptr_t*>(shape);
    if (ndim > 0 && ndim <= numpy_max_rank) {
        // NumPy allocation hooks may call Python. Take a local snapshot
        // before allocation so callers can safely revalidate their inputs.
        for (int axis = 0; axis < ndim; ++axis)
            dimensions[axis] = shape[axis];
        numpy_shape = dimensions;
    }
    auto& api = py::detail::npy_api::get();
    auto* result = api.PyArray_NewFromDescr_(
        api.PyArray_Type_, dtype.inc_ref().ptr(), ndim, numpy_shape,
        reinterpret_cast<const Py_intptr_t*>(strides), nullptr, 0, nullptr);
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<py::array>(result);
}

// Match py::array's explicit C strides, including its handling of empty axes.
// A caller matching np.empty instead can request zero strides on empty arrays.
inline pybind11::array empty_c_array(const pybind11::dtype& dtype,
                                    const pybind11::ssize_t* shape, int ndim,
                                    bool zero_empty_strides = false) {
    namespace py = pybind11;
    // NumPy's maximum rank is 32 on 1.x and 64 on 2.x. Let NumPy
    // reject a larger rank without constructing any strides.
    if (zero_empty_strides || ndim <= 0 || ndim > numpy_max_rank)
        return empty_array(dtype, shape, nullptr, ndim);

    py::ssize_t strides[numpy_max_rank];
    strides[ndim - 1] = dtype.itemsize();
    for (int axis = ndim - 1; axis > 0; --axis) {
        const auto extent = shape[axis];
        if (extent < 0 || (extent != 0 &&
            strides[axis] > std::numeric_limits<py::ssize_t>::max() / extent))
            return empty_array(dtype, shape, nullptr, ndim);
        strides[axis - 1] = strides[axis] * extent;
    }
    return empty_array(dtype, shape, strides, ndim);
}

template <typename T>
pybind11::array_t<T> empty_c_array(const pybind11::ssize_t* shape, int ndim,
                                  bool zero_empty_strides = false) {
    auto result = empty_c_array(pybind11::dtype::of<T>(), shape, ndim, zero_empty_strides);
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
