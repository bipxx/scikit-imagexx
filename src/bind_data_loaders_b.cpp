#include "skimagexx/data_loaders_b.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
py::array materialize(const py::bytes& pixels, const py::dtype& dtype,
                      const std::vector<py::ssize_t>& shape) {
    const auto item_size = dtype.itemsize();
    const auto kind = dtype.kind();
    if (!((kind == 'u' && item_size == 1) || (kind == 'f' && item_size == 4))) {
        throw py::type_error("sample pixels require uint8 or float32");
    }
    if (shape.empty() || shape.size() > 3) {
        throw py::value_error("sample shape must have one to three dimensions");
    }
    // The private decoder publishes immutable bytes, so the source allocation
    // stays bounded and stable throughout the copy.
    const auto size = PyBytes_GET_SIZE(pixels.ptr());
    auto expected = item_size;
    for (const auto extent : shape) {
        if (extent <= 0 || extent > size / expected) {
            throw py::value_error("sample buffer does not match shape");
        }
        expected *= extent;
    }
    if (expected != size) {
        throw py::value_error("sample buffer does not match shape");
    }
    py::array result(dtype, shape);
    const auto rows = static_cast<std::size_t>(shape.front());
    const auto count = static_cast<std::size_t>(size);
    skimagexx::copy_data_rows(
        reinterpret_cast<const std::uint8_t*>(PyBytes_AS_STRING(pixels.ptr())), count,
        static_cast<std::uint8_t*>(result.mutable_data()), count, rows, count / rows);
    return result;
}
} // namespace

void bind_data_loaders_b(py::module_& module) {
    module.def("data_loaders_b_materialize", &materialize,
               py::arg("pixels").noconvert(), py::arg("dtype").noconvert(),
               py::arg("shape"));
}

#ifdef SKIMAGEXX_DATA_LOADERS_B_STANDALONE
PYBIND11_MODULE(_data_loaders_b_native, module) {
    bind_data_loaders_b(module);
}
#endif
