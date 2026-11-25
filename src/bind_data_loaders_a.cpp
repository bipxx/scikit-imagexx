#include "skimagexx/data_loaders_a.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

void bind_data_loaders_a(py::module_& module) {
    module.def("data_loaders_a_copy",
        [](const py::bytes& payload, const py::dtype& dtype,
           const std::vector<py::ssize_t>& shape) {
            char* source = nullptr;
            py::ssize_t bytes = 0;
            if (PyBytes_AsStringAndSize(payload.ptr(), &source, &bytes) != 0) {
                throw py::error_already_set();
            }
            const auto itemsize = dtype.itemsize();
            if (itemsize <= 0 || bytes == 0 || bytes % itemsize != 0 ||
                dtype.attr("hasobject").cast<bool>()) {
                throw py::value_error("expected a nonempty fixed-width sample buffer");
            }
            const auto items = bytes / itemsize;
            py::ssize_t count = 1;
            for (const auto dimension : shape) {
                if (dimension <= 0 || dimension > items / count) {
                    throw py::value_error("sample shape does not match its buffer");
                }
                count *= dimension;
            }
            if (count != items) {
                throw py::value_error("sample shape does not match its buffer");
            }
            // Python bytes are immutable and the destination has no external owner.
            auto result = py::array(dtype, shape);
            auto* destination = result.mutable_data();
            {
                py::gil_scoped_release release;
                const auto size = static_cast<std::size_t>(bytes);
                skimagexx::copy_sample_data_into(source, size, destination, size);
            }
            return result;
        },
        py::arg("payload").noconvert(), py::arg("dtype").noconvert(), py::arg("shape"));
}

#ifdef SKIMAGEXX_DATA_LOADERS_A_STANDALONE
PYBIND11_MODULE(_core, module) {
    bind_data_loaders_a(module);
}
#endif
