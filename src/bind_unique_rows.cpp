#include "unique_rows.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>

namespace py = pybind11;

void bind_unique_rows(py::module_& module) {
    module.def("unique_rows_apply", [](const py::array& array, int workers) -> py::object {
        if (workers < 1 || workers > 1024)
            throw py::value_error("unique-rows worker snapshot outside [1,1024]");
        if (array.ndim() != 2 || !(array.flags() & py::array::c_style)
                || array.dtype().attr("hasobject").cast<bool>()) return py::none();
        const auto columns = std::size_t(array.shape(1));
        const auto bytes = std::size_t(array.itemsize());
        const auto rows = std::size_t(array.shape(0));
        if (!bytes || !columns || columns > std::size_t(std::numeric_limits<py::ssize_t>::max()) / bytes)
            return py::none();
        const auto width = columns * bytes;
        if (rows > std::size_t(std::numeric_limits<py::ssize_t>::max()) / width)
            return py::none();
        skimagexx::bindings::check_array_storage(array);
        auto output = py::array_t<std::int64_t>(py::ssize_t(rows));
        auto* destination = output.mutable_data();
        const skimagexx::detail::UniqueRowsInput input{
            static_cast<const unsigned char*>(array.data()), rows, width};
        std::size_t count;
        {
            py::gil_scoped_release release;
            count = skimagexx::detail::unique_rows_indices(input, destination, workers);
        }
        output.resize({py::ssize_t(count)}, false);
        return output;
    }, py::arg("array").noconvert(), py::arg("workers"));
}
