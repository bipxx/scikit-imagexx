#include <skimagexx/data_checkerboard.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;

void bind_data_checkerboard(py::module_& module) {
    module.def("data_checkerboard", []() {
        py::array_t<std::uint8_t> result(
            {static_cast<py::ssize_t>(skimagexx::data_checkerboard_rows),
             static_cast<py::ssize_t>(skimagexx::data_checkerboard_columns)});
        if (!skimagexx::data_checkerboard_u8_into(
                result.mutable_data(), skimagexx::data_checkerboard_elements)) {
            throw std::runtime_error("data_checkerboard failed");
        }
        return result;
    });
}
