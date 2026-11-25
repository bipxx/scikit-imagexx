#include <skimagexx/data_shepp_logan_phantom.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;

void bind_data_shepp_logan_phantom(py::module_& module) {
    module.def("data_shepp_logan_phantom", []() {
        py::array_t<double> result(
            {static_cast<py::ssize_t>(skimagexx::data_shepp_logan_phantom_rows),
             static_cast<py::ssize_t>(skimagexx::data_shepp_logan_phantom_columns)});
        auto* output = result.mutable_data();
        py::gil_scoped_release release;
        if (!skimagexx::data_shepp_logan_phantom_f64_into(output, skimagexx::data_shepp_logan_phantom_elements)) {
            throw std::runtime_error("data_shepp_logan_phantom failed");
        }
        return result;
    });
}
