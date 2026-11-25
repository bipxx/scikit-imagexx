#include "skimagexx/thin_plate_spline.hpp"
#include "bind_array_storage.hpp"

#include <algorithm>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using Coordinates = py::array_t<double, py::array::c_style>;

bool valid(const Coordinates& value) {
    if (value.ndim() != 2 || value.shape(1) != 2 ||
        reinterpret_cast<std::uintptr_t>(value.data()) % alignof(double)) return false;
    try {
        if (!value.owndata()) skimagexx::bindings::check_array_storage(value);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

} // namespace

void bind_thin_plate_spline(py::module_& module) {
    module.def("thin_plate_spline_fit", [](const Coordinates& src, const Coordinates& dst) -> py::object {
        if (!valid(src) || !valid(dst) || src.shape(0) != dst.shape(0)) return py::none();
        std::vector<double> weights;
        if (!skimagexx::thin_plate_spline_fit(src.data(), dst.data(),
                                             static_cast<std::size_t>(src.shape(0)), weights))
            return py::none();
        py::array_t<double> output({src.shape(0) + 3, py::ssize_t{2}});
        std::copy(weights.begin(), weights.end(), output.mutable_data());
        return output;
    }, py::arg("src").noconvert(), py::arg("dst").noconvert());
    module.def("thin_plate_spline_apply", [](const Coordinates& src, const Coordinates& weights,
                                            const Coordinates& coords) -> py::object {
        if (!valid(src) || !valid(weights) || !valid(coords) || weights.shape(0) != src.shape(0) + 3)
            return py::none();
        py::array_t<double> output({coords.shape(0), py::ssize_t{2}});
        bool supported;
        {
            py::gil_scoped_release release;
            supported = skimagexx::thin_plate_spline_apply(src.data(), weights.data(),
                static_cast<std::size_t>(src.shape(0)), coords.data(),
                static_cast<std::size_t>(coords.shape(0)), output.mutable_data());
        }
        return supported ? py::object(output) : py::none();
    }, py::arg("src").noconvert(), py::arg("weights").noconvert(), py::arg("coords").noconvert());
}
