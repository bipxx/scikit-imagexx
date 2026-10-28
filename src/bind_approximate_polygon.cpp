#include <skimagexx/approximate_polygon.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>

namespace py = pybind11;
namespace {
using Coordinates = py::array_t<double, py::array::c_style>;
using Mask = py::array_t<bool, py::array::c_style>;

skimagexx::detail::ApproximatePolygonTrig scalar_trig(void* context, double dr, double dc) {
    py::gil_scoped_acquire acquire;
    const auto values = (*static_cast<const py::function*>(context))(dr, dc).cast<py::tuple>();
    if (values.size() != 2) throw py::value_error("polygon trig provider requires two scalars");
    return {values[0].cast<double>(), values[1].cast<double>()};
}
}

void bind_approximate_polygon(py::module_& module) {
    module.def("approximate_polygon_apply", [](const py::array& coordinates, double tolerance,
               py::array keep, py::array distances, int workers, py::function trig) {
        if (workers < 1 || workers > 1024)
            throw py::value_error("polygon worker snapshot outside [1,1024]");
        if (!Coordinates::check_(coordinates) || coordinates.ndim() != 2
                || coordinates.shape(1) != 2 || coordinates.shape(0) < 1
                || reinterpret_cast<std::uintptr_t>(coordinates.data()) % alignof(double)
                || !Mask::check_(keep) || keep.ndim() != 1
                || keep.size() != coordinates.shape(0) || !keep.writeable() || !keep.owndata()
                || !Coordinates::check_(distances) || distances.ndim() != 1
                || distances.size() != coordinates.shape(0) || !distances.writeable()
                || !distances.owndata()
                || reinterpret_cast<std::uintptr_t>(distances.data()) % alignof(double)) return false;
        skimagexx::bindings::check_array_storage(coordinates);
        skimagexx::bindings::check_array_storage(keep);
        skimagexx::bindings::check_array_storage(distances);
        const auto overlaps = [](const py::array& a, const py::array& b) {
            const auto first = reinterpret_cast<std::uintptr_t>(a.data());
            const auto second = reinterpret_cast<std::uintptr_t>(b.data());
            return first < second + b.nbytes() && second < first + a.nbytes();
        };
        if (overlaps(coordinates, keep) || overlaps(coordinates, distances)
                || overlaps(keep, distances)) return false;
        const skimagexx::detail::ApproximatePolygonInput input{
            static_cast<const double*>(coordinates.data()), std::size_t(coordinates.shape(0)),
            coordinates.strides(0) / py::ssize_t(sizeof(double)),
            coordinates.strides(1) / py::ssize_t(sizeof(double))};
        auto* mask = static_cast<std::uint8_t*>(keep.mutable_data());
        auto* scratch = static_cast<double*>(distances.mutable_data());
        py::gil_scoped_release release;
        return skimagexx::detail::approximate_polygon_select(
            input, tolerance, mask, scratch, workers, scalar_trig,
            &trig);
    }, py::arg("coordinates").noconvert(), py::arg("tolerance"),
       py::arg("keep").noconvert(), py::arg("distances").noconvert(),
       py::arg("workers"), py::arg("trig"));
}
