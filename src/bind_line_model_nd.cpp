#include "skimagexx/line_model_nd.hpp"

#include <array>
#include <cstdint>
#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace {

bool packed_own_float64(const py::array& data) {
    if (!data.owndata() || !(data.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(data.data()) % alignof(double) != 0)
        return false;
    const auto dtype = data.dtype();
    return dtype.is(py::dtype::of<double>()) ||
           (dtype.equal(py::dtype::of<double>()) && dtype.char_() == 'd' &&
            dtype.attr("metadata").is_none());
}

template<std::size_t dimensions, auto predict>
py::object bind_predict(py::handle x, py::handle origin, py::handle direction, std::size_t axis) {
    const auto ndarray = py::detail::npy_api::get().PyArray_Type_;
    if (Py_TYPE(x.ptr()) != ndarray || Py_TYPE(origin.ptr()) != ndarray ||
        Py_TYPE(direction.ptr()) != ndarray || axis >= dimensions)
        return py::none();
    const auto values = py::reinterpret_borrow<py::array>(x);
    const auto center = py::reinterpret_borrow<py::array>(origin);
    const auto vector = py::reinterpret_borrow<py::array>(direction);
    constexpr auto size = static_cast<py::ssize_t>(dimensions);
    if (values.ndim() != 1 || values.size() < 128 || values.size() > 65536 ||
        center.ndim() != 1 || center.size() != size || vector.ndim() != 1 ||
        vector.size() != size || !packed_own_float64(values) ||
        !packed_own_float64(center) || !packed_own_float64(vector))
        return py::none();

    py::array_t<double> output(py::array::ShapeContainer{values.size(), size});
    auto* result = output.mutable_data();
    bool done;
    {
        py::gil_scoped_release release;
        done = predict(static_cast<const double*>(values.data()), static_cast<std::size_t>(values.size()),
                       static_cast<const double*>(center.data()), static_cast<const double*>(vector.data()),
                       axis, result);
    }
    if (!done)
        return py::none();
    return output;
}

} // namespace

void bind_line_model_nd(py::module_& module) {
    module.def("line_model_nd_two_point", [](py::handle value) -> py::object {
        if (Py_TYPE(value.ptr()) != py::detail::npy_api::get().PyArray_Type_) return py::none();
        const auto data = py::reinterpret_borrow<py::array>(value);
        if (data.ndim() != 2 || data.shape(0) != 2 ||
            (data.shape(1) != 2 && data.shape(1) != 3) ||
            !packed_own_float64(data))
            return py::none();

        std::array<double, 3> center, delta;
        const auto axes = data.shape(1);
        bool done;
        {
            py::gil_scoped_release release;
            done = skimagexx::line_model_nd_two_point(static_cast<const double*>(data.data()),
                    static_cast<std::size_t>(axes), center.data(), delta.data());
        }
        if (!done) return py::none();
        py::array_t<double> origin(axes), direction(axes);
        const auto bytes = static_cast<std::size_t>(axes) * sizeof(double);
        std::memcpy(origin.mutable_data(), center.data(), bytes);
        std::memcpy(direction.mutable_data(), delta.data(), bytes);
        return py::make_tuple(origin, direction);
    });
    module.def("line_model_nd_predict_2d", &bind_predict<2, skimagexx::line_model_nd_predict_2d>,
               py::arg("x"), py::arg("origin"), py::arg("direction"), py::arg("axis").noconvert());
    module.def("line_model_nd_predict_3d", &bind_predict<3, skimagexx::line_model_nd_predict_3d>,
               py::arg("x"), py::arg("origin"), py::arg("direction"), py::arg("axis").noconvert());
}
