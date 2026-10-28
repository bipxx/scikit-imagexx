#include "label_points.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

void bind_label_points(py::module_& module) {
    const auto apply = [](const py::tuple& indices, const py::tuple& shape,
                          int workers, const py::object& prepared) -> py::object {
        if (workers < 1 || workers > 1024)
            throw py::value_error("label-points worker snapshot outside [1,1024]");
        if (shape.size() == 0 || shape.size() > 32 || indices.size() != shape.size()) return py::none();
        std::vector<py::ssize_t> sizes;
        std::vector<std::size_t> dimensions;
        std::size_t count = 1;
        for (const auto value : shape) {
            if (!PyLong_CheckExact(value.ptr())) return py::none();
            int overflow = 0;
            const auto size = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
            if (overflow || size < 0) return py::none();
            sizes.push_back(py::ssize_t(size));
            dimensions.push_back(std::size_t(size));
        }
        for (const auto size : dimensions) if (!size) { count = 0; break; }
        if (count) for (const auto size : dimensions) {
            if (size > std::size_t(std::numeric_limits<py::ssize_t>::max()) / sizeof(std::uint64_t) / count)
                return py::none();
            count *= size;
        }
        if (!prepared.is_none()) {
            const auto output = py::reinterpret_borrow<py::array>(prepared);
            if (!(output.flags() & py::array::c_style) || !output.writeable()
                    || output.dtype().kind() != 'u' || output.itemsize() != 8
                    || !output.dtype().attr("isnative").cast<bool>()
                    || reinterpret_cast<std::uintptr_t>(output.data()) % alignof(std::uint64_t))
                return py::none();
            skimagexx::bindings::check_array_storage(output);
        }
        std::vector<py::array> owners;
        std::vector<const std::int64_t*> axes;
        std::vector<std::ptrdiff_t> strides;
        std::size_t points = 0;
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            if (!py::isinstance<py::array>(indices[axis])) return py::none();
            auto array = py::reinterpret_borrow<py::array>(indices[axis]);
            if (array.ndim() != 1 || array.dtype().kind() != 'i'
                    || array.itemsize() != 8 || !array.dtype().attr("isnative").cast<bool>())
                return py::none();
            if (!axis) points = std::size_t(array.size());
            else if (std::size_t(array.size()) != points) return py::none();
            skimagexx::bindings::check_array_storage(array);
            axes.push_back(static_cast<const std::int64_t*>(array.data()));
            strides.push_back(array.strides(0));
            owners.push_back(std::move(array));
        }
        // Public zeros output is fresh; the tuple entry allocates privately.
        auto output = !prepared.is_none()
            ? py::reinterpret_borrow<py::array_t<std::uint64_t>>(prepared)
            : count ? py::array_t<std::uint64_t>(sizes)
            : py::array_t<std::uint64_t>(sizes, std::vector<py::ssize_t>(sizes.size(), 0));
        const skimagexx::detail::LabelPointsInput input{
            axes.data(), strides.data(), dimensions.data(), dimensions.size(), points, count};
        auto* destination = output.mutable_data();
        bool accepted;
        {
            py::gil_scoped_release release;
            accepted = prepared.is_none()
                ? skimagexx::detail::label_points_into(input, destination, workers)
                : skimagexx::detail::label_points_assign(input, destination, workers);
        }
        if (!accepted) return py::none();
        return output;
    };
    module.def("label_points_apply", [apply](const py::tuple& indices,
               const py::tuple& shape, int workers) {
        return apply(indices, shape, workers, py::none());
    }, py::arg("indices").noconvert(), py::arg("shape").noconvert(), py::arg("workers"));
    module.def("label_points_apply", [apply](const py::tuple& indices,
               const py::array& output, int workers) {
        py::tuple shape(output.ndim());
        for (py::ssize_t axis = 0; axis < output.ndim(); ++axis)
            shape[axis] = output.shape(axis);
        return apply(indices, shape, workers, output);
    }, py::arg("indices").noconvert(), py::arg("output").noconvert(), py::arg("workers"));
}
