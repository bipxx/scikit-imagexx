#include "skimagexx/remove_objects_by_distance.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>

namespace py = pybind11;

namespace {

using Norm = skimagexx::ObjectDistanceNorm;

bool ordinary_array(const py::array& array) {
    const auto dtype = array.dtype();
    return (array.flags() & py::array::c_style) && array.writeable() &&
           array.attr("flags").attr("aligned").cast<bool>() &&
           dtype.attr("isnative").cast<bool>() && dtype.attr("metadata").is_none();
}

template<class T>
py::object filter(const py::array& image, const std::array<std::size_t, 3>& shape,
                  const std::vector<double>& spacing, double radius, Norm norm,
                  const py::array* priority) {
    const auto rank = static_cast<std::size_t>(image.ndim());
    const auto count = static_cast<std::size_t>(image.size());
    const auto priority_count = priority ? static_cast<std::size_t>(priority->size()) : 0;
    py::array output(image.dtype(), std::vector<py::ssize_t>(image.shape(), image.shape() + rank));
    const auto* labels = static_cast<const T*>(image.data());
    const auto* scores = priority ? static_cast<const double*>(priority->data()) : nullptr;
    auto* result = static_cast<T*>(output.mutable_data());
    bool supported;
    {
        py::gil_scoped_release release;
        supported = skimagexx::remove_objects_by_distance_into(
            labels, count, rank, shape.data(), spacing.data(), radius, norm,
            scores, priority_count, result, count);
    }
    if (supported)
        return output;
    return py::none();
}

}  // namespace

void bind_remove_objects_by_distance(py::module_& module) {
    module.def("remove_objects_by_distance_integer",
        [](py::handle labels, double radius, double p_norm, const std::vector<double>& spacing,
           py::handle scores) -> py::object {
            const auto* ndarray = py::detail::npy_api::get().PyArray_Type_;
            if (Py_TYPE(labels.ptr()) != ndarray ||
                !(0 <= radius && radius <= 1048576) || std::fegetround() != FE_TONEAREST)
                return py::none();
            Norm norm;
            if (p_norm == 1)
                norm = Norm::manhattan;
            else if (p_norm == 2)
                norm = Norm::euclidean;
            else if (std::isinf(p_norm) && p_norm > 0)
                norm = Norm::chebyshev;
            else
                return py::none();

            const auto image = py::reinterpret_borrow<py::array>(labels);
            if (image.ndim() < 1 || image.ndim() > 3 ||
                spacing.size() != static_cast<std::size_t>(image.ndim()) ||
                !ordinary_array(image))
                return py::none();
            std::array<std::size_t, 3> shape{};
            std::size_t count = 1;
            for (std::size_t axis = 0; axis < spacing.size(); ++axis) {
                const double factor = spacing[axis];
                const auto extent = image.shape(axis);
                if (!(1.0 / 16 <= factor && factor <= 16) || std::trunc(factor * 16) != factor * 16 ||
                    extent <= 0 || extent > static_cast<py::ssize_t>(skimagexx::object_distance_max_extent))
                    return py::none();
                shape[axis] = static_cast<std::size_t>(extent);
                if (count > skimagexx::object_distance_max_elements / shape[axis])
                    return py::none();
                count *= shape[axis];
            }

            py::array priority;
            const py::array* priority_ptr = nullptr;
            if (!scores.is_none()) {
                if (Py_TYPE(scores.ptr()) != ndarray)
                    return py::none();
                priority = py::reinterpret_borrow<py::array>(scores);
                if (priority.ndim() != 1 || priority.size() <= 0 ||
                    priority.dtype().char_() != 'd' ||
                    !priority.dtype().equal(py::dtype::of<double>()) || !ordinary_array(priority))
                    return py::none();
                priority_ptr = &priority;
            }
            if (!image.owndata())
                skimagexx::bindings::check_array_storage(image);
            if (priority_ptr && !priority.owndata())
                skimagexx::bindings::check_array_storage(priority);

            const auto dtype = image.dtype();
#define SKIMAGEXX_DISTANCE_DTYPE(T)                                                  \
            if (dtype.equal(py::dtype::of<T>()))                                     \
                return filter<T>(image, shape, spacing, radius, norm, priority_ptr)
            SKIMAGEXX_DISTANCE_DTYPE(std::int8_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::uint8_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::int16_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::uint16_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::int32_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::uint32_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::int64_t);
            SKIMAGEXX_DISTANCE_DTYPE(std::uint64_t);
#undef SKIMAGEXX_DISTANCE_DTYPE
            return py::none();
        }, py::arg("label_image"), py::arg("min_distance").noconvert(),
        py::arg("p_norm").noconvert(), py::arg("spacing"), py::arg("priority"));
}
