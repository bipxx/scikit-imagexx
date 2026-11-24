#include "skimagexx/isotropic_morphology.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <vector>

namespace py = pybind11;

namespace {

bool signaling_nan(double value) {
    std::uint64_t word;
    std::memcpy(&word, &value, sizeof(word));
    constexpr std::uint64_t exponent = UINT64_C(0x7ff0000000000000);
    constexpr std::uint64_t fraction = UINT64_C(0x000fffffffffffff);
    constexpr std::uint64_t quiet = UINT64_C(0x0008000000000000);
    return (word & exponent) == exponent && (word & fraction) && !(word & quiet);
}

bool contiguous_bool(const py::array& array) {
    return array.dtype().equal(py::dtype::of<bool>()) &&
           (array.flags() & py::array::c_style) &&
           array.attr("flags").attr("aligned").cast<bool>();
}

}  // namespace

void bind_isotropic_morphology(py::module_& module) {
    module.def("isotropic_morphology_mask", [](const py::array& image, double radius,
               const std::vector<double>& spacing, int operation, const py::object& out) -> py::object {
        if (image.ndim() < 1 || image.ndim() > 3 || !contiguous_bool(image) ||
            spacing.size() != static_cast<std::size_t>(image.ndim()))
            throw py::value_error("isotropic morphology requires a C-order mask and matching spacing");
        if (signaling_nan(radius) || std::fegetround() != FE_TONEAREST)
            return py::none();

        skimagexx::bindings::check_array_storage(image);
        const auto rank = static_cast<std::size_t>(image.ndim());
        void* destination_data = nullptr;
        if (!out.is_none()) {
            if (!py::isinstance<py::array>(out))
                throw py::type_error("isotropic morphology out must be an ndarray");
            auto destination = py::reinterpret_borrow<py::array>(out);
            if (!contiguous_bool(destination) || !destination.writeable() ||
                destination.ndim() != image.ndim() ||
                !std::equal(image.shape(), image.shape() + rank, destination.shape()))
                throw py::value_error("isotropic morphology out must match the C-order mask");
            skimagexx::bindings::check_array_storage(destination);
            destination_data = destination.mutable_data();
        }

        std::array<std::size_t, 3> shape{};
        for (std::size_t axis = 0; axis < rank; ++axis)
            shape[axis] = static_cast<std::size_t>(image.shape(axis));
        py::array result(image.dtype(), std::vector<py::ssize_t>(image.shape(), image.shape() + rank));
        const auto count = static_cast<std::size_t>(image.size());
        skimagexx::isotropic_morphology_into(
            static_cast<const std::uint8_t*>(image.data()), count, rank, shape.data(),
            spacing.data(), radius, static_cast<skimagexx::IsotropicOperation>(operation),
            static_cast<std::uint8_t*>(result.mutable_data()), count);
        if (out.is_none())
            return result;
        std::memcpy(destination_data, result.data(), count);
        return out;
    }, py::arg("image").noconvert(), py::arg("radius"), py::arg("spacing"),
       py::arg("operation"), py::arg("out") = py::none());
}
