#include "skimagexx/expand_labels.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

template<class T>
py::array expand(const py::array& labels, std::size_t rank, const std::size_t* shape,
                 const std::vector<double>& sampling, double distance) {
    py::array result(labels.dtype(), std::vector<py::ssize_t>(labels.shape(), labels.shape() + rank));
    skimagexx::expand_labels_into(static_cast<const T*>(labels.data()), std::size_t(labels.size()),
                                 rank, shape, sampling.data(), distance,
                                 static_cast<T*>(result.mutable_data()), std::size_t(result.size()));
    return result;
}

}  // namespace

void bind_expand_labels(py::module_& module) {
    module.def("expand_labels_integer", [](const py::array& image, double distance,
               const std::vector<double>& sampling) -> py::object {
        if (image.ndim() < 1 || image.ndim() > 3 ||
            sampling.size() != static_cast<std::size_t>(image.ndim()) ||
            !(image.flags() & py::array::c_style) || !image.attr("flags").attr("aligned").cast<bool>() ||
            !image.dtype().attr("isnative").cast<bool>())
            throw py::value_error("expand_labels requires aligned native C-order labels and matching spacing");
        if (signaling_nan(distance) || std::fegetround() != FE_TONEAREST)
            return py::none();

        skimagexx::bindings::check_array_storage(image);
        const auto rank = static_cast<std::size_t>(image.ndim());
        std::array<std::size_t, 3> shape{};
        for (std::size_t axis = 0; axis < rank; ++axis)
            shape[axis] = static_cast<std::size_t>(image.shape(axis));
        const auto dtype = image.dtype();
        if (dtype.equal(py::dtype::of<std::int8_t>()))
            return expand<std::int8_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::uint8_t>()))
            return expand<std::uint8_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::int16_t>()))
            return expand<std::int16_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::uint16_t>()))
            return expand<std::uint16_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::int32_t>()))
            return expand<std::int32_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::uint32_t>()))
            return expand<std::uint32_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::int64_t>()))
            return expand<std::int64_t>(image, rank, shape.data(), sampling, distance);
        if (dtype.equal(py::dtype::of<std::uint64_t>()))
            return expand<std::uint64_t>(image, rank, shape.data(), sampling, distance);
        throw py::type_error("expand_labels requires fixed-width integer labels");
    }, py::arg("label_image").noconvert(), py::arg("distance"), py::arg("spacing"));
}
