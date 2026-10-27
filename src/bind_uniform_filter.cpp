#include "uniform_filter.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style>;

template<class T>
py::object apply(const py::array& image, std::size_t axis,
                 std::size_t window, int workers, skimagexx::PixelType type) {
    const std::vector<py::ssize_t> dimensions(image.shape(), image.shape()+image.ndim());
    py::array_t<T> output(dimensions);
    // This is the actual prepared-buffer boundary after the final allocation.
    skimagexx::bindings::check_array_storage(image);
    if (!Array<T>::check_(image) || image.ndim()!=py::ssize_t(dimensions.size()) ||
        !std::equal(dimensions.begin(), dimensions.end(), image.shape()) ||
        reinterpret_cast<std::uintptr_t>(image.data()) % alignof(T))
        throw py::value_error("uniform input changed or is unaligned at allocation");
    const std::vector<std::size_t> shape(dimensions.begin(), dimensions.end());
    const auto* input = image.data();
    auto* destination = output.mutable_data();
    bool accepted;
    {
        py::gil_scoped_release release;
        accepted = skimagexx::detail::uniform_filter1d_into(
            input, destination, type, shape, axis, window, workers);
    }
    if (!accepted) return py::none();
    return output;
}
}

void bind_uniform_filter(py::module_& module) {
    module.def("uniform_filter1d", [](const py::array& image, std::size_t axis,
               std::size_t window, int workers) -> py::object {
        if (image.ndim()<1 || image.ndim()>3 || axis>=std::size_t(image.ndim()) ||
            !window || std::any_of(image.shape(), image.shape()+image.ndim(),
                                  [](py::ssize_t n){return n<=0;}))
            throw py::value_error("uniform filtering requires positive rank1..3 shape, axis and window");
        if (Array<float>::check_(image))
            return apply<float>(image,axis,window,workers,skimagexx::PixelType::float32);
        if (Array<double>::check_(image))
            return apply<double>(image,axis,window,workers,skimagexx::PixelType::float64);
        throw py::type_error("uniform filtering requires prepared native contiguous float32/64");
    }, py::arg("image").noconvert(), py::arg("axis"), py::arg("window"), py::arg("workers"));
}
