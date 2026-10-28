#include "compare_images.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style>;

skimagexx::PixelType input_type(const py::array& image) {
    if (Array<float>::check_(image)) return skimagexx::PixelType::float32;
    if (Array<double>::check_(image)) return skimagexx::PixelType::float64;
    throw py::type_error("comparison requires prepared native contiguous float32/64 inputs");
}
}

void bind_compare_images(py::module_& module) {
    module.def("compare_images", [](const py::array& first, const py::array& second,
               int method, const std::array<std::size_t, 6>& grid, int workers) -> py::object {
        using namespace skimagexx;
        if (method<0 || method>2) throw py::value_error("unknown comparison method");
        const auto first_type=input_type(first), second_type=input_type(second);
        const std::vector<py::ssize_t> shape(first.shape(), first.shape()+first.ndim());
        const auto count=std::size_t(first.size());
        if (second.ndim()!=first.ndim() || !std::equal(shape.begin(),shape.end(),second.shape()))
            throw py::value_error("comparison input shapes differ");
        if (method==2 && (shape.size()!=2 || grid[0]!=std::size_t(shape[0]) || grid[1]!=std::size_t(shape[1]) ||
            (grid[4] && grid[2]>grid[0]/grid[4]) || (grid[5] && grid[3]>grid[1]/grid[5])))
            throw py::value_error("comparison checker grid does not fit the input");
        const auto output_type=method==2?first_type:
            (first_type==PixelType::float64 || second_type==PixelType::float64?PixelType::float64:PixelType::float32);
        py::array output(output_type==PixelType::float32?py::dtype::of<float>():py::dtype::of<double>(),shape);
        // Validate the actual two prepared buffers once, after the last allocation.
        bindings::check_array_storage(first);
        bindings::check_array_storage(second);
        if (input_type(first)!=first_type || input_type(second)!=second_type ||
            first.ndim()!=py::ssize_t(shape.size()) || second.ndim()!=py::ssize_t(shape.size()) ||
            !std::equal(shape.begin(),shape.end(),first.shape()) || !std::equal(shape.begin(),shape.end(),second.shape()) ||
            reinterpret_cast<std::uintptr_t>(first.data())%(first_type==PixelType::float32?alignof(float):alignof(double)) ||
            reinterpret_cast<std::uintptr_t>(second.data())%(second_type==PixelType::float32?alignof(float):alignof(double)))
            throw py::value_error("comparison input changed or is unaligned at allocation");
        const detail::CompareInput a{first.data(),first_type},b{second.data(),second_type};
        const detail::CompareGrid tiles{grid[0],grid[1],grid[2],grid[3],grid[4],grid[5]};
        auto* destination=output.mutable_data();
        bool accepted;
        {
            py::gil_scoped_release release;
            accepted=detail::compare_images_into(a,b,destination,count,static_cast<detail::CompareMethod>(method),tiles,workers);
        }
        if (!accepted) return py::none();
        return output;
    },py::arg("first").noconvert(),py::arg("second").noconvert(),py::arg("method"),py::arg("grid"),py::arg("workers"));
}
