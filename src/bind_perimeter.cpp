#include "perimeter.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <cstring>
namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
}
void bind_perimeter(py::module_& module) {
    module.def("perimeter_counts", [](const py::array& input, bool cross, int workers) {
        if (input.ndim()!=2 || input.dtype().char_()!='B')
            throw py::value_error("Perimeter counts require a uint8 2D array");
        skimagexx::bindings::check_array_storage(input);
        Array<std::uint8_t> prepared(input);
        skimagexx::bindings::check_array_storage(prepared);
        std::array<std::ptrdiff_t,256> histogram{};
        const auto* data=prepared.data();
        const auto height=std::size_t(prepared.shape(0)), width=std::size_t(prepared.shape(1));
        std::size_t length;
        { py::gil_scoped_release release;
          length=skimagexx::detail::perimeter_counts(data,height,width,cross,histogram.data(),workers); }
        // Allocation follows the last native input read; no input pointer is
        // used after this new Python allocation boundary.
        py::array_t<py::ssize_t> output(length);
        static_assert(sizeof(py::ssize_t)==sizeof(std::ptrdiff_t));
        std::memcpy(output.mutable_data(),histogram.data(),length*sizeof(std::ptrdiff_t));
        return output;
    },py::arg("image").noconvert(),py::arg("cross"),py::arg("workers"));
    module.def("perimeter_weighted", [](const py::array& histogram,const py::array& weights,int workers) {
        if(histogram.ndim()!=1 || histogram.size()!=50 || !Array<py::ssize_t>::check_(histogram)
            || reinterpret_cast<std::uintptr_t>(histogram.data())%alignof(py::ssize_t)
            || weights.ndim()!=1 || weights.size()!=50 || !Array<double>::check_(weights)
            || reinterpret_cast<std::uintptr_t>(weights.data())%alignof(double))
            throw py::value_error("Perimeter weighting requires fifty native intp counts and double weights");
        skimagexx::bindings::check_array_storage(histogram);
        skimagexx::bindings::check_array_storage(weights);
        // Keep this fixed tiny operation under the GIL; the leaf has no callbacks.
        return skimagexx::detail::perimeter_weighted(static_cast<const std::ptrdiff_t*>(histogram.data()),
            static_cast<const double*>(weights.data()),workers);
    },py::arg("histogram").noconvert(),py::arg("weights").noconvert(),py::arg("workers"));
}
