#include "euler_number.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <vector>
namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style>;
}
void bind_euler_number(py::module_& module) {
    module.def("euler_counts", [](const py::array& input,bool volume,int workers) {
        py::array_t<std::ptrdiff_t> histogram(volume ? 256 : 16);
        if(input.ndim()!=(volume ? 3 : 2) || !Array<std::uint8_t>::check_(input))
            throw py::value_error("Euler counts require padded native contiguous uint8 storage of the selected rank");
        skimagexx::bindings::check_array_storage(input);
        // This exposed private entry accepts only binary bytes. Validate and
        // copy in one pass; the kernel's value invariant survives GIL release.
        const auto* source=static_cast<const std::uint8_t*>(input.data());
        std::vector<std::uint8_t> snapshot(std::size_t(input.size()));
        for(std::size_t i=0;i<snapshot.size();++i){
            const auto value=source[i];
            if(value>1)throw py::value_error("Euler counts require binary byte values");
            snapshot[i]=value;
        }
        const skimagexx::detail::EulerInput view{snapshot.data(),
            volume ? std::size_t(input.shape(0)) : 1,
            std::size_t(input.shape(volume ? 1 : 0)),std::size_t(input.shape(volume ? 2 : 1))};
        auto* output=histogram.mutable_data();
        { py::gil_scoped_release release; skimagexx::detail::euler_counts(view,volume,output,workers); }
        return histogram;
    },py::arg("image").noconvert(),py::arg("volume"),py::arg("workers"));
    module.def("euler_weighted", [](const py::array& histogram,const py::array& coefficients,bool volume,int workers)->py::object {
        const auto length=volume ? 256 : 16;
        if(histogram.ndim()!=1 || histogram.size()!=length || !Array<std::ptrdiff_t>::check_(histogram)
            || reinterpret_cast<std::uintptr_t>(histogram.data())%alignof(std::ptrdiff_t)
            || coefficients.ndim()!=1 || coefficients.size()!=length || !Array<std::int8_t>::check_(coefficients))
            throw py::value_error("Euler weighting requires matching native intp counts and int8 LUT coefficients");
        skimagexx::bindings::check_array_storage(histogram);
        skimagexx::bindings::check_array_storage(coefficients);
        std::int64_t result=0;
        const bool accepted=skimagexx::detail::euler_weighted(static_cast<const std::ptrdiff_t*>(histogram.data()),
            static_cast<const std::int8_t*>(coefficients.data()),volume,result,workers);
        if(!accepted)return py::none();
        return py::int_(result);
    },py::arg("histogram").noconvert(),py::arg("coefficients").noconvert(),py::arg("volume"),py::arg("workers"));
}
