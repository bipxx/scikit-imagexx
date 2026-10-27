#include "perimeter_crofton.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
template<class T> py::array_t<py::ssize_t> counts(const py::array& input, skimagexx::PixelType type, int workers) {
    skimagexx::bindings::check_array_storage(input);
    Array<T> prepared(input);
    if (reinterpret_cast<std::uintptr_t>(prepared.data()) % alignof(T)) {
        py::array_t<T> aligned({prepared.shape(0), prepared.shape(1)});
        skimagexx::bindings::check_array_storage(prepared);
        std::memcpy(aligned.mutable_data(), prepared.data(), std::size_t(prepared.nbytes()));
        prepared = Array<T>(aligned);
    }
    skimagexx::bindings::check_array_storage(prepared);
    py::array_t<py::ssize_t> output(16);
    const skimagexx::detail::CroftonInput view{prepared.data(), type,
        std::size_t(prepared.shape(0)), std::size_t(prepared.shape(1))};
    auto* histogram = output.mutable_data();
    { py::gil_scoped_release release; skimagexx::detail::crofton_counts(view, histogram, workers); }
    return output;
}
}
void bind_perimeter_crofton(py::module_& module) {
    module.def("crofton_counts", [](const py::array& input, int workers) -> py::object {
        if (input.ndim() != 2) throw py::value_error("Crofton counts require a 2D array");
        using P = skimagexx::PixelType;
        const auto c = input.dtype().char_();
        if (c == 'e') return counts<float>(input, P::float32, workers);
#define TYPE(C,T,PX) if(c==C) return counts<T>(input, P::PX, workers);
        TYPE('?',bool,boolean) TYPE('b',std::int8_t,int8) TYPE('B',std::uint8_t,uint8)
        TYPE('h',std::int16_t,int16) TYPE('H',std::uint16_t,uint16)
        TYPE('i',std::int32_t,int32) TYPE('I',std::uint32_t,uint32)
        TYPE('q',long long,int64) TYPE('Q',unsigned long long,uint64)
#undef TYPE
        if (c == 'l') return counts<long>(input, sizeof(long)==8 ? P::int64 : P::int32, workers);
        if (c == 'L') return counts<unsigned long>(input, sizeof(long)==8 ? P::uint64 : P::uint32, workers);
        if (c == 'f') return counts<float>(input, P::float32, workers);
        if (c == 'd') return counts<double>(input, P::float64, workers);
        throw py::type_error("Crofton counts require ordinary real storage");
    }, py::arg("image").noconvert(), py::arg("workers"));
    module.def("crofton_perimeter", [](const py::array& histogram, const py::array& coefficients, int workers) {
        if (histogram.ndim()!=1 || histogram.size()!=16 || !Array<py::ssize_t>::check_(histogram)
            || reinterpret_cast<std::uintptr_t>(histogram.data()) % alignof(py::ssize_t)
            || coefficients.ndim()!=1 || coefficients.size()!=16 || !Array<double>::check_(coefficients)
            || reinterpret_cast<std::uintptr_t>(coefficients.data()) % alignof(double))
            throw py::value_error("Crofton weighting requires sixteen native intp counts and double coefficients");
        skimagexx::bindings::check_array_storage(histogram);
        skimagexx::bindings::check_array_storage(coefficients);
        const auto* h = static_cast<const py::ssize_t*>(histogram.data());
        const auto* c = static_cast<const double*>(coefficients.data());
        py::gil_scoped_release release;
        return skimagexx::detail::crofton_perimeter(h, c, workers);
    }, py::arg("histogram").noconvert(), py::arg("coefficients").noconvert(), py::arg("workers"));
}
