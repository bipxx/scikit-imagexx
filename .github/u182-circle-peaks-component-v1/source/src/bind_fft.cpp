#include "fft_internal.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace {
using Shape = std::vector<std::size_t>;
// Bounded ndarray owners and exact immutable bytes allocation roots.
void check_storage(const py::array& input){
    auto current=input;std::vector<py::array> chain;py::object immutable_root;
    for(unsigned i=0;i<64;++i){
        if(std::any_of(chain.begin(),chain.end(),[&](auto x){return x.ptr()==current.ptr();}))throw py::value_error("FFT owner cycle");
        chain.push_back(current);auto base=current.base();
        if(!base||base.is_none())break;
        if(!py::isinstance<py::array>(base)){
            if(PyBytes_CheckExact(base.ptr())){immutable_root=base;break;}
            throw py::value_error("FFT requires ndarray-backed storage");
        }
        current=py::reinterpret_borrow<py::array>(base);
    }
    const auto base=current.base();
    if(!immutable_root && ((base&&!base.is_none())||!current.owndata()||!(current.flags()&(py::array::c_style|py::array::f_style))))throw py::value_error("FFT requires bounded contiguous root");
    const auto limit=std::numeric_limits<std::uintptr_t>::max();
    // Both storage kinds are retained by the ndarray base chain; the local
    // strong bytes reference also keeps its immutable span live during checks.
    const auto root=immutable_root?reinterpret_cast<std::uintptr_t>(PyBytes_AS_STRING(immutable_root.ptr())):reinterpret_cast<std::uintptr_t>(current.data());
    const auto bytes=immutable_root?PyBytes_GET_SIZE(immutable_root.ptr()):current.nbytes();
    if(bytes<0||std::size_t(bytes)>limit-root)throw py::value_error("FFT root overflows");
    for(const auto& a:chain){
        if(a.size()==0)continue;
        std::uintptr_t low=0,high=0;
        for(py::ssize_t d=0;d<a.ndim();++d){
            if(a.shape(d)<0)throw py::value_error("negative FFT extent");
            const auto stride=a.strides(d);
            const std::uintptr_t mag=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
            auto& extent=stride<0?low:high;const auto count=std::uintptr_t(a.shape(d)-1);
            if(count&&mag>(limit-extent)/count)throw py::value_error("FFT extent overflows");extent+=count*mag;
        }
        const auto data=reinterpret_cast<std::uintptr_t>(a.data());
        if(low>data||high>limit-data||std::size_t(a.itemsize())>limit-data-high||data-low<root||data+high+a.itemsize()>root+bytes)throw py::value_error("FFT escapes owner allocation");
    }
}
Shape dimensions(const py::array& input) {
    return Shape(input.shape(), input.shape() + input.ndim());
}
void validate(const Shape& shape, const Shape& axes) {
    if (shape.empty() || axes.empty()) throw py::value_error("FFT shape/axes must be nonempty");
    std::size_t count = 1;
    for (auto size : shape) {
        if (!size || size > std::size_t(PY_SSIZE_T_MAX) / count)
            throw py::value_error("FFT dimensions must be positive and addressable");
        count *= size;
    }
    std::vector<bool> seen(shape.size(), false);
    for (auto axis : axes) {
        if (axis >= shape.size() || seen[axis]) throw py::value_error("FFT axes must be distinct and in range");
        seen[axis] = true;
    }
}
bool single_precision(const py::array& input) {
    const auto kind = input.dtype().kind();
    const auto item = input.itemsize();
    if ((kind == 'f' && item > 8) || (kind == 'c' && item > 16) ||
        (kind != 'f' && kind != 'c' && kind != 'i' && kind != 'u' && kind != 'b')) {
        PyErr_SetString(PyExc_NotImplementedError, "owned FFT supports standard real and complex numeric dtypes");
        throw py::error_already_set();
    }
    return (kind == 'f' && item <= 4) || (kind == 'c' && item <= 8);
}
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
template<class T> void check_prepared(const Array<T>& input, const Shape& shape, bool require_alignment = true) {
    // Allocation is a callback boundary; revalidate only here before release.
    check_storage(input);
    if (dimensions(input) != shape || !Array<T>::check_(input) ||
        (require_alignment && reinterpret_cast<std::uintptr_t>(input.data()) % alignof(T)))
        throw py::value_error("FFT input changed during allocation");
}
template<class T> Array<T> prepare(const py::array& input) {
    check_storage(input);
    Array<T> result(input);
    // PocketFFT's typed buffers must be aligned; ndarray contiguity alone
    // permits an unaligned buffer offset. Copy just that admitted case.
    if (reinterpret_cast<std::uintptr_t>(result.data()) % alignof(T)) {
        const auto shape = dimensions(result);
        py::array_t<T> aligned(shape);
        check_prepared(result, shape, false);
        std::memcpy(aligned.mutable_data(), result.data(), std::size_t(result.nbytes()));
        return Array<T>(aligned);
    }
    return result;
}
template<class T> py::array real_forward(const py::array& input, const Shape& axes, int workers,
                                            bool full = false, bool forward = true) {
    auto values = prepare<T>(input);
    const auto shape = dimensions(values);
    validate(shape, axes);
    auto output_shape = shape;
    if (!full) output_shape[axes.back()] = shape[axes.back()] / 2 + 1;
    py::array_t<std::complex<T>> output(output_shape);
    check_prepared(values, shape);
    {
        py::gil_scoped_release release;
        if (full) skimagexx::detail::fft_r2c_full(shape, axes, values.data(), output.mutable_data(), forward, workers);
        else skimagexx::detail::fft_r2c(shape, axes, values.data(), output.mutable_data(), workers);
    }
    return output;
}
template<class T> py::array real_inverse(const py::array& input, const Shape& shape,
                                         const Shape& axes, int workers) {
    validate(shape, axes);
    auto expected = shape;
    expected[axes.back()] = shape[axes.back()] / 2 + 1;
    if (dimensions(input) != expected) throw py::value_error("FFT spectrum shape does not match real shape");
    auto values = prepare<std::complex<T>>(input);
    py::array_t<T> output(shape);
    check_prepared(values, expected);
    {
        py::gil_scoped_release release;
        skimagexx::detail::fft_c2r(shape, axes, values.data(), output.mutable_data(), workers);
    }
    return output;
}
template<class T> py::array complex_transform(const py::array& input, const Shape& axes,
                                              bool forward, int workers) {
    if (input.dtype().kind() != 'c')
        return real_forward<T>(input, axes, workers, true, forward);
    auto values = prepare<std::complex<T>>(input);
    const auto shape = dimensions(values);
    validate(shape, axes);
    py::array_t<std::complex<T>> output(shape);
    check_prepared(values, shape);
    {
        py::gil_scoped_release release;
        skimagexx::detail::fft_c2c(shape, axes, values.data(), output.mutable_data(), forward, workers);
    }
    return output;
}
}

namespace skimagexx::bindings {
void check_array_storage(const py::array& value) { check_storage(value); }
}

void bind_fft(py::module_& module) {
    module.def("fft_check", &check_storage, py::arg("array").noconvert());
    module.def("fft_r2c", [](const py::array& input, const Shape& axes, int workers) {
        if (input.dtype().kind() == 'c') throw py::type_error("x must be a real sequence");
        return single_precision(input) ? real_forward<float>(input, axes, workers)
                                       : real_forward<double>(input, axes, workers);
    }, py::arg("array").noconvert(), py::arg("axes"), py::arg("workers"));
    module.def("fft_c2r", [](const py::array& input, const Shape& shape, const Shape& axes, int workers) {
        return single_precision(input) ? real_inverse<float>(input, shape, axes, workers)
                                       : real_inverse<double>(input, shape, axes, workers);
    }, py::arg("spectrum").noconvert(), py::arg("shape"), py::arg("axes"), py::arg("workers"));
    module.def("fft_c2c", [](const py::array& input, const Shape& axes, bool forward, int workers) {
        return single_precision(input) ? complex_transform<float>(input, axes, forward, workers)
                                       : complex_transform<double>(input, axes, forward, workers);
    }, py::arg("array").noconvert(), py::arg("axes"), py::arg("forward"), py::arg("workers"));
}
