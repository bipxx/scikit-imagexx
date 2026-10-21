#include "skimagexx/image.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <type_traits>

namespace py = pybind11;
namespace {
void shape(const py::array& a, bool rgb) {
    if (a.ndim() != (rgb ? 3 : 2) || (rgb && a.shape(2) != 3))
        throw py::value_error(rgb ? "expected an HxWx3 RGB image" : "expected a 2D grayscale image");
    if (a.shape(0) == 0 || a.shape(1) == 0) throw py::value_error("image must be nonempty");
}
template<class T, class F> py::array execute(const py::array& image, F function) {
    // NumPy owns the uninitialized output, avoiding vector zero-fill and capsules.
    // Ensure aligned input as well as contiguous input before typed C++ access.
    auto input = py::array_t<T,py::array::c_style>::ensure(image);
    if (!input) throw py::type_error("unable to make input contiguous");
    if ((reinterpret_cast<std::uintptr_t>(input.data()) % alignof(T)) != 0) {
        input = py::array_t<T,py::array::c_style>::ensure(input.attr("copy")());
    }
    using Output = std::conditional_t<std::is_same_v<T,std::uint8_t>,double,T>;
    const auto h = static_cast<std::size_t>(input.shape(0)), w = static_cast<std::size_t>(input.shape(1));
    py::array_t<Output> output({input.shape(0),input.shape(1)});
    const auto* source = input.data();
    auto* destination = output.mutable_data();
    { py::gil_scoped_release release; function(source,destination,h,w); }
    return output;
}
template<class F> py::array dispatch(const py::array& image, F function) {
    if (image.dtype().is(py::dtype::of<float>())) return execute<float>(image,function);
    if (image.dtype().is(py::dtype::of<double>())) return execute<double>(image,function);
    if (image.dtype().is(py::dtype::of<std::uint8_t>())) return execute<std::uint8_t>(image,function);
    throw py::type_error("supported dtypes are uint8, float32, and float64");
}
}

PYBIND11_MODULE(_core, m) {
    m.def("set_num_threads", &skimagexx::set_num_threads, py::arg("count"),
          "Set the process-wide worker limit for all algorithms. Default is 1 (serial); use >1 to enable parallel work. SIMD is independent.");
    m.def("get_num_threads", &skimagexx::get_num_threads, "Return the process-wide configured worker limit.");
    m.def("has_avx2", &skimagexx::has_avx2);
    m.def("rgb2gray", [](py::array image) {
        shape(image,true);
        return dispatch(image,[](const auto* p, auto* out, auto h, auto w) { skimagexx::rgb2gray_into(p,out,h,w); });
    },py::arg("image"));
    m.def("gaussian", [](py::array image, double sigma, const std::string& mode, bool preserve_range, double truncate) {
        shape(image,false);
        skimagexx::Border border;
        if (mode == "nearest") border = skimagexx::Border::nearest;
        else if (mode == "reflect") border = skimagexx::Border::reflect;
        else throw py::value_error("supported modes are nearest and reflect");
        return dispatch(image,[=](const auto* p, auto* out, auto h, auto w) {
            using Input = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
            if constexpr (std::is_same_v<Input,std::uint8_t>)
                skimagexx::gaussian_into(p,out,h,w,sigma,border,truncate,preserve_range);
            else skimagexx::gaussian_into(p,out,h,w,sigma,border,truncate);
        });
    },py::arg("image"),py::arg("sigma")=1.0,py::kw_only(),py::arg("mode")="nearest",
       py::arg("preserve_range")=false,py::arg("truncate")=4.0);
    m.def("sobel", [](py::array image) {
        shape(image,false);
        return dispatch(image,[](const auto* p, auto* out, auto h, auto w) { skimagexx::sobel_into(p,out,h,w); });
    },py::arg("image"));
    m.def("threshold_otsu", [](py::array image) {
        shape(image,false);
        if (!image.dtype().is(py::dtype::of<std::uint8_t>())) throw py::type_error("threshold_otsu requires uint8");
        auto a = py::array_t<std::uint8_t,py::array::c_style>::ensure(image);
        const auto* data = a.data();
        const auto h = static_cast<std::size_t>(a.shape(0)), w = static_cast<std::size_t>(a.shape(1));
        py::gil_scoped_release release;
        return skimagexx::threshold_otsu(data,h,w);
    },py::arg("image"));
}
