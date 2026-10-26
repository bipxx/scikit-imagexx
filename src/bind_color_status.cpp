#include "color_detail.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <vector>
namespace py=pybind11;
namespace {
template<class T>py::tuple transform(const py::array& input,int operation){
    const std::vector<py::ssize_t> shape(input.shape(),input.shape()+input.ndim());
    if(shape.empty()||shape.back()!=3)throw py::value_error("expected an array of color triples");
    py::array_t<T> output(shape);
    skimagexx::bindings::check_array_storage(input);
    if(!py::array_t<T,py::array::c_style>::check_(input)||
       std::vector<py::ssize_t>(input.shape(),input.shape()+input.ndim())!=shape||
       reinterpret_cast<std::uintptr_t>(input.data())%alignof(T))throw py::value_error("color requires prepared aligned native C storage");
    const auto* data=static_cast<const T*>(input.data());auto* result=output.mutable_data();const auto count=std::size_t(input.size()/3);
    unsigned flags;
    {py::gil_scoped_release release;flags=skimagexx::detail::color_transform_status_into(data,result,count,static_cast<skimagexx::ColorTransform>(operation));}
    return py::make_tuple(output,flags);
}
}
void bind_color_status(py::module_& module){
    // Required third argument leaves the historical two-argument overload intact.
    module.def("color_transform",[](const py::array& input,int operation,bool capture_status){
        if(!capture_status)throw py::value_error("capture_status must be True");
        if(input.dtype().char_()=='f')return transform<float>(input,operation);
        if(input.dtype().char_()=='d')return transform<double>(input,operation);
        throw py::type_error("color status requires float32 or float64");
    },py::arg("image").noconvert(),py::arg("operation"),py::arg("capture_status"));
}
