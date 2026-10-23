#include "skimagexx/kitchen.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
namespace py=pybind11;
void bind_kitchen(py::module_& module){
    module.def("kitchen_rosenfeld",[](py::array image,const std::vector<int>& modes,double cval)->py::object{
        if(image.ndim()!=2 || modes.size()!=2 || !image.dtype().attr("isnative").cast<bool>())throw py::value_error("invalid Kitchen-Rosenfeld input plan");
        const auto type=skimagexx::bindings::real_pixel_type(image.dtype());
        auto input=py::array::ensure(image,py::array::c_style);
        if(!input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")().cast<py::array>();
        const auto dtype=type==skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
        py::array output(dtype,{image.shape(0),image.shape(1)});
        const std::array<skimagexx::GaussianBorder,2> borders{static_cast<skimagexx::GaussianBorder>(modes[0]),static_cast<skimagexx::GaussianBorder>(modes[1])};
        bool valid;
        {py::gil_scoped_release release;valid=skimagexx::kitchen_rosenfeld_into(input.data(),output.mutable_data(),type,std::size_t(image.shape(0)),std::size_t(image.shape(1)),borders,cval);}
        return valid ? py::object(output) : py::none();
    });
}
