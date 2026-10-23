#include "skimagexx/structure_tensor.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py=pybind11;
void bind_structure_tensor(py::module_& module){
    module.def("structure_tensor",[](py::array image,py::list kernels,const std::vector<std::size_t>& axes,
        const std::vector<int>& modes,double cval,bool xy)->py::object{
        if(!image.dtype().attr("isnative").cast<bool>())throw py::type_error("native byte order required");
        const auto type=skimagexx::bindings::real_pixel_type(image.dtype());
        if(image.ndim()<2 || image.ndim()>5 || modes.size()!=std::size_t(image.ndim()) || axes.size()!=std::size_t(kernels.size()))throw py::value_error("invalid structure tensor plan");
        auto input=py::array::ensure(image,py::array::c_style);
        if(!input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")().cast<py::array>();
        std::vector<std::size_t> shape;std::vector<py::ssize_t> dimensions;
        for(py::ssize_t axis=0;axis<input.ndim();++axis){shape.push_back(std::size_t(input.shape(axis)));dimensions.push_back(input.shape(axis));}
        std::vector<skimagexx::GaussianBorder> borders;
        for(auto mode:modes)borders.push_back(static_cast<skimagexx::GaussianBorder>(mode));
        std::vector<skimagexx::GaussianPass> smoothing;
        for(std::size_t i=0;i<axes.size();++i){
            if(axes[i]>=shape.size())throw py::value_error("invalid structure tensor smoothing axis");
            auto weights=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(kernels[i]);
            if(!weights || weights.ndim()!=1)throw py::value_error("one-dimensional smoothing coefficients required");
            skimagexx::GaussianPass pass;pass.axis=axes[i];pass.border=borders[axes[i]];pass.cval=cval;
            pass.weights.assign(weights.data(),weights.data()+weights.size());smoothing.push_back(std::move(pass));
        }
        py::list result;std::vector<py::array> storage;std::vector<void*> output;
        const auto dtype=type==skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
        for(std::size_t field=0;field<shape.size()*(shape.size()+1)/2;++field){
            storage.emplace_back(dtype,dimensions);output.push_back(storage.back().mutable_data());result.append(storage.back());
        }
        bool supported;
        {py::gil_scoped_release release;supported=skimagexx::structure_tensor_into(input.data(),type,output.data(),shape,smoothing,borders,cval,xy);}
        return supported ? py::object(result) : py::none();
    });
}
