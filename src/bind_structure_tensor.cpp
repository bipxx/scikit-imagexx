#include "skimagexx/structure_tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py=pybind11;
namespace {
skimagexx::PixelType tensor_type(py::dtype dtype){
    using P=skimagexx::PixelType;const auto bytes=dtype.itemsize();
    if(dtype.kind()=='b')return P::boolean;
    if(dtype.kind()=='f' && bytes==4)return P::float32;
    if(dtype.kind()=='f' && bytes==8)return P::float64;
    if(dtype.kind()=='i'){if(bytes==1)return P::int8;if(bytes==2)return P::int16;if(bytes==4)return P::int32;if(bytes==8)return P::int64;}
    if(dtype.kind()=='u'){if(bytes==1)return P::uint8;if(bytes==2)return P::uint16;if(bytes==4)return P::uint32;if(bytes==8)return P::uint64;}
    throw py::type_error("real structure tensor input required");
}
}
void bind_structure_tensor(py::module_& module){
    module.def("structure_tensor",[](py::array image,py::list kernels,const std::vector<std::size_t>& axes,
        const std::vector<int>& modes,double cval,bool xy)->py::object{
        if(!image.dtype().attr("isnative").cast<bool>())throw py::type_error("native byte order required");
        const auto type=tensor_type(image.dtype());
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
