#include "resample_detail.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
void bind_resample(py::module_& module) {
    module.def("resize_zoom",[](py::array image,const std::vector<py::ssize_t>& dimensions,
                               int order,int border,double cval,bool legacy,bool identity,bool simd) {
        if(image.dtype().kind()!='f' || image.itemsize()!=8 || image.ndim()<1)
            throw py::type_error("float64 zoom input required");
        if(dimensions.size()!=std::size_t(image.ndim()))throw py::value_error("matching zoom ranks required");
        auto input=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(image);
        if(!input)throw py::type_error("float64 zoom input required");
        if(!input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")().cast<decltype(input)>();
        std::vector<std::size_t> shape,target;
        for(py::ssize_t axis=0;axis<input.ndim();++axis) {
            if(dimensions[std::size_t(axis)]<0)throw py::value_error("negative zoom dimension");
            shape.push_back(std::size_t(input.shape(axis)));target.push_back(std::size_t(dimensions[std::size_t(axis)]));
        }
        py::array_t<double> result(dimensions);
        skimagexx::detail::ZoomOptions options;
        options.order=order;options.border=static_cast<skimagexx::GaussianBorder>(border);options.cval=cval;
        options.legacy_reflect=legacy;options.identity_copy=identity;options.allow_simd=simd;
        {py::gil_scoped_release release;skimagexx::detail::zoom_into(input.data(),result.mutable_data(),shape,target,options);}
        return result;
    });
}
