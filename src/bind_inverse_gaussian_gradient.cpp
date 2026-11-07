#include <skimagexx/inverse_gaussian_gradient.hpp>
#include "bind_array_storage.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
namespace py=pybind11;
namespace {
skimagexx::PixelType type(py::dtype d){if(d.kind()=='c'){if(d.itemsize()!=8&&d.itemsize()!=16)throw py::type_error("IGG extended complex dtype unassessed");return d.itemsize()==8?skimagexx::PixelType::float32:skimagexx::PixelType::float64;}return skimagexx::bindings::real_pixel_type(d);}
void storage(const py::array& a){if(!a.dtype().attr("isnative").cast<bool>()||!(a.flags()&py::array::c_style)||!a.attr("flags").attr("aligned").cast<bool>())throw py::value_error("IGG prepared storage must be native aligned C order");skimagexx::bindings::check_array_storage(a);}
}
void bind_inverse_gaussian_gradient(py::module_& m){
 m.def("inverse_gaussian_magnitude",[](py::array image,py::list specification,int workers){
    storage(image);std::vector<std::size_t> shape(image.shape(),image.shape()+image.ndim());
    skimagexx::InverseGaussianPasses plans;
    for(auto derivative:specification){std::vector<skimagexx::InverseGaussianPass> passes;for(auto item:derivative.cast<py::list>()){auto row=item.cast<py::tuple>();auto w=py::array_t<double,py::array::c_style>::ensure(row[1]);if(!w||w.ndim()!=1)throw py::value_error("IGG coefficient vector");passes.push_back({row[0].cast<std::size_t>(),{w.data(),w.data()+w.size()}});}plans.push_back(std::move(passes));}
    py::array output(image.dtype(),std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
    const auto pixel=type(image.dtype());const bool complex=image.dtype().kind()=='c';
    {py::gil_scoped_release release;skimagexx::inverse_gaussian_gradient_magnitude(image.data(),output.mutable_data(),pixel,complex,shape,plans,workers);}return output;
 },py::arg("image").noconvert(),py::arg("plans"),py::arg("workers"));
 m.def("inverse_gaussian_finish",[](py::array gradient,py::array alpha,std::vector<py::ssize_t> dimensions,unsigned stage,int workers){
    storage(gradient);storage(alpha);if(!gradient.dtype().is(alpha.dtype()))throw py::type_error("IGG common dtype required");
    std::vector<std::size_t> shape;for(auto d:dimensions){if(d<0)throw py::value_error("IGG extent");shape.push_back(std::size_t(d));}
    if(shape.size()>32||stage>3)throw py::value_error("IGG finish plan");
    if(stage==0){
        const auto rank=std::max(gradient.ndim(),alpha.ndim());if(dimensions.size()!=std::size_t(rank))throw py::value_error("IGG broadcast rank");
        for(py::ssize_t i=0;i<rank;++i){const auto g=i<gradient.ndim()?gradient.shape(gradient.ndim()-1-i):1,a=i<alpha.ndim()?alpha.shape(alpha.ndim()-1-i):1;if(g!=a&&g!=1&&a!=1)throw py::value_error("IGG broadcast mismatch");if(dimensions[std::size_t(rank-1-i)]!=(g==1?a:g))throw py::value_error("IGG output broadcast extent");}
    }else if(dimensions!=std::vector<py::ssize_t>(gradient.shape(),gradient.shape()+gradient.ndim()))throw py::value_error("IGG stage shape mismatch");
    py::array output(gradient.dtype(),dimensions);unsigned flags;
    const auto pixel=type(gradient.dtype());const bool complex=gradient.dtype().kind()=='c';
    {py::gil_scoped_release release;flags=skimagexx::inverse_gaussian_finish(gradient.data(),alpha.data(),output.mutable_data(),pixel,complex,{gradient.shape(),gradient.shape()+gradient.ndim()},{alpha.shape(),alpha.shape()+alpha.ndim()},shape,stage,workers);}return py::make_tuple(output,flags);
 },py::arg("gradient").noconvert(),py::arg("alpha").noconvert(),py::arg("shape"),py::arg("stage"),py::arg("workers"));
}
