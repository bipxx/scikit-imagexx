#include "skimagexx/hessian.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_hessian(py::module_& module){
 module.def("hessian_matrix",[](py::array image,py::list specifications,double cval,bool gaussian,bool xy)->py::object{
  if(!image.dtype().attr("isnative").cast<bool>() || image.ndim()<1 || image.ndim()>5)throw py::value_error("invalid Hessian input plan");
  const auto type=skimagexx::bindings::real_pixel_type(image.dtype());auto input=py::array::ensure(image,py::array::c_style);
  if(!input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")().cast<py::array>();
  std::vector<std::size_t> shape;std::vector<py::ssize_t> dimensions;
  for(py::ssize_t axis=0;axis<input.ndim();++axis){shape.push_back(std::size_t(input.shape(axis)));dimensions.push_back(input.shape(axis));}
  std::vector<std::vector<skimagexx::HessianPass>> plans;
  for(auto list:specifications){std::vector<skimagexx::HessianPass> passes;
   for(auto entry:list.cast<py::list>()){const auto tuple=entry.cast<py::tuple>();if(tuple.size()!=4)throw py::value_error("invalid Hessian pass tuple");
    auto weights=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(tuple[1]);if(!weights || weights.ndim()!=1)throw py::value_error("one-dimensional Hessian coefficients required");
    skimagexx::HessianPass p;p.filter.axis=tuple[0].cast<std::size_t>();p.filter.weights.assign(weights.data(),weights.data()+weights.size());p.filter.border=static_cast<skimagexx::GaussianBorder>(tuple[2].cast<int>());p.filter.cval=cval;p.derivative=tuple[3].cast<bool>();passes.push_back(std::move(p));
   }plans.push_back(std::move(passes));
  }
  const auto dtype=type==skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
  py::list result;std::vector<py::array> storage;std::vector<void*> output;
  for(std::size_t field=0;field<shape.size()*(shape.size()+1)/2;++field){storage.emplace_back(dtype,dimensions);output.push_back(storage.back().mutable_data());result.append(storage.back());}
  bool valid;{py::gil_scoped_release release;valid=skimagexx::hessian_matrix_into(input.data(),type,output.data(),shape,plans,gaussian,xy);}
  return valid ? py::object(result) : py::none();
 });
}
