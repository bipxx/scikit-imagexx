#include "skimagexx/sato.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py=pybind11;
namespace {
bool aligned(const py::array& a){return (a.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_)!=0;}
bool native(const py::dtype& d){const std::uint16_t marker=1;const auto order=d.byteorder();return order=='=' || order=='|' || order==(*reinterpret_cast<const unsigned char*>(&marker) ? '<' : '>');}
}
void bind_sato(py::module_& module){
 module.def("ridge_prepare",[](py::array image,bool negative){
  if(!native(image.dtype()))throw py::value_error("native ridge input required");
  const auto type=skimagexx::bindings::real_pixel_type(image.dtype());auto input=py::array::ensure(image,py::array::c_style);
  if(!aligned(input))input=input.attr("copy")().cast<py::array>();
  if(!negative && (type==skimagexx::PixelType::float32 || type==skimagexx::PixelType::float64))return input;
  const auto dtype=type==skimagexx::PixelType::float32 || type==skimagexx::PixelType::float64 ? input.dtype() : py::dtype::of<double>();
  const std::vector<py::ssize_t> shape(input.shape(),input.shape()+input.ndim());py::array output(dtype,shape);
  {py::gil_scoped_release release;skimagexx::ridge_prepare_into(input.data(),type,output.mutable_data(),input.size(),dtype.itemsize()==4,negative);}
  return output;
 });
 module.def("sato_accumulate",[](py::list fields,py::object previous,py::dtype dtype,double factor,bool value_single)->py::object{
  std::vector<py::array> storage;std::vector<const void*> data;std::vector<py::ssize_t> shape;py::dtype field_dtype;
  for(auto entry:fields){auto a=py::array::ensure(entry,py::array::c_style);
   if(!a || !native(a.dtype()) || a.dtype().kind()!='f' || (a.itemsize()!=4 && a.itemsize()!=8) || a.ndim()<2 || a.ndim()>3)throw py::value_error("native Sato Hessian fields required");
   if(storage.empty()){shape.assign(a.shape(),a.shape()+a.ndim());field_dtype=a.dtype();}
   else if(!a.dtype().is(field_dtype) || std::vector<py::ssize_t>(a.shape(),a.shape()+a.ndim())!=shape)throw py::value_error("homogeneous Sato Hessian fields required");
   if(!aligned(a))a=a.attr("copy")().cast<py::array>();storage.push_back(a);data.push_back(storage.back().data());
  }
  const auto n=shape.size();if(n<2 || fields.size()!=n*(n+1)/2 || !native(dtype) || dtype.kind()!='f' || (dtype.itemsize()!=4 && dtype.itemsize()!=8))throw py::value_error("invalid Sato output plan");
  py::array prior;const void* old=nullptr;bool previous_single=true;
  if(!previous.is_none()){prior=py::array::ensure(previous,py::array::c_style);if(!prior || !native(prior.dtype()) || prior.dtype().kind()!='f' || (prior.itemsize()!=4 && prior.itemsize()!=8) || std::vector<py::ssize_t>(prior.shape(),prior.shape()+prior.ndim())!=shape)throw py::value_error("invalid Sato previous output");
   if(!aligned(prior))prior=prior.attr("copy")().cast<py::array>();old=prior.data();previous_single=prior.itemsize()==4;}
  py::array output(dtype,shape);bool valid;
  {py::gil_scoped_release release;valid=skimagexx::sato_accumulate_into(data.data(),old,output.mutable_data(),output.size(),n,field_dtype.itemsize()==4,previous_single,dtype.itemsize()==4,value_single,factor);}
  return valid ? py::object(output) : py::none();
 });
}
