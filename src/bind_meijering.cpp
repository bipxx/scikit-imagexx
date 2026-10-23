#include "skimagexx/meijering.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py=pybind11;
void bind_meijering(py::module_& module){
 module.def("meijering_accumulate",[](py::list fields,py::object previous,py::dtype dtype,double alpha)->py::object{
  std::vector<py::array> storage;std::vector<const void*> data;std::vector<py::ssize_t> shape;py::dtype field_dtype;
  for(auto entry:fields){auto a=py::array::ensure(entry,py::array::c_style);
   if(!a || !a.dtype().attr("isnative").cast<bool>() || a.dtype().kind()!='f' || (a.itemsize()!=4 && a.itemsize()!=8) || a.ndim()<1 || a.ndim()>5)throw py::value_error("native Meijering Hessian fields required");
   if(storage.empty()){shape.assign(a.shape(),a.shape()+a.ndim());field_dtype=a.dtype();}
   else if(!a.dtype().is(field_dtype) || std::vector<py::ssize_t>(a.shape(),a.shape()+a.ndim())!=shape)throw py::value_error("homogeneous Meijering fields required");
   if(!a.attr("flags").attr("aligned").cast<bool>())a=a.attr("copy")().cast<py::array>();storage.push_back(a);data.push_back(storage.back().data());
  }
  const auto n=shape.size();if(!n || fields.size()!=n*(n+1)/2 || (dtype.kind()!='f' || dtype.itemsize()!=field_dtype.itemsize() || !dtype.attr("isnative").cast<bool>()))throw py::value_error("invalid Meijering output plan");
  py::array prior;const void* old=nullptr;
  if(!previous.is_none()){prior=py::array::ensure(previous,py::array::c_style);if(!prior || (prior.dtype().kind()!='f' || prior.itemsize()!=dtype.itemsize() || !prior.dtype().attr("isnative").cast<bool>()) || std::vector<py::ssize_t>(prior.shape(),prior.shape()+prior.ndim())!=shape)throw py::value_error("invalid Meijering previous output");
   if(!prior.attr("flags").attr("aligned").cast<bool>())prior=prior.attr("copy")().cast<py::array>();old=prior.data();}
  py::array output(dtype,shape);bool valid;
  {py::gil_scoped_release release;valid=skimagexx::meijering_accumulate_into(data.data(),old,output.mutable_data(),output.size(),n,dtype.itemsize()==4,alpha);}
  return valid ? py::object(output) : py::none();
 });
}
