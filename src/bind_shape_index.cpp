#include "skimagexx/shape_index.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py=pybind11;
void bind_shape_index(py::module_& module){
 module.def("shape_index",[](py::list fields)->py::object{
  if(fields.size()!=3)throw py::value_error("three Hessian fields required");
  std::vector<py::array> storage;storage.reserve(3);std::vector<const void*> data;
  py::dtype dtype;std::vector<py::ssize_t> shape;
  for(auto field:fields){auto array=py::array::ensure(field,py::array::c_style);
   if(!array || array.ndim()!=2 || array.dtype().kind()!='f' || !array.dtype().attr("isnative").cast<bool>() || (array.itemsize()!=4 && array.itemsize()!=8))throw py::value_error("invalid shape index Hessian fields");
   if(storage.empty()){dtype=array.dtype();shape.assign(array.shape(),array.shape()+array.ndim());}
   else if(!array.dtype().is(dtype) || array.shape(0)!=shape[0] || array.shape(1)!=shape[1])throw py::value_error("homogeneous shape index Hessian fields required");
   if(!array.attr("flags").attr("aligned").cast<bool>())array=array.attr("copy")().cast<py::array>();
   storage.push_back(std::move(array));data.push_back(storage.back().data());
  }
  py::array output(dtype,shape);bool valid;
  {py::gil_scoped_release release;valid=skimagexx::shape_index_from_hessian_into(data.data(),output.mutable_data(),std::size_t(output.size()),dtype.itemsize()==4);}
  return valid ? py::object(output) : py::none();
 });
}
