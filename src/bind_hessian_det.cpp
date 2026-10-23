#include "skimagexx/hessian_det.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py=pybind11;
namespace {
bool native_dtype(const py::dtype& dtype){
 const auto order=dtype.byteorder();const std::uint16_t marker=1;
 const bool little=*reinterpret_cast<const unsigned char*>(&marker)==1;
 return order=='=' || order=='|' || order==(little ? '<' : '>');
}
bool aligned(const py::array& array){return (array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_)!=0;}
}
void bind_hessian_det(py::module_& module){
 module.def("hessian_det_approx",[](py::array image,std::int64_t size)->py::object{
  if(image.ndim()!=2 || !native_dtype(image.dtype()))throw py::value_error("native 2D approximate Hessian input required");
  const auto type=skimagexx::bindings::real_pixel_type(image.dtype());auto input=py::array::ensure(image,py::array::c_style);
  if(!aligned(input))input=input.attr("copy")().cast<py::array>();
  py::array_t<double> result({image.shape(0),image.shape(1)});bool valid;
  {py::gil_scoped_release release;valid=skimagexx::hessian_det_approx_into(input.data(),type,result.mutable_data(),image.shape(0),image.shape(1),size);}
  return valid ? py::object(result) : py::none();
 });
 module.def("hessian_det_fields",[](py::list fields)->py::object{
  if(fields.size()==0)throw py::value_error("Hessian fields required");
  std::vector<py::array> arrays;std::vector<const void*> data;py::dtype dtype;std::vector<py::ssize_t> shape;
  for(auto entry:fields){auto a=py::array::ensure(entry,py::array::c_style);
   if(!a || a.ndim()<2 || a.ndim()>5 || a.dtype().kind()!='f' || (a.itemsize()!=4 && a.itemsize()!=8) || !native_dtype(a.dtype()))throw py::value_error("invalid Hessian fields");
   if(arrays.empty()){dtype=a.dtype();shape.assign(a.shape(),a.shape()+a.ndim());}
   else if(!a.dtype().is(dtype) || std::vector<py::ssize_t>(a.shape(),a.shape()+a.ndim())!=shape)throw py::value_error("homogeneous Hessian fields required");
   if(!aligned(a))a=a.attr("copy")().cast<py::array>();arrays.push_back(a);data.push_back(arrays.back().data());
  }
  const auto n=shape.size();if(fields.size()!=n*(n+1)/2)throw py::value_error("Hessian field count mismatch");py::array result(dtype,shape);bool valid;
  {py::gil_scoped_release release;valid=skimagexx::hessian_det_from_fields_into(data.data(),result.mutable_data(),result.size(),n,dtype.itemsize()==4);}
  return valid ? py::object(result) : py::none();
 });
}
