#include "skimagexx/frangi.hpp"
#include "frangi_exp.hpp"
#include "frangi_detail.hpp"
#include "execution.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <atomic>
namespace py=pybind11;
namespace {
struct NumpyFrangiFloatExp {
 py::object owner;
 PyUFuncGenericFunction function=nullptr;void* data=nullptr;
 skimagexx::FrangiFloatExpProvider provider;
 bool required=false;
 NumpyFrangiFloatExp():owner(py::module_::import("numpy").attr("exp")){
  if(!PyObject_TypeCheck(owner.ptr(),&PyUFunc_Type))throw std::runtime_error("native float exponential loop unavailable");
  const auto* u=reinterpret_cast<PyUFuncObject*>(owner.ptr());
  if(u->nargs==2)for(int i=0;i<u->ntypes;++i)if(u->types[2*i]==NPY_FLOAT && u->types[2*i+1]==NPY_FLOAT){function=u->functions[i];data=u->data ? u->data[i] : nullptr;break;}
  if(!function)throw std::runtime_error("native float exponential loop unavailable");
  provider={&evaluate,this};
  // Probe the active loop rather than guessing from CPU feature names or
  // environment overrides. Near-one rounding controls ridge-Hessian's finish.
  std::vector<float> inputs(1028),outputs(1028);
  for(std::size_t i=0;i<1025;++i)inputs[i]=-2e-6f+2e-6f*(float(i)/1024.f);
  inputs[1025]=-3e-8f;inputs[1026]=-2.98e-8f;inputs[1027]=-2.97e-8f;
  evaluate(inputs.data(),outputs.data(),inputs.size(),this);
  for(std::size_t i=0;i<inputs.size();++i)if(outputs[i]!=skimagexx::detail::frangi_exp(inputs[i])){required=true;break;}
 }
 static void evaluate(const float* input,float* output,std::size_t count,void* context){
  const auto& self=*static_cast<const NumpyFrangiFloatExp*>(context);
  char* args[]={reinterpret_cast<char*>(const_cast<float*>(input)),reinterpret_cast<char*>(output)};
  const npy_intp dimensions[]={npy_intp(count)},steps[]={sizeof(float),sizeof(float)};
  self.function(args,dimensions,steps,self.data);
 }
};
struct NumpyFrangiExp {
 py::object owner;
 PyUFuncGenericFunction function=nullptr;void* data=nullptr;
 skimagexx::FrangiExpProvider provider;
 NumpyFrangiExp():owner(py::module_::import("numpy").attr("exp")){
  if(!PyObject_TypeCheck(owner.ptr(),&PyUFunc_Type))throw std::runtime_error("native exponential loop unavailable");
  const auto* u=reinterpret_cast<PyUFuncObject*>(owner.ptr());
  if(u->nargs==2)for(int i=0;i<u->ntypes;++i)if(u->types[2*i]==NPY_DOUBLE && u->types[2*i+1]==NPY_DOUBLE){function=u->functions[i];data=u->data ? u->data[i] : nullptr;break;}
  if(!function)throw std::runtime_error("native double exponential loop unavailable");
  provider={&evaluate,this};
 }
 static void evaluate(const double* input,double* output,std::size_t count,void* context){
  const auto& self=*static_cast<const NumpyFrangiExp*>(context);
  char* args[]={reinterpret_cast<char*>(const_cast<double*>(input)),reinterpret_cast<char*>(output)};
  const npy_intp dimensions[]={npy_intp(count)},steps[]={sizeof(double),sizeof(double)};
  self.function(args,dimensions,steps,self.data);
 }
};
struct NumpyFrangiEigen {
 py::object owner;
 PyUFuncGenericFunction function=nullptr;void* data=nullptr;
 skimagexx::FrangiEigenProvider provider;
 NumpyFrangiEigen():owner(py::module_::import("numpy.linalg._umath_linalg").attr("eigvalsh_lo")){
  if(!PyObject_TypeCheck(owner.ptr(),&PyUFunc_Type) || owner.attr("signature").cast<std::string>()!="(m,m)->(m)")throw std::runtime_error("native symmetric LAPACK loop unavailable");
  const auto* u=reinterpret_cast<PyUFuncObject*>(owner.ptr());
  if(u->nargs==2)for(int i=0;i<u->ntypes;++i)if(u->types[2*i]==NPY_DOUBLE && u->types[2*i+1]==NPY_DOUBLE){function=u->functions[i];data=u->data ? u->data[i] : nullptr;break;}
  if(!function)throw std::runtime_error("native double symmetric LAPACK loop unavailable");
  provider={&evaluate,this};
 }
 static bool evaluate(const void* const* fields,void* output,std::size_t count,std::size_t n,bool single,void* context){
  const auto& self=*static_cast<const NumpyFrangiEigen*>(context);std::atomic<bool> valid{true};
  skimagexx::detail::partition(count,skimagexx::detail::active_workers(count*n*n*n),[&](auto begin,auto end,auto){
   const auto capacity=std::min(std::size_t(1024),end-begin);std::vector<double> matrices(capacity*n*n),values(capacity*n);
   for(auto position=begin;position<end;){const auto length=std::min(capacity,end-position);
    for(std::size_t i=0;i<length;++i){std::size_t f=0;for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){
     const double v=single ? double(static_cast<const float*>(fields[f])[position+i]) : static_cast<const double*>(fields[f])[position+i];++f;
     if(!std::isfinite(v)){valid.store(false,std::memory_order_relaxed);return;}
     matrices[i*n*n+r*n+c]=matrices[i*n*n+c*n+r]=v;
    }}
    char* args[]={reinterpret_cast<char*>(matrices.data()),reinterpret_cast<char*>(values.data())};
    const npy_intp dimensions[]={npy_intp(length),npy_intp(n)};
    const npy_intp steps[]={npy_intp(n*n*sizeof(double)),npy_intp(n*sizeof(double)),npy_intp(n*sizeof(double)),npy_intp(sizeof(double)),npy_intp(sizeof(double))};
    self.function(args,dimensions,steps,self.data);
    for(std::size_t i=0;i<length;++i)for(std::size_t d=0;d<n;++d){const auto v=values[i*n+n-1-d];if(!std::isfinite(v)){valid.store(false,std::memory_order_relaxed);return;}
     if(single)static_cast<float*>(output)[d*count+position+i]=float(v);else static_cast<double*>(output)[d*count+position+i]=v;}
    position+=length;
   }
  });return valid.load(std::memory_order_relaxed);
 }
};
}
void bind_frangi(py::module_& module){
 if(_import_array()<0 || _import_umath()<0)throw py::error_already_set();
 auto eigen=std::make_shared<NumpyFrangiEigen>();
 auto exponential=std::make_shared<NumpyFrangiExp>();
 auto float_exponential=std::make_shared<NumpyFrangiFloatExp>();
 module.def("frangi_float_exp_uses_numpy",[float_exponential](){return float_exponential->required;});
 module.def("frangi_has_fma",&skimagexx::detail::frangi_fma_available);
 module.def("frangi_exp_values_portable",[](py::array_t<float,py::array::c_style|py::array::forcecast> values){
  std::vector<py::ssize_t> shape(values.shape(),values.shape()+values.ndim());py::array_t<float> output(shape);
  {py::gil_scoped_release release;for(py::ssize_t i=0;i<values.size();++i)output.mutable_data()[i]=skimagexx::detail::frangi_exp(values.data()[i]);}
  return output;
 });
 module.def("frangi_exp_values_double",[exponential](py::array_t<double,py::array::c_style|py::array::forcecast> values){
  std::vector<py::ssize_t> shape(values.shape(),values.shape()+values.ndim());py::array_t<double> output(shape);
  {py::gil_scoped_release release;exponential->provider.evaluate(values.data(),output.mutable_data(),values.size(),exponential->provider.context);}
  return output;
 });
 module.def("frangi_exp_values_cpp",[](py::array_t<float,py::array::c_style|py::array::forcecast> values){
  std::vector<py::ssize_t> shape(values.shape(),values.shape()+values.ndim());py::array_t<float> output(shape);
  {py::gil_scoped_release release;skimagexx::detail::frangi_exp_values_into(values.data(),output.mutable_data(),values.size());}
  return output;
 });
 module.def("frangi_exp_values",[float_exponential](py::array_t<float,py::array::c_style|py::array::forcecast> values){
  std::vector<py::ssize_t> shape(values.shape(),values.shape()+values.ndim());py::array_t<float> output(shape);
  {py::gil_scoped_release release;
   if(float_exponential->required)float_exponential->provider.evaluate(values.data(),output.mutable_data(),values.size(),float_exponential->provider.context);
   else skimagexx::detail::frangi_exp_values_into(values.data(),output.mutable_data(),values.size());}
  return output;
 });
 module.def("frangi_accumulate",[eigen,exponential,float_exponential](py::list fields,py::object previous,py::dtype dtype,double alpha_den,double beta_den,py::object gamma_den,bool alpha_single,bool beta_single,bool gamma_single,bool gamma_compute_single)->py::object{
  std::vector<py::array> storage;std::vector<const void*> data;std::vector<py::ssize_t> shape;py::dtype field_dtype;
  for(auto entry:fields){auto a=py::array::ensure(entry,py::array::c_style);
   if(!a || !a.dtype().attr("isnative").cast<bool>() || a.dtype().kind()!='f' || (a.itemsize()!=4 && a.itemsize()!=8) || a.ndim()<2 || a.ndim()>3)throw py::value_error("native Frangi Hessian fields required");
   if(storage.empty()){shape.assign(a.shape(),a.shape()+a.ndim());field_dtype=a.dtype();}
   else if(!a.dtype().is(field_dtype) || std::vector<py::ssize_t>(a.shape(),a.shape()+a.ndim())!=shape)throw py::value_error("homogeneous Frangi fields required");
   if(!a.attr("flags").attr("aligned").cast<bool>())a=a.attr("copy")().cast<py::array>();storage.push_back(a);data.push_back(storage.back().data());
  }
  const auto n=shape.size();if(n<2 || fields.size()!=n*(n+1)/2 || dtype.kind()!='f' || dtype.itemsize()!=field_dtype.itemsize() || !dtype.attr("isnative").cast<bool>())throw py::value_error("invalid Frangi output plan");
  py::array prior;const void* old=nullptr;
  if(!previous.is_none()){prior=py::array::ensure(previous,py::array::c_style);if(!prior || prior.dtype().kind()!='f' || prior.itemsize()!=dtype.itemsize() || !prior.dtype().attr("isnative").cast<bool>() || std::vector<py::ssize_t>(prior.shape(),prior.shape()+prior.ndim())!=shape)throw py::value_error("invalid Frangi previous output");
   if(!prior.attr("flags").attr("aligned").cast<bool>())prior=prior.attr("copy")().cast<py::array>();old=prior.data();}
  skimagexx::FrangiParameters p;p.alpha_den=alpha_den;p.beta_den=beta_den;p.alpha_single=alpha_single;p.beta_single=beta_single;p.gamma_single=gamma_single;p.gamma_compute_single=gamma_compute_single;p.automatic_gamma=gamma_den.is_none();if(!p.automatic_gamma)p.gamma_den=gamma_den.cast<double>();
  if(n==3)p.eigen_provider=&eigen->provider;
  if(dtype.itemsize()==8)p.exp_provider=&exponential->provider;
  else if(float_exponential->required)p.float_exp_provider=&float_exponential->provider;
  py::array output(dtype,shape);bool valid;
  {py::gil_scoped_release release;valid=skimagexx::frangi_accumulate_into(data.data(),old,output.mutable_data(),output.size(),n,dtype.itemsize()==4,p);}
  return valid ? py::object(py::make_tuple(output,p.gamma_den)) : py::none();
 });
 module.def("ridge_hessian_finish",[](py::array output){
  if(output.dtype().kind()!='f' || (output.itemsize()!=4 && output.itemsize()!=8) || !output.dtype().attr("isnative").cast<bool>() || !output.writeable() || !output.attr("flags").attr("aligned").cast<bool>())throw py::value_error("writable native ridge Hessian response required");
  std::vector<std::pair<py::ssize_t,py::ssize_t>> axes;
  for(py::ssize_t axis=0;axis<output.ndim();++axis)if(output.shape(axis)>1)axes.emplace_back(output.strides(axis),output.shape(axis));
  std::sort(axes.begin(),axes.end());py::ssize_t stride=output.itemsize();
  for(const auto& axis:axes){if(axis.first!=stride)throw py::value_error("dense ridge Hessian response required");stride*=axis.second;}
  {py::gil_scoped_release release;skimagexx::ridge_hessian_finish_into(output.mutable_data(),output.size(),output.itemsize()==4);}
  return output;
 });
}
