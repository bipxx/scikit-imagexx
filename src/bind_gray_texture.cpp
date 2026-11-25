#include <skimagexx/gray_texture.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
namespace py=pybind11;
namespace {
void check_array(const py::array& array) {
    if (Py_TYPE(array.ptr()) != py::detail::npy_api::get().PyArray_Type_)
        throw py::type_error("gray texture requires a plain ndarray");
    if (!array.owndata())
        skimagexx::bindings::check_array_storage(array);
}

void check_double(const py::array& array) {
    check_array(array);
    const auto dtype = array.dtype();
    if (!dtype.is(py::dtype::of<double>()) &&
        (dtype.char_() != 'd' || !dtype.equal(py::dtype::of<double>()) ||
         !dtype.attr("metadata").is_none()))
        throw py::type_error("gray texture requires an untagged native float64 array");
}
}

void bind_gray_texture(py::module_& m){
 m.def("graycomatrix_apply",[](const py::array& image,const py::array_t<double,py::array::c_style>& distances,const py::array_t<double,py::array::c_style>& angles,std::size_t levels,bool symmetric,bool normed,int workers)->py::object{
  check_array(image);check_double(distances);check_double(angles);
  // Stock consumes the integer storage of tagged images and returns an untagged matrix.
  if(image.ndim()!=2||distances.ndim()!=1||angles.ndim()!=1)throw py::value_error("graycomatrix native shape");
  const auto kind=image.dtype().kind();const auto item=image.itemsize();skimagexx::GrayIntegerType type;
  if(kind=='b'&&item==1)type=skimagexx::GrayIntegerType::u8;
  else if(kind=='i'&&item==1)type=skimagexx::GrayIntegerType::i8;else if(kind=='u'&&item==1)type=skimagexx::GrayIntegerType::u8;
  else if(kind=='i'&&item==2)type=skimagexx::GrayIntegerType::i16;else if(kind=='u'&&item==2)type=skimagexx::GrayIntegerType::u16;
  else if(kind=='i'&&item==4)type=skimagexx::GrayIntegerType::i32;else if(kind=='u'&&item==4)type=skimagexx::GrayIntegerType::u32;
  else if(kind=='i'&&item==8)type=skimagexx::GrayIntegerType::i64;else if(kind=='u'&&item==8)type=skimagexx::GrayIntegerType::u64;
  else throw py::type_error("No matching signature found");
  skimagexx::GrayImageView view{image.data(),std::size_t(image.shape(0)),std::size_t(image.shape(1)),image.strides(0),image.strides(1),type,!py::cast<bool>(image.dtype().attr("isnative")),image.writeable()};
  std::vector<double>d(distances.data(),distances.data()+distances.size()),a(angles.data(),angles.data()+angles.size());
  skimagexx::GrayMatrix result;{py::gil_scoped_release release;result=skimagexx::graycomatrix(view,d,a,levels,symmetric,normed,workers);}
  std::vector<py::ssize_t>shape{py::ssize_t(levels),distances.size(),angles.size()};shape.insert(shape.begin(),py::ssize_t(levels));
  if(normed){py::array_t<double>out(shape);std::memcpy(out.mutable_data(),result.probabilities.data(),result.probabilities.size()*8);return out;}
  py::array_t<std::uint32_t>out(shape);std::memcpy(out.mutable_data(),result.counts.data(),result.counts.size()*4);return out;
 },py::arg("image").noconvert(),py::arg("distances").noconvert(),py::arg("angles").noconvert(),py::arg("levels"),py::arg("symmetric"),py::arg("normed"),py::arg("workers")=0);
 m.def("graycoprops_apply",[](const py::array& P,py::object prop,int workers,py::object policy){
  const auto dtype = P.dtype();
  const bool native_counts = dtype.is(py::dtype::of<std::uint32_t>()) ||
      (dtype.equal(py::dtype::of<std::uint32_t>()) && dtype.attr("metadata").is_none());
  if(native_counts)check_array(P);else check_double(P);
  if(P.ndim()!=4||P.shape(0)!=P.shape(1))throw py::value_error("graycoprops native shape");
  const auto name=py::isinstance<py::str>(prop)?py::cast<std::string>(prop):std::string{};skimagexx::GrayProperty property=skimagexx::GrayProperty::invalid;
#define PROP(S,E) if(name==S)property=skimagexx::GrayProperty::E;else
  PROP("contrast",contrast)PROP("dissimilarity",dissimilarity)PROP("homogeneity",homogeneity)PROP("ASM",ASM)PROP("energy",energy)PROP("mean",mean)PROP("variance",variance)PROP("std",stddev)PROP("entropy",entropy)PROP("correlation",correlation)property=skimagexx::GrayProperty::invalid;
#undef PROP
  skimagexx::GrayProbabilityView view{P.data(),std::size_t(P.shape(0)),std::size_t(P.shape(2)),std::size_t(P.shape(3)),{P.strides(0),P.strides(1),P.strides(2),P.strides(3)},
      native_counts ? skimagexx::GrayNumericType::u32 : skimagexx::GrayNumericType::f64};
  skimagexx::GrayProperties result;try{py::gil_scoped_release release;result=skimagexx::graycoprops(view,property,workers,[](const char* operation,unsigned flags,void* context){auto* callback=static_cast<py::object*>(context);if(!callback->is_none()){py::gil_scoped_acquire acquire;(*callback)(operation,flags);}},&policy);}catch(const std::domain_error&){throw py::value_error(py::cast<std::string>(py::str(prop))+" is an invalid property");}
  std::vector<py::ssize_t>shape{py::ssize_t(result.distances),py::ssize_t(result.angles)};std::vector<py::ssize_t>strides=result.fortran?std::vector<py::ssize_t>{8,py::ssize_t(result.distances*8)}:std::vector<py::ssize_t>{py::ssize_t(result.angles*8),8};py::array_t<double>out(shape,strides);for(std::size_t d=0;d<result.distances;++d)for(std::size_t a=0;a<result.angles;++a)std::memcpy(reinterpret_cast<char*>(out.mutable_data())+d*strides[0]+a*strides[1],&result.values[d*result.angles+a],8);return out;
 },py::arg("P").noconvert(),py::arg("prop"),py::arg("workers")=0,py::arg("policy")=py::none());
}
