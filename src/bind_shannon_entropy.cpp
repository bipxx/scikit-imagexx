#include "shannon_entropy.hpp"
#include "skimagexx/histogram_matching.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <cstdint>
#include <vector>
namespace py=pybind11;
namespace {
template<class T>using Array=py::array_t<T,py::array::c_style|py::array::forcecast>;
template<class T>py::object counts(const py::array& input,skimagexx::PixelType type,int workers){
    skimagexx::bindings::check_array_storage(input);
    Array<T> prepared(input);
    if(reinterpret_cast<std::uintptr_t>(prepared.data())%alignof(T)){
        const std::vector<py::ssize_t> shape(prepared.shape(),prepared.shape()+prepared.ndim());
        py::array_t<T> aligned(shape);
        skimagexx::bindings::check_array_storage(prepared);
        if(!Array<T>::check_(prepared)||std::vector<py::ssize_t>(prepared.shape(),prepared.shape()+prepared.ndim())!=shape)throw py::value_error("entropy input changed during allocation");
        std::memcpy(aligned.mutable_data(),prepared.data(),std::size_t(prepared.nbytes()));prepared=Array<T>(aligned);
    }
    // Last allocation/conversion boundary precedes the owned count scan.
    skimagexx::bindings::check_array_storage(prepared);
    if(!Array<T>::check_(prepared))throw py::value_error("entropy requires prepared native storage");
    const auto* data=prepared.data();const auto size=std::size_t(prepared.size());
    std::vector<std::size_t> values;
    {py::gil_scoped_release release;values=skimagexx::histogram_sorted_counts(data,size,type,workers);}
    py::array_t<py::ssize_t> output(values.size());
    for(std::size_t i=0;i<values.size();++i)output.mutable_data()[i]=py::ssize_t(values[i]);
    return output;
}
}
void bind_shannon_entropy(py::module_& module){
    module.def("shannon_counts",[](const py::array& input,int workers)->py::object{
        using P=skimagexx::PixelType;
        const auto c=input.dtype().char_();
        if(c=='e')return counts<float>(input,P::float32,workers);
#define TYPE(C,T,PX) if(c==C)return counts<T>(input,P::PX,workers);
        TYPE('?',bool,boolean) TYPE('b',std::int8_t,int8) TYPE('B',std::uint8_t,uint8)
        TYPE('h',std::int16_t,int16) TYPE('H',std::uint16_t,uint16)
        TYPE('i',std::int32_t,int32) TYPE('I',std::uint32_t,uint32)
#undef TYPE
        if(c=='l')return counts<long>(input,sizeof(long)==8?P::int64:P::int32,workers);
        if(c=='L')return counts<unsigned long>(input,sizeof(long)==8?P::uint64:P::uint32,workers);
        if(c=='q')return counts<long long>(input,P::int64,workers);
        if(c=='Q')return counts<unsigned long long>(input,P::uint64,workers);
        if(c=='f')return counts<float>(input,P::float32,workers);
        if(c=='d')return counts<double>(input,P::float64,workers);
        return py::none();
    },py::arg("image").noconvert(),py::arg("workers"));
    module.def("shannon_entropy_counts",[](const py::array& input,int workers){
        if(input.ndim()!=1||!Array<py::ssize_t>::check_(input)||reinterpret_cast<std::uintptr_t>(input.data())%alignof(py::ssize_t))
            throw py::value_error("entropy requires native contiguous intp counts");
        skimagexx::bindings::check_array_storage(input);
        const auto* data=static_cast<const py::ssize_t*>(input.data());const auto size=std::size_t(input.size());
        py::gil_scoped_release release;return skimagexx::detail::shannon_entropy_counts(data,size,workers);
    },py::arg("counts").noconvert(),py::arg("workers"));
}
