#include "map_array.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace py=pybind11;
namespace {
bool integer(const py::array& array) {
    const auto kind=array.dtype().kind();const auto width=array.itemsize();
    return (kind=='i'||kind=='u') && (width==1||width==2||width==4||width==8);
}
bool numeric(const py::array& array) {
    const auto kind=array.dtype().kind();const auto width=array.itemsize();
    return integer(array)||(kind=='b'&&width==1)||(kind=='f'&&(width==4||width==8))
        ||(kind=='c'&&(width==8||width==16));
}
template<class T>std::vector<std::uint64_t> words(const py::array& array) {
    std::vector<std::uint64_t> output(std::size_t(array.size()));
    const auto* data=static_cast<const char*>(array.data());const auto stride=array.strides(0);
    for(std::size_t i=0;i<output.size();++i) {
        T value;std::memcpy(&value,data+std::ptrdiff_t(i)*stride,sizeof(T));output[i]=std::uint64_t(value);
    }
    return output;
}
std::vector<std::uint64_t> words(const py::array& array) {
    if(array.dtype().kind()=='i')switch(array.itemsize()) {
        case 1:return words<std::int8_t>(array);case 2:return words<std::int16_t>(array);
        case 4:return words<std::int32_t>(array);case 8:return words<std::int64_t>(array);
    }
    else switch(array.itemsize()) {
        case 1:return words<std::uint8_t>(array);case 2:return words<std::uint16_t>(array);
        case 4:return words<std::uint32_t>(array);case 8:return words<std::uint64_t>(array);
    }
    throw py::type_error("Function call with ambiguous argument types");
}
}

void bind_map_array(py::module_& module) {
    module.def("map_array_apply",[](const py::array& indices,const py::array& output,
               const py::array& keys,const py::array& values,int workers) {
        // The complete literal Python prelude has finished before this boundary.
        // Match fused dtype admission first, then source argument buffer order.
        if(!integer(indices)||!integer(keys)||!numeric(output)||!numeric(values))
            throw py::type_error("Function call with ambiguous argument types");
        for(const auto* array:{&indices,&output,&keys,&values}) {
            if(array->ndim()!=1)throw py::value_error("Buffer has wrong number of dimensions (expected 1, got "+std::to_string(array->ndim())+")");
            if(!array->writeable())throw py::value_error("buffer source array is read-only");
            if(!array->dtype().attr("isnative").cast<bool>())
                throw py::value_error("Big-endian buffer not supported on little-endian compiler");
            skimagexx::bindings::check_array_storage(*array);
        }
        if(indices.size()!=output.size()||values.size()<keys.size())
            throw py::value_error("prepared map buffers have incompatible lengths");
        if(!indices.dtype().equal(keys.dtype())||!output.dtype().equal(values.dtype()))
            throw py::type_error("Function call with ambiguous argument types");
        if(workers<1||workers>1024)throw py::value_error("map worker snapshot outside [1,1024]");
        const auto stride=output.strides(0),width=output.itemsize();
        if(output.size()>1 && stride>-width && stride<width)
            throw py::value_error("overlapping map output elements are outside native coverage");
        const auto input=words(indices),mapping=words(keys);
        const skimagexx::detail::MapArrayInput plan{input.data(),input.size(),mapping.data(),values.data(),
            mapping.size(),std::size_t(width),unsigned(indices.itemsize()*8),values.strides(0)};
        const skimagexx::detail::MapArrayOutput target{const_cast<void*>(output.data()),stride};
        {
            py::gil_scoped_release release;
            skimagexx::detail::map_array_into(plan,target,workers);
        }
        return output;
    },py::arg("indices").noconvert(),py::arg("output").noconvert(),py::arg("keys").noconvert(),
       py::arg("values").noconvert(),py::arg("workers"));
}
