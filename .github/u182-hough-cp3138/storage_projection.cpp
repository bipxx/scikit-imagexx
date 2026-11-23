#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace {
void check_storage(const py::array& input){
    auto current=input;std::vector<py::array> chain;py::object immutable_root;
    for(unsigned i=0;i<64;++i){
        if(std::any_of(chain.begin(),chain.end(),[&](auto x){return x.ptr()==current.ptr();}))throw py::value_error("FFT owner cycle");
        chain.push_back(current);auto base=current.base();
        if(!base||base.is_none())break;
        if(!py::isinstance<py::array>(base)){
            if(PyBytes_CheckExact(base.ptr())){immutable_root=base;break;}
            throw py::value_error("FFT requires ndarray-backed storage");
        }
        current=py::reinterpret_borrow<py::array>(base);
    }
    const auto base=current.base();
    if(!immutable_root && ((base&&!base.is_none())||!current.owndata()||!(current.flags()&(py::array::c_style|py::array::f_style))))throw py::value_error("FFT requires bounded contiguous root");
    const auto limit=std::numeric_limits<std::uintptr_t>::max();
    // Both storage kinds are retained by the ndarray base chain; the local
    // strong bytes reference also keeps its immutable span live during checks.
    const auto root=immutable_root?reinterpret_cast<std::uintptr_t>(PyBytes_AS_STRING(immutable_root.ptr())):reinterpret_cast<std::uintptr_t>(current.data());
    const auto bytes=immutable_root?PyBytes_GET_SIZE(immutable_root.ptr()):current.nbytes();
    if(bytes<0||std::size_t(bytes)>limit-root)throw py::value_error("FFT root overflows");
    for(const auto& a:chain){
        if(a.size()==0)continue;
        std::uintptr_t low=0,high=0;
        for(py::ssize_t d=0;d<a.ndim();++d){
            if(a.shape(d)<0)throw py::value_error("negative FFT extent");
            const auto stride=a.strides(d);
            const std::uintptr_t mag=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
            auto& extent=stride<0?low:high;const auto count=std::uintptr_t(a.shape(d)-1);
            if(count&&mag>(limit-extent)/count)throw py::value_error("FFT extent overflows");extent+=count*mag;
        }
        const auto data=reinterpret_cast<std::uintptr_t>(a.data());
        if(low>data||high>limit-data||std::size_t(a.itemsize())>limit-data-high||data-low<root||data+high+a.itemsize()>root+bytes)throw py::value_error("FFT escapes owner allocation");
    }
}
}

namespace skimagexx::bindings {
void check_array_storage(const py::array& value) { check_storage(value); }
}
