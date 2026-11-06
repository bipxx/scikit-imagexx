#include "skimagexx/find_boundaries.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
namespace py=pybind11;
namespace {
bool exact_array(const py::array& a) {return Py_TYPE(a.ptr())==py::detail::npy_api::get().PyArray_Type_;}
bool descriptor(const py::array& a,skimagexx::ComponentsView& view) {
    if(!exact_array(a)||a.ndim()<1||a.ndim()>8||!a.dtype().attr("isnative").cast<bool>())return false;
    const auto kind=a.dtype().kind();const auto bytes=a.itemsize();using P=skimagexx::PixelType;
    if(kind=='b'&&bytes==1)view.type=P::boolean;
    else if(kind=='i')switch(bytes){case 1:view.type=P::int8;break;case 2:view.type=P::int16;break;case 4:view.type=P::int32;break;case 8:view.type=P::int64;break;default:return false;}
    else if(kind=='u')switch(bytes){case 1:view.type=P::uint8;break;case 2:view.type=P::uint16;break;case 4:view.type=P::uint32;break;case 8:view.type=P::uint64;break;default:return false;}
    else return false;
    view.data=a.data();
    for(py::ssize_t d=0;d<a.ndim();++d){if(a.shape(d)<=0)return false;view.shape.push_back(std::size_t(a.shape(d)));view.strides.push_back(a.strides(d));}
    return true;
}
std::pair<std::uintptr_t,std::uintptr_t> span(const py::array& a) {
    const auto limit=std::numeric_limits<std::uintptr_t>::max();std::uintptr_t low=0,high=0;
    for(py::ssize_t d=0;d<a.ndim();++d){
        auto stride=a.strides(d);auto magnitude=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
        auto& extent=stride<0?low:high;const auto count=std::uintptr_t(a.shape(d)-1);
        if(count && magnitude>(limit-extent)/count)throw py::value_error("find_boundaries array extent overflows");
        extent+=count*magnitude;
    }
    const auto address=reinterpret_cast<std::uintptr_t>(a.data());
    if(low>address||high>limit-address||std::uintptr_t(a.itemsize())>limit-address-high)throw py::value_error("find_boundaries array address overflows");
    return {address-low,address+high+a.itemsize()};
}
void storage(const py::array& a) {
    if(!a.owndata()){skimagexx::bindings::check_array_storage(a);return;}
    // Owning dense K-order arrays can be neither C nor F in rank3+. Their
    // explicit allocation size supplies the ordinary local capacity proof.
    const auto bounds=span(a);auto begin=reinterpret_cast<std::uintptr_t>(a.data());
    if(a.nbytes()<0||std::uintptr_t(a.nbytes())>std::numeric_limits<std::uintptr_t>::max()-begin||bounds.first<begin||bounds.second>begin+a.nbytes())
        throw py::value_error("find_boundaries array escapes owned allocation");
}
bool dense_output(const py::array& a,const std::vector<std::size_t>& shape) {
    if(!exact_array(a)||!a.owndata()||!a.writeable()||a.dtype().kind()!='b'||a.itemsize()!=1||std::size_t(a.ndim())!=shape.size())return false;
    std::vector<std::pair<std::ptrdiff_t,std::size_t>> axes;
    for(std::size_t d=0;d<shape.size();++d){if(std::size_t(a.shape(d))!=shape[d])return false;if(shape[d]>1)axes.emplace_back(a.strides(d),shape[d]);}
    std::sort(axes.begin(),axes.end());std::size_t expected=1;
    for(const auto& axis:axes){if(axis.first!=std::ptrdiff_t(expected))return false;expected*=axis.second;}
    return std::size_t(a.nbytes())==expected;
}
}
void bind_find_boundaries(py::module_& module) {
    module.def("find_boundaries_into",[](const py::array& input,py::array output,
        unsigned connectivity,int mode,std::uint64_t background,bool representable,int workers)->py::object {
        skimagexx::ComponentsView view;if(!descriptor(input,view))return py::none();
        skimagexx::detail::FindBoundaryParameters parameters;
        if(mode<0||mode>3)return py::none();
        parameters.mode=static_cast<skimagexx::detail::FindBoundaryMode>(mode);parameters.connectivity=connectivity;
        parameters.background_bits=background;parameters.background_representable=representable;
        std::vector<std::size_t> shape;
        if(!skimagexx::detail::find_boundaries_shape(view,parameters.mode,shape)||!dense_output(output,shape))return py::none();
        // Output allocation occurred before this new boundary. Read and check
        // CURRENT descriptors/storage once, immediately before releasing GIL.
        storage(input);storage(output);const auto a=span(input),b=span(output);
        if(a.first<b.second&&b.first<a.second)return py::none();
        std::vector<std::ptrdiff_t> strides(output.strides(),output.strides()+output.ndim());bool accepted;
        auto* destination=static_cast<std::uint8_t*>(output.mutable_data());
        {py::gil_scoped_release release;accepted=skimagexx::detail::find_boundaries_into(view,parameters,destination,strides,workers);}
        return accepted?py::object(py::bool_(true)):py::object(py::none());
    },py::arg("input").noconvert(),py::arg("output").noconvert(),py::arg("connectivity").noconvert(),py::arg("mode").noconvert(),py::arg("background_bits").noconvert(),py::arg("background_representable").noconvert(),py::arg("workers").noconvert()=0);
}
