#include <skimagexx/regionprops_state.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <memory>

namespace py=pybind11;
namespace {
struct Prepared {
    py::array owner;
    skimagexx::detail::RegionPropsState state;
};
bool descriptor(const py::array& a,skimagexx::ComponentsView& v) {
    if((a.ndim()!=2 && a.ndim()!=3) || !a.dtype().attr("isnative").cast<bool>())return false;
    const auto kind=a.dtype().kind();const auto bytes=a.itemsize();
    if((kind!='i' && kind!='u') || (bytes!=1 && bytes!=2 && bytes!=4 && bytes!=8))return false;
    using P=skimagexx::PixelType;
    if(kind=='i')v.type=bytes==1?P::int8:bytes==2?P::int16:bytes==4?P::int32:P::int64;
    else v.type=bytes==1?P::uint8:bytes==2?P::uint16:bytes==4?P::uint32:P::uint64;
    v.data=a.data();
    for(py::ssize_t i=0;i<a.ndim();++i){v.shape.push_back(std::size_t(a.shape(i)));v.strides.push_back(a.strides(i));}
    return true;
}
bool unchanged(const Prepared& p,const py::array& a) {
    skimagexx::ComponentsView current;
    return descriptor(a,current) && current.data==p.state.labels.data && current.type==p.state.labels.type
        && current.shape==p.state.labels.shape && current.strides==p.state.labels.strides;
}
bool region_shape(const Prepared& p,std::size_t index,const py::array& a) {
    if(index>=p.state.regions.size() || std::size_t(a.ndim())!=p.state.labels.shape.size())return false;
    const auto& r=p.state.regions[index];
    for(py::ssize_t d=0;d<a.ndim();++d)if(std::size_t(a.shape(d))!=r.stop[d]-r.start[d])return false;
    return true;
}
bool bool_storage(const py::array& a) {
    return a.dtype().kind()=='b' && a.itemsize()==1
        && (a.flags()&(py::array::c_style|py::array::f_style));
}
std::array<std::ptrdiff_t,3> strides(const py::array& a) {
    std::array<std::ptrdiff_t,3> result{};
    for(py::ssize_t d=0;d<a.ndim();++d)result[std::size_t(d)]=a.strides(d);
    return result;
}
}

void bind_regionprops_state(py::module_& m) {
    py::class_<Prepared,std::shared_ptr<Prepared>>(m,"_RegionPropsState");
    m.def("regionprops_state_prepare",[](const py::array& a)->py::object {
        auto p=std::make_shared<Prepared>();
        skimagexx::ComponentsView view;if(!descriptor(a,view))return py::none();
        skimagexx::bindings::check_array_storage(a);p->owner=a;
        bool ok;{py::gil_scoped_release release;ok=skimagexx::detail::prepare_regionprops_state(view,p->state);}
        if(!ok)return py::none();
        py::list geometry;
        for(const auto& r:p->state.regions) {
            py::tuple start(view.shape.size()),stop(view.shape.size());
            for(std::size_t d=0;d<view.shape.size();++d){start[d]=py::int_(r.start[d]);stop[d]=py::int_(r.stop[d]);}
            geometry.append(py::make_tuple(py::int_(r.label),start,stop));
        }
        return py::make_tuple(p,geometry);
    },py::arg("labels").noconvert());
    m.def("regionprops_state_image",[](const std::shared_ptr<Prepared>& p,std::size_t index,const py::array& labels,py::array output)->py::object {
        if(!unchanged(*p,labels) || !region_shape(*p,index,output) || !bool_storage(output) || !output.owndata() || !output.writeable())return py::none();
        skimagexx::bindings::check_array_storage(labels);skimagexx::bindings::check_array_storage(output);
        auto* data=static_cast<std::uint8_t*>(output.mutable_data());const auto steps=strides(output);
        {py::gil_scoped_release release;skimagexx::detail::regionprops_mask_strided_into(p->state,index,data,steps);}
        return py::bool_(true);
    },py::arg("state"),py::arg("index"),py::arg("labels").noconvert(),py::arg("output").noconvert());
    m.def("regionprops_state_population",[](const py::array& a)->py::object {
        if(!bool_storage(a))return py::none();skimagexx::bindings::check_array_storage(a);
        std::size_t count;{py::gil_scoped_release release;count=skimagexx::detail::regionprops_mask_population(static_cast<const std::uint8_t*>(a.data()),std::size_t(a.size()));}
        return py::int_(count);
    },py::arg("mask").noconvert());
    m.def("regionprops_state_coords",[](const std::shared_ptr<Prepared>& p,std::size_t index,const py::array& labels,const py::array& mask,py::array output)->py::object {
        if(!unchanged(*p,labels) || !region_shape(*p,index,mask) || !bool_storage(mask)
           || output.ndim()!=2 || std::size_t(output.shape(1))!=p->state.labels.shape.size()
           || !output.dtype().is(py::dtype::of<std::intptr_t>()) || !output.owndata() || !output.writeable()
           || !(output.flags()&(py::array::c_style|py::array::f_style)))return py::none();
        skimagexx::bindings::check_array_storage(labels);skimagexx::bindings::check_array_storage(mask);skimagexx::bindings::check_array_storage(output);
        const auto* data=static_cast<const std::uint8_t*>(mask.data());
        const auto count=skimagexx::detail::regionprops_mask_population(data,std::size_t(mask.size()));
        if(std::size_t(output.shape(0))!=count)return py::none();
        const auto steps=strides(mask);auto* coordinates=static_cast<std::intptr_t*>(output.mutable_data());
        const auto point_step=output.strides(0),dimension_step=output.strides(1);
        {py::gil_scoped_release release;skimagexx::detail::regionprops_coords_strided_into(p->state,index,data,steps,
            coordinates,point_step,dimension_step);}
        return py::bool_(true);
    },py::arg("state"),py::arg("index"),py::arg("labels").noconvert(),py::arg("mask").noconvert(),py::arg("output").noconvert());
}
