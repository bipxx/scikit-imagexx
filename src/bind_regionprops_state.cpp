#include <skimagexx/regionprops_state.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <array>
#include <stdexcept>
#include <memory>
#include <vector>
#include <limits>

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
bool scalar_f64(const py::object& value,double& output) {
    auto& api=py::detail::npy_api::get();
    static const py::handle type=api.PyArray_TypeObjectFromType_(py::detail::npy_api::NPY_DOUBLE_);
    if(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr()))!=type.ptr())return false;
    api.PyArray_ScalarAsCtype_(value.ptr(),&output);
    if(PyErr_Occurred())throw py::error_already_set();
    return true;
}
py::object scalar_value(const skimagexx::detail::RegionPropsScalar64& result) {
    // PyArray_Scalar copies the exact word into a NumPy scalar. No temporary
    // 0D ndarray, Python __getitem__, float conversion, or NaN canonicalization.
    auto bits=result.bits;
    auto dtype=py::dtype::of<double>();
    auto* value=py::detail::npy_api::get().PyArray_Scalar_(&bits,dtype.ptr(),nullptr);
    if(!value)throw py::error_already_set();
    return py::reinterpret_steal<py::object>(value);
}
py::object scalar_result(const skimagexx::detail::RegionPropsScalar64& result) {
    return py::make_tuple(scalar_value(result),result.fp_status);
}
bool exact_array(const py::handle& value) {
    return Py_TYPE(value.ptr())==py::detail::npy_api::get().PyArray_Type_;
}
bool scaled_workers(const py::handle& value,int& workers) {
    if(!PyLong_CheckExact(value.ptr()))return false;
    const long n=PyLong_AsLong(value.ptr());
    if(PyErr_Occurred()){PyErr_Clear();return false;}
    if(n<1 || n>std::numeric_limits<int>::max())return false;
    workers=int(n);return true;
}
bool scaled_matrix(const py::array& a,int number,bool writable) {
    return a.ndim()==2 && (a.shape(1)==2 || a.shape(1)==3)
        && std::size_t(a.shape(0))<=skimagexx::detail::regionprops_coords_scaled_max_elements
        && a.dtype().num()==number && a.dtype().attr("isnative").cast<bool>()
        && (a.flags()&py::array::f_style)
        && reinterpret_cast<std::uintptr_t>(a.data())%std::size_t(a.itemsize())==0
        && (!writable || (a.owndata() && a.writeable()));
}
py::array scaled_output(std::size_t count,std::size_t ndim,const py::dtype& dtype) {
    const auto n=py::ssize_t(count),d=py::ssize_t(ndim),bytes=dtype.itemsize();
    return py::array(dtype,std::vector<py::ssize_t>{n,d},
        std::vector<py::ssize_t>{n?bytes:0,n?n*bytes:0});
}
bool scaled_shape(const py::handle& value,std::size_t ndim,std::array<std::size_t,3>& shape) {
    if(!PyTuple_CheckExact(value.ptr()) || std::size_t(PyTuple_GET_SIZE(value.ptr()))!=ndim)return false;
    for(std::size_t d=0;d<ndim;++d) {
        auto* item=PyTuple_GET_ITEM(value.ptr(),py::ssize_t(d));
        if(!PyLong_CheckExact(item))return false;
        shape[d]=PyLong_AsSize_t(item);
        if(PyErr_Occurred()){PyErr_Clear();return false;}
    }
    return true;
}
bool scaled_disjoint(const py::array& values,const py::array& coefficient) {
    if(!values.size())return true;
    const auto first=reinterpret_cast<std::uintptr_t>(values.data());
    const auto last=first+std::size_t(values.nbytes());
    for(py::ssize_t d=0;d<coefficient.size();++d) {
        const auto p=reinterpret_cast<std::uintptr_t>(static_cast<const char*>(coefficient.data())+d*coefficient.strides(0));
        if(p<last && first<p+std::size_t(coefficient.itemsize()))return false;
    }
    return true;
}
bool scaled_coefficients(const py::handle& value,std::size_t ndim,bool spacing,std::array<double,3>& result) {
    if(!exact_array(value))return false;
    const auto a=py::reinterpret_borrow<py::array>(value);
    if(a.ndim()!=1 || std::size_t(a.size())!=ndim || !a.dtype().attr("isnative").cast<bool>()
       || !a.dtype().attr("metadata").is_none())return false;
    const int number=a.dtype().num();
    const bool f64=number==py::dtype::of<double>().num();
    const bool f32=number==py::dtype::of<float>().num();
    const bool integer=!spacing && number==py::dtype::of<std::intptr_t>().num();
    if(!f64 && !f32 && !integer)return false;
    skimagexx::bindings::check_array_storage(a);
    for(std::size_t d=0;d<ndim;++d) {
        const auto* p=static_cast<const char*>(a.data())+py::ssize_t(d)*a.strides(0);
        if(f64)std::memcpy(&result[d],p,sizeof(double));
        else if(f32) {
            std::uint32_t bits;std::memcpy(&bits,p,sizeof(bits));
            const auto exponent=(bits>>23)&255u;
            if(exponent==255u || (!exponent && (bits&0x7fffffu)))return false;
            float input;std::memcpy(&input,p,sizeof(input));result[d]=double(input);
        } else {
            std::intptr_t input;std::memcpy(&input,p,sizeof(input));
            // Conversion is exact and cannot set an inexact FP flag before refusal.
            if(input<-(std::int64_t(1)<<53) || input>(std::int64_t(1)<<53))return false;
            result[d]=double(input);
        }
    }
    return true;
}
}

void bind_regionprops_state(py::module_& m) {
    m.def("regionprops_state_coords_scaled_local",[](const py::object& value,const py::object& budget)->py::object {
        int workers;if(!exact_array(value) || !scaled_workers(budget,workers))return py::none();
        const auto mask=py::reinterpret_borrow<py::array>(value);
        if((mask.ndim()!=2 && mask.ndim()!=3) || !bool_storage(mask)
           || std::size_t(mask.size())>skimagexx::detail::regionprops_coords_scaled_max_elements)return py::none();
        std::array<std::size_t,3> shape{};py::tuple snapshot(mask.ndim());
        for(py::ssize_t d=0;d<mask.ndim();++d) {
            shape[std::size_t(d)]=std::size_t(mask.shape(d));
            if(shape[std::size_t(d)]>skimagexx::detail::regionprops_coords_scaled_max_extent)return py::none();
            snapshot[d]=py::int_(shape[std::size_t(d)]);
        }
        skimagexx::bindings::check_array_storage(mask);
        const auto* data=static_cast<const std::uint8_t*>(mask.data());
        const auto ndim=std::size_t(mask.ndim()),elements=std::size_t(mask.size());
        const auto steps=strides(mask);
        std::size_t count;{py::gil_scoped_release release;count=skimagexx::detail::regionprops_mask_population(data,elements);}
        auto output=scaled_output(count,ndim,py::dtype::of<std::intptr_t>());
        auto* coordinates=static_cast<std::intptr_t*>(output.mutable_data());
        bool ok;{py::gil_scoped_release release;ok=skimagexx::detail::regionprops_coords_scaled_local_into(data,ndim,shape,steps,count,coordinates,workers);}
        if(!ok)return py::none();
        return py::make_tuple(output,snapshot);
    },py::arg("mask"),py::arg("workers"));
    m.def("regionprops_state_coords_scaled_origin_into",[](const py::object& value,const py::object& origin,const py::object& mask_shape,const py::object& budget)->py::object {
        int workers;if(!exact_array(value) || !exact_array(origin) || !scaled_workers(budget,workers))return py::none();
        auto coordinates=py::reinterpret_borrow<py::array>(value);const auto origins=py::reinterpret_borrow<py::array>(origin);
        if(!scaled_matrix(coordinates,py::dtype::of<std::intptr_t>().num(),true)
           || origins.ndim()!=1 || origins.size()!=coordinates.shape(1)
           || origins.dtype().num()!=py::dtype::of<std::intptr_t>().num()
           || !origins.dtype().attr("isnative").cast<bool>() || !origins.dtype().attr("metadata").is_none())return py::none();
        std::array<std::size_t,3> shape{};const auto ndim=std::size_t(coordinates.shape(1));
        if(!scaled_shape(mask_shape,ndim,shape))return py::none();
        skimagexx::bindings::check_array_storage(coordinates);skimagexx::bindings::check_array_storage(origins);
        if(!scaled_disjoint(coordinates,origins))return py::none();
        std::array<std::intptr_t,3> offsets{};
        for(std::size_t d=0;d<ndim;++d)std::memcpy(&offsets[d],static_cast<const char*>(origins.data())+py::ssize_t(d)*origins.strides(0),sizeof(std::intptr_t));
        auto* data=static_cast<std::intptr_t*>(coordinates.mutable_data());const auto count=std::size_t(coordinates.shape(0));
        bool ok;{py::gil_scoped_release release;ok=skimagexx::detail::regionprops_coords_scaled_origin_into(data,count,ndim,shape,offsets,workers);}
        return ok?py::object(py::bool_(true)):py::object(py::none());
    },py::arg("indices"),py::arg("origin"),py::arg("mask_shape"),py::arg("workers"));
    m.def("regionprops_state_coords_scaled_multiply_f64",[](const py::object& value,const py::object& spacing,const py::object& budget)->py::object {
        int workers;if(!exact_array(value) || !scaled_workers(budget,workers))return py::none();
        const auto coordinates=py::reinterpret_borrow<py::array>(value);
        if(!scaled_matrix(coordinates,py::dtype::of<std::intptr_t>().num(),false))return py::none();
        const auto count=std::size_t(coordinates.shape(0)),ndim=std::size_t(coordinates.shape(1));std::array<double,3> coefficients{};
        if(!scaled_coefficients(spacing,ndim,true,coefficients))return py::none();
        skimagexx::bindings::check_array_storage(coordinates);
        auto output=scaled_output(count,ndim,py::dtype::of<double>());
        const auto* data=static_cast<const std::intptr_t*>(coordinates.data());auto* result=static_cast<double*>(output.mutable_data());
        bool ok;{py::gil_scoped_release release;ok=skimagexx::detail::regionprops_coords_scaled_multiply_f64_into(data,count,ndim,coefficients,result,workers);}
        if(!ok)return py::none();return output;
    },py::arg("absolute"),py::arg("spacing"),py::arg("workers"));
    m.def("regionprops_state_coords_scaled_add_f64_into",[](const py::object& value,const py::object& offset,const py::object& budget)->py::object {
        int workers;if(!exact_array(value) || !scaled_workers(budget,workers))return py::none();
        auto values=py::reinterpret_borrow<py::array>(value);
        if(!scaled_matrix(values,py::dtype::of<double>().num(),true) || !values.dtype().attr("metadata").is_none())return py::none();
        const auto count=std::size_t(values.shape(0)),ndim=std::size_t(values.shape(1));std::array<double,3> coefficients{};
        if(!scaled_coefficients(offset,ndim,false,coefficients))return py::none();
        skimagexx::bindings::check_array_storage(values);
        if(!scaled_disjoint(values,py::reinterpret_borrow<py::array>(offset)))return py::none();
        auto* data=static_cast<double*>(values.mutable_data());
        bool ok;{py::gil_scoped_release release;ok=skimagexx::detail::regionprops_coords_scaled_add_f64_into(data,count,ndim,coefficients,workers);}
        return ok?py::object(py::bool_(true)):py::object(py::none());
    },py::arg("scaled"),py::arg("offset"),py::arg("workers"));
    m.def("regionprops_state_area_bbox_get",[](const py::object& self)->py::object {
        // Preserve the public dependency reads before deciding native arithmetic.
        py::object count=self.attr("image").attr("size");
        py::object pixel_area=self.attr("_pixel_area");
        double scale;
        if(!PyLong_CheckExact(count.ptr()) || !scalar_f64(pixel_area,scale))
            return py::make_tuple(count,pixel_area);
        const auto size=PyLong_AsSize_t(count.ptr());
        if(PyErr_Occurred()) {PyErr_Clear();return py::make_tuple(count,pixel_area);}
        skimagexx::detail::RegionPropsScalar64 result;
        try{result=skimagexx::detail::regionprops_scaled_count_f64(size,scale);}
        catch(const std::invalid_argument&){return py::make_tuple(count,pixel_area);}
        if(result.fp_status)return py::make_tuple(count,pixel_area);
        return scalar_value(result);
    },py::arg("self"));
    m.def("regionprops_state_extent_get",[](const py::object& self)->py::object {
        py::object area=self.attr("area");
        py::object bbox=self.attr("area_bbox");
        double first,denominator;
        if(!scalar_f64(area,first) || !scalar_f64(bbox,denominator))
            return py::make_tuple(area,bbox);
        skimagexx::detail::RegionPropsScalar64 result;
        try{result=skimagexx::detail::regionprops_extent_f64(first,denominator);}
        catch(const std::invalid_argument&){return py::make_tuple(area,bbox);}
        // Policy has no event to deliver for status0. Exceptional/refused
        // operands are returned once for the existing source-faithful policy tail.
        if(result.fp_status)return py::make_tuple(area,bbox);
        return scalar_value(result);
    },py::arg("self"));
    m.def("regionprops_state_scaled_count_f64",[](std::size_t count,const py::object& pixel_area)->py::object {
        double scale;if(!scalar_f64(pixel_area,scale))return py::none();
        // SDK helper owns the exported numeric/FENV boundary. Refusal is an
        // explicit primitive-provider route with already evaluated operands.
        skimagexx::detail::RegionPropsScalar64 result;
        try{result=skimagexx::detail::regionprops_scaled_count_f64(count,scale);}
        catch(const std::invalid_argument&){return py::none();}
        return scalar_result(result);
    },py::arg("count"),py::arg("pixel_area"));
    m.def("regionprops_state_extent_f64",[](const py::object& area,const py::object& bbox)->py::object {
        double first,denominator;
        if(!scalar_f64(area,first)||!scalar_f64(bbox,denominator))return py::none();
        skimagexx::detail::RegionPropsScalar64 result;
        try{result=skimagexx::detail::regionprops_extent_f64(first,denominator);}
        catch(const std::invalid_argument&){return py::none();}
        return scalar_result(result);
    },py::arg("area"),py::arg("bbox"));
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
