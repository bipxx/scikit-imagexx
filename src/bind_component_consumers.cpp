#include "skimagexx/component_consumers.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL SKIMAGEXX_COMPONENT_CONSUMERS_ARRAY_API
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <limits>
namespace py=pybind11;
using namespace skimagexx;
namespace {
PyArrayObject* array(const py::array& a){return reinterpret_cast<PyArrayObject*>(a.ptr());}
py::array checked(PyObject* result){if(!result)throw py::error_already_set();return py::reinterpret_steal<py::array>(result);}
std::vector<py::ssize_t> shape(const py::array& a){return {a.shape(),a.shape()+a.ndim()};}
ComponentsView view(const py::array& a) {
    ComponentsView v;const auto kind=a.dtype().kind();const auto size=a.itemsize();using P=PixelType;
    if(kind=='b')v.type=P::boolean;
    else if(kind=='i')switch(size){case 1:v.type=P::int8;break;case 2:v.type=P::int16;break;case 4:v.type=P::int32;break;case 8:v.type=P::int64;break;default:throw py::type_error("unsupported consumer dtype");}
    else if(kind=='u')switch(size){case 1:v.type=P::uint8;break;case 2:v.type=P::uint16;break;case 4:v.type=P::uint32;break;case 8:v.type=P::uint64;break;default:throw py::type_error("unsupported consumer dtype");}
    else if(kind=='f' && size==2){v.type=P::uint16;v.half=true;}
    else if(kind=='f' && size==4)v.type=P::float32;
    else if(kind=='f' && size==8)v.type=P::float64;
    else throw py::type_error("unsupported consumer dtype");
    v.data=a.data();v.shape.assign(a.shape(),a.shape()+a.ndim());v.strides.assign(a.strides(),a.strides()+a.ndim());v.swap=!a.dtype().attr("isnative").cast<bool>();return v;
}
ConsumerRaster writable(const py::array& a) {
    if(!a.writeable())throw py::value_error("assignment destination is read-only");
    ConsumerRaster result;result.data=const_cast<void*>(a.data());result.shape.assign(a.shape(),a.shape()+a.ndim());result.strides.assign(a.strides(),a.strides()+a.ndim());result.itemsize=a.itemsize();return result;
}
py::array_t<std::intptr_t> indices(const py::array& a) {
    const auto plan=view(a);py::array_t<std::intptr_t> result(shape(a));
    if(a.dtype().kind()=='b') {
        py::array_t<std::uint8_t> bits(shape(a));
        {py::gil_scoped_release release;components_prepare_bool_into(plan,bits.mutable_data());detail::partition(std::size_t(a.size()),detail::active_workers(std::size_t(a.size())),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)result.mutable_data()[i]=bits.data()[i];});}
    } else {py::gil_scoped_release release;components_prepare_index_into(plan,result.mutable_data());}
    return result;
}
py::array_t<std::uint64_t> words(const py::array& a) {
    if(a.dtype().kind()!='i' && a.dtype().kind()!='u')throw py::type_error("integer consumer keys required");
    auto prepared=indices(a);py::array_t<std::uint64_t> result(shape(a));
    if(a.size())std::memcpy(result.mutable_data(),prepared.data(),std::size_t(a.size())*sizeof(std::uint64_t));return result;
}
py::array contiguous(const py::array& a) {
    return checked(PyArray_FromArray(array(a),nullptr,NPY_ARRAY_CARRAY_RO|NPY_ARRAY_ENSUREARRAY));
}
void same_shape(const py::array& a,const py::array& b) {if(shape(a)!=shape(b))throw py::value_error("consumer shape mismatch");}
void native_buffer(const py::array& a) {if(!a.dtype().attr("isnative").cast<bool>())throw py::value_error("Big-endian buffer not supported on little-endian compiler");}
void pod_buffer(const py::array& a) {
    const auto kind=a.dtype().kind();if((kind!='b' && kind!='i' && kind!='u' && kind!='f') || (a.itemsize()!=1 && a.itemsize()!=2 && a.itemsize()!=4 && a.itemsize()!=8))throw py::type_error("No matching signature found");
}
py::array flattened(const py::array& a) {
    npy_intp length=-1;PyArray_Dims dims{&length,1};return checked(PyArray_Newshape(array(a),&dims,NPY_CORDER));
}
py::array cast(const py::array& a,py::dtype dtype,NPY_ORDER order) {
    auto* descriptor=reinterpret_cast<PyArray_Descr*>(dtype.ptr());
    Py_INCREF(descriptor);auto result=checked(PyArray_NewLikeArray(array(a),order,descriptor,0));
    if(PyArray_CopyInto(array(result),array(a))<0)throw py::error_already_set();return result;
}
py::array scalar(py::dtype dtype,py::handle value) {
    py::array output(dtype,std::vector<py::ssize_t>{});
    if(PyObject_SetItem(output.ptr(),py::tuple().ptr(),value.ptr())<0)throw py::error_already_set();return output;
}
}
void bind_component_consumers(py::module_& m) {
    // Each translation unit owns/imports its own NumPy API table. Headers2,
    // target1.23 intentionally support both NumPy1.24 and modern NumPy2.
    if(_import_array()<0)throw py::error_already_set();
    m.attr("component_consumers_interface_version")=component_consumers_interface_version;
    m.attr("_consumer_numpy_header_abi")=py::int_(NPY_ABI_VERSION);
    m.attr("_consumer_numpy_target_api")=py::int_(NPY_FEATURE_VERSION);
    m.attr("_consumer_numpy_runtime_abi")=py::int_(PyArray_GetNDArrayCVersion());
    m.attr("_consumer_numpy_runtime_api")=py::int_(PyArray_GetNDArrayCFeatureVersion());
#ifdef PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF
    m.attr("_consumer_gil_assertions")=true;
#else
    m.attr("_consumer_gil_assertions")=false;
#endif
    py::class_<ConsumerCounts>(m,"_ConsumerCounts")
        .def_property_readonly("maximum",[](const ConsumerCounts& c){return py::int_(c.maximum);})
        .def_readonly("empty",&ConsumerCounts::empty)
        .def_property_readonly("keys",[](const ConsumerCounts& c){py::array_t<std::uint64_t> a(c.keys.size());if(!c.keys.empty())std::memcpy(a.mutable_data(),c.keys.data(),c.keys.size()*8);return a;})
        .def_property_readonly("sizes",[](const ConsumerCounts& c){py::array_t<std::intptr_t> a(c.sizes.size());if(!c.sizes.empty())std::memcpy(a.mutable_data(),c.sizes.data(),c.sizes.size()*sizeof(std::intptr_t));return a;});
    m.def("consumer_copy",[](const py::array& a,py::object dtype,bool keep_order){return cast(a,dtype.is_none() ? a.dtype():py::dtype::from_args(dtype),keep_order ? NPY_KEEPORDER:NPY_CORDER);});
    m.def("consumer_assign",[](const py::array& out,py::handle source){
        py::slice all{py::none(),py::none(),py::none()};if(PyObject_SetItem(out.ptr(),all.ptr(),source.ptr())<0)throw py::error_already_set();return out;
    });
    m.def("consumer_indices",&indices);
    m.def("consumer_empty",[](std::vector<py::ssize_t> dimensions,py::dtype dtype){return py::array(dtype,dimensions);});
    m.def("consumer_arange",[](py::handle start,py::handle stop,py::handle step){return checked(PyArray_ArangeObj(start.ptr(),stop.ptr(),step.ptr(),nullptr));});
    m.def("consumer_sequence",[](std::size_t count,std::uint64_t offset,bool zero,py::dtype dtype){
        py::array_t<std::uint64_t> raw(count);
        {py::gil_scoped_release release;detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)raw.mutable_data()[i]=zero && i==0 ? 0 : offset+i-std::size_t(zero);});}
        return cast(raw,dtype,NPY_CORDER);
    });
    m.def("consumer_min_negative",[](const py::array& a){const auto plan=view(a);bool negative;{py::gil_scoped_release release;negative=consumer_min_negative(plan);}return negative;});
    m.def("consumer_bool_keys",[](const py::array& a){auto prepared=indices(a);py::array_t<std::uint8_t> out(shape(a));{py::gil_scoped_release release;consumer_truth_into(prepared.data(),out.mutable_data(),a.size(),false);}return out;});
    m.def("consumer_isolated_labels",[](const py::array_t<std::uint8_t,py::array::c_style>& keys){
        py::array_t<std::intptr_t> output(shape(keys));const auto count=std::size_t(keys.size());
        {py::gil_scoped_release release;detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)output.mutable_data()[i]=keys.data()[i] ? std::intptr_t(i+1):0;});}
        return output;
    });
    m.def("consumer_invert",[](const py::array& a,const py::array& out){
        if(out.dtype().kind()!='b')throw py::type_error("boolean consumer output required");
        if(!out.writeable())throw py::value_error("output array is read-only");same_shape(a,out);
        auto prepared=indices(a);py::array_t<std::uint8_t> bits(shape(a));auto destination=writable(out);
        {py::gil_scoped_release release;consumer_truth_into(prepared.data(),bits.mutable_data(),a.size(),false);consumer_bool_write(bits.data(),destination,true);}return out;
    });
    m.def("consumer_count",[](const py::array_t<std::intptr_t,py::array::c_style>& a){ConsumerCounts counts;{py::gil_scoped_release release;counts=consumer_count(a.data(),a.size());}return counts;});
    m.def("consumer_filter",[](const py::array_t<std::intptr_t,py::array::c_style>& labels,const ConsumerCounts& counts,double threshold,const py::array& out){same_shape(labels,out);pod_buffer(out);auto target=writable(out);{py::gil_scoped_release release;consumer_filter_into(labels.data(),counts,threshold,target);}return out;});
    m.def("consumer_border",[](std::vector<std::size_t> dimensions,std::vector<std::size_t> stops,std::vector<std::size_t> starts){py::array_t<bool> out(std::vector<py::ssize_t>(dimensions.begin(),dimensions.end()));{py::gil_scoped_release release;consumer_border_into(reinterpret_cast<std::uint8_t*>(out.mutable_data()),dimensions,stops,starts);}return out;});
    m.def("consumer_clear",[](const py::array_t<std::intptr_t,py::array::c_style>& labels,const py::array& border,std::size_t number,const py::array& out,py::handle bgval){
        same_shape(labels,border);same_shape(labels,out);pod_buffer(out);if(border.dtype().kind()!='b')throw py::type_error("boolean border required");auto target=writable(out);auto fill=scalar(out.dtype(),bgval);auto bits=contiguous(border);std::vector<std::uint8_t> remove;
        {py::gil_scoped_release release;remove=consumer_border_ids(labels.data(),static_cast<const std::uint8_t*>(bits.data()),labels.size(),number);consumer_clear_into(labels.data(),remove.data(),number,fill.data(),target);}return out;
    });
    m.def("consumer_unique",[](const py::array& a){auto input=words(a);std::vector<std::uint64_t> keys;{py::gil_scoped_release release;keys=consumer_sorted_unique(input.data(),input.size());}py::array_t<std::uint64_t> raw(keys.size());if(!keys.empty())std::memcpy(raw.mutable_data(),keys.data(),keys.size()*8);return cast(raw,a.dtype(),NPY_CORDER);});
    m.def("consumer_map",[](const py::array& index,const py::array& keys,const py::array& values,py::object supplied){
        if(index.dtype().kind()!='i' && index.dtype().kind()!='u')throw py::type_error("The dtype of an array to be remapped should be integer.");
        native_buffer(index);native_buffer(values);pod_buffer(values);if(values.size()<keys.size())throw py::value_error("insufficient map values");
        auto flat=flattened(index);if(!flat.writeable())throw py::value_error("buffer source array is read-only");
        if(!keys.writeable() && PyArray_EquivTypes(PyArray_DESCR(array(keys)),PyArray_DESCR(array(index))))throw py::value_error("buffer source array is read-only");
        auto indices=words(flat);auto old=words(keys);py::array out=supplied.is_none() ? py::array(values.dtype(),shape(index)) : supplied.cast<py::array>();
        native_buffer(out);pod_buffer(out);same_shape(index,out);if(!(out.flags()&py::array::c_style))throw py::value_error("C-order map output required");
        if(!values.writeable() && PyArray_EquivTypes(PyArray_DESCR(array(values)),PyArray_DESCR(array(out))))throw py::value_error("buffer source array is read-only");
        auto destination=writable(out);auto source=cast(values,out.dtype(),NPY_CORDER);const auto index_bits=unsigned(index.itemsize()*8);
        {py::gil_scoped_release release;consumer_map_into(indices.data(),indices.size(),old.data(),source.data(),old.size(),destination.itemsize,index_bits,destination.data);}return out;
    },py::arg("index"),py::arg("keys"),py::arg("values"),py::arg("out")=py::none());
    m.def("consumer_max",[](const py::array& a)->py::object {
        if(!a.size())throw py::value_error("zero-size array to reduction operation maximum which has no identity");
        auto prepared=words(a);std::uint64_t maximum=prepared.data()[0];
        for(py::ssize_t i=1;i<a.size();++i){const auto x=prepared.data()[i];if(a.dtype().kind()=='i' ? std::int64_t(x)>std::int64_t(maximum) : x>maximum)maximum=x;}
        py::array_t<std::uint64_t> raw(1);raw.mutable_data()[0]=maximum;auto typed=cast(raw,a.dtype(),NPY_CORDER);return typed.attr("__getitem__")(0);
    });
    const auto numpy=py::module_::import("numpy");
    const py::object zeros=numpy.attr("zeros"),maximum=numpy.attr("maximum").attr("reduce");
    m.def("consumer_dense_max",[maximum](py::object keys){
        // Noninteger constructors need NumPy's scalar/type protocol. Integer
        // map maxima retain consumer_max's C++ scan and scalar reconstruction.
        return maximum(keys,py::arg("axis")=py::none());
    });
    m.def("consumer_dense",[zeros](py::object supplied_keys,py::object supplied_values,py::handle length,py::object dtype){
        // Allocation is the shape/dtype authority, including object zeros.
        // No upstream ArrayMap or map_array implementation is called.
        py::array out=zeros(length,py::arg("dtype")=dtype).cast<py::array>();
        const bool array_keys=PyArray_Check(supplied_keys.ptr()),array_values=PyArray_Check(supplied_values.ptr());
        py::array keys=array_keys ? py::reinterpret_borrow<py::array>(supplied_keys) : py::array();
        py::array values=array_values ? py::reinterpret_borrow<py::array>(supplied_values) : py::array();
        const auto key_kind=array_keys ? keys.dtype().kind() : '\0',value_kind=array_values ? values.dtype().kind() : '\0';
        const bool typed=array_keys && array_values && out.ndim()==1 && keys.ndim()==1 && values.ndim()==1 && keys.size()==values.size()
            && (key_kind=='i' || key_kind=='u') && (value_kind=='b' || value_kind=='i' || value_kind=='u' || value_kind=='f')
            && values.itemsize()<=8 && PyArray_EquivTypes(PyArray_DESCR(array(values)),PyArray_DESCR(array(out)));
        if(typed) {
            auto old=words(keys);const auto count=std::size_t(out.size());bool in_bounds=true;
            {py::gil_scoped_release release;for(py::ssize_t i=0;i<old.size();++i)if(old.data()[i]>=count){in_bounds=false;break;}}
            if(in_bounds) {
                auto source=contiguous(values);const auto bytes=std::size_t(out.itemsize());
                {py::gil_scoped_release release;consumer_scatter_into(old.data(),source.data(),std::size_t(old.size()),bytes,out.mutable_data(),count);}
                return out;
            }
        }
        // Checked indexing preserves negative/bool indices, bounds messages,
        // broadcast/cast warnings and the reference's partial-error ordering.
        if(PyObject_SetItem(out.ptr(),supplied_keys.ptr(),supplied_values.ptr())<0)throw py::error_already_set();
        return out;
    });
    m.def("consumer_scatter",[](const py::array& cache,py::handle index,py::handle values){
        const auto kind=cache.dtype().kind();
        const bool integer_scalar=(PyLong_Check(index.ptr()) && !PyBool_Check(index.ptr())) || PyArray_IsScalar(index.ptr(),Integer);
        if(integer_scalar && cache.ndim()==1 && cache.writeable() && (kind=='b' || kind=='i' || kind=='u' || kind=='f') && cache.itemsize()<=8) {
            auto location=PyNumber_AsSsize_t(index.ptr(),PyExc_IndexError);
            if(PyErr_Occurred()) {
                // NumPy distinguishes large scalar integers from integers
                // outside its index protocol. Keep that checked authority.
                PyErr_Clear();
                if(PyObject_SetItem(cache.ptr(),index.ptr(),values.ptr())<0)throw py::error_already_set();
                return;
            }
            const auto original=location;if(location<0)location+=cache.size();
            if(location<0 || location>=cache.size())throw py::index_error("index "+std::to_string(original)+" is out of bounds for axis 0 with size "+std::to_string(cache.size()));
            auto target=writable(cache);auto value=scalar(cache.dtype(),values);const std::uint64_t key=std::uint64_t(location);
            const auto count=std::size_t(cache.size());
            {py::gil_scoped_release release;consumer_scatter_into(&key,value.data(),1,target.itemsize,target.data,count);}return;
        }
        // NumPy's native assignment provider is also the reference authority
        // for advanced-index broadcasting, casting and partial-error state.
        if(PyObject_SetItem(cache.ptr(),index.ptr(),values.ptr())<0)throw py::error_already_set();
    });
    m.def("consumer_compact",[](const py::array& a){
        auto plan=view(a);py::array_t<std::uint8_t> prepared(shape(a));{py::gil_scoped_release release;consumer_nonzero_into(plan,prepared.mutable_data());}
        std::vector<std::intptr_t> positions;positions.reserve(a.size());for(py::ssize_t i=0;i<a.size();++i)if(prepared.data()[i])positions.push_back(i);
        py::array_t<std::intptr_t> key_storage(std::vector<py::ssize_t>{py::ssize_t(positions.size()),1});
        if(!positions.empty())std::memcpy(key_storage.mutable_data(),positions.data(),positions.size()*sizeof(std::intptr_t));
        npy_intp key_count=npy_intp(positions.size());PyArray_Dims key_shape{&key_count,1};
        auto keys=checked(PyArray_Newshape(array(key_storage),&key_shape,NPY_CORDER));
        auto source=contiguous(a);py::array values(a.dtype(),std::vector<py::ssize_t>{py::ssize_t(positions.size())});for(std::size_t i=0;i<positions.size();++i)std::memcpy(static_cast<char*>(values.mutable_data())+i*a.itemsize(),static_cast<const char*>(source.data())+positions[i]*a.itemsize(),a.itemsize());return py::make_tuple(keys,values);
    });
}
#ifdef SKIMAGEXX_COMPONENT_CONSUMERS_STANDALONE
void bind_components(py::module_&);
PYBIND11_MODULE(_component_consumers_core,m){bind_components(m);bind_component_consumers(m);m.def("set_num_threads",&set_num_threads);m.def("get_num_threads",&get_num_threads);m.def("has_avx2",&has_avx2);}
#endif
