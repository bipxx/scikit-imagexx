#include "skimagexx/h_extrema.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include "half.hpp"

namespace py=pybind11;
namespace {
std::array<std::atomic<std::uint64_t>,6> counters{};
std::array<std::atomic<std::uint64_t>,2> real_counters{};
// Complete direct-C-slot ownership and metadata stamps, established before
// scalar conversion/output allocation. Unknown foreign roots are explicit gaps.
struct Node {
    py::array array;py::object base;py::dtype dtype;const void* data;
    py::ssize_t bytes;int flags;std::vector<py::ssize_t> shape,strides;
    explicit Node(py::array value):array(std::move(value)),base(array.base()),dtype(array.dtype()),
        data(array.data()),bytes(array.nbytes()),flags(array.flags()),
        shape(array.shape(),array.shape()+array.ndim()),strides(array.strides(),array.strides()+array.ndim()) {
        if(bytes<0)throw py::value_error("H-extrema array byte size overflows");
    }
    void check()const {
        if(array.base().ptr()!=base.ptr() || array.data()!=data || array.nbytes()!=bytes ||
           !array.dtype().is(dtype) || array.flags()!=flags || std::size_t(array.ndim())!=shape.size())
            throw py::value_error("H-extrema owner metadata changed during intake");
        for(std::size_t axis=0;axis<shape.size();++axis)
            if(array.shape(py::ssize_t(axis))!=shape[axis] || array.strides(py::ssize_t(axis))!=strides[axis])
                throw py::value_error("H-extrema owner metadata changed during intake");
    }
};
struct Owners {
    std::vector<Node> nodes;
    explicit Owners(py::array value) {
        for(unsigned depth=0;;++depth) {
            if(depth==64)throw py::value_error("H-extrema owner chain is too deep");
            for(const auto& node:nodes)if(node.array.ptr()==value.ptr())
                throw py::value_error("H-extrema owner chain is cyclic");
            nodes.emplace_back(std::move(value));const auto base=nodes.back().base;
            if(!base || base.is_none())break;
            if(!py::isinstance<py::array>(base))throw py::value_error("H-extrema requires ndarray-backed storage");
            value=py::reinterpret_borrow<py::array>(base);
        }
        const auto& root=nodes.back().array;
        if(!root.owndata() || !(root.flags()&(py::array::c_style|py::array::f_style)))
            throw py::value_error("H-extrema requires a contiguous owning ndarray root");
        check();
    }
    void check()const {for(const auto& node:nodes)node.check();}
};
struct Input {
    std::vector<std::size_t> shape;std::vector<std::ptrdiff_t> strides;
    skimagexx::ReconstructionView view;
    explicit Input(const Owners& owners) {
        using T=skimagexx::ReconstructionType;
        const auto& first=owners.nodes.front();const auto& array=first.array;
        const auto dtype=array.dtype();const auto kind=dtype.kind();const auto bytes=array.itemsize();
        if(kind=='u' && bytes==1)view.type=T::uint8;else if(kind=='i' && bytes==1)view.type=T::int8;
        else if(kind=='u' && bytes==2)view.type=T::uint16;else if(kind=='i' && bytes==2)view.type=T::int16;
        else if(kind=='u' && bytes==4)view.type=T::uint32;else if(kind=='i' && bytes==4)view.type=T::int32;
        else if(kind=='u' && bytes==8)view.type=T::uint64;else if(kind=='i' && bytes==8)view.type=T::int64;
        else if(kind=='f' && bytes==8)view.type=T::float64;
        else throw py::type_error("H-extrema integer stage requires integer image or float64 reconstruction");
        for(py::ssize_t axis=0;axis<array.ndim();++axis) {
            if(array.shape(axis)<0)throw py::value_error("H-extrema negative array dimension");
            shape.push_back(std::size_t(array.shape(axis)));strides.push_back(array.strides(axis));
        }
        view.data=first.data;view.shape={shape.data(),shape.size()};view.strides={strides.data(),strides.size()};
        const auto& root=owners.nodes.back();view.allocation=root.data;view.allocation_bytes=std::size_t(root.bytes);
        const std::uint16_t one=1;const bool little=*reinterpret_cast<const unsigned char*>(&one)==1;
        const char order=dtype.byteorder();view.swapped=(order=='<' && !little)||(order=='>' && little);
    }
};
skimagexx::ReconstructionMethod method(const py::object& value) {
    if(!PyLong_CheckExact(value.ptr()))throw py::type_error("H-extrema method must be an exact integer");
    const auto result=PyLong_AsLong(value.ptr());if(PyErr_Occurred())throw py::error_already_set();
    if(result!=0 && result!=1)throw py::value_error("H-extrema invalid method");
    return result==0 ? skimagexx::ReconstructionMethod::dilation : skimagexx::ReconstructionMethod::erosion;
}
std::uint64_t integer_height(const py::object& value) {
    if(!PyLong_CheckExact(value.ptr()))throw py::type_error("H-extrema height must be an exact integer");
    const auto result=PyLong_AsUnsignedLongLong(value.ptr());if(PyErr_Occurred())throw py::error_already_set();
    return result;
}
py::dict info(skimagexx::HExtremaInfo value) {
    py::dict out;out["worker_limit"]=value.workers;out["simd"]=value.simd;return out;
}
skimagexx::ReconstructionSpan<unsigned char> output(py::array& array) {
    return {static_cast<unsigned char*>(array.mutable_data()),std::size_t(array.nbytes())};
}

// Generic byte-storage validation is deliberately separate from integer-v1
// Input. Empty/bool/complex providers must keep their source error precedence.
struct StorageGuard {
    Owners owners;
    explicit StorageGuard(py::array image):owners(std::move(image)){check();}
    std::size_t check()const {
        owners.check();
        const auto& first=owners.nodes.front();const auto& root=owners.nodes.back();
        const auto item=first.array.itemsize();
        if(item<=0)throw py::type_error("H-extrema requires positive storage itemsize");
        const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        auto add=[limit](std::size_t a,std::size_t b) {
            if(b>limit || a>limit-b)throw py::value_error("H-extrema storage extent overflows");
            return a+b;
        };
        auto multiply=[limit](std::size_t a,std::size_t b) {
            if(b && a>limit/b)throw py::value_error("H-extrema storage extent overflows");
            return a*b;
        };
        std::size_t count=1,low=0,high=0;
        for(auto length:first.shape) {
            if(length<0)throw py::value_error("H-extrema negative storage dimension");
            count=multiply(count,std::size_t(length));
        }
        if(!count)return 0; // No pixel may be read; source reduction handles it.
        for(std::size_t axis=0;axis<first.shape.size();++axis) {
            if(first.shape[axis]<=1)continue;
            const auto stride=first.strides[axis];
            if(stride==std::numeric_limits<py::ssize_t>::min())
                throw py::value_error("H-extrema storage extent overflows");
            const auto span=multiply(std::size_t(first.shape[axis]-1),std::size_t(stride<0 ? -stride : stride));
            if(stride<0)low=add(low,span);else high=add(high,span);
        }
        high=add(high,std::size_t(item));
        const auto data=reinterpret_cast<std::uintptr_t>(first.data);
        const auto base=reinterpret_cast<std::uintptr_t>(root.data);
        const auto bound=std::numeric_limits<std::uintptr_t>::max();
        if(!data || !base || data<low || high>bound-data || std::size_t(root.bytes)>bound-base ||
           data-low<base || data+high>base+std::size_t(root.bytes))
            throw py::value_error("H-extrema view outside owning allocation");
        return count;
    }
};
struct RealInput {
    std::vector<std::size_t> shape;std::vector<std::ptrdiff_t> strides;
    skimagexx::ReconstructionView view;
    explicit RealInput(const Owners& owners) {
        using T=skimagexx::ReconstructionType;
        const auto& first=owners.nodes.front();const auto& array=first.array;
        const auto dtype=array.dtype();const auto kind=dtype.kind();const auto bytes=array.itemsize();
        if(kind=='b' && bytes==1)view.type=T::boolean;
        else if(kind=='u' && bytes==1)view.type=T::uint8;else if(kind=='i' && bytes==1)view.type=T::int8;
        else if(kind=='u' && bytes==2)view.type=T::uint16;else if(kind=='i' && bytes==2)view.type=T::int16;
        else if(kind=='u' && bytes==4)view.type=T::uint32;else if(kind=='i' && bytes==4)view.type=T::int32;
        else if(kind=='u' && bytes==8)view.type=T::uint64;else if(kind=='i' && bytes==8)view.type=T::int64;
        else if(kind=='f' && bytes==2)view.type=T::float16;
        else if(kind=='f' && bytes==4)view.type=T::float32;
        else if(kind=='f' && bytes==8)view.type=T::float64;
        else throw py::type_error("H-extrema real stage requires a supported real dtype");
        for(auto length:first.shape)shape.push_back(std::size_t(length));
        for(auto stride:first.strides)strides.push_back(stride);
        view.data=first.data;view.shape={shape.data(),shape.size()};view.strides={strides.data(),strides.size()};
        const auto& root=owners.nodes.back();view.allocation=root.data;view.allocation_bytes=std::size_t(root.bytes);
        const std::uint16_t one=1;const bool little=*reinterpret_cast<const unsigned char*>(&one)==1;
        const char order=dtype.byteorder();view.swapped=(order=='<' && !little)||(order=='>' && little);
    }
};
bool unsigned_height(std::uint64_t value,double& result) {
    unsigned high=0;for(auto word=value;word>1;word>>=1)++high;
    if(high>52 && (value&((std::uint64_t(1)<<(high-52))-1)))return false;
    result=double(value);return true; // Exact conversion: no rounding/status.
}
bool float32_height(std::uint32_t bits,double& result) {
    const auto exponent=(bits>>23)&255;auto fraction=bits&0x7fffff;
    if(exponent==255)return false;
    std::uint64_t encoded=std::uint64_t(bits&0x80000000u)<<32;
    if(exponent)encoded|=(std::uint64_t(exponent+896)<<52)|(std::uint64_t(fraction)<<29);
    else if(fraction) {
        unsigned shift=0;while(!(fraction&0x800000)){fraction<<=1;++shift;}
        encoded|=(std::uint64_t(897-shift)<<52)|(std::uint64_t(fraction&0x7fffff)<<29);
    }
    std::memcpy(&result,&encoded,sizeof(result));return true;
}
bool bytes_height(const void* pointer,char kind,std::size_t bytes,bool swapped,double& result) {
    if(bytes!=1 && bytes!=2 && bytes!=4 && bytes!=8)return false;
    std::uint64_t bits=0;
    // Native-width loads also work on big-endian hosts and unaligned arrays.
    if(bytes==1)bits=*static_cast<const unsigned char*>(pointer);
    else if(bytes==2){std::uint16_t word;std::memcpy(&word,pointer,2);bits=word;}
    else if(bytes==4){std::uint32_t word;std::memcpy(&word,pointer,4);bits=word;}
    else std::memcpy(&bits,pointer,8);
    if(swapped){auto word=bits;bits=0;for(std::size_t i=0;i<bytes;++i){bits=(bits<<8)|(word&255);word>>=8;}}
    if(kind=='b'){result=bits ? 1. : 0.;return true;}
    if(kind=='i' && (bits&(std::uint64_t(1)<<(bytes*8-1))))return false;
    if(kind=='i' || kind=='u')return unsigned_height(bits,result);
    if(kind!='f')return false;
    if(bytes==2) {
        const auto word=std::uint16_t(bits);
        if((word&0x7c00)==0x7c00)return false;
        const float value=skimagexx::detail::decode_half(word);
        std::uint32_t single;std::memcpy(&single,&value,4);return float32_height(single,result);
    }
    if(bytes==4)return float32_height(std::uint32_t(bits),result);
    if(((bits>>52)&2047)==2047)return false;
    std::memcpy(&result,&bits,8);return true;
}
bool current_height(const py::object& height,const py::tuple& scalar_types,
                    const StorageGuard* storage,double& result) {
    if(PyFloat_CheckExact(height.ptr())) {
        result=PyFloat_AS_DOUBLE(height.ptr());return true;
    }
    if(PyBool_Check(height.ptr())){result=height.ptr()==Py_True ? 1. : 0.;return true;}
    if(PyLong_CheckExact(height.ptr())) {
        const auto value=PyLong_AsUnsignedLongLong(height.ptr());
        if(PyErr_Occurred()){PyErr_Clear();return false;}
        return unsigned_height(value,result);
    }
    auto& api=py::detail::npy_api::get();
    if(storage) {
        if(Py_TYPE(height.ptr())!=api.PyArray_Type_)return false;
        const auto& array=storage->owners.nodes.front().array;
        if(storage->check()!=1)return false;
        const auto dtype=array.dtype();const std::uint16_t one=1;
        const bool little=*reinterpret_cast<const unsigned char*>(&one)==1;
        const char order=dtype.byteorder();
        return bytes_height(array.data(),dtype.kind(),std::size_t(array.itemsize()),
                            (order=='<' && !little)||(order=='>' && little),result);
    }
    constexpr char kinds[]={'b','u','i','u','i','u','i','u','i','f','f','f'};
    constexpr std::size_t sizes[]={1,1,1,2,2,4,4,8,8,2,4,8};
    for(std::size_t i=0;i<12;++i) {
        if(reinterpret_cast<PyObject*>(Py_TYPE(height.ptr()))!=scalar_types[i].ptr())continue;
        alignas(double) std::array<unsigned char,8> bytes{};
        api.PyArray_ScalarAsCtype_(height.ptr(),bytes.data());
        if(PyErr_Occurred())throw py::error_already_set();
        return bytes_height(bytes.data(),kinds[i],sizes[i],false,result);
    }
    return false; // No __float__/__index__/array protocol admission probes.
}
}

void bind_h_extrema(py::module_& module) {
    module.attr("h_extrema_interface")=skimagexx::h_extrema_interface;
    module.attr("h_extrema_integer_widths")=py::make_tuple(8,16,32,64);
    module.attr("h_extrema_full_API")=false;
    module.def("h_extrema_validate_metadata",[](py::array image) {
        Owners owner(image);Input input(owner);owner.check();
        return skimagexx::h_extrema_validate_view(input.view);
    },py::arg("image").noconvert());
    module.def("h_extrema_counts",[](bool reset) {
        py::dict out;const char* names[]={"range","seed","mask"};
        for(std::size_t i=0;i<3;++i){py::dict row;row["attempted"]=counters[2*i].load();row["completed"]=counters[2*i+1].load();out[names[i]]=row;}
        if(reset)for(auto& count:counters)count=0;return out;
    },py::arg("reset")=false);
    module.def("h_extrema_integer_range",[](py::array image) {
        ++counters[0];Owners owner(image);Input input(owner);owner.check();skimagexx::HExtremaRange result;
        {py::gil_scoped_release release;result=skimagexx::h_extrema_integer_range(input.view);}
        ++counters[1];return result.bits;
    },py::arg("image").noconvert());
    module.def("h_extrema_integer_seed",[](py::array image,py::object height,py::object requested_method) {
        ++counters[2];Owners owner(image);const auto h=integer_height(height);const auto selected=method(requested_method);
        Input input(owner);owner.check();skimagexx::h_extrema_validate_view(input.view);
        const auto dtype=py::reinterpret_borrow<py::dtype>(image.dtype().attr("newbyteorder")("="));
        py::array shifted(dtype,owner.nodes.front().shape);const auto destination=output(shifted);
        owner.check();skimagexx::HExtremaInfo result;
        {py::gil_scoped_release release;result=skimagexx::h_extrema_integer_seed_into(input.view,h,selected,destination);}
        ++counters[3];return py::make_tuple(shifted,info(result));
    },py::arg("image").noconvert(),py::arg("height"),py::arg("method"));
    module.def("h_extrema_integer_mask",[](py::array image,py::array reconstructed,py::object height,py::object requested_method) {
        ++counters[4];Owners first(image),second(reconstructed);
        if(!PyFloat_CheckExact(height.ptr()))throw py::type_error("H-extrema comparison height must be an exact float");
        const auto h=PyFloat_AS_DOUBLE(height.ptr());const auto selected=method(requested_method);
        Input a(first),b(second);first.check();second.check();
        skimagexx::h_extrema_validate_view(a.view);skimagexx::h_extrema_validate_view(b.view);
        py::array mask(py::dtype::of<std::uint8_t>(),first.nodes.front().shape);
        const auto destination=output(mask);
        first.check();second.check();skimagexx::HExtremaInfo result;
        {py::gil_scoped_release release;result=skimagexx::h_extrema_integer_mask_into(a.view,b.view,h,selected,destination);}
        ++counters[5];return py::make_tuple(mask,info(result));
    },py::arg("image").noconvert(),py::arg("reconstructed").noconvert(),py::arg("height"),py::arg("method"));
    module.attr("h_extrema_real_interface")=skimagexx::h_extrema_real_interface;
    py::class_<StorageGuard>(module,"_HExtremaStorage")
        .def(py::init<py::array>(),py::arg("image").noconvert())
        .def("check",&StorageGuard::check);
    module.def("h_extrema_real_counts",[](bool reset) {
        py::dict out;out["attempted"]=real_counters[0].load();out["completed"]=real_counters[1].load();
        if(reset)for(auto& count:real_counters)count=0;return out;
    },py::arg("reset")=false);
    const char* formats[]={"?","u1","i1","u2","i2","u4","i4","u8","i8","f2","f4","f8"};
    py::tuple scalar_types(12);
    for(std::size_t i=0;i<12;++i)scalar_types[i]=py::dtype::from_args(py::str(formats[i])).attr("type");
    module.def("h_extrema_real_mask",[scalar_types](py::array image,py::array reconstructed,
            py::object height,py::object requested_method)->py::object {
        ++real_counters[0];StorageGuard first(image),second(reconstructed);
        std::unique_ptr<StorageGuard> height_storage;
        if(py::isinstance<py::array>(height))height_storage=std::make_unique<StorageGuard>(py::reinterpret_borrow<py::array>(height));
        const auto selected=method(requested_method);RealInput a(first.owners),b(second.owners);
        skimagexx::h_extrema_validate_real_view(a.view);skimagexx::h_extrema_validate_real_view(b.view);
        if(height_storage) {
            const auto& shape=height_storage->owners.nodes.front().shape;
            if(shape.size()>a.shape.size())return py::none();
            for(auto length:shape)if(length!=1)return py::none();
        }
        py::array mask(py::dtype::of<std::uint8_t>(),first.owners.nodes.front().shape);
        const auto destination=output(mask);
        // Allocation/provider boundaries precede both value capture and reads.
        first.check();second.check();if(height_storage)height_storage->check();
        skimagexx::h_extrema_validate_real_view(a.view);skimagexx::h_extrema_validate_real_view(b.view);
        double exact_height=0;
        if(!current_height(height,scalar_types,height_storage.get(),exact_height))return py::none();
        first.check();second.check();if(height_storage)height_storage->check();
        skimagexx::HExtremaRealInfo result;
        {py::gil_scoped_release release;result=skimagexx::h_extrema_real_mask_into(a.view,b.view,exact_height,selected,destination);}
        if(!result.admitted)return py::none();
        ++real_counters[1];return py::make_tuple(mask,info(result.execution));
    },py::arg("image").noconvert(),py::arg("reconstructed").noconvert(),py::arg("height"),py::arg("method"));
}
