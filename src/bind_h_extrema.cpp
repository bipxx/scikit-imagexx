#include "skimagexx/h_extrema.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace py=pybind11;
namespace {
std::array<std::atomic<std::uint64_t>,6> counters{};
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
}
