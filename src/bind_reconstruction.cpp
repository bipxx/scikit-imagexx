#include "skimagexx/reconstruction.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <atomic>
#include <limits>
#include <vector>
namespace py=pybind11;
namespace {
std::array<std::atomic<std::uint64_t>,4> counters{};
struct Input {
    py::array a;std::vector<std::size_t> shape;std::vector<std::ptrdiff_t> strides;
    skimagexx::ReconstructionView v;
    explicit Input(py::array value):a(std::move(value)) {
        using T=skimagexx::ReconstructionType;auto dtype=a.dtype();const auto kind=dtype.kind();const auto bytes=a.itemsize();
        if(kind=='b' && bytes==1)v.type=T::boolean;
        else if(kind=='u' && bytes==1)v.type=T::uint8;else if(kind=='i' && bytes==1)v.type=T::int8;
        else if(kind=='u' && bytes==2)v.type=T::uint16;else if(kind=='i' && bytes==2)v.type=T::int16;
        else if(kind=='u' && bytes==4)v.type=T::uint32;else if(kind=='i' && bytes==4)v.type=T::int32;
        else if(kind=='u' && bytes==8)v.type=T::uint64;else if(kind=='i' && bytes==8)v.type=T::int64;
        else if(kind=='f' && bytes==2)v.type=T::float16;else if(kind=='f' && bytes==4)v.type=T::float32;else if(kind=='f' && bytes==8)v.type=T::float64;
        else throw py::type_error("reconstruction requires a real supported ndarray dtype");
        for(py::ssize_t d=0;d<a.ndim();++d){shape.push_back(std::size_t(a.shape(d)));strides.push_back(a.strides(d));}
        v.data=a.data();v.shape={shape.data(),shape.size()};v.strides={strides.data(),strides.size()};v.swapped=!dtype.attr("isnative").cast<bool>();
        py::object owner=a;for(unsigned depth=0;depth<64;++depth){auto base=owner.attr("base");if(base.is_none())break;if(!py::isinstance<py::array>(base))throw py::value_error("private reconstruction requires ndarray-backed storage");owner=base;}
        auto root=owner.cast<py::array>();v.allocation=root.data();v.allocation_bytes=std::size_t(root.nbytes());
    }
};
struct Stamp {
    py::array a;const void* data;py::dtype dtype;int flags;std::vector<py::ssize_t> shape,strides;
    explicit Stamp(py::array x):a(std::move(x)),data(a.data()),dtype(a.dtype()),flags(a.flags()),shape(a.shape(),a.shape()+a.ndim()),strides(a.strides(),a.strides()+a.ndim()){}
    void check()const{if(a.data()!=data||!a.dtype().is(dtype)||a.flags()!=flags||std::size_t(a.ndim())!=shape.size())throw py::value_error("reconstruction buffer metadata changed during conversion");for(std::size_t d=0;d<shape.size();++d)if(a.shape(py::ssize_t(d))!=shape[d]||a.strides(py::ssize_t(d))!=strides[d])throw py::value_error("reconstruction buffer metadata changed during conversion");}
};
template<class T> skimagexx::ReconstructionSpan<T> writable(std::vector<T>& v){return {v.data(),v.size()};}
}
void bind_reconstruction(py::module_& m) {
    m.attr("reconstruction_interface")=skimagexx::reconstruction_interface;
    m.attr("reconstruction_capabilities")=skimagexx::reconstruction_capabilities;
    m.attr("reconstruction_parallel_flood")=false;
    m.def("reconstruction_counts",[](bool reset){py::dict out;for(std::size_t i=0;i<2;++i){py::dict row;row["attempted"]=counters[2*i].load();row["completed"]=counters[2*i+1].load();out[i ? "run" : "ordered"]=row;}if(reset)for(auto& c:counters)c=0;return out;},py::arg("reset")=false);
    m.def("reconstruction_negative_zero",[](py::array seed,py::array mask){Input a(seed),b(mask);py::gil_scoped_release release;return skimagexx::reconstruction_negative_zero(a.v,b.v);},py::arg("seed").noconvert(),py::arg("mask").noconvert());
    m.def("reconstruction_ordered",[](py::array seed,py::array mask,int method){++counters[0];Input a(seed),b(mask);bool result;{py::gil_scoped_release release;result=skimagexx::reconstruction_ordered(a.v,b.v,static_cast<skimagexx::ReconstructionMethod>(method));}++counters[1];return result;},py::arg("seed").noconvert(),py::arg("mask").noconvert(),py::arg("method"));
    m.def("reconstruction_run",[](py::array seed,py::array mask,py::array footprint,py::array offset,int method,py::object zero,int queue){
        ++counters[2];Stamp sa(seed),sb(mask),sf(footprint),so(offset);
        if(!offset.dtype().equal(py::dtype::of<std::size_t>())||offset.ndim()!=1||!(offset.flags()&py::array::c_style)||reinterpret_cast<std::uintptr_t>(offset.data())%alignof(std::size_t))throw py::type_error("reconstruction offset must be native contiguous size_t vector");
        skimagexx::ReconstructionZero z;if(!zero.is_none()){z.supplied=true;z.negative=zero.cast<bool>();}
        sa.check();sb.check();sf.check();so.check();Input a(seed),b(mask),fp(footprint);
        auto offsets=skimagexx::ReconstructionSpan<const std::size_t>{static_cast<const std::size_t*>(offset.data()),std::size_t(offset.size())};
        skimagexx::ReconstructionFootprint f{fp.v,offsets};auto requested=static_cast<skimagexx::ReconstructionQueue>(queue);auto r=skimagexx::reconstruction_requirements(a.v,b.v,f,requested);
        std::vector<py::ssize_t> padded;for(std::size_t d=0;d<a.shape.size();++d)padded.push_back(py::ssize_t(a.shape[d]+fp.shape[d]-1));
        auto dtype=r.output_type==skimagexx::ReconstructionType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
        py::array out(dtype,padded),bounds(dtype,std::vector<py::ssize_t>{py::ssize_t(r.samples)});
        std::vector<std::size_t> to(r.samples),from(r.padded_samples),nodes(r.samples),positions(r.samples),heads(r.bucket_heads);std::vector<std::ptrdiff_t> neighbors(r.neighbor_capacity);
        skimagexx::ReconstructionBuffers buffers{{static_cast<unsigned char*>(out.mutable_data()),std::size_t(out.nbytes())},{static_cast<unsigned char*>(bounds.mutable_data()),std::size_t(bounds.nbytes())},writable(to),writable(from),writable(nodes),writable(positions),writable(heads),writable(neighbors)};
        sa.check();sb.check();sf.check();so.check();
        skimagexx::ReconstructionInfo result;{py::gil_scoped_release release;result=skimagexx::reconstruction_into(a.v,b.v,f,static_cast<skimagexx::ReconstructionMethod>(method),buffers,z,requested);}
        ++counters[3];py::dict info;info["queue"]=int(result.queue);info["preprocessing_workers"]=result.preprocessing_workers;info["preprocessing_simd"]=result.preprocessing_simd;info["parallel_flood"]=result.parallel_flood;
        return py::make_tuple(out,info);
    },py::arg("seed").noconvert(),py::arg("mask").noconvert(),py::arg("footprint").noconvert(),py::arg("offset").noconvert(),py::arg("method"),py::arg("zero")=py::none(),py::arg("queue")=0);
}
