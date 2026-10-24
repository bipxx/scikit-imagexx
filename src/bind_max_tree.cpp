#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "skimagexx/max_tree.hpp"
#include "max_tree_stages.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py=pybind11;
namespace stages=skimagexx::max_tree_stages;
namespace {
struct Count {std::uint64_t attempted=0,completed=0;};
// Python entry/return holds the GIL. These test-observable counters are not a
// worker pool/state; resetting them requires callers to have no calls in flight.
std::array<Count,6> counts{};
const char* names[]={"aggregate","build","area","extension","filter","maxima"};
void extent(const py::array& value) {
    py::array owner=value;
    for(unsigned depth=0;;++depth) {
        if(depth>64)throw py::value_error("max-tree array base chain is too deep");
        auto base=owner.base();if(!base || base.is_none())break;
        if(!py::isinstance<py::array>(base))throw py::value_error("max-tree requires a NumPy allocation owner");
        owner=py::reinterpret_borrow<py::array>(base);
    }
    if(!owner.owndata() || !(owner.flags()&(py::array::c_style|py::array::f_style)))throw py::value_error("uncertified max-tree allocation extent");
    auto begin=reinterpret_cast<std::uintptr_t>(owner.data()),pointer=reinterpret_cast<std::uintptr_t>(value.data());
    const auto bytes=static_cast<std::size_t>(owner.nbytes()),size=static_cast<std::size_t>(value.nbytes());
    if(bytes>std::numeric_limits<std::uintptr_t>::max()-begin || pointer<begin || pointer>begin+bytes || size>begin+bytes-pointer)
        throw py::value_error("max-tree array exceeds allocation extent");
}
py::array array(py::handle object,int ndim=-1,bool either=false) {
    if(!py::isinstance<py::array>(object))throw py::type_error("max-tree buffers must be NumPy arrays");
    auto a=py::reinterpret_borrow<py::array>(object);
    if((ndim>=0 && a.ndim()!=ndim) || !a.writeable() ||
       !(a.flags()&(either?(py::array::c_style|py::array::f_style):py::array::c_style)))
        throw py::value_error("max-tree buffers require writable dense native arrays of the declared rank");
    const auto width=static_cast<std::size_t>(a.itemsize());
    if(!width || reinterpret_cast<std::uintptr_t>(a.data())%width || !py::cast<bool>(a.dtype().attr("isnative")))
        throw py::value_error("max-tree buffers must be naturally aligned and native endian");
    extent(a);return a;
}
template<class T> py::array typed(py::handle object,int ndim=1,bool either=false) {
    auto a=array(object,ndim,either);
    if(!a.dtype().equal(py::dtype::of<T>()))throw py::type_error("max-tree buffer dtype mismatch");
    return a;
}
skimagexx::MaxTreeType type(const py::array& a) {
#define TYPE(code,T) if(a.dtype().equal(py::dtype::of<T>()))return skimagexx::MaxTreeType::code;
    TYPE(u8,std::uint8_t) TYPE(u16,std::uint16_t) TYPE(u32,std::uint32_t) TYPE(u64,std::uint64_t)
    TYPE(i8,std::int8_t) TYPE(i16,std::int16_t) TYPE(i32,std::int32_t) TYPE(i64,std::int64_t)
    TYPE(f32,float) TYPE(f64,double)
#undef TYPE
    throw py::type_error("max-tree requires one of ten native real dtypes");
}
stages::Image image(const py::array& a) {return {a.data(),type(a),static_cast<std::size_t>(a.size())};}
template<class T> skimagexx::MaxTreeSpan<const T> read(const py::array& a) {return {static_cast<const T*>(a.data()),static_cast<std::size_t>(a.size())};}
template<class T> skimagexx::MaxTreeSpan<T> write(const py::array& a) {return {static_cast<T*>(const_cast<void*>(a.data())),static_cast<std::size_t>(a.size())};}
stages::Tree tree(const py::array& parent,const py::array& order) {return {read<std::int64_t>(parent),read<std::int64_t>(order)};}
stages::Coordinates coordinates(const py::array& a) {
    if(a.ndim()!=2 || a.strides(0)<0 || a.strides(1)<0 || a.strides(0)%8 || a.strides(1)%8)throw py::value_error("invalid max-tree coordinate strides");
    return {static_cast<double*>(const_cast<void*>(a.data())),static_cast<std::size_t>(a.size()),static_cast<std::size_t>(a.shape(1)),static_cast<std::size_t>(a.strides(0)/8),static_cast<std::size_t>(a.strides(1)/8)};
}
struct Stamp {
    py::array value;py::dtype dtype;const void* pointer;py::ssize_t size;int flags;
    std::vector<py::ssize_t> shape,strides;
    explicit Stamp(py::array a):value(std::move(a)),dtype(value.dtype()),pointer(value.data()),size(value.nbytes()),flags(value.flags()),shape(value.shape(),value.shape()+value.ndim()),strides(value.strides(),value.strides()+value.ndim()) {}
    void verify() const {
        if(value.data()!=pointer || value.nbytes()!=size || std::size_t(value.ndim())!=shape.size())throw py::value_error("array storage changed during scalar conversion");
        if(!value.dtype().is(dtype) || value.flags()!=flags)throw py::value_error("array dtype/flags changed during scalar conversion");
        for(py::ssize_t d=0;d<value.ndim();++d)if(value.shape(d)!=shape[std::size_t(d)] || value.strides(d)!=strides[std::size_t(d)])throw py::value_error("array metadata changed during scalar conversion");
        extent(value);
    }
};
long integer(py::handle value) {
    py::object index=py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if(!index)throw py::error_already_set();
    auto result=PyLong_AsLong(index.ptr());
    if(PyErr_Occurred())throw py::error_already_set();
    return result;
}
double scalar(py::handle value) {double result=PyFloat_AsDouble(value.ptr());if(PyErr_Occurred())throw py::error_already_set();return result;}
}

void bind_max_tree(py::module_& m) {
    m.attr("max_tree_interface")=skimagexx::max_tree_interface;
    m.attr("max_tree_type_capabilities")=skimagexx::max_tree_type_capabilities;
    m.attr("max_tree_operation_capabilities")=skimagexx::max_tree_operation_capabilities;
    m.attr("max_tree_parallel_traversal")=skimagexx::max_tree_parallel_traversal;
    m.attr("max_tree_stage_interface")=stages::interface;
    m.attr("max_tree_stage_capabilities")=stages::capabilities;
    m.def("max_tree_counts",[](bool reset) {
        py::dict result;
        for(std::size_t i=0;i<counts.size();++i){py::dict row;row["attempted"]=counts[i].attempted;row["completed"]=counts[i].completed;result[names[i]]=row;}
        if(reset)counts={};
        return result;
    },py::arg("reset")=false);
    m.def("max_tree_run",[](py::object input_obj,py::object output_obj,py::object parent_obj,py::object order_obj,
        py::object operation_obj,py::object connectivity_obj,py::object threshold_obj,bool provided) {
        ++counts[0].attempted;
        auto a=array(input_obj);auto dtype=type(a);
        if(a.ndim()!=2 && a.ndim()!=3)throw py::value_error("prepared max-tree spatial rank must be2or3");
        auto parent=typed<std::int64_t>(parent_obj,-1),order=typed<std::int64_t>(order_obj,1);
        std::array<Stamp,3> captured{{Stamp(a),Stamp(parent),Stamp(order)}};
        const auto operation=integer(operation_obj);
        if(operation<0 || operation>5)throw py::value_error("invalid max-tree operation");
        const auto connectivity=provided?1:integer(connectivity_obj);
        if(connectivity<std::numeric_limits<int>::min() || connectivity>std::numeric_limits<int>::max())throw py::value_error("connectivity exceeds C int");
        const double threshold=operation>=2?scalar(threshold_obj):0;
        for(const auto& stamp:captured)stamp.verify();
        py::array out;
        if(operation) {
            out=operation==1?typed<std::uint64_t>(output_obj,-1):array(output_obj);
            if(operation!=1 && !out.dtype().equal(a.dtype()))throw py::type_error("max-tree output dtype mismatch");
        }else if(!output_obj.is_none())throw py::value_error("tree operation output must beNone");
        const auto n=static_cast<std::size_t>(a.size()),rank=static_cast<std::size_t>(a.ndim());
        if(n>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(8*rank))throw py::value_error("max-tree workspace size overflow");
        std::vector<std::int64_t> roots(provided?0:n),scratch(provided?0:n),low(operation>=4?n*rank:0),high(operation>=4?n*rank:0);
        std::vector<std::uint64_t> inverted((operation==3 || operation==5)?(n*static_cast<std::size_t>(a.itemsize())+7)/8:0);
        std::vector<double> attribute(operation>=2?n:0);
        skimagexx::MaxTreeImage input{a.data(),dtype,n,rank,{{1,1,1}}};
        for(std::size_t d=0;d<rank;++d)input.shape[d]=static_cast<std::size_t>(a.shape(py::ssize_t(d)));
        skimagexx::MaxTreeParameters params{skimagexx::MaxTreeOperation(operation),int(connectivity),threshold};
        skimagexx::MaxTreeWorkspace w{
            provided?skimagexx::MaxTreeSpan<std::int64_t>{nullptr,0}:write<std::int64_t>(parent),
            provided?skimagexx::MaxTreeSpan<std::int64_t>{nullptr,0}:write<std::int64_t>(order),
            {roots.data(),roots.size()},{scratch.data(),scratch.size()},
            {inverted.empty()?nullptr:inverted.data(),inverted.empty()?0:n},
            {attribute.data(),attribute.size()},{low.data(),low.size()},{high.data(),high.size()}};
        auto supplied=provided?skimagexx::MaxTreeProvided{read<std::int64_t>(parent),read<std::int64_t>(order)}:skimagexx::MaxTreeProvided{{nullptr,0},{nullptr,0}};
        skimagexx::MaxTreeBuffer output{operation?const_cast<void*>(out.data()):nullptr,operation?static_cast<std::size_t>(out.size()):0};
        skimagexx::MaxTreeResult result{};const char* unsupported=nullptr;
        {
            py::gil_scoped_release release;
            // Unsupported prepared metadata is explicit; the owned wrapper may
            // then choose its source-provider stage route, never upstream.
            unsupported=skimagexx::max_tree_ineligibility(input,params,w,output,supplied);
            if(!unsupported)result=skimagexx::max_tree_into(input,params,w,output,supplied);
        }
        if(unsupported){PyErr_SetString(PyExc_NotImplementedError,unsupported);throw py::error_already_set();}
        ++counts[0].completed;
        py::dict info;info["tree_reused"]=result.tree_reused;info["preparation_workers"]=result.preparation_workers;return info;
    },py::arg("image"),py::arg("output"),py::arg("parent"),py::arg("traverser"),py::arg("operation"),py::arg("connectivity"),py::arg("threshold"),py::arg("provided")=false);
    m.def("max_tree_build_stage",[](py::object a0,py::object a1,py::object a2,py::object a3,py::object a4,py::object a5,py::object a6,py::object a7) {
        ++counts[1].attempted;
        auto a=array(a0,1),mask=typed<std::uint8_t>(a1),structure=typed<std::int32_t>(a2),shape=typed<std::int32_t>(a3),points=typed<std::int32_t>(a4,2),parent=typed<std::int64_t>(a5),order=typed<std::int64_t>(a6),roots=typed<std::int64_t>(a7);
        auto input=image(a);auto msk=read<std::uint8_t>(mask);auto st=read<std::int32_t>(structure);auto sh=read<std::int32_t>(shape);auto pt=read<std::int32_t>(points);auto p=write<std::int64_t>(parent);auto s=read<std::int64_t>(order);auto r=write<std::int64_t>(roots);
        {py::gil_scoped_release release;stages::build(input,msk,st,sh,pt,p,s,r);}
        ++counts[1].completed;
    });
    m.def("max_tree_area_stage",[](py::object a0,py::object a1,py::object a2,py::object a3) {
        ++counts[2].attempted;
        auto a=array(a0,1),p=typed<std::int64_t>(a1),s=typed<std::int64_t>(a2),attribute=typed<double>(a3);
        auto input=image(a);auto supplied=tree(p,s);auto attr=write<double>(attribute);
        {py::gil_scoped_release release;stages::area(input,supplied,attr);}
        ++counts[2].completed;
    });
    m.def("max_tree_extension_stage",[](py::object a0,py::object a1,py::object a2,py::object a3,py::object a4,py::object a5,py::object a6) {
        ++counts[3].attempted;
        auto a=array(a0,1),shape=typed<std::int32_t>(a1),p=typed<std::int64_t>(a2),s=typed<std::int64_t>(a3),attribute=typed<double>(a4),low=typed<double>(a5,2,true),high=typed<double>(a6,2,true);
        if(low.shape(0)!=a.size() || high.shape(0)!=a.size())throw py::value_error("coordinate row count mismatch");
        auto input=image(a);auto sh=read<std::int32_t>(shape);auto supplied=tree(p,s);auto attr=write<double>(attribute);auto lo=coordinates(low);auto hi=coordinates(high);
        {py::gil_scoped_release release;stages::extension(input,sh,supplied,attr,lo,hi);}
        ++counts[3].completed;
    });
    m.def("max_tree_filter_stage",[](py::object a0,py::object a1,py::object a2,py::object a3,py::object a4,py::object threshold_obj) {
        ++counts[4].attempted;
        auto a=array(a0,1),output=array(a1,1),p=typed<std::int64_t>(a2),s=typed<std::int64_t>(a3),attribute=typed<double>(a4);
        if(!a.dtype().equal(output.dtype()))throw py::type_error("direct-filter output dtype mismatch");
        auto input=image(a);auto supplied=tree(p,s);auto attr=read<double>(attribute);
        skimagexx::MaxTreeBuffer out{const_cast<void*>(output.data()),static_cast<std::size_t>(output.size())};
        std::array<Stamp,5> captured{{Stamp(a),Stamp(output),Stamp(p),Stamp(s),Stamp(attribute)}};
        const double threshold=scalar(threshold_obj);
        for(const auto& stamp:captured)stamp.verify();
        {py::gil_scoped_release release;stages::direct_filter(input,out,supplied,attr,threshold);}
        ++counts[4].completed;
    });
    m.def("max_tree_maxima_stage",[](py::object a0,py::object a1,py::object a2,py::object a3) {
        ++counts[5].attempted;
        auto a=array(a0,1),output=typed<std::uint64_t>(a1),p=typed<std::int64_t>(a2),s=typed<std::int64_t>(a3);
        auto input=image(a);auto out=write<std::uint64_t>(output);auto supplied=tree(p,s);
        {py::gil_scoped_release release;stages::local_maxima(input,out,supplied);}
        ++counts[5].completed;
    });
}
