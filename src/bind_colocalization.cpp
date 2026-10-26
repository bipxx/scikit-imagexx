#include "skimagexx/colocalization.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace py=pybind11;
namespace {
using Type=skimagexx::ColocType;
using View=skimagexx::ColocView;
constexpr auto size_limit=std::size_t(std::numeric_limits<py::ssize_t>::max());
std::size_t checked_product(std::size_t a,std::size_t b) {
    if(b && a>size_limit/b)throw py::value_error("colocalization allocation size overflows");
    return a*b;
}

// Direct ndarray C slots only: Python .base/.shape/.dtype properties are never
// called. Retain every owner and a builtin-buffer export through every stage.
// Concurrent external writes/resize while the GIL is released are a caller
// precondition; callback-driven changes while held are checked before reads.
class Owner {
    struct Stamp {
        py::array array;
        void* data;
        PyObject* base;
        PyObject* dtype;
        int flags;
        py::ssize_t item;
        std::vector<py::ssize_t> shape,strides;
    };
    struct Lease {
        Py_buffer value{};
        ~Lease(){if(value.obj)PyBuffer_Release(&value);}
    };
    std::vector<std::unique_ptr<Lease>> leases_;
    std::vector<Stamp> stamps_;
    std::vector<py::object> owners_;
    std::uintptr_t allocation_=0;
    std::size_t bytes_=0;

    static std::size_t count(const Stamp& stamp) {
        for(auto length:stamp.shape)if(length<0)throw py::value_error("colocalization negative array extent");
        if(std::find(stamp.shape.begin(),stamp.shape.end(),0)!=stamp.shape.end())return 0;
        std::size_t result=1;
        for(auto length:stamp.shape)result=checked_product(result,std::size_t(length));
        return result;
    }
public:
    Owner(const py::array& input,const py::object& mmap_type) {
        py::object current=input;
        for(unsigned depth=0;depth<64;++depth) {
            if(std::any_of(owners_.begin(),owners_.end(),[&](const auto& old){return old.ptr()==current.ptr();}))
                throw py::value_error("colocalization array owner cycle");
            owners_.push_back(current);
            if(PyMemoryView_Check(current.ptr())) {
                // A memoryview alone does not certify its exporter: an ndarray
                // exporter can be resized with refcheck=False while the old
                // memoryview keeps its stale pointer. Pin its export AND follow
                // the direct C owner to stamp the actual allocation as well.
                auto lease=std::make_unique<Lease>();
                if(PyObject_GetBuffer(current.ptr(),&lease->value,PyBUF_SIMPLE)<0)throw py::error_already_set();
                leases_.push_back(std::move(lease));
                auto* base=PyMemoryView_GET_BASE(current.ptr());
                if(!base)throw py::value_error("colocalization requires a bounded memoryview exporter");
                current=py::reinterpret_borrow<py::object>(base);continue;
            }
            if(!py::isinstance<py::array>(current)) {
                const bool trusted=PyBytes_CheckExact(current.ptr()) || PyByteArray_CheckExact(current.ptr()) ||
                    Py_TYPE(current.ptr())==reinterpret_cast<PyTypeObject*>(mmap_type.ptr());
                if(!trusted)throw py::value_error("colocalization requires a bounded ndarray or builtin-buffer owner");
                auto lease=std::make_unique<Lease>();
                if(PyObject_GetBuffer(current.ptr(),&lease->value,PyBUF_SIMPLE)<0)throw py::error_already_set();
                if(lease->value.len<0)throw py::value_error("colocalization invalid buffer extent");
                allocation_=reinterpret_cast<std::uintptr_t>(lease->value.buf);
                bytes_=std::size_t(lease->value.len);leases_.push_back(std::move(lease));check();return;
            }
            auto array=py::reinterpret_borrow<py::array>(current);
            const auto* raw=py::detail::array_proxy(array.ptr());
            if(array.ndim()<0 || array.ndim()>py::ssize_t(skimagexx::colocalization_max_rank))
                throw py::value_error("colocalization array rank exceeds bound");
            Stamp stamp{array,raw->data,raw->base,reinterpret_cast<PyObject*>(raw->descr),raw->flags,array.itemsize(),{}, {}};
            if(stamp.item<0)throw py::value_error("colocalization negative element size");
            for(py::ssize_t axis=0;axis<array.ndim();++axis){stamp.shape.push_back(array.shape(axis));stamp.strides.push_back(array.strides(axis));}
            count(stamp);stamps_.push_back(std::move(stamp));
            if(raw->base){current=py::reinterpret_borrow<py::object>(raw->base);continue;}
            if(!(raw->flags&py::detail::npy_api::NPY_ARRAY_OWNDATA_) || !(raw->flags&(py::array::c_style|py::array::f_style)))
                throw py::value_error("colocalization requires an owned contiguous root allocation");
            allocation_=reinterpret_cast<std::uintptr_t>(raw->data);
            bytes_=checked_product(count(stamps_.back()),std::size_t(stamps_.back().item));check();return;
        }
        throw py::value_error("colocalization owner chain exceeds bound");
    }
    void check()const {
        const auto limit=std::numeric_limits<std::uintptr_t>::max();
        if((bytes_ && !allocation_) || bytes_>limit-allocation_)throw py::value_error("colocalization owner allocation overflows");
        for(const auto& stamp:stamps_) {
            const auto* raw=py::detail::array_proxy(stamp.array.ptr());
            if(raw->data!=stamp.data || raw->base!=stamp.base || reinterpret_cast<PyObject*>(raw->descr)!=stamp.dtype ||
                raw->flags!=stamp.flags || stamp.array.itemsize()!=stamp.item || stamp.array.ndim()!=py::ssize_t(stamp.shape.size()))
                throw py::value_error("colocalization owner metadata changed during the call");
            for(py::ssize_t axis=0;axis<stamp.array.ndim();++axis)
                if(stamp.array.shape(axis)!=stamp.shape[std::size_t(axis)] || stamp.array.strides(axis)!=stamp.strides[std::size_t(axis)])
                    throw py::value_error("colocalization owner metadata changed during the call");
            if(!count(stamp))continue;
            std::uintptr_t negative=0,positive=0;
            for(std::size_t axis=0;axis<stamp.shape.size();++axis) {
                const auto stride=stamp.strides[axis];
                const auto magnitude=stride<0 ? std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
                const auto length=std::uintptr_t(stamp.shape[axis]-1);
                auto& extent=stride<0 ? negative:positive;
                if(length && magnitude>(limit-extent)/length)throw py::value_error("colocalization byte extent overflows");
                extent+=length*magnitude;
            }
            const auto data=reinterpret_cast<std::uintptr_t>(stamp.data),item=std::uintptr_t(stamp.item);
            if(negative>data || positive>limit-data || item>limit-data-positive || data-negative<allocation_ ||
                data+positive+item>allocation_+bytes_)
                throw py::value_error("colocalization array escapes its current owner allocation");
        }
    }
    const void* allocation()const{return reinterpret_cast<const void*>(allocation_);}
    std::size_t bytes()const{return bytes_;}
};

class Guards {
    py::object mmap_type_;
    std::vector<py::array> arrays_;
    std::vector<std::unique_ptr<Owner>> owners_;
public:
    explicit Guards(py::object mmap_type):mmap_type_(std::move(mmap_type)){}
    Owner& add(const py::array& input) {
        for(std::size_t i=0;i<arrays_.size();++i)if(arrays_[i].ptr()==input.ptr())return *owners_[i];
        auto owner=std::make_unique<Owner>(input,mmap_type_);
        arrays_.push_back(input);owners_.push_back(std::move(owner));return *owners_.back();
    }
    void check()const{for(const auto& owner:owners_)owner->check();}
};

bool real_type(const py::array& array,Type& result) {
    const auto dtype=array.dtype();
    if(dtype.has_fields())return false;
    // Only NumPy's legacy builtin real descriptors. A user DType with the same
    // kind/itemsize must not inherit the builtin loop contract. NPY_HALF=23 is
    // the stable public NumPy1/2 legacy type number; LONGDOUBLE is not float64.
    const int number=dtype.num();
    if(!((number>=0 && number<=12) || number==23))return false;
    const auto size=array.itemsize();
    switch(dtype.kind()) {
    case 'b':if(size==1){result=Type::boolean;return true;}return false;
    case 'u':
        if(size==1)result=Type::uint8;else if(size==2)result=Type::uint16;
        else if(size==4)result=Type::uint32;else if(size==8)result=Type::uint64;else return false;return true;
    case 'i':
        if(size==1)result=Type::int8;else if(size==2)result=Type::int16;
        else if(size==4)result=Type::int32;else if(size==8)result=Type::int64;else return false;return true;
    case 'f':
        if(size==2)result=Type::float16;else if(size==4)result=Type::float32;else if(size==8)result=Type::float64;else return false;return true;
    default:return false;
    }
}
const char* dtype_name(Type type) {
    switch(type) {
    case Type::boolean:return "bool";case Type::uint8:return "uint8";case Type::int8:return "int8";
    case Type::uint16:return "uint16";case Type::int16:return "int16";
    case Type::uint32:return "uint32";case Type::int32:return "int32";
    case Type::uint64:return "uint64";case Type::int64:return "int64";
    case Type::float16:return "float16";case Type::float32:return "float32";case Type::float64:return "float64";
    }throw py::value_error("unknown colocalization dtype");
}
struct Input {
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    View value;
    Input(Guards& guards,const py::array& input,Type type) {
        auto& owner=guards.add(input);
        for(py::ssize_t axis=0;axis<input.ndim();++axis){shape.push_back(std::size_t(input.shape(axis)));strides.push_back(input.strides(axis));}
        const std::uint16_t one=1;const bool little=*reinterpret_cast<const unsigned char*>(&one)!=0;
        const auto order=input.dtype().byteorder();
        value={input.data(),owner.allocation(),owner.bytes(),{shape.data(),shape.size()},
               {strides.data(),strides.size()},type,(order=='>' && little)||(order=='<' && !little)};
    }
};
py::array output(Guards& guards,const Input& input,const skimagexx::ColocOrder& order,Type type) {
    std::vector<py::ssize_t> shape(input.shape.begin(),input.shape.end()),strides(shape.size(),0);
    std::size_t stride=skimagexx::coloc_itemsize(type);
    for(std::size_t i=0;i<order.rank;++i) {
        const auto axis=order.axes[i];strides[axis]=py::ssize_t(stride);
        stride=checked_product(stride,std::max(input.shape[axis],std::size_t(1)));
    }
    // Dtype construction and NumPy allocation can run Python finalizers. The
    // entry point checks the complete owner set after these allocations and
    // obtaining the writable buffer, immediately before its next pixel read.
    py::array result(py::dtype(dtype_name(type)),shape,strides);
    guards.add(result);return result;
}
skimagexx::ColocBuffer buffer(py::array& array) {
    if(array.nbytes()<0)throw py::value_error("colocalization negative output bytes");
    return {array.mutable_data(),std::size_t(array.nbytes())};
}
py::object scalar(const skimagexx::ColocScalar& value) {
    // NumPy's scalar constructor consumes exact raw storage; it is not float().
    const auto dtype=py::dtype(dtype_name(value.type));
    std::array<unsigned char,8> raw{};
    const auto item=skimagexx::coloc_itemsize(value.type);
    if(item==2){const auto bits=std::uint16_t(value.bits);std::memcpy(raw.data(),&bits,2);}
    else if(item==4){const auto bits=std::uint32_t(value.bits);std::memcpy(raw.data(),&bits,4);}
    else if(item==8)std::memcpy(raw.data(),&value.bits,8);
    else raw[0]=static_cast<unsigned char>(value.bits);
    auto result=py::reinterpret_steal<py::object>(py::detail::npy_api::get().PyArray_Scalar_(raw.data(),dtype.ptr(),nullptr));
    if(!result)throw py::error_already_set();return result;
}
skimagexx::ColocReductionPlan reduction(int profile,std::size_t buffer_elements) {
    if(profile!=1 && profile!=2)throw py::value_error("colocalization requires an audited NumPy profile");
    return {profile==1 ? skimagexx::ColocProfile::numpy_1244:skimagexx::ColocProfile::numpy_254,buffer_elements};
}
}

void bind_colocalization(py::module_& module) {
    const py::object mmap_type=py::module_::import("mmap").attr("mmap");
    py::class_<Guards>(module,"ColocalizationGuards")
        .def(py::init([mmap_type](){return std::make_unique<Guards>(mmap_type);}))
        .def("add",[](Guards& guards,py::array array){guards.add(array);guards.check();},py::arg("array").noconvert())
        .def("check",&Guards::check);
    module.attr("colocalization_interface")=skimagexx::colocalization_interface;
    module.def("coloc_binary",[](Guards& guards,py::array array)->py::object {
        guards.add(array);Type type;if(!real_type(array,type))return py::none();
        Input input(guards,array,type);skimagexx::ColocPredicate valid;
        bool finite;guards.check();{py::gil_scoped_release release;
            finite=skimagexx::coloc_all_finite(input.value);
            if(finite)valid=skimagexx::coloc_binary_valid(input.value);
        }
        if(!finite)return py::none();
        if(!valid.value)return py::make_tuple(false,py::none());
        const auto order=skimagexx::coloc_keep_order(input.value);
        auto result=output(guards,input,order,Type::boolean);const auto target=buffer(result);
        guards.check();{py::gil_scoped_release release;skimagexx::coloc_binary_into(input.value,order,target);}
        return py::make_tuple(true,result);
    },py::arg("guards"),py::arg("array").noconvert());
    module.def("coloc_count",[](Guards& guards,py::array array){
        Input input(guards,array,Type::boolean);
        if(array.dtype().kind()!='b' || array.itemsize()!=1)throw py::type_error("colocalization count requires bool");
        std::uint64_t result;guards.check();{py::gil_scoped_release release;result=skimagexx::coloc_count_nonzero(input.value);}return result;
    },py::arg("guards"),py::arg("array").noconvert());
    module.def("coloc_joint_count",[](Guards& guards,py::array a,py::array b){
        if(a.dtype().kind()!='b' || a.itemsize()!=1 || b.dtype().kind()!='b' || b.itemsize()!=1)throw py::type_error("colocalization joint count requires bool");
        Input first(guards,a,Type::boolean),second(guards,b,Type::boolean);
        std::uint64_t result;guards.check();{py::gil_scoped_release release;result=skimagexx::coloc_joint_count(first.value,second.value);}return result;
    },py::arg("guards"),py::arg("image0").noconvert(),py::arg("image1").noconvert());
    module.def("coloc_min_negative",[](Guards& guards,py::array array)->py::object {
        guards.add(array);Type type;if(!real_type(array,type))return py::none();
        Input input(guards,array,type);skimagexx::ColocPredicate result;
        bool finite;guards.check();{py::gil_scoped_release release;
            finite=skimagexx::coloc_all_finite(input.value);
            if(finite)result=skimagexx::coloc_min_negative(input.value);
        }
        if(!finite)return py::none();return py::bool_(result.value);
    },py::arg("guards"),py::arg("array").noconvert());
    module.def("coloc_gather",[](Guards& guards,py::array array,py::array mask)->py::object {
        guards.add(array);guards.add(mask);Type type;if(!real_type(array,type))return py::none();
        if(mask.dtype().kind()!='b' || mask.itemsize()!=1)throw py::type_error("colocalization gather requires bool mask");
        Input input(guards,array,type),selection(guards,mask,Type::boolean);std::uint64_t selected;
        guards.check();{py::gil_scoped_release release;selected=skimagexx::coloc_count_nonzero(selection.value);}
        if(selected>size_limit)throw py::value_error("colocalization gather size overflows");
        py::array result(array.dtype(),std::vector<py::ssize_t>{py::ssize_t(selected)});
        guards.add(result);const auto target=buffer(result);guards.check();
        {py::gil_scoped_release release;skimagexx::coloc_gather_into(input.value,selection.value,target,std::size_t(selected));}
        return result;
    },py::arg("guards"),py::arg("array").noconvert(),py::arg("mask").noconvert());
    module.def("coloc_sum",[](Guards& guards,py::array array,int profile,std::size_t buffer_elements)->py::object {
        guards.add(array);Type type;if(!real_type(array,type))return py::none();
        Input input(guards,array,type);const auto plan=reduction(profile,buffer_elements);
        skimagexx::ColocScalar result;guards.check();
        try {py::gil_scoped_release release;result=skimagexx::coloc_sum(input.value,plan);}
        catch(const skimagexx::ColocProviderRequired&){return py::none();}
        auto boxed=scalar(result);guards.check();return py::make_tuple(boxed,result.fp_status);
    },py::arg("guards"),py::arg("array").noconvert(),py::arg("profile"),py::arg("buffer_elements"));
    module.def("coloc_product",[](Guards& guards,py::array a,py::object second)->py::object {
        guards.add(a);Type type;if(!real_type(a,type))return py::none();
        Input first(guards,a,type);std::unique_ptr<Input> other;
        if(!second.is_none()) {
            if(!py::isinstance<py::array>(second))throw py::type_error("colocalization product requires an ndarray");
            auto b=py::reinterpret_borrow<py::array>(second);guards.add(b);Type btype;
            if(!real_type(b,btype) || (btype!=type && btype!=Type::boolean))return py::none();
            other=std::make_unique<Input>(guards,b,btype);
        }
        const auto order=other ? skimagexx::coloc_keep_order(first.value,other->value):skimagexx::coloc_keep_order(first.value);
        const auto result_type=other ? type:skimagexx::coloc_square_type(type);
        auto result=output(guards,first,order,result_type);const auto target=buffer(result);unsigned status=0;
        guards.check();try {py::gil_scoped_release release;
            if(other)skimagexx::coloc_multiply_into(first.value,other->value,order,target,status);
            else skimagexx::coloc_square_into(first.value,order,target,status);
        }
        catch(const skimagexx::ColocProviderRequired&){return py::none();}
        return py::make_tuple(result,status);
    },py::arg("guards"),py::arg("image0").noconvert(),py::arg("image1")=py::none());
}
