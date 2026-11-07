#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <skimagexx/finite_radon.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <limits>
#include <algorithm>
namespace py=pybind11;
namespace {
struct Input {skimagexx::FiniteRadonInput view;std::vector<std::uint32_t> objects;};
Input input(const py::array& a){
    skimagexx::bindings::check_array_storage(a);using T=skimagexx::PixelType;T t;const auto kind=a.dtype().kind();const auto n=a.itemsize();bool half=false,complex=false;
    if(kind=='b')t=T::boolean;
    else if(kind=='f'){if(n==2){t=T::float32;half=true;}else if(n==4)t=T::float32;else if(n==8)t=T::float64;else throw py::type_error("finite Radon extended precision frontier");}
    else if(kind=='c'){complex=true;if(n==8)t=T::float32;else if(n==16)t=T::float64;else throw py::type_error("finite Radon extended complex frontier");}
    else if(kind=='u'){switch(n){case 1:t=T::uint8;break;case 2:t=T::uint16;break;case 4:t=T::uint32;break;case 8:t=T::uint64;break;default:throw py::type_error("finite Radon unsigned width");}}
    else if(kind=='i'||kind=='m'){switch(n){case 1:t=T::int8;break;case 2:t=T::int16;break;case 4:t=T::int32;break;case 8:t=T::int64;break;default:throw py::type_error("finite Radon signed width");}}
    else if(kind=='O')t=T::uint32;
    else throw py::type_error("finite Radon dtype resolution belongs to metadata boundary");
    skimagexx::CropView v;v.data=a.data();v.itemsize=std::size_t(n);for(py::ssize_t d=0;d<a.ndim();++d){v.shape.push_back(a.shape(d));v.strides.push_back(a.strides(d));}
    Input out;out.view={v,t,!a.dtype().attr("isnative").cast<bool>(),half,complex};
    if(kind=='O'){
        if(a.ndim()!=2||a.shape(0)>4097||a.shape(1)>4096)throw py::value_error("finite Radon object allocation frontier");
        out.objects.resize(std::size_t(a.size()));std::uint64_t maximum=0;
        for(py::ssize_t r=0;r<a.shape(0);++r)for(py::ssize_t c=0;c<a.shape(1);++c){PyObject* value;std::memcpy(&value,static_cast<const unsigned char*>(a.data())+r*a.strides(0)+c*a.strides(1),sizeof(value));
            if(!PyLong_CheckExact(value)&&!PyBool_Check(value))throw py::type_error("finite Radon arbitrary object arithmetic provider frontier");
            const auto integer=PyLong_AsUnsignedLongLong(value);if(PyErr_Occurred())throw py::error_already_set();
            maximum=std::max(maximum,std::uint64_t(integer));if(integer>0xffffffffULL)throw py::value_error("finite Radon object assignment range frontier");out.objects[std::size_t(r*a.shape(1)+c)]=std::uint32_t(integer);
        }
        if(a.shape(1)&&maximum>0xffffffffULL/std::uint64_t(a.shape(1)))throw py::value_error("finite Radon object projection overflow frontier");
        out.view.image.data=out.objects.data();out.view.image.itemsize=4;out.view.image.strides={a.shape(1)*4,4};out.view.swapped=false;
    }
    return out;
}
struct NoticeContext {py::object complex_warning;};
void invalid_cast_notice(){
    // Public generated NumPy ufunc C-API signatures/slots, pinned in the
    // family ABI contract. Headers2.x can build for the lowest1.23 feature
    // API; dispatch MUST use the loaded runtime, not the header ABI version.
    // PyUFunc_GiveFloatingpointErrors is slot46, introduced in NumPy2.0.
    if(PyArray_GetNDArrayCFeatureVersion()>=0x00000012u){
        using GiveErrors=int(*)(const char*,int);
        const auto give=reinterpret_cast<GiveErrors>(PyUFunc_API[46]);
        if(give("cast",NPY_FPE_INVALID)<0)throw py::error_already_set();
    }else{
        // NumPy1.x PyUFunc_GetPyValues(slot25) returns the owned policy object;
        // PyUFunc_handlefperr(slot29) applies warn/raise/call/log/ignore/print.
        using GetValues=int(*)(char*,int*,int*,PyObject**);
        using HandleErrors=int(*)(int,PyObject*,int,int*);
        const auto get=reinterpret_cast<GetValues>(PyUFunc_API[25]);
        const auto handle=reinterpret_cast<HandleErrors>(PyUFunc_API[29]);
        int buffer=0,mask=0,first=1;PyObject* object=nullptr;
        if(get(const_cast<char*>("cast"),&buffer,&mask,&object)<0)throw py::error_already_set();
        const int error=handle(mask,object,NPY_FPE_INVALID,&first);
        Py_XDECREF(object);if(error<0)throw py::error_already_set();
    }
}
void notice(void* opaque,unsigned event){
    py::gil_scoped_acquire acquire;
    if(event==1){auto& ctx=*static_cast<NoticeContext*>(opaque);if(PyErr_WarnEx(ctx.complex_warning.ptr(),"Casting complex values to real discards the imaginary part",1)<0)throw py::error_already_set();}
    else if(event==2){
        invalid_cast_notice();
    }
}
NoticeContext context(bool complex){
    NoticeContext ctx;if(complex){auto np=py::module_::import("numpy");const auto version=np.attr("__version__").cast<std::string>();ctx.complex_warning=version.rfind("1.",0)==0?np.attr("ComplexWarning"):py::module_::import("numpy.exceptions").attr("ComplexWarning");}return ctx;
}
py::array transformed(const skimagexx::FiniteRadonTransform& out){py::array_t<std::uint32_t> result({py::ssize_t(out.rows),py::ssize_t(out.columns)});if(!out.values.empty())std::memcpy(result.mutable_data(),out.values.data(),out.values.size()*4);return result;}
}
void bind_finite_radon(py::module_& m){
    if(_import_array()<0||_import_umath()<0)throw py::error_already_set();
    m.def("frt2",[](const py::array& a){auto in=input(a);auto ctx=context(in.view.complex);skimagexx::FiniteRadonTransform out;{py::gil_scoped_release release;out=skimagexx::frt2(in.view,0,notice,&ctx);}return transformed(out);},py::arg("a").noconvert());
    m.def("_finite_radon_inverse_prefix",[](const py::array& a){auto in=input(a);auto ctx=context(in.view.complex);skimagexx::FiniteRadonTransform out;{py::gil_scoped_release release;out=skimagexx::finite_radon_inverse_prefix(in.view,0,notice,&ctx);}return transformed(out);},py::arg("a").noconvert());
    m.def("ifrt2",[](const py::array& a,unsigned bits,bool weak,bool signed_sum){auto in=input(a);skimagexx::InverseFiniteRadonTransform out;{py::gil_scoped_release release;out=skimagexx::ifrt2(in.view,{bits,weak,signed_sum});}py::array_t<double> result({py::ssize_t(out.rows),py::ssize_t(out.columns)});if(!out.values.empty())std::memcpy(result.mutable_data(),out.values.data(),out.values.size()*8);return result;},py::arg("a").noconvert(),py::arg("accumulator_bits"),py::arg("weak_scalar"),py::arg("signed_sum"));
}
