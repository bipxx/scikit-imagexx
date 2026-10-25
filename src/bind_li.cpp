#include "skimagexx/li.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
namespace {
struct NumpyLiLog {
    struct Loop {PyUFuncGenericFunction function;void* data;};
    py::object owner;
    Loop loops[2];
    static Loop select(const py::object& object,int type) {
        if(!PyObject_TypeCheck(object.ptr(),&PyUFunc_Type))throw std::runtime_error("native logarithm ufunc required");
        const auto* u=reinterpret_cast<PyUFuncObject*>(object.ptr());
        if(u->nargs==2)for(int i=0;i<u->ntypes;++i)if(u->types[2*i]==type && u->types[2*i+1]==type)
            return {u->functions[i],u->data ? u->data[i] : nullptr};
        throw std::runtime_error("native floating logarithm unavailable");
    }
    NumpyLiLog():owner(py::module_::import("numpy").attr("log")),loops{select(owner,NPY_DOUBLE),select(owner,NPY_FLOAT)} {}
    template<class T> static double run(const Loop& loop,double value) {
        alignas(64) T input=T(value);
        alignas(64) T output;
        char* args[]={reinterpret_cast<char*>(&input),reinterpret_cast<char*>(&output)};
        const npy_intp n=1,steps[]={sizeof(T),sizeof(T)};
        loop.function(args,&n,steps,loop.data);return double(output);
    }
    static double logarithm(double value,bool single,void* context) {
        const auto& loop=static_cast<NumpyLiLog*>(context)->loops[int(single)];
        return single ? run<float>(loop,value) : run<double>(loop,value);
    }
    static void poll(void*) {py::gil_scoped_acquire acquire;if(PyErr_CheckSignals()<0)throw py::error_already_set();}
};
bool dtype_matches(const py::dtype& a,const py::dtype& b) {return a.is(b) || (a.itemsize()==b.itemsize() && a.kind()==b.kind() && a.equal(b));}
skimagexx::PixelType pixel_type(const py::dtype& dtype) {
    using Type = skimagexx::PixelType;
    const auto kind = dtype.kind();
    const auto size = dtype.itemsize();
    if (size > 1 && !dtype.attr("isnative").cast<bool>()) throw py::type_error("native byte order is required");
    if (kind == 'b' && size == 1) return Type::boolean;
    if (kind == 'u') {
        if (size == 1) return Type::uint8;
        if (size == 2) return Type::uint16;
        if (size == 4) return Type::uint32;
        if (size == 8) return Type::uint64;
    }
    if (kind == 'i') {
        if (size == 1) return Type::int8;
        if (size == 2) return Type::int16;
        if (size == 4) return Type::int32;
        if (size == 8) return Type::int64;
    }
    if (kind == 'f') {
        if (size == 4) return Type::float32;
        if (size == 8) return Type::float64;
    }
    throw py::type_error("unsupported dtype for native conversion");
}
} // namespace

void bind_li(py::module_& m) {
    // This translation unit owns its NumPy C API tables.
    if(_import_array()<0)throw py::error_already_set();
    if(_import_umath()<0)throw py::error_already_set();
    const auto li_log=std::make_shared<NumpyLiLog>();
    py::class_<skimagexx::LiImage,std::shared_ptr<skimagexx::LiImage>>(m,"_LiImage",py::module_local())
        .def_property_readonly("kind",[](const skimagexx::LiImage& image){return int(image.kind());})
        .def_property_readonly("value",&skimagexx::LiImage::value)
        .def_property_readonly("minimum",&skimagexx::LiImage::minimum)
        .def_property_readonly("maximum",&skimagexx::LiImage::maximum)
        .def("spacing",&skimagexx::LiImage::spacing,py::call_guard<py::gil_scoped_release>())
        .def("run",[li_log](const skimagexx::LiImage& image,double tolerance,bool has_guess,double guess,py::object trace,std::size_t reduction_block){
            struct Call {NumpyLiLog* logarithms;py::object trace;};
            Call context{li_log.get(),trace};
            const skimagexx::LiMath math{
                [](double value,bool single,void* data){return NumpyLiLog::logarithm(value,single,static_cast<Call*>(data)->logarithms);},
                NumpyLiLog::poll,&context,
                trace.is_none() ? nullptr : +[](double value,bool updated,void* data){py::gil_scoped_acquire acquire;static_cast<Call*>(data)->trace(value,updated);},reduction_block};
            skimagexx::LiResult result;
            {py::gil_scoped_release release;result=image.run(tolerance,has_guess,guess,&math);}
            return py::make_tuple(result.value,result.updated,result.status);
        },py::arg("tolerance"),py::arg("has_guess"),py::arg("guess"),py::arg("trace")=py::none(),py::arg("reduction_block")=0);
    m.def("li_prepare",[](py::array image){
        auto prepare=[&](auto dummy){
            using T=decltype(dummy);auto a=py::array_t<T,py::array::c_style>::ensure(image);
            if(!a)throw py::type_error("Li input conversion failed");
            if(reinterpret_cast<std::uintptr_t>(a.data())%alignof(T))a=py::array_t<T,py::array::c_style>::ensure(a.attr("copy")());
            const auto type=pixel_type(a.dtype());const auto* data=a.data();const auto count=std::size_t(a.size());
            py::gil_scoped_release release;
            return std::make_shared<skimagexx::LiImage>(data,count,type);
        };
        if(dtype_matches(image.dtype(),py::dtype::of<bool>()))return prepare(bool{});
        if(dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>()))return prepare(std::uint8_t{});
        if(image.dtype().kind()=='f' && image.itemsize()==4)return prepare(float{});
        if(image.dtype().kind()=='f' && image.itemsize()==8)return prepare(double{});
        throw py::type_error("Li input types are boolean, uint8, float32 and float64");
    });
}
