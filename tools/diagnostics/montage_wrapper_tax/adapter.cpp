// Diagnostic only: calls existing genuine ELF symbols; no product math/linking.
#include <Python.h>
#include <skimagexx/montage.hpp>
#include <dlfcn.h>
#include <new>
#include <cstdint>

using namespace skimagexx::detail;
using Entry=void(*)(const MontageInput&,MontageGrid,const void*,MontageOutput,int);
struct Prepared {
    Py_buffer input{},fill{},output{};
    void* handle=nullptr; Entry entries[2]{};
    MontageInput descriptor{}; MontageGrid grid{}; MontageOutput destination{};
    ~Prepared() {
        if(output.obj)PyBuffer_Release(&output);
        if(fill.obj)PyBuffer_Release(&fill);
        if(input.obj)PyBuffer_Release(&input);
        if(handle)dlclose(handle);
    }
};
static constexpr auto capsule_name="montage_diagnostic.prepared";
static void release(PyObject* capsule) {
    delete static_cast<Prepared*>(PyCapsule_GetPointer(capsule,capsule_name));
}
static PyObject* prepare(PyObject*,PyObject* args) {
    const char* core;PyObject *input,*fill,*output;Py_ssize_t rows,columns,padding;
    if(!PyArg_ParseTuple(args,"sOOOnnn",&core,&input,&fill,&output,&rows,&columns,&padding))return nullptr;
    if(rows<1||columns<1||padding<0){PyErr_SetString(PyExc_ValueError,"invalid frozen grid");return nullptr;}
    auto* state=new(std::nothrow) Prepared;
    if(!state)return PyErr_NoMemory();
    if(PyObject_GetBuffer(input,&state->input,PyBUF_STRIDES)<0
       ||PyObject_GetBuffer(fill,&state->fill,PyBUF_STRIDES)<0
       ||PyObject_GetBuffer(output,&state->output,PyBUF_STRIDES|PyBUF_WRITABLE)<0){delete state;return nullptr;}
    const auto& a=state->input;const auto& f=state->fill;const auto& b=state->output;
    if(a.ndim!=4||f.ndim!=1||b.ndim!=3||a.itemsize!=f.itemsize||a.itemsize!=b.itemsize
       ||!PyBuffer_IsContiguous(&f,'C')||!PyBuffer_IsContiguous(&b,'C')
       ||a.shape[3]<1||f.shape[0]!=a.shape[3]||b.shape[2]!=a.shape[3]
       ||a.shape[0]>rows*columns
       ||b.shape[0]!=(a.shape[1]+padding)*rows+padding
       ||b.shape[1]!=(a.shape[2]+padding)*columns+padding){
        delete state;PyErr_SetString(PyExc_ValueError,"prepared buffers differ from frozen grid");return nullptr;
    }
    state->handle=dlopen(core,RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL);
    if(!state->handle){delete state;PyErr_SetString(PyExc_RuntimeError,"genuine core must already be loaded");return nullptr;}
    const char* symbols[]={
      "_ZN9skimagexx6detail12montage_intoERKNS0_12MontageInputENS0_11MontageGridEPKvNS0_13MontageOutputEi",
      "_ZN9skimagexx6detail20montage_generic_intoERKNS0_12MontageInputENS0_11MontageGridEPKvNS0_13MontageOutputEi"};
    Dl_info info[2]{};
    for(int i=0;i<2;++i){
        void* symbol=dlsym(state->handle,symbols[i]);
        if(!symbol||!dladdr(symbol,&info[i])){delete state;PyErr_SetString(PyExc_RuntimeError,"required genuine export absent");return nullptr;}
        state->entries[i]=reinterpret_cast<Entry>(symbol);
    }
    if(info[0].dli_fbase!=info[1].dli_fbase){delete state;PyErr_SetString(PyExc_RuntimeError,"routes resolved to different core instances");return nullptr;}
    state->descriptor.data=a.buf;state->descriptor.itemsize=a.itemsize;
    for(int i=0;i<4;++i){state->descriptor.shape[i]=a.shape[i];state->descriptor.strides[i]=a.strides[i];}
    state->grid={std::size_t(rows),std::size_t(columns),std::size_t(padding)};
    state->destination={b.buf,std::size_t(b.shape[0]),std::size_t(b.shape[1])};
    auto* capsule=PyCapsule_New(state,capsule_name,release);
    if(!capsule)delete state;
    return capsule;
}
static PyObject* call(PyObject*,PyObject* args) {
    PyObject* capsule;int route,loops;
    if(!PyArg_ParseTuple(args,"Oii",&capsule,&route,&loops))return nullptr;
    auto* state=static_cast<Prepared*>(PyCapsule_GetPointer(capsule,capsule_name));
    if(!state)return nullptr;
    if(route<0||route>1||loops<1||loops>100000){PyErr_SetString(PyExc_ValueError,"invalid diagnostic route/loops");return nullptr;}
    const auto entry=state->entries[route];
    Py_BEGIN_ALLOW_THREADS
    for(int i=0;i<loops;++i)entry(state->descriptor,state->grid,state->fill.buf,state->destination,1);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
static PyObject* identity(PyObject*,PyObject* capsule) {
    auto* state=static_cast<Prepared*>(PyCapsule_GetPointer(capsule,capsule_name));
    if(!state)return nullptr;
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K}",
      "input",(unsigned long long)(uintptr_t)state->input.buf,
      "fill",(unsigned long long)(uintptr_t)state->fill.buf,
      "output",(unsigned long long)(uintptr_t)state->output.buf,
      "wrapper",(unsigned long long)(uintptr_t)state->entries[0],
      "generic",(unsigned long long)(uintptr_t)state->entries[1]);
}
static PyMethodDef methods[]={
 {"prepare",prepare,METH_VARARGS,nullptr},{"call",call,METH_VARARGS,nullptr},
 {"identity",identity,METH_O,nullptr},{nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_montage_diagnostic",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__montage_diagnostic(){return PyModule_Create(&module);}
