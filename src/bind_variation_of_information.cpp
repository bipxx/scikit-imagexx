#include <skimagexx/variation_of_information.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <utility>
namespace py=pybind11;
namespace {
struct Log2Provider {
    py::object callable;
    static bool batch(const double* values,double* logarithms,std::size_t count,void* context) {
        // Native entropy stays on the calling thread. Enter the caller's Python
        // context only for the explicitly supplied public whole-array provider.
        py::gil_scoped_acquire acquire;
        const auto& owner=*static_cast<Log2Provider*>(context);
        py::array_t<double> input(count);
        if(count)std::memcpy(input.mutable_data(),values,count*sizeof(double));
        PyObject* returned=PyObject_CallOneArg(owner.callable.ptr(),input.ptr());
        if(!returned)throw py::error_already_set();
        auto result=py::reinterpret_steal<py::object>(returned);
        if(Py_TYPE(result.ptr())!=Py_TYPE(input.ptr()))
            throw py::type_error("VI log2 provider must return a plain ndarray");
        auto array=py::reinterpret_borrow<py::array>(result);
        if(array.ndim()!=1||array.size()!=static_cast<py::ssize_t>(count)
           ||array.itemsize()!=8||array.dtype().kind()!='f'
           ||!py::cast<bool>(array.dtype().attr("isnative"))
           ||!(array.flags()&py::array::c_style)||!array.owndata()
           ||!py::cast<bool>(array.attr("flags").attr("aligned"))
           ||(count&&array.strides(0)!=static_cast<py::ssize_t>(sizeof(double))))
            throw py::type_error("VI log2 provider must return owned contiguous native float64 words");
        skimagexx::bindings::check_array_storage(array);
        if(count)std::memcpy(logarithms,array.data(),count*sizeof(double));
        return true;
    }
};
struct InnerProductProvider {
    py::object callable;
    static bool product(const double* left,const double* right,std::size_t count,
        double* output,void* context) {
        py::gil_scoped_acquire acquire;
        const auto& owner=*static_cast<InnerProductProvider*>(context);
        py::array_t<double> a(count),b(count);
        if(count) {
            std::memcpy(a.mutable_data(),left,count*sizeof(double));
            std::memcpy(b.mutable_data(),right,count*sizeof(double));
        }
        PyObject* returned=PyObject_CallFunctionObjArgs(owner.callable.ptr(),a.ptr(),b.ptr(),nullptr);
        if(!returned)throw py::error_already_set();
        auto result=py::reinterpret_steal<py::object>(returned);
        // np.matmul on two plain1D f64 arrays returns a plain np.float64.
        // Public dtype/type and buffer protocol transport its exact scalar
        // word without conversion, private NumPy layouts or Python arithmetic.
        const auto scalar_type=a.dtype().attr("type");
        if(reinterpret_cast<PyObject*>(Py_TYPE(result.ptr()))!=scalar_type.ptr())
            throw py::type_error("VI matmul provider must return a plain native float64 scalar");
        const auto buffer=py::reinterpret_borrow<py::buffer>(result).request();
        if(buffer.ndim!=0||buffer.size!=1||buffer.itemsize!=sizeof(double)
           ||buffer.format!=py::format_descriptor<double>::format()||!buffer.ptr)
            throw py::type_error("VI matmul provider must return one native float64 buffer word");
        std::memcpy(output,buffer.ptr,sizeof(double));
        return true;
    }
};
}

void bind_variation_of_information(py::module_& module) {
    module.def("variation_of_information_csr",[](const py::array& data,
        const py::array& indices,const py::array& indptr,std::size_t rows,
        std::size_t columns,int workers,py::object log2_provider,py::object matmul_provider)->py::object {
        // Stable typed buffers cross this independently callable native ABI.
        for(const auto* array:{&data,&indices,&indptr}) {
            const bool floating=array==&data;
            if(array->ndim()!=1||array->itemsize()!=8||array->dtype().kind()!=(floating?'f':'i')
                ||!py::cast<bool>(array->dtype().attr("isnative"))
                ||!(array->flags()&py::array::c_style)
                ||!py::cast<bool>(array->attr("flags").attr("aligned")))return py::none();
            skimagexx::bindings::check_array_storage(*array);
        }
        if(indices.size()!=data.size()||rows==0||indptr.size()!=static_cast<py::ssize_t>(rows+1))return py::none();
        if(!log2_provider.is_none()&&!PyCallable_Check(log2_provider.ptr()))
            throw py::type_error("VI log2 provider must be callable");
        if(!matmul_provider.is_none()&&!PyCallable_Check(matmul_provider.ptr()))
            throw py::type_error("VI matmul provider must be callable");
        if(!matmul_provider.is_none()&&log2_provider.is_none())
            throw py::type_error("VI matmul provider requires an explicit log2 provider");
        Log2Provider provider{std::move(log2_provider)};
        InnerProductProvider product{std::move(matmul_provider)};
        const bool supplied=!provider.callable.is_none();
        const bool product_supplied=!product.callable.is_none();
        skimagexx::VariationInformation result;bool accepted;
        {
            py::gil_scoped_release release;
            skimagexx::VariationTableView table{static_cast<const double*>(data.data()),
                static_cast<const std::int64_t*>(indices.data()),static_cast<const std::int64_t*>(indptr.data()),
                rows,columns,static_cast<std::size_t>(data.size())};
            if(product_supplied)
                accepted=skimagexx::variation_of_information_compatible(table,result,
                    {Log2Provider::batch,&provider},{InnerProductProvider::product,&product},workers);
            else accepted=supplied
                    ?skimagexx::variation_of_information_compatible(table,result,{Log2Provider::batch,&provider},workers)
                    :skimagexx::variation_of_information(table,result,workers);
        }
        if(!accepted)return py::none();
        py::array_t<double> out(2);std::memcpy(out.mutable_data(),result.values.data(),2*sizeof(double));
        return out;
    },py::arg("data").noconvert(),py::arg("indices").noconvert(),py::arg("indptr").noconvert(),
      py::arg("rows"),py::arg("columns"),py::arg("workers")=0,
      py::arg("log2_provider")=py::none(),py::arg("matmul_provider")=py::none());
}
