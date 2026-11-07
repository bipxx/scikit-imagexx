#include <skimagexx/variation_of_information.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
namespace py=pybind11;

void bind_variation_of_information(py::module_& module) {
    module.def("variation_of_information_csr",[](const py::array& data,
        const py::array& indices,const py::array& indptr,std::size_t rows,
        std::size_t columns,int workers)->py::object {
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
        skimagexx::VariationInformation result;bool accepted;
        {
            py::gil_scoped_release release;
            accepted=skimagexx::variation_of_information({static_cast<const double*>(data.data()),
                static_cast<const std::int64_t*>(indices.data()),static_cast<const std::int64_t*>(indptr.data()),
                rows,columns,static_cast<std::size_t>(data.size())},result,workers);
        }
        if(!accepted)return py::none();
        py::array_t<double> out(2);std::memcpy(out.mutable_data(),result.values.data(),2*sizeof(double));
        return out;
    },py::arg("data").noconvert(),py::arg("indices").noconvert(),py::arg("indptr").noconvert(),
      py::arg("rows"),py::arg("columns"),py::arg("workers")=0);
}
