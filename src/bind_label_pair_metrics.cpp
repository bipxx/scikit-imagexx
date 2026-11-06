#include <skimagexx/label_pair_metrics.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
namespace py=pybind11;

void bind_label_pair_metrics(py::module_& module) {
    module.def("label_pair_counts",[](const py::array& truth,const py::array& test,
        const py::array& ignored,bool normalize,int workers)->py::object {
        // External NumPy representation/storage boundary; the SDK entry owns
        // value/count/FP validation once and then uses established invariants.
        for(const auto* value:{&truth,&test,&ignored}) {
            if(value->ndim()!=1||value->dtype().kind()!='i'||value->itemsize()!=8
               ||!py::cast<bool>(value->dtype().attr("isnative"))
               ||!(value->flags()&py::array::c_style)
               ||!py::cast<bool>(value->attr("flags").attr("aligned")))return py::none();
            skimagexx::bindings::check_array_storage(*value);
        }
        if(truth.size()!=test.size())return py::none();
        const skimagexx::detail::LabelPairInput input{
            static_cast<const std::int64_t*>(truth.data()),static_cast<const std::int64_t*>(test.data()),
            static_cast<std::size_t>(truth.size()),static_cast<const std::int64_t*>(ignored.data()),
            static_cast<std::size_t>(ignored.size()),normalize};
        skimagexx::detail::LabelPairCsr counts;bool accepted;
        {
            py::gil_scoped_release release;
            accepted=skimagexx::detail::label_pair_counts(input,counts,workers);
        }
        if(!accepted)return py::none();
        py::array_t<std::int64_t> indptr(static_cast<py::ssize_t>(counts.indptr.size()));
        py::array_t<std::int64_t> indices(static_cast<py::ssize_t>(counts.indices.size()));
        py::array_t<double> data(static_cast<py::ssize_t>(counts.data.size()));
        {
            py::gil_scoped_release release;
            std::memcpy(indptr.mutable_data(),counts.indptr.data(),counts.indptr.size()*sizeof(std::int64_t));
            std::memcpy(indices.mutable_data(),counts.indices.data(),counts.indices.size()*sizeof(std::int64_t));
            std::memcpy(data.mutable_data(),counts.data.data(),counts.data.size()*sizeof(double));
        }
        return py::make_tuple(py::make_tuple(counts.rows,counts.columns),indptr,indices,data,
                              counts.kept_samples,counts.normalization_fp_status);
    },py::arg("truth").noconvert(),py::arg("test").noconvert(),py::arg("ignored").noconvert(),
      py::arg("normalize"),py::arg("workers")=0);
}
