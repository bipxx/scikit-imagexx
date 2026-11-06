#include <skimagexx/label_pair_metrics.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
namespace py=pybind11;
namespace {
py::object rand_scalar(const skimagexx::detail::LabelPairScalar64& result) {
    auto bits=result.bits;auto dtype=py::dtype::of<double>();
    auto* scalar=py::detail::npy_api::get().PyArray_Scalar_(&bits,dtype.ptr(),nullptr);
    if(!scalar)throw py::error_already_set();
    return py::make_tuple(py::reinterpret_steal<py::object>(scalar),result.fp_status,bits);
}
}

void bind_label_pair_metrics(py::module_& module) {
    // Validate the source buffer before NumPy reshape/conversion may copy it.
    // The existing typed entries validate their own independently callable ABI.
    module.def("label_pair_check_storage",&skimagexx::bindings::check_array_storage,
               py::arg("array").noconvert());
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
    module.def("label_pair_rand_terms",[](const py::array& truth,const py::array& test,
        const py::array& ignored,int workers)->py::object {
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
            static_cast<std::size_t>(ignored.size()),false};
        skimagexx::detail::LabelPairRandTerms terms;bool accepted;
        {py::gil_scoped_release release;accepted=skimagexx::detail::label_pair_rand_terms(input,terms,workers);}
        if(!accepted)return py::none();
        return py::make_tuple(terms.joint,terms.row,terms.column);
    },py::arg("truth").noconvert(),py::arg("test").noconvert(),py::arg("ignored").noconvert(),py::arg("workers")=0);
    module.def("label_pair_rand_divide",[](double first,double second)->py::object {
        skimagexx::detail::LabelPairScalar64 result;
        if(!skimagexx::detail::label_pair_rand_divide(first,second,result))return py::none();
        return rand_scalar(result);
    });
    module.def("label_pair_rand_fscore",[](double joint,double row,double column,double alpha)->py::object {
        skimagexx::detail::LabelPairScalar64 result;
        if(!skimagexx::detail::label_pair_rand_fscore({joint,row,column},alpha,result))return py::none();
        return rand_scalar(result);
    });
    module.def("label_pair_rand_error",[](std::uint64_t bits)->py::object {
        skimagexx::detail::LabelPairScalar64 result;
        if(!skimagexx::detail::label_pair_rand_error(bits,result))return py::none();
        return rand_scalar(result);
    });
}
