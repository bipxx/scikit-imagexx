#include "skimagexx/symmetric_eigen.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace py=pybind11;
void bind_symmetric_eigen(py::module_& m) {
    m.def("symmetric_eigenvalues",[](py::object elements,bool legacy_scalars)->py::object {
        const auto np=py::module_::import("numpy");
        const auto array_type=np.attr("ndarray"),memmap_type=np.attr("memmap");
        const auto float32_type=np.attr("float32"),float64_type=np.attr("float64");
        const auto array_like=[&](py::handle x){const auto t=py::type::of(x);return t.is(array_type) || t.is(memmap_type);};
        const auto field_like=[&](py::handle x){const auto t=py::type::of(x);return array_like(x) || t.is(float32_type) || t.is(float64_type);};
        if(!PyList_CheckExact(elements.ptr()) && !PyTuple_CheckExact(elements.ptr()) && !array_like(elements))return py::none();
        if(array_like(elements) && py::reinterpret_borrow<py::array>(elements).ndim()==0)return py::none();
        auto sequence=py::reinterpret_borrow<py::sequence>(elements);const auto length=sequence.size();
        if(!length)return py::none();
        const auto original_first=sequence[0];if(!field_like(original_first))return py::none();
        const auto first=py::array::ensure(original_first);if(!first)return py::none();
        const auto scalar_type=first.dtype().attr("type");
        if(!scalar_type.is(float32_type) && !scalar_type.is(float64_type))return py::none();
        const bool single=first.itemsize()==4,pair=length==3;
        const auto dimension=pair ? std::size_t(2) : std::size_t(first.ndim());
        if(!dimension || dimension>64 || length<dimension*(dimension+1)/2)return py::none();
        if(!pair && first.ndim()>30){
            // NumPy's supported rank differs between major releases. Probe
            // metadata only, matching the reference's intermediate matrix.
            py::tuple probe(first.ndim()+2);
            for(py::ssize_t i=0;i<first.ndim()+2;++i)probe[i]=py::int_(0);
            try{np.attr("empty")(probe);}
            catch(py::error_already_set& error){if(error.matches(PyExc_ValueError)){error.restore();PyErr_Clear();return py::none();}throw;}
        }
        const auto count=dimension*(dimension+1)/2;
        const std::vector<py::ssize_t> shape(first.shape(),first.shape()+first.ndim());
        py::tuple target_shape(shape.size());for(std::size_t i=0;i<shape.size();++i)target_shape[i]=py::int_(shape[i]);
        std::vector<py::array> raw;raw.reserve(count);bool middle_scalar=false;
        for(std::size_t field=0;field<count;++field){
            const auto original=sequence[py::ssize_t(field)];if(!field_like(original))return py::none();
            if(field==1)middle_scalar=!array_like(original);
            auto array=py::array::ensure(original);if(!array || array.dtype().kind()!='f' || !array.dtype().attr("type").is(scalar_type))return py::none();
            const bool same=array.ndim()==first.ndim() && std::equal(array.shape(),array.shape()+array.ndim(),first.shape());
            if(!same){
                try{array=py::array::ensure(np.attr("broadcast_to")(array,target_shape));}
                catch(py::error_already_set& error){if(error.matches(PyExc_ValueError) || error.matches(PyExc_TypeError)){error.restore();PyErr_Clear();return py::none();}throw;}
            }
            raw.push_back(std::move(array));
        }
        skimagexx::PairPrecision precision=skimagexx::PairPrecision::native;
        if(pair && single && first.ndim()==0 && legacy_scalars)precision=middle_scalar ? skimagexx::PairPrecision::wide_scalar : skimagexx::PairPrecision::wide_scalar_narrow_square;
        auto execute=[&](auto dummy)->py::object {
            using T=decltype(dummy);
            py::dtype dtype=(pair || !single) ? first.dtype() : py::dtype::of<T>();
            if(!dtype.attr("isnative").cast<bool>())dtype=py::reinterpret_borrow<py::dtype>(dtype.attr("newbyteorder")("="));
            auto out_shape=shape;
            if(pair)out_shape.insert(out_shape.begin(),py::ssize_t(dimension));else out_shape.push_back(py::ssize_t(dimension));
            py::array output(dtype,out_shape);
            std::vector<py::array_t<T,py::array::c_style|py::array::forcecast>> prepared;prepared.reserve(count);
            std::vector<const void*> fields;fields.reserve(count);
            for(auto& input:raw){
                auto array=py::array_t<T,py::array::c_style|py::array::forcecast>::ensure(input);
                if(!array)return py::none();
                if(reinterpret_cast<std::uintptr_t>(array.data())%alignof(T))array=decltype(array)::ensure(array.attr("copy")());
                prepared.push_back(std::move(array));fields.push_back(prepared.back().data());
            }
            bool supported;
            {py::gil_scoped_release release;supported=skimagexx::symmetric_eigenvalues_into(fields.data(),output.mutable_data(),std::size_t(first.size()),dimension,single,pair,precision,!pair);}
            if(!supported)return py::none();
            if(pair && !first.dtype().attr("isnative").cast<bool>())return output.attr("astype")(first.dtype(),py::arg("copy")=false);
            if(pair)return output;
            // Preserve NumPy's reversed-last-axis/transposed view without a
            // second full output copy; the solver writes ascending AoS data.
            auto leading_shape=shape;leading_shape.insert(leading_shape.begin(),py::ssize_t(dimension));
            std::vector<py::ssize_t> strides{ -output.strides(output.ndim()-1) };
            for(py::ssize_t axis=0;axis<first.ndim();++axis)strides.push_back(output.strides(axis));
            const auto* start=static_cast<const char*>(output.data())+(dimension-1)*std::size_t(output.strides(output.ndim()-1));
            return py::array(dtype,leading_shape,strides,start,output);
        };
        return single ? execute(float{}) : execute(double{});
    });
}
