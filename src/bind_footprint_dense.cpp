#include "skimagexx/footprint_dense.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <limits>
namespace py=pybind11;

void bind_footprint_dense(py::module_& module) {
    module.def("footprint_dense_into",[](py::array output,
        py::array_t<double,py::array::c_style> axis,int kind,double limit,
        py::array zero,py::array one) {
        if(kind<0 || kind>3)throw py::value_error("invalid dense footprint kind");
        if(!(output.flags()&py::array::c_style) || !output.writeable())
            throw py::value_error("dense footprint output must be writable C-contiguous");
        if(output.dtype().attr("hasobject").cast<bool>())
            throw py::type_error("dense footprint atoms cannot contain Python references");
        if(axis.ndim()!=1)throw py::value_error("dense footprint axis must be one-dimensional");
        for(const auto& atom:{zero,one})
            if(atom.size()!=1 || !atom.dtype().attr("__eq__")(output.dtype()).cast<bool>() || !(atom.flags()&py::array::c_style))
                throw py::value_error("dense footprint atoms must match output dtype");
        if(kind!=0) {
            const auto dimensions=kind==3 ? 3:2;
            if(output.ndim()!=dimensions)throw py::value_error("dense footprint output rank mismatch");
            for(py::ssize_t d=0;d<output.ndim();++d)
                if(output.shape(d)!=axis.size())throw py::value_error("dense footprint output shape mismatch");
        }
        // Reject overlapping buffers before releasing the GIL. No dtype may
        // expose Python references to the byte-writing C++ kernel.
        const auto np=py::module_::import("numpy");
        for(const auto& input:{py::array(axis),zero,one})
            if(np.attr("shares_memory")(output,input).cast<bool>())
                throw py::value_error("disjoint dense footprint buffers required");
        py::gil_scoped_release release;
        skimagexx::footprint_dense_into(output.mutable_data(),std::size_t(output.size()),
            std::size_t(output.itemsize()),zero.data(),one.data(),axis.data(),
            std::size_t(axis.size()),static_cast<skimagexx::DenseFootprint>(kind),limit);
    });
}
