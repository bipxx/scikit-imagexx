#include "skimagexx/footprint_decomposition.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace {
py::tuple plans(const std::vector<skimagexx::FootprintCross>& input) {
    py::tuple result(input.size());std::size_t i=0;
    for(const auto& step:input)result[i++]=py::make_tuple(step.vertical,step.horizontal,step.repetitions);
    return result;
}
}
void bind_footprint_decomposition(py::module_& module) {
    module.def("footprint_sphere_counts",&skimagexx::footprint_sphere_counts);
    module.def("footprint_cross_costs",[](py::array_t<double,py::array::c_style> costs,double limit) {
        if(costs.ndim()!=1)throw py::value_error("cross costs must be one-dimensional");
        std::vector<skimagexx::FootprintCross> result;
        {py::gil_scoped_release release;result=skimagexx::footprint_cross_costs(costs.data(),std::size_t(costs.size()),limit);}
        return plans(result);
    });
    module.def("footprint_cross_sums",[](py::array_t<std::ptrdiff_t,py::array::c_style> sums,std::size_t height) {
        if(sums.ndim()!=1)throw py::value_error("cross sums must be one-dimensional");
        std::vector<skimagexx::FootprintCross> result;
        {py::gil_scoped_release release;result=skimagexx::footprint_cross_sums(sums.data(),std::size_t(sums.size()),height);}
        return plans(result);
    });
    module.def("footprint_element_into",[](py::array output,py::array zero,py::array one,
                                          int kind,int dimensions,int axis,int direction,
                                          std::size_t vertical,std::size_t horizontal) {
        if(!(output.flags()&py::array::c_style) || !output.writeable())throw py::value_error("element output must be writable C-contiguous");
        if(output.dtype().attr("hasobject").cast<bool>())throw py::type_error("element atoms cannot contain Python references");
        for(const auto& atom:{zero,one})
            if(atom.size()!=1 || !atom.dtype().attr("__eq__")(output.dtype()).cast<bool>() || !(atom.flags()&py::array::c_style))
                throw py::value_error("element atoms must match output dtype");
        if(kind==2) {
            if(vertical>std::size_t(PY_SSIZE_T_MAX/2) || horizontal>std::size_t(PY_SSIZE_T_MAX/2))throw py::value_error("cross shape overflow");
            if(output.ndim()!=2 || output.shape(0)!=py::ssize_t(2*vertical+1) || output.shape(1)!=py::ssize_t(2*horizontal+1))throw py::value_error("cross output shape mismatch");
        } else {
            if(output.ndim()!=dimensions)throw py::value_error("element output rank mismatch");
            for(py::ssize_t d=0;d<output.ndim();++d)if(output.shape(d)!=3)throw py::value_error("element output shape mismatch");
        }
        const auto np=py::module_::import("numpy");
        for(const auto& atom:{zero,one})if(np.attr("shares_memory")(output,atom).cast<bool>())throw py::value_error("disjoint element buffers required");
        py::gil_scoped_release release;
        skimagexx::footprint_element_into(output.mutable_data(),std::size_t(output.itemsize()),zero.data(),one.data(),kind,dimensions,axis,direction,vertical,horizontal);
    });
}
