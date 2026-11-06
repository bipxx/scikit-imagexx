#include "skimagexx/regular_grid.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py=pybind11;
namespace {
skimagexx::ComponentsView view(const py::array& a){
    using P=skimagexx::PixelType;P type;const auto kind=a.dtype().kind();const auto bytes=a.itemsize();
    if(kind=='b')type=P::boolean;
    else if(kind=='i'){switch(bytes){case 1:type=P::int8;break;case 2:type=P::int16;break;case 4:type=P::int32;break;case 8:type=P::int64;break;default:throw py::type_error("regular_grid integer width unsupported");}}
    else if(kind=='u'){switch(bytes){case 1:type=P::uint8;break;case 2:type=P::uint16;break;case 4:type=P::uint32;break;case 8:type=P::uint64;break;default:throw py::type_error("regular_grid integer width unsupported");}}
    else if(kind=='f' && bytes==4)type=P::float32;
    else if(kind=='f' && bytes==8)type=P::float64;
    else throw py::type_error("regular_grid shape dtype unsupported");
    return {a.data(),type,false,!a.dtype().attr("isnative").cast<bool>(),{a.shape(),a.shape()+a.ndim()},{a.strides(),a.strides()+a.ndim()}};
}
[[noreturn]] void translate(const skimagexx::RegularGridError& error){
    auto* kind=error.kind==skimagexx::RegularGridErrorKind::division_zero ? PyExc_ZeroDivisionError : error.kind==skimagexx::RegularGridErrorKind::complex_power ? PyExc_TypeError : PyExc_NotImplementedError;
    PyErr_SetString(kind,error.what());throw py::error_already_set();
}
py::tuple slices(const std::vector<skimagexx::RegularGridSlice>& values){
    py::tuple out(values.size());for(std::size_t i=0;i<values.size();++i){const auto& value=values[i];if(value.kind==skimagexx::RegularGridSliceKind::all)out[i]=py::make_tuple(py::none(),py::none());else out[i]=py::make_tuple(value.start,value.step);}return out;
}
}
void bind_regular_grid(py::module_& m){
    py::class_<skimagexx::RegularGridPrepared>(m,"RegularGridPrepared")
        .def_readonly("space_size",&skimagexx::RegularGridPrepared::space_size)
        .def_property_readonly("ndim",[](const skimagexx::RegularGridPrepared& p){return p.sorted_dimensions.size();});
    m.def("regular_grid_prepare",[](const py::array& a,unsigned bytes){try{return skimagexx::regular_grid_prepare(view(a),bytes);}catch(const skimagexx::RegularGridError& error){translate(error);}});
    m.def("regular_grid_suffix_product",[](const skimagexx::RegularGridPrepared& p,std::size_t first){return skimagexx::regular_grid_suffix_product(p,first);});
    m.def("regular_grid_finish",[](const skimagexx::RegularGridPrepared& p,double points){try{return slices(skimagexx::regular_grid_finish(p,points));}catch(const skimagexx::RegularGridError& error){translate(error);}});
}
