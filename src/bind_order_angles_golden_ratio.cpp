#include <skimagexx/order_angles_golden_ratio.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bind_array_storage.hpp"
namespace py=pybind11;
namespace {
skimagexx::ComponentsView view(const py::array& a) {
    skimagexx::bindings::check_array_storage(a);
    if(a.ndim()!=1)throw py::value_error("golden angles require a1D array");
    const auto kind=py::str(a.dtype().attr("kind")).cast<std::string>();const auto n=a.itemsize();
    skimagexx::ComponentsView v;v.data=a.data();v.shape={std::size_t(a.shape(0))};v.strides={a.strides(0)};v.swap=!py::cast<bool>(a.dtype().attr("isnative"));
    using T=skimagexx::PixelType;
    if(kind=="b"&&n==1)v.type=T::boolean;
    else if(kind=="u"&&n==1)v.type=T::uint8;else if(kind=="i"&&n==1)v.type=T::int8;
    else if(kind=="u"&&n==2)v.type=T::uint16;else if(kind=="i"&&n==2)v.type=T::int16;
    else if(kind=="u"&&n==4)v.type=T::uint32;else if(kind=="i"&&n==4)v.type=T::int32;
    else if(kind=="u"&&n==8)v.type=T::uint64;else if(kind=="i"&&n==8)v.type=T::int64;
    else if(kind=="f"&&n==2){v.type=T::float32;v.half=true;}
    else if(kind=="f"&&n==4)v.type=T::float32;else if(kind=="f"&&n==8)v.type=T::float64;
    else throw py::type_error("golden angles POD dtype frontier");
    return v;
}
struct State {
    py::array owner;
    skimagexx::GoldenAngleOrder order;
    std::int64_t first;
    State(py::array a,bool weak):owner(a),order(view(a),weak?skimagexx::GoldenScalarPolicy::weak_input:skimagexx::GoldenScalarPolicy::wide),first(order.begin()){}
};
}
void bind_order_angles_golden_ratio(py::module_& m) {
    py::class_<State>(m,"GoldenAngleState")
        .def_property_readonly("first",[](const State& s){return s.first;})
        .def_property_readonly("worker_budget",[](const State& s){return s.order.worker_budget();})
        .def_property_readonly("remaining_indices",[](const State& s){return s.order.remaining();});
    m.def("_order_angles_begin",[](py::array input,bool weak){return State(input,weak);},py::arg("input").noconvert(),py::arg("weak"));
    m.def("_order_angles_resume",[](State& state){return state.order.resume();});
    m.def("_order_angles_next",[](State& state,py::array live){return state.order.next(view(live));},py::arg("state"),py::arg("live").noconvert());
}
