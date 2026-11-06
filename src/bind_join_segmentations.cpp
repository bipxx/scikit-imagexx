#include "skimagexx/join_segmentations.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py=pybind11;
namespace {
skimagexx::JoinType type(const py::dtype& d) {
    if((d.kind()!='i' && d.kind()!='u' && d.kind()!='f') || (d.kind()=='f' && d.itemsize()!=8))throw py::type_error("join stage dtype unsupported");
    if(!d.attr("isnative").cast<bool>())throw py::value_error("join stage requires native byte order");
    return {unsigned(d.itemsize()),d.kind()=='i',d.kind()=='f'};
}
skimagexx::JoinType input(const py::array& a) {
    if(!(a.flags()&py::array::c_style))throw py::value_error("join stage requires C-order input");
    return type(a.dtype());
}
std::vector<py::ssize_t> shape(const py::array& a){return {a.shape(),a.shape()+a.ndim()};}
}
void bind_join_segmentations(py::module_& m) {
    m.def("join_encode",[](const py::array& left,const py::array& right,std::uint64_t factor,py::dtype product,py::dtype sum){
        auto lt=input(left),rt=input(right),mt=type(product),et=type(sum);
        if(shape(left)!=shape(right))throw py::value_error("join stage shape mismatch");
        py::array out(sum,shape(left));
        {py::gil_scoped_release release;skimagexx::join_encode(left.data(),lt,right.data(),rt,left.size(),factor,mt,et,out.mutable_data());}
        return out;
    });
    m.def("join_divmod",[](const py::array& labels,std::uint64_t factor,py::dtype result){
        auto lt=input(labels),dt=type(result);py::array quotient(result,shape(labels)),remainder(result,shape(labels));
        {py::gil_scoped_release release;skimagexx::join_divmod(labels.data(),lt,labels.size(),factor,dt,quotient.mutable_data(),remainder.mutable_data());}
        return py::make_tuple(quotient,remainder);
    });
}
