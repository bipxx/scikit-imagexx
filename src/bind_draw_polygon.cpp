#include <skimagexx/draw_polygon.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
namespace py=pybind11;
namespace {
using Plan=skimagexx::detail::DrawPolygonPlan;
using Int=std::intptr_t;
void coordinates(const py::array& a,std::size_t count) {
    if(a.ndim()!=1 || std::size_t(a.size())!=count || !a.dtype().is(py::dtype::of<Int>())
       || !(a.flags()&py::array::c_style) || !a.owndata() || !a.writeable())
        throw py::value_error("polygon coordinates require fresh owning contiguous intp outputs");
    skimagexx::bindings::check_array_storage(a);
}
}
void bind_draw_polygon(py::module_& m) {
    py::class_<Plan,std::shared_ptr<Plan>>(m,"_DrawPolygonPlan")
        .def_property_readonly("count",[](const Plan& p){return p.count;});
    m.def("draw_polygon_prepare",[](const py::array& r,const py::array& c,
        const std::array<Int,2>& first,const std::array<Int,2>& last,int workers)->py::object {
        const std::uint16_t endian=1;
        const char native_order=*reinterpret_cast<const unsigned char*>(&endian) ? '<' : '>';
        for(const auto* a:{&r,&c}) {
            if(a->ndim()!=1 || (a->dtype().num()!=py::dtype::of<double>().num() || (a->dtype().byteorder()!='=' && a->dtype().byteorder()!=native_order)) || !(a->flags()&py::array::c_style) || !a->writeable())
                return py::none();
            skimagexx::bindings::check_array_storage(*a);
        }
        if(r.size()!=c.size())return py::none();
        const skimagexx::detail::DrawPolygonVertices vertices{static_cast<const double*>(r.data()),static_cast<const double*>(c.data()),std::size_t(r.size())};
        const skimagexx::detail::DrawPolygonScan scan{first,last};auto plan=std::make_shared<Plan>();
        bool accepted;
        {py::gil_scoped_release release;accepted=skimagexx::detail::draw_polygon_prepare(vertices,scan,*plan,workers);}
        if(!accepted)return py::none();
        return py::cast(plan);
    },py::arg("rows").noconvert(),py::arg("columns").noconvert(),py::arg("first"),py::arg("last"),py::arg("workers"));
    m.def("draw_polygon_coords_into",[](const Plan& plan,py::array rows,py::array columns,int workers) {
        coordinates(rows,plan.count);coordinates(columns,plan.count);
        const auto a=reinterpret_cast<std::uintptr_t>(rows.data()),b=reinterpret_cast<std::uintptr_t>(columns.data());
        const auto bytes=plan.count*sizeof(Int);
        if(rows.is(columns) || (bytes && (a<b ? b-a<bytes : a-b<bytes)))
            throw py::value_error("polygon coordinate outputs must be independent");
        auto* rr=static_cast<Int*>(rows.mutable_data());auto* cc=static_cast<Int*>(columns.mutable_data());
        {py::gil_scoped_release release;skimagexx::detail::draw_polygon_coords_into(plan,rr,cc,workers);}
        return true;
    },py::arg("plan"),py::arg("rows").noconvert(),py::arg("columns").noconvert(),py::arg("workers"));
    m.def("draw_polygon_mask_into",[](const Plan& plan,py::array output,int workers) {
        if(output.ndim()!=2 || !output.dtype().is(py::dtype::of<bool>()) || !(output.flags()&py::array::c_style)
           || !output.owndata() || !output.writeable())throw py::value_error("polygon mask requires a fresh owning C bool output");
        skimagexx::bindings::check_array_storage(output);
        const auto rows=std::size_t(output.shape(0)),columns=std::size_t(output.shape(1));
        auto* data=static_cast<std::uint8_t*>(output.mutable_data());
        {py::gil_scoped_release release;skimagexx::detail::draw_polygon_mask_into(plan,rows,columns,data,workers);}
        return true;
    },py::arg("plan"),py::arg("output").noconvert(),py::arg("workers"));
}
