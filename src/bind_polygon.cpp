#include "polygon.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>
namespace py = pybind11;
namespace {
void check_column(const py::array& column) {
    if(column.ndim()!=1 || !py::array_t<double,py::array::c_style>::check_(column)
       || reinterpret_cast<std::uintptr_t>(column.data())%alignof(double))
        throw py::value_error("polygon columns require aligned native contiguous float64 storage");
    skimagexx::bindings::check_array_storage(column);
}
void check_polygon_size(py::ssize_t vertices,py::ssize_t count) {
    if(vertices==0 && count!=0)
        throw py::value_error("polygon vertices must be nonempty for nonempty work");
}
}
void bind_polygon(py::module_& module) {
    module.def("points_in_poly",[](const py::array& x,const py::array& y,
                                   const py::array& vx,const py::array& vy,int workers) {
        py::array_t<bool> output(x.size());
        check_column(x);check_column(y);check_column(vx);check_column(vy);
        if(x.size()!=y.size() || vx.size()!=vy.size())
            throw py::value_error("polygon coordinate column lengths differ");
        check_polygon_size(vx.size(),x.size());
        const skimagexx::detail::PolygonView polygon{static_cast<const double*>(vx.data()),
            static_cast<const double*>(vy.data()),std::size_t(vx.size())};
        const auto* px=static_cast<const double*>(x.data());const auto* pydata=static_cast<const double*>(y.data());
        auto* out=reinterpret_cast<std::uint8_t*>(output.mutable_data());
        const auto count=std::size_t(x.size());
        {py::gil_scoped_release release;skimagexx::detail::points_in_poly(polygon,px,pydata,count,out,workers);}
        return output;
    },py::arg("x").noconvert(),py::arg("y").noconvert(),py::arg("vx").noconvert(),py::arg("vy").noconvert(),py::arg("workers"));
    module.def("grid_points_in_poly",[](const py::array& vx,const py::array& vy,
                                        py::ssize_t rows,py::ssize_t columns,int workers) {
        if(rows<0 || columns<0)throw py::value_error("negative dimensions are not allowed");
        if(rows && columns>std::numeric_limits<py::ssize_t>::max()/rows)
            throw py::value_error("polygon grid size exceeds addressable storage");
        py::array_t<std::uint8_t> output({rows,columns});
        check_column(vx);check_column(vy);
        if(vx.size()!=vy.size())throw py::value_error("polygon coordinate column lengths differ");
        check_polygon_size(vx.size(),rows*columns);
        const skimagexx::detail::PolygonView polygon{static_cast<const double*>(vx.data()),
            static_cast<const double*>(vy.data()),std::size_t(vx.size())};
        auto* out=output.mutable_data();
        {py::gil_scoped_release release;skimagexx::detail::grid_points_in_poly(polygon,std::size_t(rows),std::size_t(columns),out,workers);}
        return output;
    },py::arg("vx").noconvert(),py::arg("vy").noconvert(),py::arg("rows"),py::arg("columns"),py::arg("workers"));
}
