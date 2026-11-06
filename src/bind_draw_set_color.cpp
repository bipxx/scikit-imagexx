#include <skimagexx/draw_set_color.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py=pybind11;
void bind_draw_set_color(py::module_& m) {
    m.def("draw_set_color",[](py::array image,py::array rows,py::array columns,py::array color,py::array alpha,int workers,bool integer_product){
        const auto f64=[](const py::array& a){return a.dtype().num()==py::dtype::of<double>().num()&&a.dtype().attr("isnative").cast<bool>();};
        const auto index=[](const py::array& a){return a.dtype().num()==py::dtype::of<py::ssize_t>().num()&&a.dtype().attr("isnative").cast<bool>()&&a.ndim()==1&&(a.size()==0||a.strides(0)==sizeof(py::ssize_t));};
        const bool float32=image.dtype().num()==py::dtype::of<float>().num()&&image.dtype().attr("isnative").cast<bool>();
        if((!f64(image)&&!float32)||(image.ndim()!=2&&image.ndim()!=3)||!image.writeable()||!index(rows)||!index(columns)||rows.size()!=columns.size())return false;
        if(!f64(color)||color.ndim()!=1||!f64(alpha)||alpha.ndim()!=1||alpha.size()!=rows.size()||color.size()!=(image.ndim()==2?1:image.shape(2)))return false;
        if(!rows.attr("flags").attr("aligned").cast<bool>()||!columns.attr("flags").attr("aligned").cast<bool>()||!color.attr("flags").attr("aligned").cast<bool>()||!alpha.attr("flags").attr("aligned").cast<bool>())return false;
        if((color.size()&&color.strides(0)!=8)||(alpha.size()&&alpha.strides(0)!=8))return false;
        for(const auto* a:{&image,&rows,&columns,&color,&alpha})skimagexx::bindings::check_array_storage(*a);
        skimagexx::detail::SetColorView v;v.image=image.mutable_data();v.image_float32=float32;
        v.shape={std::size_t(image.shape(0)),std::size_t(image.shape(1)),image.ndim()==2?1:std::size_t(image.shape(2))};
        v.strides={image.strides(0),image.strides(1),image.ndim()==2?0:image.strides(2)};
        v.rows=static_cast<const std::intptr_t*>(rows.data());v.columns=static_cast<const std::intptr_t*>(columns.data());v.count=std::size_t(rows.size());
        v.color=static_cast<const double*>(color.data());v.alpha=static_cast<const double*>(alpha.data());v.integer_product=integer_product;
        py::gil_scoped_release release;return skimagexx::detail::draw_set_color(v,workers);
    },py::arg("image"),py::arg("rows"),py::arg("columns"),py::arg("color"),py::arg("alpha"),py::arg("workers"),py::arg("integer_product")=false);
}
