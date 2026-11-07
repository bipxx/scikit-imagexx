#include <skimagexx/rolling_ball.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
namespace py=pybind11;
namespace {
template<class T> void run(const py::array& image,const py::array& difference,
                          const std::vector<std::ptrdiff_t>& shape,py::array& out,bool safe,int workers) {
    skimagexx::RollingBallGeometry g{{image.shape(),image.shape()+image.ndim()},
                                   {image.strides(),image.strides()+image.ndim()},shape};
    std::vector<T> values;
    {
        py::gil_scoped_release release;
        values=skimagexx::detail::rolling_ball_from_difference(
            static_cast<const T*>(image.data()),g,static_cast<const T*>(difference.data()),safe,workers);
    }
    if(values.size()!=std::size_t(out.size()))throw py::value_error("rolling_ball output size mismatch");
    if(!values.empty())std::memcpy(out.mutable_data(),values.data(),values.size()*sizeof(T));
}
}
void bind_rolling_ball(py::module_& m) {
    m.def("rolling_ball_apply",[](py::array image,py::array difference,
           const std::vector<std::ptrdiff_t>& shape,py::array out,bool safe,int workers) {
        if(image.dtype().has_fields()||!image.dtype().attr("isnative").cast<bool>()||
           !(image.dtype().num()==py::dtype::of<float>().num()||image.dtype().num()==py::dtype::of<double>().num()))
            throw py::type_error("rolling_ball prepared input must be native float32/float64");
        if(image.dtype().num()!=difference.dtype().num()||image.dtype().num()!=out.dtype().num()||
           !difference.dtype().attr("isnative").cast<bool>()||!out.dtype().attr("isnative").cast<bool>()||
           !(difference.flags()&py::array::c_style)||difference.ndim()!=1||
           !(out.flags()&py::array::c_style)||!out.writeable()||!out.owndata()||out.size()!=image.size())
            throw py::value_error("rolling_ball requires matching prepared difference and owning dense output");
        std::size_t count=1;
        if(shape.size()!=std::size_t(image.ndim())||shape.empty())throw py::value_error("rolling_ball kernel rank mismatch");
        for(auto extent:shape) {
            if(extent<=0||std::size_t(extent)>std::size_t(PY_SSIZE_T_MAX)/count)
                throw py::value_error("rolling_ball kernel shape invalid");
            count*=std::size_t(extent);
        }
        if(count!=std::size_t(difference.size()))throw py::value_error("rolling_ball kernel span mismatch");
        skimagexx::bindings::check_array_storage(image);
        skimagexx::bindings::check_array_storage(difference);
        skimagexx::bindings::check_array_storage(out);
        if(reinterpret_cast<std::uintptr_t>(difference.data())%std::size_t(difference.itemsize()))
            throw py::value_error("rolling_ball difference must be aligned");
        const auto overlaps=[&](const py::array& a) {
            if(!a.size()||!out.size())return false;
            auto low=reinterpret_cast<std::uintptr_t>(a.data()),high=low;
            // check_array_storage already authenticated these bounded spans.
            for(py::ssize_t axis=0;axis<a.ndim();++axis) {
                const auto stride=a.strides(axis),count=a.shape(axis)-1;
                if(stride<0)low-=(std::uintptr_t(-(stride+1))+1)*std::uintptr_t(count);
                else high+=std::uintptr_t(stride)*std::uintptr_t(count);
            }
            high+=std::size_t(a.itemsize());
            const auto target=reinterpret_cast<std::uintptr_t>(out.data());
            return low<target+out.nbytes()&&target<high;
        };
        if(overlaps(image)||overlaps(difference))throw py::value_error("rolling_ball output overlaps input");
        if(image.dtype().num()==py::dtype::of<float>().num())run<float>(image,difference,shape,out,safe,workers);
        else run<double>(image,difference,shape,out,safe,workers);
    },py::arg("image").noconvert(),py::arg("difference").noconvert(),py::arg("kernel_shape"),
       py::arg("output").noconvert(),py::arg("nansafe"),py::arg("workers"));
}
