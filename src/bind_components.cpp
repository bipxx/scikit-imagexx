#include "skimagexx/components.hpp"
#include "skimagexx/image.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <limits>
namespace py=pybind11;
namespace {
skimagexx::ComponentsView view(const py::array& image,bool boolean) {
    skimagexx::ComponentsView result;const auto kind=image.dtype().kind();const auto size=image.itemsize();
    if(boolean) {
        if(kind!='b')throw py::type_error("boolean component input required");
        result.type=skimagexx::PixelType::boolean;
    } else {
        using P=skimagexx::PixelType;
        if(kind=='i')switch(size){case 1:result.type=P::int8;break;case 2:result.type=P::int16;break;case 4:result.type=P::int32;break;case 8:result.type=P::int64;break;default:throw py::type_error("unsupported component input dtype");}
        else if(kind=='u')switch(size){case 1:result.type=P::uint8;break;case 2:result.type=P::uint16;break;case 4:result.type=P::uint32;break;case 8:result.type=P::uint64;break;default:throw py::type_error("unsupported component input dtype");}
        else if(kind=='f' && size==2){result.type=P::uint16;result.half=true;}
        else if(kind=='f' && size==4)result.type=P::float32;
        else if(kind=='f' && size==8)result.type=P::float64;
        else throw py::type_error("unsupported component input dtype");
    }
    result.data=image.data();result.swap=!image.dtype().attr("isnative").cast<bool>();
    result.shape.assign(image.shape(),image.shape()+image.ndim());
    result.strides.assign(image.strides(),image.strides()+image.ndim());return result;
}
std::intptr_t background(py::handle value) {
    if(!PyNumber_Check(value.ptr()) || PyComplex_Check(value.ptr()))throw py::type_error("an integer is required");
    auto number=py::reinterpret_steal<py::object>(PyNumber_Long(value.ptr()));
    if(!number)throw py::error_already_set();
    if constexpr(sizeof(long)==sizeof(std::intptr_t)) {
        const auto result=PyLong_AsLong(number.ptr());
        if(PyErr_Occurred())throw py::error_already_set();return std::intptr_t(result);
    }
    int overflow=0;const auto wide=PyLong_AsLongLongAndOverflow(number.ptr(),&overflow);
    if(overflow || wide<std::numeric_limits<std::intptr_t>::min() || wide>std::numeric_limits<std::intptr_t>::max()) {
        PyErr_Clear();PyErr_SetString(PyExc_OverflowError,"int too big to convert");throw py::error_already_set();
    }
    if(PyErr_Occurred())throw py::error_already_set();return std::intptr_t(wide);
}
}
void bind_components(py::module_& module) {
    module.attr("components_interface_version")=skimagexx::components_interface_version;
    module.def("components_background",&background);
    module.def("components_prepare_index",[](const py::array& image) {
        auto plan=view(image,false);py::array_t<std::intptr_t> output(std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        skimagexx::ComponentsConversionInfo info;
        {py::gil_scoped_release release;info=skimagexx::components_prepare_index_into(plan,output.mutable_data());}
        return py::make_tuple(output,info.invalid_cast);
    });
    module.def("components_prepare_bool",[](const py::array& image) {
        auto plan=view(image,true);py::array_t<std::uint8_t> output(std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        {py::gil_scoped_release release;skimagexx::components_prepare_bool_into(plan,output.mutable_data());}return output;
    });
    module.def("components_label",[](const py::array& image,std::intptr_t bg,int connectivity,bool boolean,py::dtype dtype) {
        const auto expected=boolean ? py::dtype::of<std::uint8_t>() : py::dtype::of<std::intptr_t>();
        if(!image.dtype().is(expected) || !(image.flags()&py::array::c_style) || !image.attr("flags").attr("aligned").cast<bool>())throw py::value_error("prepared native aligned C component input required");
        const bool wide=dtype.kind()=='i' && dtype.itemsize()==sizeof(std::intptr_t) && dtype.attr("isnative").cast<bool>();
        if(!wide && (!boolean || !dtype.is(py::dtype::of<std::int32_t>())))throw py::value_error("invalid component output dtype");
        if(boolean && bg!=0 && bg!=1)throw py::value_error("boolean component background must be0or1");
        std::vector<std::size_t> shape(image.shape(),image.shape()+image.ndim());
        py::array output(dtype,std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));std::size_t count;
        {py::gil_scoped_release release;
            if(boolean)count=skimagexx::components_label_bool_into(static_cast<const std::uint8_t*>(image.data()),output.mutable_data(),wide,shape,std::uint8_t(bg),connectivity);
            else count=skimagexx::components_label_index_into(static_cast<const std::intptr_t*>(image.data()),static_cast<std::intptr_t*>(output.mutable_data()),shape,bg,connectivity);
        }return py::make_tuple(output,py::int_(count));
    });
}
#ifdef SKIMAGEXX_COMPONENTS_STANDALONE
PYBIND11_MODULE(_components_core,module) {
    bind_components(module);module.def("set_num_threads",&skimagexx::set_num_threads);
    module.def("get_num_threads",&skimagexx::get_num_threads);module.def("has_avx2",&skimagexx::has_avx2);
}
#endif
