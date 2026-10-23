#include "skimagexx/footprint_util.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_footprint_util(py::module_& module) {
    const auto numpy=py::module_::import("numpy");
    const py::object reference_aware_pad=numpy.attr("pad");
    module.def("footprint_mirror",[](py::array image)->py::object {
        py::tuple indices(image.ndim());
        for(py::ssize_t axis=0;axis<image.ndim();++axis)indices[axis]=py::slice(py::none(),py::none(),py::int_(-1));
        return image.attr("__getitem__")(indices);
    });
    module.def("footprint_pad",[numpy,reference_aware_pad](py::array image,const std::vector<std::pair<std::size_t,std::size_t>>& padding)->py::array {
        if(!image.ndim())throw py::type_error("`pad_width` must be of integral type.");
        if(padding.size()!=std::size_t(image.ndim()))throw py::value_error("one footprint pad pair per axis required");
        const auto kind=image.dtype().kind();
        if(image.dtype().attr("hasobject").cast<bool>() || std::string("buifcmMSUV").find(kind)==std::string::npos) {
            // Pointer-bearing records require dtype-specific reference counts,
            // copy hooks and scalar assignment; never copy their raw pointers.
            return reference_aware_pad(image,py::cast(padding)).cast<py::array>();
        }
        const bool fortran=bool(image.flags()&py::array::f_style) && !bool(image.flags()&py::array::c_style);
        std::vector<py::ssize_t> shape;skimagexx::FootprintPadView view;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            const auto [left,right]=padding[std::size_t(axis)];
            if(left>1 || right>1 || left+right!=std::size_t(image.shape(axis)%2==0))throw py::value_error("invalid footprint padding");
            shape.push_back(image.shape(axis)+py::ssize_t(left+right));
            view.shape.push_back(std::size_t(image.shape(axis)));view.before.push_back(left);view.input_strides.push_back(image.strides(axis));
        }
        auto result=numpy.attr("empty")(py::cast(shape),py::arg("dtype")=image.dtype(),py::arg("order")=fortran ? "F":"C").cast<py::array>();
        auto zero=numpy.attr("empty")(py::tuple(),py::arg("dtype")=image.dtype()).cast<py::array>();
        // np.pad expands its default constant through np.asarray(0), so
        // assignment receives a NumPy integer scalar (significant for void).
        const auto scalar_zero=numpy.attr("asarray")(py::int_(0)).attr("__getitem__")(py::tuple());
        zero.attr("__setitem__")(py::tuple(),scalar_zero);
        view.input=image.data();view.output=result.mutable_data();view.zero=zero.data();view.itemsize=std::size_t(image.itemsize());view.fortran=fortran;
        for(py::ssize_t axis=0;axis<result.ndim();++axis)view.output_strides.push_back(result.strides(axis));
        {py::gil_scoped_release release;skimagexx::footprint_pad_into(view);}
        return result;
    });
}
