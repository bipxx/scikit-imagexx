#include "mesh_surface_area.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <cstring>
namespace py = pybind11;
void bind_mesh_surface_area(py::module_& module) {
    module.def("mesh_surface_area",[](const py::array& input,int profile,std::size_t buffer_elements,int workers)->py::object {
        if(profile!=1 && profile!=2)throw py::value_error("mesh area requires an audited NumPy profile");
        if(input.ndim()!=3 || input.shape(1)<3 || input.shape(2)!=3)
            throw py::value_error("mesh area requires gathered shape (F,K,3) with K at least three");
        const bool single=py::array_t<float,py::array::c_style>::check_(input);
        const bool wide=py::array_t<double,py::array::c_style>::check_(input);
        if((!single && !wide) || reinterpret_cast<std::uintptr_t>(input.data())%(single ? alignof(float) : alignof(double)))
            throw py::value_error("mesh area requires aligned native contiguous float32 or float64 storage");
        const py::dtype dtype(single ? "float32" : "float64");
        skimagexx::bindings::check_array_storage(input);
        const skimagexx::detail::MeshAreaInput view{input.data(),
            single ? skimagexx::ColocType::float32 : skimagexx::ColocType::float64,
            std::size_t(input.shape(0)),std::size_t(input.shape(1))};
        const skimagexx::ColocReductionPlan plan{profile==1 ? skimagexx::ColocProfile::numpy_1244 : skimagexx::ColocProfile::numpy_254,buffer_elements};
        skimagexx::ColocScalar result;
        try {py::gil_scoped_release release;result=skimagexx::detail::mesh_surface_area(view,plan,workers);}
        catch(const skimagexx::ColocProviderRequired&){return py::none();}
        std::array<unsigned char,8> raw{};
        if(single){const auto bits=std::uint32_t(result.bits);std::memcpy(raw.data(),&bits,4);}
        else std::memcpy(raw.data(),&result.bits,8);
        auto boxed=py::reinterpret_steal<py::object>(py::detail::npy_api::get().PyArray_Scalar_(raw.data(),dtype.ptr(),nullptr));
        if(!boxed)throw py::error_already_set();
        return boxed;
    },py::arg("actual_verts").noconvert(),py::arg("profile"),py::arg("buffer_elements"),py::arg("workers"));
}
