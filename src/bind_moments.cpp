#include "moments_internal.hpp"
#include "coords_internal.hpp"
#include "inertia_internal.hpp"
#include "skimagexx/symmetric_eigen.hpp"
#include <cfenv>
#include <limits>
#include <stdexcept>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>

namespace py = pybind11;
namespace {
using Shape = std::vector<py::ssize_t>;
Shape dimensions(const py::array& value) {
    return Shape(value.shape(), value.shape() + value.ndim());
}
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
template<class T> void check(const py::array& value, const Shape& shape, bool aligned = true) {
    skimagexx::bindings::check_array_storage(value);
    if (dimensions(value) != shape || !Array<T>::check_(value) ||
        (aligned && reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T)))
        throw py::value_error("moments input changed during allocation");
}
template<class T> Array<T> prepare(const py::array& value) {
    skimagexx::bindings::check_array_storage(value);
    Array<T> result(value);
    if (reinterpret_cast<std::uintptr_t>(result.data()) % alignof(T)) {
        const auto shape = dimensions(result);
        py::array_t<T> aligned(shape);
        check<T>(result, shape, false);
        std::memcpy(aligned.mutable_data(), result.data(), std::size_t(result.nbytes()));
        return Array<T>(aligned);
    }
    return result;
}
template<class T> py::tuple project(const py::array& calc, const py::array& powers,
                                    std::size_t axis, int workers) {
    auto values = prepare<T>(calc), weights = prepare<T>(powers);
    const auto shape = dimensions(values), power_shape = dimensions(weights);
    if (axis >= shape.size() || power_shape.size() != 2 || power_shape[0] != shape[axis])
        throw py::value_error("moments projection shape/axis mismatch");
    auto output_shape = shape;
    output_shape[axis] = power_shape[1];
    py::array_t<T> output(output_shape);
    // Preparation and output allocation may run callbacks. Recheck the two
    // established typed buffers here before releasing the GIL for native reads.
    check<T>(values, shape);
    check<T>(weights, power_shape);
    const std::vector<std::size_t> native_shape(shape.begin(), shape.end());
    unsigned status;
    {
        py::gil_scoped_release release;
        status = skimagexx::detail::moments_project(values.data(), native_shape, weights.data(),
            std::size_t(power_shape[1]), axis, output.mutable_data(), workers);
    }
    return py::make_tuple(output, status);
}

// The tiny assembly retains the GIL; scalar conversions precede the final
// allocation/buffer boundary. Exceptional arithmetic is left to the literal
// NumPy path so its warning/error order is not reconstructed here.
template<class T> py::object inertia(const py::array& diagonal,const py::array& mixed,
                                     const py::object& sum,const py::object& zero) {
    if(diagonal.ndim()!=1 || mixed.ndim()!=1)
        throw py::value_error("inertia assembly requires one-dimensional inputs");
    const auto n=diagonal.shape(0);
    if(n && n>std::numeric_limits<py::ssize_t>::max()/n)
        throw py::value_error("inertia tensor size overflow");
    const auto pairs=n*(n-1)/2;
    const T total=sum.cast<T>(),mass=zero.cast<T>();
    py::array output(diagonal.dtype(),Shape{n,n});
    check<T>(diagonal,Shape{n});
    check<T>(mixed,Shape{pairs});
    auto* result=static_cast<T*>(output.mutable_data());
    // Capture only native arithmetic; the established buffers remain live.
    std::fenv_t saved;
    if(std::feholdexcept(&saved))throw std::runtime_error("cannot capture inertia floating environment");
    skimagexx::detail::inertia_tensor(static_cast<const T*>(diagonal.data()),
        static_cast<const T*>(mixed.data()),std::size_t(n),total,mass,
        result);
    const int flags=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
    std::fesetenv(&saved);
    if(flags)return py::none();
    return output;
}


// NumPy eigvalsh's default UPLO is lower, including nonsymmetric supplied T.
// Reuse the existing finite symmetric solver; no new eigenvalue math.
template<class T> py::object inertia_eigenvalues(const py::array& input) {
    if(input.ndim()!=2 || input.shape(0)!=input.shape(1) ||
       input.shape(0)<1 || input.shape(0)>64)return py::none();
    auto values=prepare<T>(input);
    const auto n=values.shape(0),count=n*(n+1)/2;
    py::array_t<T> output(n);
    std::vector<T> packed(static_cast<std::size_t>(count));
    std::vector<const void*> fields(static_cast<std::size_t>(count));
    // Preparation/output allocation is the last callback boundary before
    // reading the established typed C buffer. The tiny solve retains the GIL.
    check<T>(values,Shape{n,n});
    const auto* data=values.data();
    std::size_t field=0;
    for(py::ssize_t r=0;r<n;++r)for(py::ssize_t c=r;c<n;++c,++field) {
        packed[field]=data[c*n+r];fields[field]=packed.data()+field;
    }
    if(!skimagexx::symmetric_eigenvalues_into(fields.data(),output.mutable_data(),
        1,std::size_t(n),std::is_same_v<T,float>,false))return py::none();
    return output;
}


// Powers are the frontend's private native C copy, not the external point view.
template<class T>py::object coords_reduce(const py::array& powers,int workers){
    if(powers.ndim()!=3)return py::none();
    const auto shape=dimensions(powers);
    const auto N=shape[0],D=shape[1],K=shape[2];
    if(N<1||D<1||D>3||K<2)return py::none();
    py::array_t<T> output(Shape(std::size_t(D),K));
    // One actual pointer boundary after output allocation, no FFT guards on
    // earlier NumPy-only power/intermediate arrays.
    check<T>(powers,shape);
    const auto* input=static_cast<const T*>(powers.data());auto* result=output.mutable_data();
    unsigned flags;
    {
        py::gil_scoped_release release;
        flags=skimagexx::detail::moments_coords_reduce(input,
            std::size_t(N),std::size_t(D),std::size_t(K),result,workers);
    }
    if(flags)return py::none();
    return output;
}

}

void bind_moments(py::module_& module) {
    module.def("moments_coords_reduce",[](const py::array& powers,int workers)->py::object {
        if(powers.dtype().char_()=='f')return coords_reduce<float>(powers,workers);
        if(powers.dtype().char_()=='d')return coords_reduce<double>(powers,workers);
        return py::none();
    },py::arg("powers").noconvert(),py::arg("workers"));
    module.def("inertia_eigvals",[](const py::array& input)->py::object {
        if(input.dtype().char_()=='f')return inertia_eigenvalues<float>(input);
        if(input.dtype().char_()=='d')return inertia_eigenvalues<double>(input);
        return py::none();
    },py::arg("T").noconvert());
    module.def("inertia_tensor",[](const py::array& diagonal,const py::array& mixed,
                                    const py::object& total,const py::object& mass)->py::object {
        if(py::array_t<float,0>::check_(diagonal) && py::array_t<float,0>::check_(mixed))
            return inertia<float>(diagonal,mixed,total,mass);
        if(py::array_t<double,0>::check_(diagonal) && py::array_t<double,0>::check_(mixed))
            return inertia<double>(diagonal,mixed,total,mass);
        throw py::type_error("inertia assembly requires matching native float32 or float64");
    },py::arg("diagonal").noconvert(),py::arg("mixed").noconvert(),py::arg("total"),py::arg("mass"));
    module.def("moments_project", [](const py::array& calc, const py::array& powers,
                                     std::size_t axis, int workers) {
        // The source frontend chooses the common dtype once from actual
        // array descriptors, after its literal coordinate/power promotion.
        const auto kind = calc.dtype().kind();
        const auto bytes = calc.itemsize();
        if (kind != powers.dtype().kind() || bytes != powers.itemsize())
            throw py::type_error("moments projection requires matching component types");
        if (kind == 'f' && bytes == 4) return project<float>(calc, powers, axis, workers);
        if (kind == 'f' && bytes == 8) return project<double>(calc, powers, axis, workers);
        if (kind == 'c' && bytes == 8) return project<std::complex<float>>(calc, powers, axis, workers);
        if (kind == 'c' && bytes == 16) return project<std::complex<double>>(calc, powers, axis, workers);
        PyErr_SetString(PyExc_NotImplementedError, "moments projection supports standard float and complex types");
        throw py::error_already_set();
    }, py::arg("calc").noconvert(), py::arg("powers").noconvert(),
       py::arg("axis"), py::arg("workers"));
}
