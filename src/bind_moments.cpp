#include "moments_internal.hpp"
#include "coords_internal.hpp"
#include "inertia_internal.hpp"
#include "skimagexx/symmetric_eigen.hpp"
#include <cfenv>
#include <limits>
#include <stdexcept>
#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>

namespace py = pybind11;
namespace bindings = skimagexx::bindings;
namespace {
constexpr auto max_rank = bindings::numpy_max_rank;
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

void check_storage(const py::array& value) {
    // A contiguous owning root already spans its full logical buffer.
    if (value.owndata() && (value.flags() & (py::array::c_style | py::array::f_style))
        && !value.base())
        return;
    bindings::check_array_storage(value);
}

template<class T> void check(const py::array& value, const py::ssize_t* shape,
                             std::size_t rank, bool aligned = true) {
    check_storage(value);
    if (std::size_t(value.ndim()) != rank ||
        !std::equal(shape, shape + rank, value.shape()) || !Array<T>::check_(value) ||
        (aligned && reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T)))
        throw py::value_error("moments input changed during allocation");
}
template<class T> Array<T> prepare(const py::array& value) {
    check_storage(value);
    const auto* array_type = py::detail::npy_api::get().PyArray_Type_;
    const bool prepared = Py_TYPE(value.ptr()) == array_type && Array<T>::check_(value)
        && bindings::is_native_byteorder(value);
    auto result = prepared ? py::reinterpret_borrow<Array<T>>(value) : Array<T>(value);
    if (reinterpret_cast<std::uintptr_t>(result.data()) % alignof(T)) {
        py::ssize_t shape[max_rank];
        const auto rank = result.ndim();
        std::copy_n(result.shape(), rank, shape);
        auto aligned = bindings::empty_c_array<T>(shape, static_cast<int>(rank));
        check<T>(result, shape, std::size_t(rank), false);
        std::memcpy(aligned.mutable_data(), result.data(), std::size_t(result.nbytes()));
        return py::reinterpret_steal<Array<T>>(aligned.release());
    }
    return result;
}
template<class T> py::tuple project(const py::array& calc, const py::array& powers,
                                    std::size_t axis, int workers) {
    auto values = prepare<T>(calc), weights = prepare<T>(powers);
    const auto rank = std::size_t(values.ndim());
    if (axis >= rank || weights.ndim() != 2 || weights.shape(0) != values.shape(axis))
        throw py::value_error("moments projection shape/axis mismatch");
    py::ssize_t shape[max_rank], output_shape[max_rank];
    std::copy_n(values.shape(), rank, shape);
    std::copy_n(shape, rank, output_shape);
    const py::ssize_t power_shape[]{weights.shape(0), weights.shape(1)};
    output_shape[axis] = power_shape[1];
    auto output = bindings::empty_c_array<T>(output_shape, static_cast<int>(rank));
    // Preparation and output allocation may run callbacks. Recheck the two
    // established typed buffers here before releasing the GIL for native reads.
    check<T>(values, shape, rank);
    check<T>(weights, power_shape, 2);
    const std::vector<std::size_t> native_shape(shape, shape + rank);
    const auto* input = values.data();
    const auto* power_data = weights.data();
    auto* result = output.mutable_data();
    unsigned status;
    {
        py::gil_scoped_release release;
        status = skimagexx::detail::moments_project(input, native_shape, power_data,
            std::size_t(power_shape[1]), axis, result, workers);
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
    const py::ssize_t output_shape[]{n,n}, diagonal_shape[]{n}, mixed_shape[]{pairs};
    auto output=bindings::empty_c_array(diagonal.dtype(),output_shape,2);
    check<T>(diagonal,diagonal_shape,1);
    check<T>(mixed,mixed_shape,1);
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
    const py::ssize_t output_shape[]{n}, input_shape[]{n,n};
    auto output=bindings::empty_c_array<T>(output_shape);
    const void* small_fields[6];
    std::vector<const void*> large_fields;
    if(count>6) large_fields.resize(static_cast<std::size_t>(count));
    auto* fields=count>6 ? large_fields.data() : small_fields;
    // Preparation/output allocation is the last callback boundary before
    // reading the established typed C buffer. The tiny solve retains the GIL.
    check<T>(values,input_shape,2);
    const auto* data=values.data();
    std::size_t field=0;
    for(py::ssize_t r=0;r<n;++r)for(py::ssize_t c=r;c<n;++c,++field)
        fields[field]=data+c*n+r;
    if(!skimagexx::symmetric_eigenvalues_into(fields,output.mutable_data(),
        1,std::size_t(n),std::is_same_v<T,float>,false))return py::none();
    return output;
}


// Powers are the frontend's private native C copy, not the external point view.
template<class T>py::object coords_reduce(const py::array& powers,int workers){
    if(powers.ndim()!=3)return py::none();
    const py::ssize_t shape[]{powers.shape(0),powers.shape(1),powers.shape(2)};
    const auto N=shape[0],D=shape[1],K=shape[2];
    if(N<1||D<1||D>3||K<2)return py::none();
    const py::ssize_t output_shape[]{K,K,K};
    auto output=bindings::empty_c_array<T>(output_shape,static_cast<int>(D));
    // One actual pointer boundary after output allocation, no FFT guards on
    // earlier NumPy-only power/intermediate arrays.
    check<T>(powers,shape,3);
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
