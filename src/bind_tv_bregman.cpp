#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "skimagexx/tv_bregman.hpp"
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
// Bounds must come from an actual NumPy allocation, never an arbitrary
// __array_interface__ provider or unchecked as_strided base.
void allocation_bounds(const py::array& value) {
    py::array owner = value;
    for (unsigned depth = 0; ; ++depth) {
        if (depth > 64) throw py::value_error("array base chain is too deep");
        auto base = owner.base();
        if (!base || base.is_none()) break;
        if (!py::isinstance<py::array>(base))
            throw py::value_error("array storage requires a NumPy allocation owner");
        owner = py::reinterpret_borrow<py::array>(base);
    }
    if (!owner.owndata() || !(owner.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("array allocation extent is uncertified");
    auto begin = reinterpret_cast<std::uintptr_t>(owner.data());
    auto pointer = reinterpret_cast<std::uintptr_t>(value.data());
    auto bytes = static_cast<std::size_t>(owner.nbytes());
    auto size = static_cast<std::size_t>(value.nbytes());
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin ||
        pointer < begin || pointer > begin + bytes || size > begin + bytes - pointer)
        throw py::value_error("array exceeds its allocation extent");
}
py::array prepared(py::handle object, const char* name) {
    if (!py::isinstance<py::array>(object)) throw py::type_error(std::string(name) + " must be a NumPy array");
    auto a = py::reinterpret_borrow<py::array>(object);
    if (a.ndim() != 3 || !(a.flags() & py::array::c_style) || !a.writeable())
        throw py::value_error(std::string(name) + " must be writable dense C rank3");
    if (!a.dtype().is(py::dtype::of<float>()) && !a.dtype().is(py::dtype::of<double>()))
        throw py::type_error(std::string(name) + " must have native float32 or float64 dtype");
    if (reinterpret_cast<std::uintptr_t>(a.data()) % static_cast<std::size_t>(a.itemsize()))
        throw py::value_error(std::string(name) + " must be naturally aligned");
    allocation_bounds(a);
    return a;
}
template<class T> py::object run(const py::array& image, const py::array& out,
    double weight, int maximum, double eps, bool isotropic, bool tracing) {
    auto rows = static_cast<std::size_t>(image.shape(0));
    auto cols = static_cast<std::size_t>(image.shape(1));
    auto channels = static_cast<std::size_t>(image.shape(2));
    if (rows > static_cast<std::size_t>(PY_SSIZE_T_MAX) - 2 ||
        cols > static_cast<std::size_t>(PY_SSIZE_T_MAX) - 2 ||
        out.shape(0) != static_cast<py::ssize_t>(rows + 2) ||
        out.shape(1) != static_cast<py::ssize_t>(cols + 2) || out.shape(2) != image.shape(2))
        throw py::value_error("out must have padded shape (rows+2,cols+2,channels)");
    const auto count = static_cast<std::size_t>(out.size());
    // Uninitialized aux storage is completely overwritten by the kernel's
    // copy-before-reset stage. No Python allocation provider is substituted.
    std::unique_ptr<T[]> dx(new T[count]), dy(new T[count]), bx(new T[count]), by(new T[count]);
    std::vector<skimagexx::TvBregmanTrace> trace;
    if (tracing && maximum > 0) trace.resize(static_cast<std::size_t>(maximum));
    skimagexx::TvBregmanImage input{image.data(), sizeof(T) == 4 ? skimagexx::TvBregmanType::float32 : skimagexx::TvBregmanType::float64,
        rows, cols, channels, static_cast<std::size_t>(image.size())};
    skimagexx::TvBregmanWorkspace workspace{{const_cast<void*>(out.data()),count},{dx.get(),count},{dy.get(),count},{bx.get(),count},{by.get(),count}};
    skimagexx::TvBregmanParameters params{weight,maximum,eps,isotropic};
    skimagexx::TvBregmanResult result;
    {
        py::gil_scoped_release release;
        result = skimagexx::tv_bregman_into(input,workspace,params,
            trace.empty() ? nullptr : trace.data(),trace.size());
    }
    if (!tracing) return py::none();
    py::list steps;
    for (std::size_t i=0;i<result.iterations;++i)
        steps.append(py::make_tuple(trace[i].rmse_bits,trace[i].stopped));
    py::dict answer;
    answer["iterations"] = result.iterations;
    answer["rmse"] = result.rmse;
    answer["initialization_workers"] = result.initialization_workers;
    answer["trace"] = steps;
    return answer;
}
}

void bind_tv_bregman(py::module_& m) {
    m.attr("tv_bregman_interface") = skimagexx::tv_bregman_interface;
    m.attr("tv_bregman_capabilities") = skimagexx::tv_bregman_capabilities;
    m.attr("tv_bregman_parallel_sweep") = skimagexx::tv_bregman_parallel_sweep;
    m.def("tv_bregman_run", [](py::object image_obj, py::object weight_obj,
        py::object maximum_obj, py::object eps_obj, py::object iso_obj,
        py::object out_obj, bool tracing) -> py::object {
        auto image = prepared(image_obj,"image");
        const double weight = py::cast<double>(weight_obj);
        py::object index = py::reinterpret_steal<py::object>(PyNumber_Index(maximum_obj.ptr()));
        if (!index) throw py::error_already_set();
        const long long integer = PyLong_AsLongLong(index.ptr());
        if (PyErr_Occurred()) throw py::error_already_set();
        if (integer < INT_MIN || integer > INT_MAX) throw std::overflow_error("max_num_iter does not fit C int");
        const double eps = py::cast<double>(eps_obj);
        py::object iso = py::reinterpret_steal<py::object>(PyNumber_Long(iso_obj.ptr()));
        if (!iso) throw py::error_already_set();
        const long character = PyLong_AsLong(iso.ptr());
        if (PyErr_Occurred()) throw py::error_already_set();
        if (character < -128 || character > 127) throw std::overflow_error("isotropic does not fit signed char");
        auto out = prepared(out_obj,"out");
        if (!image.dtype().is(out.dtype())) throw py::type_error("out dtype must match image");
        if (image.itemsize() == 4) return run<float>(image,out,weight,static_cast<int>(integer),eps,character!=0,tracing);
        return run<double>(image,out,weight,static_cast<int>(integer),eps,character!=0,tracing);
    }, py::arg("image"),py::arg("weight"),py::arg("max_num_iter"),py::arg("eps"),
       py::arg("isotropic"),py::arg("out"),py::kw_only(),py::arg("trace")=false);
}
